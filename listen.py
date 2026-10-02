"""Hearing: voice activity, end-of-turn detection and streaming speech-to-text."""

import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from moonshine_voice import ModelArch, Transcriber, get_model_for_language

from models import model_path

SAMPLE_RATE = 16000
# Silero VAD works on 512-sample frames at 16 kHz, so the whole listener ticks at 32 ms.
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE

STT_SIZES = {
    "tiny": ModelArch.TINY_STREAMING,
    "small": ModelArch.SMALL_STREAMING,
    "medium": ModelArch.MEDIUM_STREAMING,
}


def onnx_session(filename: str, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path(filename)), sess_options=options, providers=["CPUExecutionProvider"])


class VoiceActivity:
    """Silero VAD v6: the probability that a 32 ms frame contains speech."""

    CONTEXT_SAMPLES = 64

    def __init__(self):
        self.session = onnx_session("silero_vad.onnx", threads=1)
        self.sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        audio = np.concatenate([self.context, frame])[None, :]
        probability, self.state = self.session.run(
            None, {"input": audio, "state": self.state, "sr": self.sample_rate}
        )
        self.context = frame[-self.CONTEXT_SAMPLES :]
        return float(probability[0, 0])


class TurnDetector:
    """Smart Turn v3.2: hears from intonation whether a pause ends the turn or is just thinking."""

    WINDOW_SAMPLES = 8 * SAMPLE_RATE

    def __init__(self, threads: int):
        self.session = onnx_session("smart-turn-v3.2-cpu.onnx", threads)
        self.mel_filters = slaney_mel_filters(n_fft=400, n_mels=80)
        # Periodic Hann window, the one Whisper's feature extractor uses.
        self.window = np.hanning(401)[:-1]

    def probability_finished(self, audio: np.ndarray) -> float:
        features = self.whisper_features(audio)[None]
        (probability,) = self.session.run(None, {"input_features": features})[0]
        return float(probability[0])

    def whisper_features(self, audio: np.ndarray) -> np.ndarray:
        """Whisper log-mel features over the last 8 seconds, matching transformers' WhisperFeatureExtractor."""
        audio = audio[-self.WINDOW_SAMPLES :].astype(np.float64)
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        audio = np.pad(audio, (0, self.WINDOW_SAMPLES - len(audio)))

        padded = np.pad(audio, 200, mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160]
        power = np.abs(np.fft.rfft(frames * self.window, axis=1)) ** 2
        log_mel = np.log10(np.maximum(self.mel_filters @ power.T, 1e-10))[:, :-1]
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        return ((log_mel + 4.0) / 4.0).astype(np.float32)


def slaney_mel_filters(n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz):
        return np.where(hz < 1000, 3 * hz / 200, 15 + np.log(np.maximum(hz, 1e-10) / 1000) * 27 / np.log(6.4))

    def mel_to_hz(mel):
        return np.where(mel < 15, 200 * mel / 3, 1000 * np.exp((mel - 15) * np.log(6.4) / 27))

    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, n_fft // 2 + 1)
    hz_points = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), n_mels + 2))
    slopes = hz_points[None, :] - fft_freqs[:, None]
    widths = np.diff(hz_points)
    rising = -slopes[:, :-2] / widths[:-1]
    falling = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0, np.minimum(rising, falling)) * (2.0 / (hz_points[2:] - hz_points[:-2]))
    return filters.T


class SpeechToText:
    """Moonshine v2 streaming: transcribes while you talk, so the text is ready the moment you stop."""

    def __init__(self, language: str, size: str, update_interval: float = 0.3):
        path, arch = get_model_for_language(language, STT_SIZES[size])
        self.transcriber = Transcriber(path, arch)
        self.update_interval = update_interval
        # A transcription pass can take a few hundred ms, so it runs on its own thread
        # instead of holding up voice activity and end-of-turn detection.
        self.jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self.run, daemon=True).start()

    def start(self):
        self.jobs.put(("start", None))

    def add(self, frame: np.ndarray):
        self.jobs.put(("audio", frame))

    def finish(self) -> str:
        """Waits for the audio still in the queue to be transcribed and returns the whole turn."""
        text = Future()
        self.jobs.put(("finish", text))
        return text.result()

    def run(self):
        stream = None
        while True:
            job, value = self.jobs.get()
            if job == "start":
                stream = self.transcriber.create_stream(update_interval=self.update_interval)
                stream.start()
            elif job == "audio":
                stream.add_audio(value.tolist(), SAMPLE_RATE)
            elif job == "finish":
                transcript = stream.stop()
                stream.close()
                value.set_result(" ".join(line.text.strip() for line in transcript.lines if line.text.strip()))


@dataclass
class Turn:
    text: str
    # How long the listener waited in silence before deciding the turn was over.
    wait_ms: float
    smart_turn_ms: float
    stt_ms: float
    finished_probability: float
    speech_end: float


@dataclass
class ListenerSettings:
    start_threshold: float = 0.5
    # Lower threshold once speaking, as Silero recommends, so soft word endings don't count as silence.
    continue_threshold: float = 0.35
    start_ms: float = 64
    # Speech needed to interrupt the assistant: long enough to ignore coughs and leftover echo.
    barge_in_ms: float = 250
    # Silence before Smart Turn is asked whether the turn is over.
    pause_ms: float = 200
    # Silence after which the turn ends even if Smart Turn thinks the user is mid-sentence.
    max_pause_ms: float = 1800
    finished_threshold: float = 0.5
    preroll_ms: float = 320


class Listener:
    """Turns a stream of 32 ms microphone frames into user turns, on its own thread.

    Callbacks run on the listener thread: on_speech_start() when someone starts talking,
    on_turn(Turn) when they are done.
    """

    def __init__(self, stt: SpeechToText, turn_detector: TurnDetector, settings: ListenerSettings):
        self.vad = VoiceActivity()
        self.stt = stt
        self.turn_detector = turn_detector
        self.settings = settings
        self.on_speech_start = lambda: None
        self.on_turn = lambda turn: None
        # Set by the conversation while the assistant's voice is playing.
        self.assistant_speaking = False
        self.barge_in = True

        self.frames: queue.Queue = queue.Queue()
        self.preroll: deque = deque(maxlen=int(settings.preroll_ms / FRAME_MS))
        self.in_turn = False
        self.speech_run = 0
        self.silence_run = 0
        self.pause_checked = False
        self.turn_audio: list[np.ndarray] = []
        self.last_speech_at = 0.0
        self.smart_turn_ms = 0.0
        self.finished_probability = 0.0
        threading.Thread(target=self.run, daemon=True).start()

    def push(self, frame: np.ndarray):
        self.frames.put((frame, time.monotonic()))

    def reset(self):
        """Forgets any half-heard turn, for when the client goes away."""
        self.frames.put((None, time.monotonic()))

    def run(self):
        while True:
            frame, arrived_at = self.frames.get()
            if frame is None:
                self.forget_turn()
            else:
                self.process(frame, arrived_at)

    def forget_turn(self):
        if self.in_turn:
            self.stt.finish()
        self.clear_turn()
        self.preroll.clear()
        self.assistant_speaking = False

    def process(self, frame: np.ndarray, arrived_at: float):
        if self.assistant_speaking and not self.barge_in and not self.in_turn:
            return
        probability = self.vad(frame)
        if not self.in_turn:
            self.wait_for_speech(frame, probability, arrived_at)
        else:
            self.follow_turn(frame, probability, arrived_at)

    def wait_for_speech(self, frame: np.ndarray, probability: float, arrived_at: float):
        self.preroll.append(frame)
        self.speech_run = self.speech_run + 1 if probability >= self.settings.start_threshold else 0
        needed_ms = self.settings.barge_in_ms if self.assistant_speaking else self.settings.start_ms
        if self.speech_run * FRAME_MS < needed_ms:
            return
        self.in_turn = True
        self.silence_run = 0
        self.pause_checked = False
        self.last_speech_at = arrived_at
        self.turn_audio = list(self.preroll)
        self.stt.start()
        for buffered in self.preroll:
            self.stt.add(buffered)
        self.preroll.clear()
        self.on_speech_start()

    def follow_turn(self, frame: np.ndarray, probability: float, arrived_at: float):
        self.turn_audio.append(frame)
        self.stt.add(frame)
        if probability >= self.settings.continue_threshold:
            self.silence_run = 0
            self.pause_checked = False
            self.last_speech_at = arrived_at
            return

        self.silence_run += 1
        silence_ms = self.silence_run * FRAME_MS
        if not self.pause_checked and silence_ms >= self.settings.pause_ms:
            self.pause_checked = True
            started = time.monotonic()
            self.finished_probability = self.turn_detector.probability_finished(np.concatenate(self.turn_audio))
            self.smart_turn_ms = (time.monotonic() - started) * 1000
            if self.finished_probability >= self.settings.finished_threshold:
                self.end_turn()
        elif silence_ms >= self.settings.max_pause_ms:
            self.end_turn()

    def end_turn(self):
        decided_at = time.monotonic()
        text = self.stt.finish()
        stt_ms = (time.monotonic() - decided_at) * 1000
        self.clear_turn()
        self.on_turn(
            Turn(
                text=text,
                wait_ms=(decided_at - self.last_speech_at) * 1000,
                smart_turn_ms=self.smart_turn_ms,
                stt_ms=stt_ms,
                finished_probability=self.finished_probability,
                speech_end=self.last_speech_at,
            )
        )

    def clear_turn(self):
        self.in_turn = False
        self.speech_run = 0
        self.turn_audio = []
        self.vad.reset()
