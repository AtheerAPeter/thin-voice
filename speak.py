"""Speaking: interchangeable on-device voices behind one small interface.

Every voice turns one piece of text into float32 mono audio chunks at `sample_rate`,
and stops early when `stop` is set. Only the chosen engine is imported, so the others
cost no memory.
"""

import threading
from collections.abc import Iterator

import numpy as np

# Spoken once at startup: it warms the engine up and tells the face whether the voice is high or low.
SAMPLE_SENTENCES = {
    "en": "Hello, it is nice to meet you. How can I help you today?",
    "de": "Hallo, schön dich kennenzulernen. Wie kann ich dir heute helfen?",
}

DEFAULT_VOICES = {
    "moonshine": {"en": "kokoro_af_heart", "de": "piper_de_DE-thorsten-medium"},
    # Pocket TTS takes its accent from the voice sample, so German needs a native German speaker.
    "pocket": {"en": "alba", "de": "juergen"},
    "supertonic": {"en": "F1", "de": "F1"},
}


class MoonshineVoice:
    """Kokoro-82M or Piper through Moonshine's C++ runtime: no PyTorch, small and quick."""

    LANGUAGE_TAGS = {"en": "en-us", "de": "de-de"}

    def __init__(self, language: str, voice: str):
        from moonshine_voice import TextToSpeech

        self.tts = TextToSpeech().language(self.LANGUAGE_TAGS[language]).voice(voice).load()
        _, self.sample_rate = self.tts.synthesize("Ready.")

    def stream(self, text: str, stop: threading.Event) -> Iterator[np.ndarray]:
        samples, _ = self.tts.synthesize(text)
        yield np.asarray(samples, dtype=np.float32)


class PocketVoice:
    """Kyutai Pocket TTS: streams audio out while it is still generating, and can clone a voice."""

    LANGUAGES = {"en": "english", "de": "german"}

    def __init__(self, language: str, voice: str, threads: int):
        import torch
        from pocket_tts import TTSModel

        torch.set_num_threads(threads)
        self.model = TTSModel.load_model(language=self.LANGUAGES[language])
        self.voice_state = self.model.get_state_for_audio_prompt(voice)
        self.sample_rate = self.model.sample_rate

    def stream(self, text: str, stop: threading.Event) -> Iterator[np.ndarray]:
        for chunk in self.model.generate_audio_stream(self.voice_state, text, stop=stop):
            yield chunk.numpy().astype(np.float32)


class SupertonicVoice:
    """Supertonic 3 (archived September 2026): renders a whole piece at once, very fast."""

    def __init__(self, language: str, voice: str, threads: int, steps: int = 5):
        from supertonic import TTS

        self.tts = TTS(intra_op_num_threads=threads, inter_op_num_threads=1)
        self.style = self.tts.get_voice_style(voice)
        self.language = language
        self.steps = steps
        self.sample_rate = self.tts.sample_rate

    def stream(self, text: str, stop: threading.Event) -> Iterator[np.ndarray]:
        audio, _ = self.tts.synthesize(text, voice_style=self.style, lang=self.language, total_steps=self.steps)
        yield audio[0].astype(np.float32)


def load_voice(engine: str, language: str, voice: str | None, threads: int):
    voice = voice or DEFAULT_VOICES[engine][language]
    if engine == "moonshine":
        loaded = MoonshineVoice(language, voice)
    elif engine == "pocket":
        loaded = PocketVoice(language, voice, threads)
    elif engine == "supertonic":
        loaded = SupertonicVoice(language, voice, threads)
    else:
        raise ValueError(f"Unknown voice engine: {engine}")
    sample = np.concatenate(list(loaded.stream(SAMPLE_SENTENCES[language], threading.Event())))
    loaded.pitch_hz = typical_pitch_hz(sample, loaded.sample_rate)
    return loaded


def typical_pitch_hz(audio: np.ndarray, sample_rate: int) -> float:
    """Median pitch of the voiced 40 ms frames, found by autocorrelation between 70 and 400 Hz."""
    frame, hop = int(0.04 * sample_rate), int(0.01 * sample_rate)
    shortest, longest = int(sample_rate / 400), int(sample_rate / 70)
    pitches = []
    for start in range(0, len(audio) - frame, hop):
        window = audio[start : start + frame]
        if np.sqrt(np.mean(window**2)) < 0.02:
            continue
        window = window - window.mean()
        correlation = np.correlate(window, window, "full")[frame - 1 :]
        lag = shortest + int(np.argmax(correlation[shortest:longest]))
        if correlation[lag] > 0.3 * correlation[0]:
            pitches.append(sample_rate / lag)
    return float(np.median(pitches))
