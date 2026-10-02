"""The conversation loop: listen, think, speak, and stop talking when interrupted."""

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from listen import Listener, Turn
from think import LanguageModel, speakable_pieces

# Older turns are dropped so the prompt stays small and fast to prefill.
MAX_HISTORY_MESSAGES = 20


class Conversation:
    """Wires a Listener, a LanguageModel and a voice together for one client.

    `send_event` delivers JSON-able dicts to the client and `send_audio` delivers
    int16 PCM at `voice.sample_rate`; the browser and the benchmark provide their own.
    """

    def __init__(
        self,
        listener: Listener,
        llm: LanguageModel,
        voice,
        system_prompt: str,
        send_event: Callable[[dict], Awaitable[None]],
        send_audio: Callable[[bytes], Awaitable[None]],
    ):
        self.listener = listener
        self.llm = llm
        self.voice = voice
        self.send_event = send_event
        self.send_audio = send_audio
        self.loop = asyncio.get_running_loop()
        self.system = {"role": "system", "content": system_prompt}
        self.history: list[dict] = []
        self.reply_task: asyncio.Task | None = None
        self.stop_voice = threading.Event()
        # The voice engines are not thread-safe; a cancelled reply may still be finishing a piece.
        self.voice_lock = threading.Lock()
        # What the user said when they were cut off by their own pause: it is joined with what they say next.
        self.unanswered = ""
        self.audio_sent = False
        self.client_playing = False

        listener.on_speech_start = lambda: self.loop.call_soon_threadsafe(self.speech_started)
        listener.on_turn = lambda turn: self.loop.call_soon_threadsafe(self.turn_ended, turn)

    def close(self):
        self.interrupt()
        self.listener.on_speech_start = lambda: None
        self.listener.on_turn = lambda turn: None
        self.listener.reset()

    def set_client_playing(self, playing: bool):
        self.client_playing = playing
        self.update_speaking_state()

    def update_speaking_state(self):
        replying = self.reply_task is not None and not self.reply_task.done()
        self.listener.assistant_speaking = self.client_playing or (replying and self.audio_sent)

    def speech_started(self):
        self.loop.create_task(self.send_event({"type": "user_speaking"}))
        if self.client_playing or (self.reply_task and not self.reply_task.done()):
            self.interrupt()

    def interrupt(self):
        self.stop_voice.set()
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        if self.client_playing or self.audio_sent:
            self.loop.create_task(self.send_event({"type": "stop"}))
        self.client_playing = False
        self.update_speaking_state()

    def turn_ended(self, turn: Turn):
        text = f"{self.unanswered} {turn.text}".strip()
        self.unanswered = ""
        if not text:
            self.loop.create_task(self.send_event({"type": "ignored"}))
            return
        self.reply_task = self.loop.create_task(self.reply(text, turn))

    async def reply(self, user_text: str, turn: Turn):
        await self.send_event({"type": "user", "text": user_text})
        self.history = self.history[-MAX_HISTORY_MESSAGES:]
        self.history.append({"role": "user", "content": user_text})
        self.stop_voice = stop = threading.Event()
        self.audio_sent = False
        spoken: list[str] = []
        pieces: asyncio.Queue = asyncio.Queue()
        timing = {"asked": time.monotonic()}

        async def think():
            tokens = self.timed_tokens(self.llm.stream([self.system, *self.history]), timing)
            async for piece in speakable_pieces(tokens):
                await pieces.put((piece, time.monotonic()))
            await pieces.put(None)

        async def speak():
            while (item := await pieces.get()) is not None:
                piece, ready_at = item
                await self.send_event({"type": "assistant", "text": piece})
                async for audio in self.synthesize(piece, stop):
                    if not self.audio_sent:
                        self.audio_sent = True
                        self.update_speaking_state()
                        await self.send_event(self.metrics(turn, timing, ready_at))
                    await self.send_audio(to_pcm16(audio))
                spoken.append(piece)

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(think())
                group.create_task(speak())
        except asyncio.CancelledError:
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken) + " …"})
            else:
                self.history.pop()
                self.unanswered = user_text
            raise
        except Exception as error:
            self.history.pop()
            await self.send_event({"type": "error", "message": str(error)})
            raise
        else:
            self.history.append({"role": "assistant", "content": " ".join(spoken)})
            await self.send_event({"type": "done"})
        finally:
            stop.set()
            self.update_speaking_state()

    async def timed_tokens(self, tokens: AsyncIterator[str], timing: dict) -> AsyncIterator[str]:
        async for token in tokens:
            timing.setdefault("first_token", time.monotonic())
            yield token

    async def synthesize(self, text: str, stop: threading.Event) -> AsyncIterator[np.ndarray]:
        """Runs the voice on a worker thread and hands its chunks over as they come."""
        chunks: asyncio.Queue = asyncio.Queue()

        def work():
            with self.voice_lock:
                for audio in self.voice.stream(text, stop):
                    if stop.is_set():
                        break
                    self.loop.call_soon_threadsafe(chunks.put_nowait, audio)
            self.loop.call_soon_threadsafe(chunks.put_nowait, None)

        self.loop.run_in_executor(None, work)
        while (audio := await chunks.get()) is not None:
            yield audio

    def metrics(self, turn: Turn, timing: dict, piece_ready_at: float) -> dict:
        now = time.monotonic()
        metrics = {
            "type": "metrics",
            "total_ms": round((now - turn.speech_end) * 1000),
            "wait_ms": round(turn.wait_ms),
            "smart_turn_ms": round(turn.smart_turn_ms),
            "stt_ms": round(turn.stt_ms),
            "llm_first_token_ms": round((timing["first_token"] - timing["asked"]) * 1000),
            "first_piece_ms": round((piece_ready_at - timing["asked"]) * 1000),
            "tts_first_audio_ms": round((now - piece_ready_at) * 1000),
        }
        print(
            f"  reply in {metrics['total_ms']} ms after you stopped "
            f"(pause {metrics['wait_ms']}, stt {metrics['stt_ms']}, llm {metrics['llm_first_token_ms']}, "
            f"first clause {metrics['first_piece_ms']}, voice {metrics['tts_first_audio_ms']})"
        )
        return metrics


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
