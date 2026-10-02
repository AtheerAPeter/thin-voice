"""Measures the whole pipeline without a microphone: spoken questions are fed in at real-time pace.

    uv run bench.py
    uv run bench.py --voice-engine moonshine --llm qwen3.5-2b
"""

import argparse
import asyncio
import itertools
import os
import threading
import time

import numpy as np

from conversation import Conversation
from listen import FRAME_SAMPLES, SAMPLE_RATE, STT_SIZES, Listener, ListenerSettings, SpeechToText, TurnDetector
from main import LLM_FILES, resident_mb, start_llama_server
from speak import DEFAULT_VOICES, MoonshineVoice, load_voice
from think import SYSTEM_PROMPTS, LanguageModel, llama_session

QUESTIONS = {
    "en": [
        "What's a good name for a cat that sleeps all day?",
        "Why is the sky blue?",
        "Give me one quick tip for falling asleep faster.",
        "How far away is the moon?",
    ],
    "de": [
        "Wie nenne ich eine Katze, die den ganzen Tag schläft?",
        "Warum ist der Himmel blau?",
        "Hast du einen Tipp, wie ich schneller einschlafe?",
        "Wie weit ist der Mond entfernt?",
    ],
}
READER_VOICES = {"en": "kokoro_am_adam", "de": "piper_de_DE-karlsson-low"}
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


def spoken_questions(language: str) -> list[tuple[str, np.ndarray]]:
    """The questions read aloud by a different voice than the assistant's, at 16 kHz."""
    reader = MoonshineVoice(language, READER_VOICES[language])
    spoken = []
    for question in QUESTIONS[language]:
        audio = np.concatenate(list(reader.stream(question, threading.Event())))
        spoken.append((question, resample(audio, reader.sample_rate, SAMPLE_RATE)))
    return spoken


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    source_times = np.arange(len(audio)) / source_rate
    target_times = np.arange(int(len(audio) * target_rate / source_rate)) / target_rate
    return np.interp(target_times, source_times, audio).astype(np.float32)


async def feed_microphone(listener: Listener, speech: np.ndarray):
    """Plays the question into the listener at real-time pace, then keeps sending quiet room noise."""
    noise = np.random.default_rng(0)
    padded = np.concatenate([np.zeros(SAMPLE_RATE // 2, dtype=np.float32), speech])
    started = time.monotonic()
    for index in itertools.count():
        start = index * FRAME_SAMPLES
        frame = padded[start : start + FRAME_SAMPLES]
        if len(frame) < FRAME_SAMPLES:
            room = (noise.standard_normal(FRAME_SAMPLES - len(frame)) * 0.002).astype(np.float32)
            frame = np.concatenate([frame, room])
        listener.push(frame)
        await asyncio.sleep(max(0.0, started + (index + 1) * FRAME_SECONDS - time.monotonic()))


async def run(args: argparse.Namespace):
    llama = start_llama_server(LLM_FILES[args.llm], args.llm_port, args.threads)
    try:
        listener = Listener(SpeechToText(args.language, args.stt), TurnDetector(args.threads), ListenerSettings())
        voice = load_voice(args.voice_engine, args.language, args.voice, args.threads)
        app_mb, llm_mb = resident_mb(os.getpid()), resident_mb(llama.pid)
        questions = spoken_questions(args.language)

        results: list[dict] = []
        replied = asyncio.Event()

        async def send_event(event: dict):
            if event["type"] == "user":
                print(f"\nheard:   {event['text']}")
            elif event["type"] == "assistant":
                print(f"replied: {event['text']}")
            elif event["type"] == "metrics":
                results.append(event)
            elif event["type"] in ("done", "error"):
                replied.set()

        async def send_audio(pcm: bytes):
            pass

        async with llama_session() as session:
            llm = LanguageModel(f"http://127.0.0.1:{args.llm_port}", session)
            Conversation(listener, llm, voice, SYSTEM_PROMPTS[args.language], send_event, send_audio)
            for _, speech in questions:
                replied.clear()
                microphone = asyncio.create_task(feed_microphone(listener, speech))
                await asyncio.wait_for(replied.wait(), timeout=60)
                microphone.cancel()

        print(f"\n{args.voice_engine} voice, {args.llm}, Moonshine {args.stt}, {args.threads} threads")
        for key in ["total_ms", "wait_ms", "smart_turn_ms", "stt_ms", "llm_first_token_ms", "first_piece_ms", "tts_first_audio_ms"]:
            values = [result[key] for result in results]
            print(f"  {key:20s} median {np.median(values):6.0f}   max {max(values):6.0f}")
        print(f"  memory: app {app_mb:.0f} MB + llm {llm_mb:.0f} MB = {app_mb + llm_mb:.0f} MB")
    finally:
        llama.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", choices=["en", "de"], default="en")
    parser.add_argument("--stt", choices=list(STT_SIZES))
    parser.add_argument("--voice-engine", choices=list(DEFAULT_VOICES), default="pocket")
    parser.add_argument("--voice")
    parser.add_argument("--llm", choices=list(LLM_FILES), default="lfm2.5-1.2b")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--llm-port", type=int, default=8767)
    args = parser.parse_args()
    args.stt = args.stt or ("medium" if args.language == "en" else "small")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
