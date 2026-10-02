"""thin-voice: a local voice assistant small enough for an 8 GB laptop.

    uv run main.py                          # English, Pocket TTS, LFM2.5-1.2B
    uv run main.py --voice-engine moonshine # Kokoro instead of Pocket TTS
    uv run main.py --llm qwen3.5-2b         # smarter, slower brain
    uv run main.py --language de            # German
"""

import argparse
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from conversation import Conversation
from listen import STT_SIZES, Listener, ListenerSettings, SpeechToText, TurnDetector
from models import model_path
from speak import DEFAULT_VOICES, load_voice
from think import SYSTEM_PROMPTS, LanguageModel, llama_session

ROOT = Path(__file__).parent
LLM_FILES = {
    "lfm2.5-1.2b": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    "qwen3.5-2b": "Qwen3.5-2B-Q4_K_M.gguf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talk to a local LLM.")
    parser.add_argument("--language", choices=["en", "de"], default="en")
    parser.add_argument("--stt", choices=list(STT_SIZES), help="Moonshine size (default: medium, small for German)")
    parser.add_argument("--voice-engine", choices=list(DEFAULT_VOICES), default="pocket")
    parser.add_argument("--voice", help="voice id for the chosen engine")
    parser.add_argument("--llm", choices=list(LLM_FILES), default="lfm2.5-1.2b")
    parser.add_argument("--llm-url", help="use an already running OpenAI-compatible server instead of starting llama-server")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per model (an M1 Air has 4 performance cores)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--llm-port", type=int, default=8766)
    args = parser.parse_args()
    # German Moonshine only comes in tiny and small.
    args.stt = args.stt or ("medium" if args.language == "en" else "small")
    return args


def start_llama_server(model_file: str, port: int, threads: int) -> subprocess.Popen:
    log = open(ROOT / "llama-server.log", "w")
    process = subprocess.Popen(
        [
            "llama-server",
            "--model", str(model_path(model_file)),
            "--port", str(port),
            "--ctx-size", "4096",
            # One conversation, so one slot gets the whole context and keeps its prompt cache.
            "--parallel", "1",
            "--n-gpu-layers", "99",
            "--threads", str(threads),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(240):
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited, see {ROOT / 'llama-server.log'}")
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    return process
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("llama-server did not become ready within a minute")


def resident_mb(pid: int) -> float:
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])) / 1024


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(ROOT / "web" / "index.html")


async def config(request: web.Request) -> web.Response:
    voice = request.app["voice"]
    lam_heads = sorted(path.stem for path in (ROOT / "web" / "avatars" / "lam").glob("*.zip"))
    pictures = sorted(path.stem for path in (ROOT / "web" / "avatars" / "pictures").glob("*.json"))
    return web.json_response(
        {
            "language": request.app["language"],
            "sample_rate": voice.sample_rate,
            "voice_pitch_hz": round(voice.pitch_hz),
            "lam_heads": lam_heads,
            "pictures": pictures,
        }
    )


async def talk(request: web.Request) -> web.WebSocketResponse:
    app = request.app
    socket = web.WebSocketResponse()
    await socket.prepare(request)
    if app["busy"]:
        await socket.send_json({"type": "busy"})
        await socket.close()
        return socket

    async def send_event(event: dict):
        if not socket.closed:
            await socket.send_json(event)

    async def send_audio(pcm: bytes):
        if not socket.closed:
            await socket.send_bytes(pcm)

    app["busy"] = True
    listener: Listener = app["listener"]
    conversation = Conversation(listener, app["llm"], app["voice"], app["system_prompt"], send_event, send_audio)
    try:
        async for message in socket:
            if message.type == WSMsgType.BINARY:
                listener.push(np.frombuffer(message.data, dtype="<i2").astype(np.float32) / 32768)
            elif message.type == WSMsgType.TEXT:
                event = message.json()
                if event["type"] == "playing":
                    conversation.set_client_playing(event["value"])
                elif event["type"] == "barge_in":
                    listener.barge_in = event["value"]
    finally:
        conversation.close()
        app["busy"] = False
    return socket


def main():
    args = parse_args()

    started = time.monotonic()
    print("Starting the language model ...")
    llama = None if args.llm_url else start_llama_server(LLM_FILES[args.llm], args.llm_port, args.threads)
    llm_url = args.llm_url or f"http://127.0.0.1:{args.llm_port}"

    print(f"Loading speech-to-text (Moonshine {args.stt}) ...")
    listener = Listener(SpeechToText(args.language, args.stt), TurnDetector(args.threads), ListenerSettings())
    print(f"Loading the voice ({args.voice_engine}) ...")
    voice = load_voice(args.voice_engine, args.language, args.voice, args.threads)

    app_mb = resident_mb(os.getpid())
    llm_mb = resident_mb(llama.pid) if llama else 0
    print(f"Ready in {time.monotonic() - started:.0f}s. Memory: app {app_mb:.0f} MB + llm {llm_mb:.0f} MB = {app_mb + llm_mb:.0f} MB")

    async def on_startup(app: web.Application):
        app["llm"] = LanguageModel(llm_url, llama_session())

    async def on_cleanup(app: web.Application):
        await app["llm"].session.close()

    app = web.Application()
    app["listener"] = listener
    app["voice"] = voice
    app["system_prompt"] = SYSTEM_PROMPTS[args.language]
    app["language"] = args.language
    app["busy"] = False
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/config", config)
    app.router.add_get("/talk", talk)
    app.router.add_static("/avatars/", ROOT / "web" / "avatars")
    try:
        web.run_app(app, host="127.0.0.1", port=args.port, print=lambda _: print(f"Open http://localhost:{args.port}"))
    finally:
        if llama:
            llama.terminate()


if __name__ == "__main__":
    main()
