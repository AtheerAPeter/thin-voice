"""Model files that are not fetched by their own libraries, downloaded into ./models on first use."""

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

URLS = {
    "silero_vad.onnx": "https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx",
    "smart-turn-v3.2-cpu.onnx": "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
    "LFM2.5-1.2B-Instruct-Q4_K_M.gguf": "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    "Qwen3.5-2B-Q4_K_M.gguf": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
}


def model_path(filename: str) -> Path:
    path = MODELS_DIR / filename
    if not path.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        print(f"Downloading {filename} ...")
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(URLS[filename], partial)
        partial.rename(path)
    return path
