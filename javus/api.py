"""HTTP API for serving the Javus Alpha PyTorch checkpoint."""
from __future__ import annotations

import os
import tempfile
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import JavusConfig
from .model import JavusLanguageModel
from .tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "javus-alpha.pt"
MODEL: Optional[JavusLanguageModel] = None
TOKENIZER = ByteTokenizer()
DEVICE = torch.device("cpu")


def checkpoint_path() -> Path:
    configured = os.getenv("JAVUS_CHECKPOINT_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT


def obtain_checkpoint() -> Path:
    path = checkpoint_path()
    if path.is_file():
        return path

    url = os.getenv("JAVUS_CHECKPOINT_URL")
    if not url:
        raise RuntimeError(
            "Checkpoint topilmadi. JAVUS_CHECKPOINT_PATH yoki JAVUS_CHECKPOINT_URL sozlang."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "Javus-AI/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
        output.write(response.read())
    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checkpoint fayli bo'sh.")
    temporary.replace(path)
    return path


def load_model() -> None:
    global MODEL
    path = obtain_checkpoint()
    payload = torch.load(path, map_location=DEVICE, weights_only=True)
    if not isinstance(payload, dict) or "config" not in payload or "model" not in payload:
        raise RuntimeError("Checkpoint formati Javus Alpha uchun mos emas.")
    config = JavusConfig(**payload["config"])
    model = JavusLanguageModel(config).to(DEVICE)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    MODEL = model


def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.getenv("JAVUS_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="API kalit noto'g'ri yoki yuborilmagan.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    torch.set_num_threads(max(1, int(os.getenv("JAVUS_TORCH_THREADS", "2"))))
    try:
        load_model()
        print("Javus Alpha checkpoint yuklandi.", flush=True)
    except Exception as exc:
        # Keep health/status endpoints available so a missing checkpoint is diagnosable.
        print(f"Javus Alpha model yuklanmadi: {exc}", flush=True)
    yield


app = FastAPI(
    title="Javus AI API",
    version="0.1.0",
    description="Javus Alpha PyTorch text-generation API.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=80, ge=1, le=128)
    temperature: float = Field(default=0.8, gt=0.05, le=2.0)
    top_k: int = Field(default=40, ge=1, le=256)


class ChatResponse(BaseModel):
    model: str
    reply: str


@app.get("/")
def root():
    return {"name": "Javus AI API", "model": "Javus Alpha", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "model_unavailable",
        "model": "Javus Alpha",
        "loaded": MODEL is not None,
        "device": str(DEVICE),
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(request: ChatRequest):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model yuklanmagan. Server loglari va checkpoint sozlamalarini tekshiring.",
        )
    # Match the prompt format used by the supervised conversation corpus.
    formatted_prompt = f"Foydalanuvchi: {request.message.strip()}\nJavus:"
    prompt_tokens = TOKENIZER.encode(formatted_prompt)
    if not prompt_tokens:
        raise HTTPException(status_code=400, detail="Xabar bo'sh.")
    context = prompt_tokens[-MODEL.config.block_size:]
    tokens = torch.tensor([context], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        generated = MODEL.generate(
            tokens,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    new_tokens = generated[0, len(context):].tolist()
    reply = TOKENIZER.decode(new_tokens).strip()
    return ChatResponse(model="Javus Alpha", reply=reply)
