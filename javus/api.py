"""HTTP API for serving the Javus Alpha PyTorch checkpoint."""
from __future__ import annotations

import os
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import JavusConfig
from .model import JavusLanguageModel
from .tokenizer import ByteTokenizer, CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "javus-alpha.pt"
MODEL: Optional[JavusLanguageModel] = None
TOKENIZER = ByteTokenizer()
DEVICE = torch.device("cpu")
MODEL_NAME = "Javus Alpha"


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
    request = urllib.request.Request(url, headers={"User-Agent": "Javus-AI/0.2"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
        output.write(response.read())
    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checkpoint fayli bo'sh.")
    temporary.replace(path)
    return path


def load_model() -> None:
    global MODEL, TOKENIZER
    path = obtain_checkpoint()
    payload = torch.load(path, map_location=DEVICE, weights_only=True)
    if not isinstance(payload, dict) or "config" not in payload or "model" not in payload:
        raise RuntimeError("Checkpoint formati Javus Alpha uchun mos emas.")
    config = JavusConfig(**payload["config"])
    if "tokenizer" in payload:
        TOKENIZER = CharTokenizer.from_state(payload["tokenizer"])
        if TOKENIZER.vocab_size != config.vocab_size:
            raise RuntimeError("Checkpoint tokenizer hajmi model config bilan mos emas.")
    else:
        # Backward compatibility for the currently deployed byte-token model.
        TOKENIZER = ByteTokenizer()
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
        print(f"{MODEL_NAME} checkpoint yuklandi.", flush=True)
    except Exception as exc:
        print(f"{MODEL_NAME} model yuklanmadi: {exc}", flush=True)
    yield


app = FastAPI(
    title="Javus AI API",
    version="0.2.0",
    description="Javus Alpha PyTorch text-generation API.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=80, ge=1, le=128)
    temperature: float = Field(default=0.7, gt=0.05, le=2.0)
    top_k: int = Field(default=20, ge=1, le=256)


class ChatResponse(BaseModel):
    model: str
    reply: str


def clean_reply(reply: str) -> str:
    """Stop when the model begins generating the next training conversation."""
    for marker in ("\nFoydalanuvchi:", "\n\nFoydalanuvchi:", "\nUser:", "\n\nUser:", "\nПользователь:", "\n\nПользователь:"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]
    return reply.strip()


@app.get("/")
def root():
    return {"name": "Javus AI API", "model": MODEL_NAME, "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "model_unavailable",
        "model": MODEL_NAME,
        "loaded": MODEL is not None,
        "device": str(DEVICE),
        "tokenizer": "unicode-character" if isinstance(TOKENIZER, CharTokenizer) else "utf8-byte-legacy",
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(request: ChatRequest):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model yuklanmagan. Server loglari va checkpoint sozlamalarini tekshiring.",
        )
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
    reply = clean_reply(TOKENIZER.decode(new_tokens))
    return ChatResponse(model=MODEL_NAME, reply=reply)
