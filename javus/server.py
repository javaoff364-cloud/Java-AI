from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import JavusConfig
from .model import JavusLanguageModel
from .tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT = CHECKPOINT_DIR / "javus-alpha.pt"
LEGACY_CHECKPOINT = CHECKPOINT_DIR / "javus-small.pt"
MODEL_NAME = "Javus Alpha"
app = FastAPI(title="Javus AI API", version="0.1.0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = ByteTokenizer()


class ChatRequest(BaseModel):
    messages: list[dict] = Field(min_length=1, max_length=20)
    max_new_tokens: int = Field(default=80, ge=1, le=256)


def load_model():
    global model
    if model is not None:
        return model
    checkpoint_path = CHECKPOINT if CHECKPOINT.exists() else LEGACY_CHECKPOINT
    if not checkpoint_path.exists():
        raise HTTPException(status_code=503, detail=f"{MODEL_NAME} hali o'qitilmagan.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = JavusConfig(**checkpoint["config"])
    model = JavusLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@app.get("/health")
def health():
    return {
        "name": "Javus AI",
        "model": MODEL_NAME,
        "version": "0.1.0",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    latest = request.messages[-1]
    if latest.get("role") != "user" or not isinstance(latest.get("content"), str):
        raise HTTPException(status_code=400, detail="Oxirgi xabar user matni bo'lishi kerak.")
    active_model = load_model()
    prompt = tokenizer.encode(latest["content"][:2000])
    if not prompt:
        raise HTTPException(status_code=400, detail="Xabar bo'sh.")
    tokens = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = active_model.generate(tokens, request.max_new_tokens)
    generated = output[0, len(prompt):].tolist()
    return {"reply": tokenizer.decode(generated), "model": MODEL_NAME, "experimental": True}
