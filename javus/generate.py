import argparse
from pathlib import Path

import torch

from .config import JavusConfig
from .model import JavusLanguageModel
from .bpe_tokenizer import ByteBPETokenizer
from .tokenizer import ByteTokenizer, CharTokenizer

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
CHECKPOINT = CHECKPOINT_DIR / "javus-alpha.pt"
LEGACY_CHECKPOINT = CHECKPOINT_DIR / "javus-small.pt"
MODEL_NAME = "Javus Alpha 2"


def clean_reply(reply: str) -> str:
    for marker in ("\nUser:", "\n\nUser:"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]
    return reply.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate text with Javus Alpha")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    checkpoint_path = CHECKPOINT if CHECKPOINT.exists() else LEGACY_CHECKPOINT
    if not checkpoint_path.exists():
        raise SystemExit("Javus Alpha checkpoint topilmadi. Avval python -m javus.train buyrug'ini bajaring.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = JavusConfig(**checkpoint["config"])
    state = checkpoint.get("tokenizer")
    if isinstance(state, dict) and state.get("type") == "byte_bpe":
        tokenizer = ByteBPETokenizer.from_state(state)
    elif state is not None:
        tokenizer = CharTokenizer.from_state(state)
    else:
        tokenizer = ByteTokenizer()
    model = JavusLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    formatted_prompt = f"User: {args.prompt.strip()}\nJavus:"
    prompt = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long, device=device)
    output = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    reply_tokens = output[0, prompt.shape[1]:].tolist()
    print(f"[{MODEL_NAME}]")
    print(clean_reply(tokenizer.decode(reply_tokens)))


if __name__ == "__main__":
    main()
