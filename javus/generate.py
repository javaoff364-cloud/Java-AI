import argparse
from pathlib import Path

import torch

from .config import JavusConfig
from .model import JavusLanguageModel
from .tokenizer import ByteTokenizer

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
CHECKPOINT = CHECKPOINT_DIR / "javus-alpha.pt"
LEGACY_CHECKPOINT = CHECKPOINT_DIR / "javus-small.pt"
MODEL_NAME = "Javus Alpha"


def main():
    parser = argparse.ArgumentParser(description="Generate text with Javus Alpha")
    parser.add_argument("--prompt", default="Assalomu alaykum")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    checkpoint_path = CHECKPOINT if CHECKPOINT.exists() else LEGACY_CHECKPOINT
    if not checkpoint_path.exists():
        raise SystemExit("Javus Alpha checkpoint topilmadi. Avval python -m javus.train buyrug'ini bajaring.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = JavusConfig(**checkpoint["config"])
    model = JavusLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = ByteTokenizer()
    formatted_prompt = f"Foydalanuvchi: {args.prompt.strip()}\nJavus:"
    prompt = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long, device=device)
    output = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    reply_tokens = output[0, prompt.shape[1]:].tolist()
    print(f"[{MODEL_NAME}]")
    print(tokenizer.decode(reply_tokens).strip())


if __name__ == "__main__":
    main()
