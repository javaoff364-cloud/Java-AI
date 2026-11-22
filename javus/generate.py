import argparse
from pathlib import Path

import torch

from .config import JavusConfig
from .model import JavusLanguageModel
from .tokenizer import ByteTokenizer

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "javus-small.pt"


def main():
    parser = argparse.ArgumentParser(description="Generate text with Javus")
    parser.add_argument("--prompt", default="Assalomu alaykum")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if not CHECKPOINT.exists():
        raise SystemExit("Model topilmadi. Avval python -m javus.train buyrug'ini bajaring.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    config = JavusConfig(**checkpoint["config"])
    model = JavusLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = ByteTokenizer()
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(prompt, args.max_new_tokens)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
