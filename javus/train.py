import argparse
import random
from pathlib import Path

import torch

from .config import JavusConfig
from .model import JavusLanguageModel
from .tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.txt"
CHECKPOINT = ROOT / "checkpoints" / "javus-small.pt"


def main():
    parser = argparse.ArgumentParser(description="Train Javus small Transformer from scratch")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps va batch-size kamida 1 bo'lishi kerak")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()
    text = DATA_PATH.read_text(encoding="utf-8")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(data) < 20:
        raise SystemExit("Dataset juda kichik. data/train.txt fayliga ko'proq matn kiriting.")
    config = JavusConfig()
    if len(data) < config.block_size + 2:
        config.block_size = max(16, min(config.block_size, len(data) // 2))
    model = JavusLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    for step in range(1, args.steps + 1):
        starts = torch.randint(0, len(data) - config.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i:i + config.block_size] for i in starts]).to(device)
        y = torch.stack([data[i + 1:i + config.block_size + 1] for i in starts]).to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step {step:>5}/{args.steps} | loss {loss.item():.4f} | device {device}")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "model": model.state_dict(), "step": args.steps}, CHECKPOINT)
    print(f"Checkpoint saqlandi: {CHECKPOINT}")


if __name__ == "__main__":
    main()
