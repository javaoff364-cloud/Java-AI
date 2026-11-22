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


def save_checkpoint(model, optimizer, config, global_step, device):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": global_step,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    temporary = CHECKPOINT.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(CHECKPOINT)


def main():
    parser = argparse.ArgumentParser(description="Train Javus Small Transformer from scratch")
    parser.add_argument("--steps", type=int, default=100, help="Har bir training partiyasidagi qadamlar")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forever", action="store_true", help="Har 100 qadamdan keyin saqlab, davom etish")
    parser.add_argument("--fresh", action="store_true", help="Mavjud checkpointni e'tiborsiz qoldirib, noldan boshlash")
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

    checkpoint = None
    if CHECKPOINT.is_file() and not args.fresh:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        config = JavusConfig(**checkpoint["config"])
        print(f"Mavjud model yuklandi: {checkpoint.get('step', 0)} qadamdan davom etiladi.")
    else:
        config = JavusConfig()

    if len(data) < config.block_size + 2:
        config.block_size = max(16, min(config.block_size, len(data) // 2))

    model = JavusLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    global_step = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("step", 0))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    model.train()
    batch_number = 0
    try:
        while True:
            batch_number += 1
            losses = []
            for _ in range(args.steps):
                starts = torch.randint(0, len(data) - config.block_size - 1, (args.batch_size,))
                x = torch.stack([data[i:i + config.block_size] for i in starts]).to(device)
                y = torch.stack([data[i + 1:i + config.block_size + 1] for i in starts]).to(device)
                _, loss = model(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                losses.append(loss.item())

            save_checkpoint(model, optimizer, config, global_step, device)
            average_loss = sum(losses) / len(losses)
            print(
                f"batch {batch_number} | {args.steps} qadam tugadi | "
                f"jami {global_step} qadam | average loss {average_loss:.4f} | "
                f"device {device}",
                flush=True,
            )
            if not args.forever:
                break
    except KeyboardInterrupt:
        print("To'xtatish so'rovi olindi. Oxirgi checkpoint saqlangan.", flush=True)


if __name__ == "__main__":
    main()
