"""Bounded, resumable training for Javus Alpha."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch

from .config import JavusConfig
from .model import JavusLanguageModel
from .smart_training import AdaptiveLearningRate, SmartTrainingState, evaluate
from .bpe_tokenizer import ByteBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.txt"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT = CHECKPOINT_DIR / "javus-alpha.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "javus-alpha-best.pt"
LEGACY_CHECKPOINT = CHECKPOINT_DIR / "javus-small.pt"
PREVIOUS_CHECKPOINT = CHECKPOINT_DIR / "javus-alpha-pre-alpha2.pt"
METRICS = CHECKPOINT_DIR / "javus-alpha-metrics.jsonl"
MODEL_NAME = "Javus Alpha 2"


def split_corpus(text: str, tokenizer: ByteBPETokenizer, seed: int = 42):
    """Validate records and make a reproducible, conversation-level train/validation split."""
    records = [
        item.strip()
        for item in text.split("\n\n")
        if item.strip() and not item.lstrip().startswith("#")
    ]
    if len(records) < 20:
        raise ValueError("Training uchun kamida 20 ta to'liq suhbat namunasi kerak.")

    prompts = set()
    for number, record in enumerate(records, start=1):
        if not record.startswith("User: "):
            raise ValueError(f"{number}-namuna 'User: ' bilan boshlanishi kerak.")
        if record.count("\nJavus: ") != 1:
            raise ValueError(f"{number}-namunada aynan bitta 'Javus: ' javobi bo'lishi kerak.")
        prompt, answer = record[len("User: "):].split("\nJavus: ", 1)
        if not prompt.strip() or not answer.strip():
            raise ValueError(f"{number}-namunada savol yoki javob bo'sh.")
        normalized = " ".join(prompt.casefold().split())
        if normalized in prompts:
            raise ValueError(f"Takroriy savol topildi: {prompt[:100]}")
        prompts.add(normalized)

    # Shuffle whole conversations so a source-ordered dataset does not create
    # a validation set containing only the final source category.
    random.Random(seed).shuffle(records)
    split = max(1, min(len(records) - 1, int(len(records) * 0.9)))
    train_text = "\n\n".join(records[:split]) + "\n\n"
    val_text = "\n\n".join(records[split:])
    return (
        torch.tensor(tokenizer.encode(train_text), dtype=torch.long),
        torch.tensor(tokenizer.encode(val_text), dtype=torch.long),
        len(records),
    )


def save_checkpoint(model, optimizer, config, tokenizer, global_step, device,
                    smart_state, corpus_sha256):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_name": MODEL_NAME,
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "smart_training": smart_state.to_dict(),
        "step": global_step,
        "corpus_sha256": corpus_sha256,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    temporary = CHECKPOINT.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(CHECKPOINT)


def main():
    parser = argparse.ArgumentParser(description="Train Javus Alpha with resumable checkpoints")
    parser.add_argument("--steps", type=int, default=100,
                        help="Maximum optimizer steps in this run (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bpe-merges", type=int, default=128)
    parser.add_argument("--forever", action="store_true",
                        help="Explicitly continue batches until manually stopped")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints and create a new tokenizer/model")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    if args.steps > 10000:
        parser.error("--steps cannot exceed 10000 in one run")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = DATA_PATH.read_text(encoding="utf-8")
    corpus_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    source_checkpoint = CHECKPOINT if CHECKPOINT.is_file() else LEGACY_CHECKPOINT
    checkpoint = None
    if source_checkpoint.is_file() and not args.fresh:
        candidate = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        state = candidate.get("tokenizer")
        if not isinstance(state, dict) or state.get("type") != "byte_bpe":
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            if not PREVIOUS_CHECKPOINT.exists():
                shutil.copy2(source_checkpoint, PREVIOUS_CHECKPOINT)
            print("Preserved the previous model; initializing Javus Alpha 2 with byte-BPE.", flush=True)
            tokenizer = ByteBPETokenizer.from_text(text, args.bpe_merges)
        else:
            old_tokenizer = ByteBPETokenizer.from_state(state)
            expected_tokenizer = ByteBPETokenizer.from_text(text, args.bpe_merges)
            if (old_tokenizer.to_dict() != expected_tokenizer.to_dict()
                    or candidate.get("corpus_sha256") != corpus_sha256):
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                archive = CHECKPOINT_DIR / f"javus-alpha-before-dataset-{str(candidate.get('corpus_sha256', 'unknown'))[:12]}.pt"
                if not archive.exists():
                    shutil.copy2(source_checkpoint, archive)
                print("Training corpus or BPE vocabulary changed; archived the old checkpoint and starting a fresh model.", flush=True)
                tokenizer = expected_tokenizer
            else:
                tokenizer = old_tokenizer
                checkpoint = candidate
                print(f"{MODEL_NAME}: resuming from step {checkpoint.get('step', 0)}.", flush=True)
    else:
        tokenizer = ByteBPETokenizer.from_text(text, args.bpe_merges)

    train_data, val_data, record_count = split_corpus(text, tokenizer, seed=args.seed)
    if len(train_data) < 40:
        raise SystemExit("Dataset is too small; add more complete conversation examples.")

    if checkpoint is not None:
        config = JavusConfig(**checkpoint["config"])
        if config.vocab_size != tokenizer.vocab_size:
            raise SystemExit("Checkpoint vocabulary and model config do not match; use --fresh.")
    else:
        config = JavusConfig(vocab_size=tokenizer.vocab_size)

    if len(train_data) <= config.block_size + 1:
        config.block_size = max(8, min(config.block_size, (len(train_data) - 2) // 2))

    model = JavusLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    global_step = 0
    smart_state = SmartTrainingState()

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("step", 0))
        smart_state = SmartTrainingState.from_dict(checkpoint.get("smart_training"))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    controller = AdaptiveLearningRate(optimizer, smart_state)
    batch_number = 0
    try:
        while True:
            batch_number += 1
            losses = []
            model.train()
            for _ in range(args.steps):
                upper = len(train_data) - config.block_size
                starts = torch.randint(0, upper, (args.batch_size,))
                x = torch.stack([train_data[i:i + config.block_size] for i in starts]).to(device)
                y = torch.stack([train_data[i + 1:i + config.block_size + 1] for i in starts]).to(device)
                _, loss = model(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                losses.append(float(loss.item()))

            scores = evaluate(model, train_data, val_data, config.block_size,
                              batch_size=args.batch_size, device=device)
            validation_loss = scores["validation"]
            improved = validation_loss is not None and validation_loss < smart_state.best_val_loss
            lr_changed = controller.update(validation_loss) if validation_loss is not None else False
            record = {
                "model": MODEL_NAME, "step": global_step,
                "batch_loss": sum(losses) / len(losses),
                "train_loss": scores["train"], "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "lr_reduced": lr_changed, "dataset_records": record_count,
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "corpus_sha256": corpus_sha256,
            }
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            with METRICS.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            save_checkpoint(model, optimizer, config, tokenizer, global_step,
                            device, smart_state, corpus_sha256)
            if improved:
                shutil.copy2(CHECKPOINT, BEST_CHECKPOINT)

            print(
                f"{MODEL_NAME} | batch {batch_number} | +{args.steps} steps | "
                f"total {global_step} | records {record_count} | vocab {tokenizer.vocab_size} | "
                f"train loss {scores['train']:.4f} | validation loss "
                f"{validation_loss if validation_loss is not None else 'N/A'} | "
                f"LR {optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )
            if not args.forever:
                break
    except KeyboardInterrupt:
        print("Stop requested. Saving the latest safe checkpoint...", flush=True)
        save_checkpoint(model, optimizer, config, tokenizer, global_step,
                        device, smart_state, corpus_sha256)


if __name__ == "__main__":
    main()
