"""Export a Javus PyTorch checkpoint to a NumPy-only NPZ bundle."""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("javus-alpha.npz"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    arrays = {f"w__{key}": value.detach().cpu().numpy() for key, value in checkpoint["model"].items()}
    for key, value in config.items():
        arrays[f"cfg__{key}"] = np.asarray(value)
    arrays["meta__model_name"] = np.asarray("Javus Alpha")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
