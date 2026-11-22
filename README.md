# Javus AI

**Javus AI** is an independent Python/PyTorch neural-network research project. **Javus Alpha 2** is a from-scratch, decoder-only Transformer experiment intended to grow into an English conversational assistant. It does not load pretrained model weights or call an external LLM.

## Alpha 2 foundation

- Python 3.10+ and PyTorch
- Decoder-only Transformer with causal self-attention and pre-normalized residual blocks
- GPT-style parameter initialization and tied input/output embeddings
- Deterministic byte-level BPE tokenizer learned from the current corpus
- UTF-8 byte fallback, so unseen Unicode text can be encoded without an unknown-character token
- English-only seed corpus for this development phase
- AdamW, cross-entropy, gradient clipping, validation monitoring, adaptive learning rate
- Resumable checkpoints, JSONL metrics, pytest, and bounded GitHub Actions training

**Important:** the current corpus is only a small starter set. A better tokenizer and architecture do not create language knowledge by themselves. Javus will not become a useful assistant until we add a large, carefully reviewed training corpus and evaluate it on questions it has never seen.

## Install in Termux

A Linux environment through Ubuntu proot is recommended for PyTorch compatibility:

```bash
pkg update && pkg upgrade -y
pkg install -y git proot-distro
proot-distro install ubuntu
proot-distro login ubuntu
```

Inside Ubuntu:

```bash
apt update
apt install -y python3 python3-venv python3-pip git
git clone https://github.com/javaoff364-cloud/Java-AI.git
cd Java-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

`data/train.txt` is intended to contain 1,000 English conversation examples (the existing authored seed plus filtered Dolly 15k examples). Run `python scripts/import_dolly.py` to rebuild it reproducibly; the GitHub Actions **Import Dolly training data** workflow can run this on the dataset branch. The importer filters empty, duplicate, overly long, non-English, and URL-containing examples. Dolly is licensed CC BY-SA 3.0, so retain attribution and license details. Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k. Each conversation is separated by a blank line:

```text
User: Why is the sky blue?
Javus: Air molecules scatter shorter blue wavelengths of sunlight more than longer red wavelengths.
```

Keep examples accurate, natural, diverse, and free of private information. Track source and license information when adding external data. Keep evaluation questions separate from training data to avoid testing memorization.

## Tests

```bash
python -m pytest -q
```

## Train Javus Alpha 2

Run one bounded session (100 optimizer steps by default):

```bash
python -m javus.train --steps 100 --batch-size 8
```

The first Alpha 2 run detects the older tokenizer checkpoint, preserves it as `checkpoints/javus-alpha-pre-alpha2.pt`, and initializes a new byte-BPE model. Later runs resume only when the saved tokenizer vocabulary matches the current corpus.

To intentionally start the Alpha 2 model from scratch:

```bash
python -m javus.train --steps 100 --batch-size 8 --fresh
```

To change the number of BPE merge rules on a fresh model:

```bash
python -m javus.train --steps 100 --bpe-merges 128 --fresh
```

Training is bounded by default. The optional `--forever` flag is for deliberate local experiments only; it should not be used in GitHub Actions.

## Checkpoints and metrics

- `checkpoints/javus-alpha.pt` — latest model, optimizer, BPE merges, step, and RNG state
- `checkpoints/javus-alpha-best.pt` — best validation checkpoint
- `checkpoints/javus-alpha-metrics.jsonl` — per-run metrics
- `checkpoints/javus-alpha-pre-alpha2.pt` — preserved previous model when migrating to Alpha 2

## Generate text

```bash
python -m javus.generate --prompt "Hello, how are you?" --max-new-tokens 40
```

## HTTP API

After a checkpoint exists:

```bash
export JAVUS_CHECKPOINT_PATH=checkpoints/javus-alpha.pt
uvicorn javus.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /` — API information
- `GET /health` — model status and tokenizer type
- `POST /chat` — text generation
- `GET /docs` — interactive API documentation

Set `JAVUS_API_KEY` to protect `POST /chat`. Never commit API keys or private checkpoint URLs.

## GitHub Actions

The training workflow runs a finite number of steps (100 by default; manual choices include 250, 500, or 1000), saves a checkpoint, and uploads artifacts for 30 days. It is scheduled every six hours. A separate CI workflow runs tests on pushes and pull requests.

## Roadmap

- [x] Python Transformer prototype
- [x] Byte-level BPE tokenizer foundation
- [x] English-only starter corpus
- [x] Checkpoint and resume support
- [x] Bounded training workflow
- [x] Core, dataset, and tokenizer tests
- [ ] Expand the English corpus substantially with reviewed/licensed data
- [ ] Build a held-out assistant evaluation benchmark
- [ ] Improve conversation formatting and generation quality
- [ ] Train and evaluate larger model sizes
- [ ] Research vision and audio after text quality is established

## License

Apache-2.0
