# Javus AI

**Javus AI** is an independent Python/PyTorch neural-network research project. Its first model, **Javus Alpha**, is a small experimental text-only Transformer — not yet a production-ready assistant.

## Current stack

- Python 3.10+ and PyTorch
- Decoder-only Transformer with causal self-attention
- Unicode character tokenizer with a checkpointed vocabulary
- Multilingual starter corpus: English, Uzbek (Latin and Cyrillic), Russian, Turkish, Arabic, Chinese, and Japanese
- AdamW, cross-entropy, gradient clipping, validation monitoring, and adaptive learning rate
- Resumable checkpoints, JSONL metrics, pytest, and GitHub Actions

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

## Dataset and language support

`data/train.txt` contains manually authored starter examples in English, Uzbek Latin/Cyrillic, Russian, Turkish, Arabic, Chinese, and Japanese. The corpus is UTF-8 and includes non-Latin scripts and emoji.

The tokenizer maps Unicode code points to IDs and stores its exact vocabulary in each checkpoint. It can round-trip characters present in its vocabulary. A character absent from the vocabulary maps to `<unk>`; add new scripts/symbols to the corpus and retrain with a compatible vocabulary before expecting the model to generate them.

This is a small seed corpus for testing the training pipeline, not a large language dataset. It does not make Javus fluent in all listed languages. A useful assistant requires a much larger, reviewed, balanced, and legally usable corpus, plus a separate benchmark and human evaluation.

When adding data:
- keep each conversation example complete and separate by a blank line;
- check spelling, facts, duplicates, and language balance;
- do not include private or sensitive information;
- record data sources and licenses;
- keep validation/test examples separate from training examples.

## Tests

```bash
python -m pytest -q
```

## Train Javus Alpha

Run one bounded session (100 optimizer steps by default):

```bash
python -m javus.train --steps 100 --batch-size 8
```

The run saves a checkpoint after the batch. A later run resumes from that checkpoint. To explicitly continue repeated batches until stopped manually:

```bash
python -m javus.train --steps 100 --batch-size 8 --forever
```

To intentionally ignore existing model state and start from scratch:

```bash
python -m javus.train --steps 100 --fresh
```

Files created locally:
- `checkpoints/javus-alpha.pt` — latest model, optimizer, tokenizer, step, and RNG state;
- `checkpoints/javus-alpha-best.pt` — best validation checkpoint;
- `checkpoints/javus-alpha-metrics.jsonl` — per-run metrics;
- `checkpoints/javus-alpha-pre-multilingual.pt` — preserved previous checkpoint if a new vocabulary is required.

When a new dataset contains characters missing from an older checkpoint vocabulary, the trainer preserves the old checkpoint and starts a compatible Unicode model. This avoids corrupting embedding dimensions or silently training new characters as unknown tokens.

## Training behavior

1. The trainer splits complete conversation paragraphs into 90% training and 10% validation data.
2. It performs the requested maximum number of optimizer steps per run.
3. It evaluates train and validation loss and adjusts learning rate when validation stops improving.
4. It saves model weights, optimizer state, tokenizer vocabulary, global step, and random states.
5. The next run resumes from the latest checkpoint when the vocabulary is compatible.
6. Ctrl+C requests a safe checkpoint save.

## GitHub Actions

The training workflow is deliberately **bounded**. It no longer runs an infinite loop inside one GitHub Actions job. It runs a finite number of steps (100 by default; manual choices include 250, 500, or 1000), saves a checkpoint, uploads training artifacts for 30 days, and can resume from the Actions cache on a later scheduled or manual run. The workflow is scheduled every six hours.

GitHub-hosted runners are temporary and have execution limits. For long-term training, regularly download and back up the checkpoint artifacts.

## Generate text

```bash
python -m javus.generate --prompt "Assalomu alaykum" --max-new-tokens 40
```

## HTTP API

Run the FastAPI server after a checkpoint exists:

```bash
export JAVUS_CHECKPOINT_PATH=checkpoints/javus-alpha.pt
uvicorn javus.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /` — API information
- `GET /health` — model status
- `POST /chat` — text generation
- `GET /docs` — interactive API documentation

Set `JAVUS_API_KEY` to protect `POST /chat`. Never commit API keys or private checkpoint URLs.

## Roadmap

- [x] Python Transformer prototype
- [x] Unicode character tokenizer
- [x] Multilingual starter corpus
- [x] Checkpoint and resume support
- [x] Bounded GitHub Actions training
- [x] Core and dataset tests
- [ ] Larger balanced and licensed multilingual corpus
- [ ] Separate evaluation benchmark
- [ ] Improve tokenizer and model architecture
- [ ] Vision and audio research modules
- [ ] Javus AI 1.0

## License

Apache-2.0
