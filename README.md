# Javus AI

**Javus AI** is an independent Python/PyTorch neural-network research project. **Javus Alpha 2** is a from-scratch, decoder-only Transformer experiment intended to grow into a bilingual Uzbek + English conversational assistant. It does not load pretrained model weights or call an external LLM.

## Alpha 2 foundation

- Python 3.10+ and PyTorch
- Decoder-only Transformer with causal self-attention and pre-normalized residual blocks
- GPT-style parameter initialization and tied input/output embeddings
- Deterministic byte-level BPE tokenizer learned from the current corpus
- UTF-8 byte fallback, including Uzbek apostrophes and other Unicode characters
- English + Uzbek instruction data from Hugging Face Aya Collection
- AdamW, cross-entropy, gradient clipping, validation monitoring, adaptive learning rate
- Resumable checkpoints, JSONL metrics, pytest, and bounded GitHub Actions training

**Important:** the current 1,000-example corpus is only a pipeline smoke-test dataset, not enough to create a capable assistant. A useful model needs much larger, carefully reviewed, diverse data and held-out evaluation. A low training loss alone does not prove answer quality.

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
pip install -e ".[data,test]"
```

## Hugging Face dataset

The importer uses **CohereLabs/aya_collection_language_split**, streaming the `english` and `uzbek` subsets rather than downloading the entire multilingual collection. The dataset provides instruction/response pairs and is published under Apache-2.0. The importer:
- takes equal numbers of examples from English and Uzbek,
- filters empty, duplicate, overly long, URL-containing, and invalid-Unicode rows,
- shuffles records deterministically,
- writes UTF-8 atomically and refuses to save a partial corpus.

Run locally:

```bash
python scripts/import_dolly.py --target 1000
```

Or open **Actions → Import bilingual Hugging Face dataset → Run workflow** to build, test, and commit the selected corpus size to the selected branch. This workflow is manual-only; it does not start model training.

Source: https://huggingface.co/datasets/CohereLabs/aya_collection_language_split  
License: Apache-2.0. Keep the source attribution and license information when redistributing data. Review examples for quality and suitability before using them in a release.

The conversation format is:

```text
User: O‘zbekiston haqida gapirib ber.
Javus: O‘zbekiston Markaziy Osiyoda joylashgan.
```

Keep evaluation questions separate from training data to avoid testing memorization.

## Tests

```bash
python -m pytest -q
```

## Train Javus Alpha 2

Run one bounded session (100 optimizer steps by default):

```bash
python -m javus.train --steps 100 --batch-size 8
```

The first Alpha 2 run detects an older tokenizer checkpoint, preserves it, and initializes a byte-BPE model. A checkpoint is resumed only when both the corpus SHA-256 and BPE vocabulary match exactly; changed training data starts a fresh run while archiving the previous checkpoint. Train/validation splitting shuffles complete conversation records with a fixed seed, preventing source-order bias and keeping each answer paired with its prompt.

To intentionally start the Alpha 2 model from scratch:

```bash
python -m javus.train --steps 100 --batch-size 8 --fresh
```

Training is bounded by default. The optional `--forever` flag is for deliberate local experiments only; it must not be used in GitHub Actions.

## Checkpoints and metrics

- `checkpoints/javus-alpha.pt` — latest model, optimizer, BPE merges, step, and RNG state
- `checkpoints/javus-alpha-best.pt` — best validation checkpoint
- `checkpoints/javus-alpha-metrics.jsonl` — per-run metrics
- `checkpoints/javus-alpha-pre-alpha2.pt` — preserved previous model when migrating to Alpha 2

## Generate text

```bash
python -m javus.generate --prompt "Salom, qalaysan?" --max-new-tokens 40
```

## HTTP API

After a checkpoint exists:

```bash
export JAVUS_CHECKPOINT_PATH=checkpoints/javus-alpha.pt
uvicorn javus.api:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /`, `GET /health`, `POST /chat`, and `GET /docs`. Set `JAVUS_API_KEY` to protect `POST /chat`. Never commit API keys or private checkpoint URLs.

## GitHub Actions

The training workflow is manual-only and bounded. It runs tests first, trains for one selected step budget, saves a resumable checkpoint, and uploads checkpoints, metrics, and logs as artifacts for 30 days. It does not start recurring training automatically. A separate CI workflow runs tests on pushes and pull requests.

## Roadmap

- [x] Python Transformer prototype
- [x] Byte-level BPE tokenizer foundation
- [x] Checkpoint and resume support
- [x] Bounded GitHub Actions training
- [x] Core, dataset, and tokenizer tests
- [x] Hugging Face English + Uzbek data importer
- [ ] Expand the bilingual corpus substantially with reviewed/licensed data
- [ ] Build a held-out bilingual assistant evaluation benchmark
- [ ] Improve conversation formatting and generation quality
- [ ] Train and evaluate larger model sizes
- [ ] Research vision and audio after text quality is established

## License

Apache-2.0
