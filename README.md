# Javus AI

**Javus AI** is an independent Python/PyTorch neural-network research project. **Javus Alpha 2** is a text-only, from-scratch decoder-only Transformer experiment for a bilingual Uzbek + English assistant. Image and audio inputs are intentionally out of scope for this phase. It does not load pretrained model weights or call an external LLM.

## Alpha 2 foundation

- Python 3.10+ and PyTorch
- Decoder-only Transformer with causal self-attention and pre-normalized residual blocks
- GPT-style parameter initialization and tied input/output embeddings
- Deterministic byte-level BPE tokenizer learned from the current corpus
- UTF-8 byte fallback, including Uzbek apostrophes and other Unicode characters
- English + Uzbek instruction data from Hugging Face Aya Collection
- AdamW, cross-entropy, gradient clipping, validation monitoring, adaptive learning rate
- Resumable checkpoints, JSONL metrics, pytest, and manually triggered, strictly bounded GitHub Actions training

The default BPE setup uses a 256-byte base vocabulary plus up to 128 learned merges (384 tokens). The default Small Transformer uses 128 hidden dimensions, 4 attention heads, 4 layers, and a 128-token context window. Its parameter count is under one million (the exact count depends on the tokenizer vocabulary). This is a compact research baseline, not a production-scale assistant.

**Important:** the current 1,000-example corpus is only a pipeline smoke-test dataset, not enough to create a capable assistant. A useful model needs much larger, carefully reviewed, diverse data and held-out evaluation. A low training loss alone does not prove answer quality.

## GitHub-first development

GitHub is the primary development and training environment for this phase. Changes are made on feature branches and merged through pull requests after CI checks. Training and dataset import workflows are started manually from the Actions tab; pushes and pull requests run tests only.

To work locally when needed, clone the repository and install the project:

```bash
git clone https://github.com/javaoff364-cloud/Java-AI.git
cd Java-AI
python -m pip install --upgrade pip
python -m pip install -e ".[data,test]"
```

For Termux, use a supported Linux environment (such as Ubuntu in proot) if native PyTorch installation is unavailable.

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

Every training invocation is bounded: `--steps` sets the maximum optimizer steps for that run, and the CLI does not provide an unbounded/forever mode. GitHub Actions training is manual-only and has a job timeout. A later run can resume from the saved checkpoint.

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
