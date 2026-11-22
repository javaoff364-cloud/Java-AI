#!/usr/bin/env python3
"""Build a balanced English + Uzbek Javus corpus from Hugging Face Aya Collection.

The script streams only the two requested language subsets and writes a reproducible
1,000-record UTF-8 corpus. Install the optional data extra before running it.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.txt"
DATASET = "CohereLabs/aya_collection_language_split"
TARGET = 1000
MAX_CHARS = 3500
BAD = re.compile(r"\b(?:https?://|www\.)", re.IGNORECASE)


def clean_pair(prompt: object, answer: object) -> str | None:
    """Return a safe, complete chat record or None for unusable source rows."""
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    prompt = " ".join(prompt.split())
    answer = " ".join(answer.split())
    if not prompt or not answer or len(prompt) + len(answer) > MAX_CHARS:
        return None
    if BAD.search(prompt) or BAD.search(answer):
        return None
    if "\ufffd" in prompt or "\ufffd" in answer:
        return None
    return f"User: {prompt}\nJavus: {answer}"


def unique_records(rows: Iterable[dict], limit: int) -> list[str]:
    records: list[str] = []
    seen: set[str] = set()
    for row in rows:
        record = clean_pair(row.get("inputs"), row.get("targets"))
        if record is None:
            continue
        prompt = record.split("\nJavus: ", 1)[0].casefold()
        if prompt in seen:
            continue
        seen.add(prompt)
        records.append(record)
        if len(records) >= limit:
            break
    return records


def load_language(language: str, limit: int, seed: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face importer uchun 'pip install -e .[data]' buyrug'ini bajaring."
        ) from exc

    try:
        stream = load_dataset(DATASET, language, split="train", streaming=True)
        stream = stream.shuffle(seed=seed, buffer_size=10_000)
        records = unique_records(stream, limit)
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face {language!r} subsetini yuklab bo'lmadi: {exc}"
        ) from exc
    if len(records) < limit:
        raise RuntimeError(
            f"{language} subsetidan {limit} ta emas, faqat {len(records)} ta sifatli "
            "namuna olindi. Importni to'xtatdim; qisman dataset yozilmadi."
        )
    return records


def authored_seed_records() -> list[str]:
    """Keep only the hand-written starter examples before imported Context records."""
    if not TRAIN.exists():
        return []
    text = TRAIN.read_text(encoding="utf-8")
    records = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip() and not part.lstrip().startswith("#")
    ]
    authored = []
    for record in records:
        if "\nContext:" in record:
            break
        if record.startswith("User: ") and "\nJavus: " in record:
            authored.append(record)
    return authored


def main() -> None:
    parser = argparse.ArgumentParser(description="Import bilingual Javus data from Hugging Face")
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=TRAIN)
    args = parser.parse_args()
    if args.target < 20 or args.target % 2:
        parser.error("--target kamida 20 va juft son bo'lishi kerak")

    seeds = authored_seed_records() if args.output.resolve() == TRAIN.resolve() else []
    if len(seeds) >= args.target:
        raise SystemExit("Qo'lda yozilgan seed namunalar target sonidan ko'p; targetni oshiring.")

    # Keep the starter examples, then add equal numbers from English and Uzbek.
    remaining = args.target - len(seeds)
    per_language = remaining // 2
    english = load_language("english", per_language, args.seed)
    uzbek = load_language("uzbek", remaining - per_language, args.seed)

    combined = seeds + english + uzbek
    random.Random(args.seed).shuffle(combined)
    header = (
        "# Javus bilingual instruction corpus\n"
        f"# Total records: {len(combined)} (English + Uzbek)\n"
        f"# Source: https://huggingface.co/datasets/{DATASET}\n"
        "# Dataset license: Apache-2.0; retain this attribution when redistributing.\n\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(header + "\n\n".join(combined) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"Yakunlandi: {len(combined)} ta namuna | qo'lda: {len(seeds)} | "
        f"English: {len(english)} | Uzbek: {len(uzbek)} | {args.output}"
    )


if __name__ == "__main__":
    main()
