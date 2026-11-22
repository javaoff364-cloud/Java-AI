#!/usr/bin/env python3
"""Build a reproducible English + Uzbek corpus from Hugging Face Aya Collection.

Only the selected language subsets are streamed; the full 155 GB collection is
never downloaded. Install the optional data extra before running this script.
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


def assemble_corpus(english: list[str], uzbek: list[str], target: int, seed: int = 42) -> list[str]:
    """Validate language quotas and return a deterministic, balanced record list."""
    if target < 20 or target % 2:
        raise ValueError("target kamida 20 va juft son bo'lishi kerak")
    half = target // 2
    if len(english) != half or len(uzbek) != half:
        raise ValueError(
            f"Balanslangan corpus uchun {half} ta English va {half} ta Uzbek namuna kerak."
        )
    combined = list(english) + list(uzbek)
    if len({record.split("\nJavus: ", 1)[0].casefold() for record in combined}) != target:
        raise ValueError("Corpus ichida takroriy savollar bor.")
    random.Random(seed).shuffle(combined)
    return combined


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
            "namuna olindi. Qisman dataset yozilmadi."
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Import bilingual Javus data from Hugging Face")
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=TRAIN)
    args = parser.parse_args()
    if args.target < 20 or args.target % 2:
        parser.error("--target kamida 20 va juft son bo'lishi kerak")

    quota = args.target // 2
    english = load_language("english", quota, args.seed)
    uzbek = load_language("uzbek", quota, args.seed)
    records = assemble_corpus(english, uzbek, args.target, args.seed)

    header = (
        "# Javus bilingual instruction corpus\n"
        f"# Total records: {len(records)} (English: {quota}, Uzbek: {quota})\n"
        f"# Source: https://huggingface.co/datasets/{DATASET}\n"
        "# Dataset license: Apache-2.0; retain this attribution when redistributing.\n\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(header + "\n\n".join(records) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"Yakunlandi: {len(records)} ta namuna | English: {quota} | "
        f"Uzbek: {quota} | {args.output}"
    )


if __name__ == "__main__":
    main()
