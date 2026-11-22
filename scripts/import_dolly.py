#!/usr/bin/env python3
"""Reproducibly build a 1,000-example English Javus corpus from Dolly 15k.

Keeps the existing authored seed examples and fills to TARGET using filtered Dolly
instruction/response pairs. Uses only Python standard library + dataset-viewer API.
"""
from __future__ import annotations
import json, re, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.txt"
TARGET = 1000
DATASET = "databricks/databricks-dolly-15k"
API = "https://datasets-server.huggingface.co/rows"
MAX_CHARS = 3500
BAD = re.compile(r"\b(?:http[s]?://|www\.)", re.I)
ENGLISH = re.compile(r"^[\x00-\x7F\u00A0-\u024F]+$")

def fetch_rows():
    rows = []
    for offset in range(0, 15000, 100):
        url = f"{API}?dataset={DATASET}&config=default&split=train&offset={offset}&length=100"
        req = urllib.request.Request(url, headers={"User-Agent": "Javus-Dataset-Builder/1.0"})
        with urllib.request.urlopen(req, timeout=45) as response:
            payload = json.load(response)
        rows.extend(item["row"] for item in payload.get("rows", []))
        if len(payload.get("rows", [])) < 100:
            break
    return rows

def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def valid(prompt, answer):
    if not prompt or not answer or len(prompt) > 1200 or len(answer) > MAX_CHARS:
        return False
    if not ENGLISH.match(prompt) or not ENGLISH.match(answer):
        return False
    if BAD.search(prompt) or BAD.search(answer):
        return False
    if prompt.lower() == answer.lower() or len(answer) < 12:
        return False
    return True

def parse_seed(text):
    examples = []
    for block in text.split("\n\n"):
        user = re.search(r"^User:\s*(.+(?:\n(?!Javus:).+)*)", block, re.M)
        javus = re.search(r"^Javus:\s*(.+(?:\n(?!User:).+)*)", block, re.M)
        if user and javus:
            examples.append((clean(user.group(1)), clean(javus.group(1))))
    return examples

def main():
    seed_text = TRAIN.read_text(encoding="utf-8")
    seed = parse_seed(seed_text)
    seen = {p.casefold() for p, _ in seed}
    chosen = []
    for row in fetch_rows():
        instruction, context, response = clean(row.get("instruction")), clean(row.get("context")), clean(row.get("response"))
        prompt = instruction
        if context:
            prompt = f"{instruction}\nContext: {context}"
        if not valid(prompt, response) or prompt.casefold() in seen:
            continue
        seen.add(prompt.casefold())
        chosen.append((prompt, response))
        if len(seed) + len(chosen) == TARGET:
            break
    if len(seed) + len(chosen) != TARGET:
        raise SystemExit(f"Could not reach {TARGET}; got {len(seed)+len(chosen)} examples. No file written.")
    output = [
        "# Javus Alpha English instruction corpus (1,000 examples)",
        "# Authored seed examples + filtered Databricks Dolly 15k records.",
        "# Dolly source: https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "# Dolly license: CC BY-SA 3.0. Keep attribution and license with redistributed corpus.",
        "",
    ]
    for prompt, answer in seed + chosen:
        output.extend([f"User: {prompt}", f"Javus: {answer}", ""])
    TRAIN.write_text("\n".join(output), encoding="utf-8")
    print(f"Saved exactly {TARGET} unique prompt/response examples ({len(seed)} seed + {len(chosen)} Dolly).")

if __name__ == "__main__":
    main()
