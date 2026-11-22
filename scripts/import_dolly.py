#!/usr/bin/env python3
"""Reproducibly build a 1,000-example English Javus corpus from Dolly 15k.

Keeps the existing authored seed examples and fills to TARGET using filtered Dolly
instruction/response pairs. Uses only Python standard library + dataset-viewer API.
"""
from __future__ import annotations
import json, re, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.txt"
TARGET = 1000
DATASET = "databricks/databricks-dolly-15k"
SOURCE_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl?download=true"
MAX_CHARS = 3500
BAD = re.compile(r"\b(?:http[s]?://|www\.)", re.I)
ENGLISH = re.compile(r"^[\x00-\x7F\u00A0-\u024F]+$")

def fetch_rows():
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Javus-Dataset-Builder/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                raw = response.read().decode("utf-8")
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            time.sleep(min(20, 2 ** attempt))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError("Could not download Dolly JSONL after retries")

