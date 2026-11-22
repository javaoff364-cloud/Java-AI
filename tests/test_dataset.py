from pathlib import Path

import pytest

from javus.bpe_tokenizer import ByteBPETokenizer
from javus.train import split_corpus


def test_training_corpus_is_english_only():
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "train.txt"
    corpus = corpus_path.read_text(encoding="utf-8")
    records = [part.strip() for part in corpus.split("\\n\\n") if part.strip() and not part.lstrip().startswith("#")]
    assert len(records) == 1000
    assert all(record.startswith("User:") for record in records)
    assert all("Javus:" in record for record in records)
    assert all(all(ord(char) < 128 for char in record) for record in records)
    assert "What is artificial intelligence?" in corpus
    assert "How should an AI assistant handle uncertainty?" in corpus
    assert "�" not in corpus


def test_split_is_reproducible_and_keeps_conversations_whole():
    text = "\\n\\n".join(
        f"User: Question {i}?\\nJavus: Answer number {i}." for i in range(30)
    )
    tokenizer = ByteBPETokenizer.from_text(text, num_merges=8)
    train_a, val_a, count_a = split_corpus(text, tokenizer, seed=17)
    train_b, val_b, count_b = split_corpus(text, tokenizer, seed=17)
    assert count_a == count_b == 30
    assert train_a.tolist() == train_b.tolist()
    assert val_a.tolist() == val_b.tolist()
    assert len(train_a) > len(val_a) > 0


def test_split_rejects_duplicate_prompts():
    text = "User: Same question?\\nJavus: First answer.\\n\\n" + "\\n\\n".join(
        f"User: Other question {i}?\\nJavus: Answer {i}." for i in range(19)
    ) + "\\n\\nUser: Same question?\\nJavus: Different answer."
    tokenizer = ByteBPETokenizer.from_text(text, num_merges=4)
    with pytest.raises(ValueError, match="Takroriy"):
        split_corpus(text, tokenizer)


def test_split_rejects_missing_answer():
    text = "\\n\\n".join(
        [f"User: Question {i}?" + ("\\nJavus: Answer." if i else "") for i in range(20)]
    )
    tokenizer = ByteBPETokenizer.from_text(text, num_merges=4)
    with pytest.raises(ValueError, match="Javus"):
        split_corpus(text, tokenizer)
