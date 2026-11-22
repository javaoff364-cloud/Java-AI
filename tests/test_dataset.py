from pathlib import Path


def test_training_corpus_is_english_only():
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "train.txt"
    corpus = corpus_path.read_text(encoding="utf-8")
    records = [part.strip() for part in corpus.split("\n\n") if part.strip() and not part.lstrip().startswith("#")]
    assert len(records) >= 50
    assert all(record.startswith("User:") for record in records)
    assert all("Javus:" in record for record in records)
    assert all(all(ord(char) < 128 for char in record) for record in records)
    assert "What is artificial intelligence?" in corpus
    assert "How should an AI assistant handle uncertainty?" in corpus
    assert "�" not in corpus
