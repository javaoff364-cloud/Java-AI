from pathlib import Path


def test_training_corpus_has_conversational_examples():
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "train.txt"
    corpus = corpus_path.read_text(encoding="utf-8")
    assert len(corpus.encode("utf-8")) >= 5000
    assert corpus.count("Foydalanuvchi:") >= 40
    assert corpus.count("Javus:") >= 40
