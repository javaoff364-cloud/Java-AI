from pathlib import Path


def test_training_corpus_has_multilingual_examples_and_unicode():
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "train.txt"
    corpus = corpus_path.read_text(encoding="utf-8")
    assert len(corpus.encode("utf-8")) >= 5000
    assert corpus.count("Foydalanuvchi:") >= 40
    assert corpus.count("Javus:") >= 100
    for marker in ("User:", "Пользователь:", "مرحبا", "你好", "ありがとう", "😊", "O‘", "Ў"):
        assert marker in corpus
    assert "\ufffd" not in corpus
