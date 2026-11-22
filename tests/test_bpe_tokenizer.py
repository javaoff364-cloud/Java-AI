from javus.bpe_tokenizer import ByteBPETokenizer


def test_byte_bpe_round_trip_english():
    text = "Hello, Javus! A language model learns patterns from text."
    tokenizer = ByteBPETokenizer.from_text(text, num_merges=64)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == 256 + len(tokenizer.merges)


def test_byte_bpe_round_trip_unicode_and_unseen_text():
    training = "Hello world. Javus is an assistant."
    tokenizer = ByteBPETokenizer.from_text(training, num_merges=32)
    unseen = "こんにちは — O‘zbekiston 😊"
    assert tokenizer.decode(tokenizer.encode(unseen)) == unseen


def test_byte_bpe_state_round_trip_is_stable():
    text = "the the the model model learns learns"
    original = ByteBPETokenizer.from_text(text, num_merges=32)
    restored = ByteBPETokenizer.from_state(original.to_dict())
    assert restored.encode(text) == original.encode(text)
    assert restored.decode(original.encode(text)) == text


def test_byte_bpe_training_is_deterministic():
    text = "one two two three three three"
    first = ByteBPETokenizer.from_text(text, num_merges=24)
    second = ByteBPETokenizer.from_text(text, num_merges=24)
    assert first.to_dict() == second.to_dict()
