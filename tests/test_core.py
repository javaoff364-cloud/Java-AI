import pytest
import torch

from javus.config import JavusConfig
from javus.model import JavusLanguageModel
from javus.tokenizer import ByteTokenizer


def tiny_config():
    return JavusConfig(vocab_size=256, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)


def test_byte_tokenizer_round_trip_uzbek():
    tokenizer = ByteTokenizer()
    text = "Assalomu alaykum! O‘zbekiston — go‘zal."
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert max(tokenizer.encode(text)) < tokenizer.vocab_size


def test_config_rejects_incompatible_attention_heads():
    config = tiny_config()
    config.n_embd = 30
    with pytest.raises(ValueError):
        config.validate()


def test_model_forward_returns_logits_and_loss():
    model = JavusLanguageModel(tiny_config())
    tokens = torch.randint(0, 256, (2, 12))
    targets = torch.randint(0, 256, (2, 12))
    logits, loss = model(tokens, targets)
    assert logits.shape == (2, 12, 256)
    assert loss is not None and torch.isfinite(loss)


def test_model_rejects_context_longer_than_block_size():
    model = JavusLanguageModel(tiny_config())
    with pytest.raises(ValueError, match="block_size"):
        model(torch.randint(0, 256, (1, 17)))


def test_causal_attention_does_not_see_future_tokens():
    torch.manual_seed(7)
    model = JavusLanguageModel(tiny_config()).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 99, 100]])
    with torch.inference_mode():
        logits_a, _ = model(first)
        logits_b, _ = model(changed_future)
    assert torch.allclose(logits_a[:, :2], logits_b[:, :2], atol=1e-6)


def test_generate_appends_requested_number_of_tokens():
    model = JavusLanguageModel(tiny_config()).eval()
    prompt = torch.tensor([[65, 66, 67]])
    result = model.generate(prompt, max_new_tokens=5)
    assert result.shape == (1, 8)


def test_top_k_one_selects_only_the_most_likely_token():
    torch.manual_seed(3)
    model = JavusLanguageModel(tiny_config()).eval()
    prompt = torch.tensor([[65, 66, 67]])
    result = model.generate(prompt, max_new_tokens=4, top_k=1)
    assert result.shape == (1, 7)
    assert model.training is False


def test_generate_rejects_invalid_sampling_parameters():
    model = JavusLanguageModel(tiny_config())
    prompt = torch.tensor([[65, 66]])
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, temperature=0)
    with pytest.raises(ValueError, match="top_k"):
        model.generate(prompt, top_k=0)
