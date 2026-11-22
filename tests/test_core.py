import torch

from javus.config import JavusConfig
from javus.model import JavusLanguageModel
from javus.tokenizer import ByteTokenizer


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    sample = "Salom, Javus! O'zbekcha:  salom"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample


def test_model_forward_and_loss():
    config = JavusConfig(block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    model = JavusLanguageModel(config)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)


def test_causal_generation_shape():
    config = JavusConfig(block_size=16, n_embd=32, n_head=4, n_layer=1, dropout=0.0)
    model = JavusLanguageModel(config)
    result = model.generate(torch.tensor([[65, 66]]), max_new_tokens=3)
    assert result.shape == (1, 5)
