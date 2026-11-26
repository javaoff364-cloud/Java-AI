import math

import pytest
import torch

from javus.config import JavusConfig
from javus.model import JavusLanguageModel


def small_config(**overrides):
    values = {
        "vocab_size": 384,
        "block_size": 32,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return JavusConfig(**values)


def test_small_model_forward_and_loss_are_finite():
    model = JavusLanguageModel(small_config())
    tokens = torch.randint(0, 384, (2, 16))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 16, 384)
    assert loss is not None and torch.isfinite(loss)


def test_model_rejects_non_matrix_input():
    model = JavusLanguageModel(small_config())
    with pytest.raises(ValueError, match="batch, sequence"):
        model(torch.ones(8, dtype=torch.long))


def test_model_rejects_empty_sequence():
    model = JavusLanguageModel(small_config())
    with pytest.raises(ValueError, match="bo'sh"):
        model(torch.empty((1, 0), dtype=torch.long))


def test_model_rejects_targets_with_different_shape():
    model = JavusLanguageModel(small_config())
    tokens = torch.randint(0, 384, (2, 16))
    targets = torch.randint(0, 384, (2, 15))
    with pytest.raises(ValueError, match="targets"):
        model(tokens, targets)


def test_generation_rejects_negative_token_count_and_non_finite_temperature():
    model = JavusLanguageModel(small_config())
    prompt = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(prompt, max_new_tokens=-1)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, temperature=float("nan"))
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, temperature=float("inf"))


def test_default_small_configuration_is_valid_and_compact():
    config = JavusConfig(vocab_size=384)
    config.validate()
    model = JavusLanguageModel(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 1_500_000
    assert math.isfinite(float(parameter_count))


@pytest.mark.parametrize(
    "changes",
    [
        {"n_head": 0},
        {"n_embd": 0},
        {"vocab_size": -1},
        {"block_size": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
    ],
)
def test_config_rejects_invalid_values(changes):
    config = small_config(**changes)
    with pytest.raises(ValueError):
        config.validate()
