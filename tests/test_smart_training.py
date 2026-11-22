import torch

from javus.config import JavusConfig
from javus.model import JavusLanguageModel
from javus.smart_training import AdaptiveLearningRate, SmartTrainingState, evaluate


def test_smart_training_reduces_learning_rate_after_plateau():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    controller = AdaptiveLearningRate(optimizer, patience=2, factor=0.5)
    assert controller.update(1.0) is False
    assert controller.update(1.1) is False
    assert controller.update(1.2) is True
    assert optimizer.param_groups[0]["lr"] == 0.0005
    assert controller.state.reductions == 1


def test_smart_training_state_round_trip():
    state = SmartTrainingState(best_val_loss=0.75, bad_evaluations=2, reductions=1)
    assert SmartTrainingState.from_dict(state.to_dict()) == state


def test_validation_evaluation_handles_short_validation_split():
    config = JavusConfig(block_size=4, n_embd=8, n_head=2, n_layer=1, dropout=0.0)
    model = JavusLanguageModel(config)
    train_data = torch.randint(0, 256, (20,))
    validation_data = torch.randint(0, 256, (3,))
    scores = evaluate(model, train_data, validation_data, block_size=4, batches=2)
    assert scores["train"] is not None
    assert scores["validation"] is not None
    assert torch.isfinite(torch.tensor(scores["validation"]))


def test_validation_evaluation_processes_data_in_batches():
    config = JavusConfig(block_size=8, n_embd=16, n_head=2, n_layer=1, dropout=0.0)
    model = JavusLanguageModel(config)
    train_data = torch.randint(0, 256, (32,))
    validation_data = torch.randint(0, 256, (17,))
    scores = evaluate(
        model, train_data, validation_data, block_size=8, batch_size=4, batches=2
    )
    assert scores["validation"] is not None
    assert torch.isfinite(torch.tensor(scores["validation"]))
