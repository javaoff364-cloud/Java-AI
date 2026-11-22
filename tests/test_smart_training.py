import torch

from javus.smart_training import AdaptiveLearningRate, SmartTrainingState


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
