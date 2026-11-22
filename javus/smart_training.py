"""Smart-training utilities for Javus Alpha."""

from dataclasses import asdict, dataclass


@dataclass
class SmartTrainingState:
    best_val_loss: float = float("inf")
    bad_evaluations: int = 0
    reductions: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict | None):
        return cls(**value) if value else cls()


class AdaptiveLearningRate:
    """Reduce LR when validation loss stops improving."""

    def __init__(self, optimizer, state=None, patience=3, factor=0.5, min_lr=1e-5):
        self.optimizer = optimizer
        self.state = state or SmartTrainingState()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr

    def update(self, validation_loss: float) -> bool:
        if validation_loss < self.state.best_val_loss:
            self.state.best_val_loss = validation_loss
            self.state.bad_evaluations = 0
            return False

        self.state.bad_evaluations += 1
        if self.state.bad_evaluations < self.patience:
            return False

        changed = False
        for group in self.optimizer.param_groups:
            old_lr = group["lr"]
            new_lr = max(self.min_lr, old_lr * self.factor)
            if new_lr < old_lr:
                group["lr"] = new_lr
                changed = True

        self.state.bad_evaluations = 0
        if changed:
            self.state.reductions += 1
        return changed


@__import__("torch").no_grad()
def evaluate(model, train_data, val_data, block_size, batch_size=8, batches=10, device="cpu"):
    import torch
    import torch.nn.functional as F

    model.eval()
    result = {}
    if len(train_data) > block_size + 1:
        losses = []
        for _ in range(batches):
            starts = torch.randint(0, len(train_data) - block_size - 1, (batch_size,))
            x = torch.stack([train_data[i:i + block_size] for i in starts]).to(device)
            y = torch.stack([train_data[i + 1:i + block_size + 1] for i in starts]).to(device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        result["train"] = sum(losses) / len(losses)
    else:
        result["train"] = None

    # Validation target tokens use preceding training text as context. This
    # keeps the model's block_size unchanged even when the validation split
    # is shorter than one full training window.
    if len(val_data) > 0 and len(train_data) >= block_size:
        combined = torch.cat((train_data, val_data))
        split_index = len(train_data)
        x = torch.stack([
            combined[index - block_size:index]
            for index in range(split_index, len(combined))
        ]).to(device)
        targets = val_data.to(device)
        logits, _ = model(x)
        result["validation"] = float(F.cross_entropy(logits[:, -1, :], targets).item())
    else:
        result["validation"] = None

    model.train()
    return result
