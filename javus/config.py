from dataclasses import asdict, dataclass
import math


@dataclass
class JavusConfig:
    """Configuration for the text-only Javus decoder Transformer."""

    vocab_size: int = 256
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        dimensions = (self.vocab_size, self.block_size, self.n_embd, self.n_head, self.n_layer)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in dimensions):
            raise ValueError("Model o'lchamlari musbat butun son bo'lishi kerak")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd n_head ga qoldiqsiz bo'linishi kerak")
        if not isinstance(self.dropout, (int, float)) or not math.isfinite(self.dropout):
            raise ValueError("dropout chekli son bo'lishi kerak")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 0.0 dan katta yoki teng va 1.0 dan kichik bo'lishi kerak")

    def to_dict(self) -> dict:
        return asdict(self)
