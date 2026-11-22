from dataclasses import asdict, dataclass


@dataclass
class JavusConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd n_head ga qoldiqsiz bo'linishi kerak")
        if min(self.vocab_size, self.block_size, self.n_embd, self.n_head, self.n_layer) <= 0:
            raise ValueError("Model o'lchamlari musbat bo'lishi kerak")

    def to_dict(self) -> dict:
        return asdict(self)
