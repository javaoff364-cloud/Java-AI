"""English character tokenizer for Javus Alpha.

Character-level tokens preserve English text. The vocabulary is fitted from the corpus
and serialized into each checkpoint; the tokenizer can still represent Unicode when present.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CharTokenizer:
    """Maps Unicode code points to stable integer IDs."""

    itos: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        common = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " '’‘-_.,!?;:()[]{}<>/@#$%&*+=\\\n\r\t"
        )
        chars = sorted(set(text) | set(common))
        return cls(itos=["<unk>", *chars])

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        values = state.get("itos")
        if not isinstance(values, list) or not values or values[0] != "<unk>":
            raise ValueError("Checkpoint tokenizer lug'ati noto'g'ri.")
        if any(not isinstance(char, str) for char in values):
            raise ValueError("Tokenizer lug'atida noto'g'ri qiymat bor.")
        if len(set(values)) != len(values):
            raise ValueError("Tokenizer lug'atida takroriy token bor.")
        return cls(itos=values)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def stoi(self) -> dict[str, int]:
        return {char: index for index, char in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        lookup = self.stoi
        return [lookup.get(char, 0) for char in text]

    def decode(self, tokens: list[int]) -> str:
        output = []
        for token in tokens:
            index = int(token)
            output.append(self.itos[index] if 0 <= index < len(self.itos) else "�")
        return "".join(char for char in output if char != "<unk>")

    def to_dict(self) -> dict:
        return asdict(self)


class ByteTokenizer:
    """Legacy UTF-8 byte tokenizer, retained for old checkpoints only."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(max(0, min(255, int(token))) for token in tokens).decode("utf-8", errors="replace")
