"""Deterministic byte-level BPE tokenizer for Javus Alpha 2.

The base vocabulary contains all 256 byte values, so every UTF-8 string is encodable.
Merge rules are learned from the training corpus and stored in the checkpoint.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class ByteBPETokenizer:
    merges: list[tuple[int, int]]

    @classmethod
    def from_text(cls, text: str, num_merges: int = 128) -> "ByteBPETokenizer":
        if num_merges < 0:
            raise ValueError("num_merges cannot be negative")
        words = Counter(text.encode("utf-8").split())
        sequences = [list(word) for word in words]
        frequencies = list(words.values())
        merges: list[tuple[int, int]] = []
        for _ in range(num_merges):
            pairs: Counter[tuple[int, int]] = Counter()
            for sequence, frequency in zip(sequences, frequencies):
                for pair, count in _pair_counts(sequence).items():
                    pairs[pair] += count * frequency
            if not pairs:
                break
            pair, count = min(pairs.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break
            merged_id = 256 + len(merges)
            sequences = [_merge_pair(sequence, pair, merged_id) for sequence in sequences]
            merges.append(pair)
        return cls(merges=merges)

    @classmethod
    def from_state(cls, state: dict) -> "ByteBPETokenizer":
        if state.get("type") != "byte_bpe" or not isinstance(state.get("merges"), list):
            raise ValueError("Checkpoint BPE tokenizer state is invalid.")
        merges = []
        for pair in state["merges"]:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(not isinstance(value, int) or value < 0 for value in pair)):
                raise ValueError("Checkpoint BPE merge is invalid.")
            merges.append((pair[0], pair[1]))
        for index, (left, right) in enumerate(merges):
            if left >= 256 + index or right >= 256 + index:
                raise ValueError("BPE merges must reference tokens created earlier.")
        return cls(merges=merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        ranks = {pair: 256 + index for index, pair in enumerate(self.merges)}
        while len(tokens) > 1:
            candidates = [(ranks[(tokens[i], tokens[i + 1])], i)
                          for i in range(len(tokens) - 1)
                          if (tokens[i], tokens[i + 1]) in ranks]
            if not candidates:
                break
            _, index = min(candidates)
            pair = (tokens[index], tokens[index + 1])
            tokens = _merge_pair(tokens, pair, ranks[pair])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        vocabulary = [bytes([value]) for value in range(256)]
        for left, right in self.merges:
            vocabulary.append(vocabulary[left] + vocabulary[right])
        data = b"".join(vocabulary[int(token)] for token in tokens
                        if 0 <= int(token) < len(vocabulary))
        return data.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "byte_bpe", "merges": [list(pair) for pair in self.merges]}


def _pair_counts(sequence: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(sequence, sequence[1:]))


def _merge_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    output = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            output.append(new_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output
