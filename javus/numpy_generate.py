"""Run Javus Alpha using NumPy only (no PyTorch required on Android)."""
import argparse
import math
from pathlib import Path

import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def linear(x, weights, bias):
    y = x @ weights.T
    return y if bias is None else y + bias


def layer_norm(x, prefix, p, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * p[prefix + ".weight"] + p[prefix + ".bias"]


def gelu(x):
    # tanh approximation to GELU, stable and fast with NumPy on Android
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


class JavusNumpy:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.p = {k[3:]: z[k] for k in z.files if k.startswith("w__")}
        self.block_size = int(z["cfg__block_size"])
        self.n_embd = int(z["cfg__n_embd"])
        self.n_head = int(z["cfg__n_head"])
        self.n_layer = int(z["cfg__n_layer"])
        self.head_dim = self.n_embd // self.n_head
        self.embedding = self.p["token_embedding.weight"]

    def forward(self, tokens):
        p = self.p
        n = len(tokens)
        x = self.embedding[tokens] + p["position_embedding.weight"][:n]
        for i in range(self.n_layer):
            b = f"blocks.{i}."
            h = layer_norm(x, b + "ln1", p)
            qkv = linear(h, p[b + "attn.qkv.weight"], p[b + "attn.qkv.bias"])
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(n, self.n_head, self.head_dim).transpose(1, 0, 2)
            k = k.reshape(n, self.n_head, self.head_dim).transpose(1, 0, 2)
            v = v.reshape(n, self.n_head, self.head_dim).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            scores = np.where(mask[None, :, :], -1e9, scores)
            attn = softmax(scores)
            a = (attn @ v).transpose(1, 0, 2).reshape(n, self.n_embd)
            a = linear(a, p[b + "attn.proj.weight"], p[b + "attn.proj.bias"])
            x = x + a
            h = layer_norm(x, b + "ln2", p)
            ff = linear(h, p[b + "ff.net.0.weight"], p[b + "ff.net.0.bias"])
            ff = gelu(ff)
            ff = linear(ff, p[b + "ff.net.2.weight"], p[b + "ff.net.2.bias"])
            x = x + ff
        x = layer_norm(x, "final_norm", p)
        return x @ self.embedding.T

    def generate(self, prompt, max_new_tokens=60, temperature=0.8, seed=None):
        rng = np.random.default_rng(seed)
        tokens = list(prompt.encode("utf-8"))
        if not tokens:
            tokens = [32]
        for _ in range(max_new_tokens):
            context = np.asarray(tokens[-self.block_size:], dtype=np.int64)
            logits = self.forward(context)[-1] / max(temperature, 1e-5)
            probs = softmax(logits)
            nxt = int(rng.choice(len(probs), p=probs))
            tokens.append(nxt)
        return bytes(tokens).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Generate text with Javus Alpha (NumPy)")
    parser.add_argument("--model", default="checkpoints/javus-alpha.npz")
    parser.add_argument("--prompt", default="Assalomu alaykum")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model topilmadi: {model_path}")
    model = JavusNumpy(model_path)
    print("[Javus Alpha - NumPy]")
    print(model.generate(args.prompt, args.max_new_tokens, args.temperature, args.seed))


if __name__ == "__main__":
    main()
