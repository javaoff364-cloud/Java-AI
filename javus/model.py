import torch
from torch import nn
from torch.nn import functional as F

from .config import JavusConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: JavusConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.proj(output))


class FeedForward(nn.Module):
    def __init__(self, config: JavusConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: JavusConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class JavusLanguageModel(nn.Module):
    def __init__(self, config: JavusConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layer)))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / (2 * config.n_layer) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ff.net[2].weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2:
            raise ValueError("tokens [batch, sequence] shaklida bo'lishi kerak")
        batch, seq_len = tokens.shape
        if seq_len < 1:
            raise ValueError("Kirish ketma-ketligi bo'sh bo'lmasligi kerak")
        if seq_len > self.config.block_size:
            raise ValueError("Kirish ketma-ketligi block_size dan uzun")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("targets shakli tokens bilan bir xil bo'lishi kerak")
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.drop(self.token_embedding(tokens) + self.position_embedding(positions))
        x = self.final_norm(self.blocks(x))
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ):
        """Generate tokens with temperature and optional top-k sampling."""
        import math

        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("tokens bo'sh bo'lmagan [batch, sequence] tensor bo'lishi kerak")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens manfiy bo'lmasligi kerak")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature chekli va musbat bo'lishi kerak")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k kamida 1 bo'lishi kerak")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = tokens[:, -self.config.block_size:]
                logits, _ = self(context)
                logits = logits[:, -1, :] / temperature
                if top_k is not None and top_k < logits.size(-1):
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                tokens = torch.cat((tokens, next_token), dim=1)
            return tokens
        finally:
            self.train(was_training)
