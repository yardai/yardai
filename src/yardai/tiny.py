"""A tiny GPT and a synthetic dataset, so distributed-training ideas can be tested on a laptop CPU.

This is a research harness, not the YardAI model. Real runs will start from an existing
open-weight base model (see ``yardai.models``).
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

VOCAB = "0123456789+=\n"
STOI = {c: i for i, c in enumerate(VOCAB)}


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        d_model: int = 64,
        n_layer: int = 2,
        n_head: int = 4,
        seq_len: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.size(1)
        mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        x = self.blocks(x, mask=mask)
        return self.head(self.ln(x))


def lm_loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x, y = batch
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def arithmetic_corpus(n_lines: int, seed: int) -> torch.Tensor:
    """Lines like ``37+5=42``. Learnable structure, zero licensing concerns."""
    rng = random.Random(seed)
    text = "".join(
        f"{a}+{b}={a + b}\n"
        for a, b in ((rng.randrange(100), rng.randrange(100)) for _ in range(n_lines))
    )
    return torch.tensor([STOI[c] for c in text], dtype=torch.long)


def batches(
    data: torch.Tensor, batch_size: int, seq_len: int, seed: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    while True:
        ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=g)
        x = torch.stack([data[i : i + seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
        yield x, y


@torch.no_grad()
def evaluate(model: nn.Module, data: torch.Tensor, seq_len: int, n_batches: int = 8) -> float:
    model.eval()
    it = batches(data, 32, seq_len, seed=12345)
    loss = sum(lm_loss(model, next(it)).item() for _ in range(n_batches)) / n_batches
    model.train()
    return loss
