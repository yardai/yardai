"""DiLoCo-style low-communication training: workers and coordinator.

The idea (Douillard et al., 2023, "DiLoCo: Distributed Low-Communication Training of
Language Models"):

1. Every worker starts a round from the current global weights.
2. It trains *locally* for H steps with its own AdamW optimizer — no network traffic.
3. It sends one pseudo-gradient: ``global_weights - local_weights``.
4. The coordinator averages pseudo-gradients from whichever workers reported and applies
   them with an outer Nesterov-SGD step.

Compared with data-parallel training (an all-reduce every step), this communicates
H times less often, and the pseudo-gradients compress well on top of that.

Fault tolerance in this reference implementation is simple by design: a worker that is
offline for a round just doesn't report, and the average is taken over those that did.
A worker that comes back pulls the latest global weights and carries on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import torch
from torch import nn

from yardai.compression import Compressor, Packet
from yardai.config import DiLoCoConfig

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[nn.Module, Batch], torch.Tensor]


def _params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


class Worker:
    """A device contributing compute. Holds its own model replica and inner optimizer."""

    def __init__(
        self,
        worker_id: int,
        model: nn.Module,
        data: Iterator[Batch],
        loss_fn: LossFn,
        cfg: DiLoCoConfig,
        compressor: Compressor,
    ):
        self.worker_id = worker_id
        self.model = model
        self.data = data
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.compressor = compressor
        self.inner_opt = torch.optim.AdamW(
            _params(model), lr=cfg.inner_lr, weight_decay=cfg.weight_decay
        )
        self._residuals: list[torch.Tensor] | None = None
        self.bytes_sent = 0
        self.tokens_trained = 0
        """Total training tokens processed, accepted or not (input to compute credits)."""
        self.last_round_tokens = 0

    @torch.no_grad()
    def pull(self, global_params: Sequence[torch.Tensor]) -> None:
        for p, g in zip(_params(self.model), global_params):
            p.copy_(g)

    def train_round(self, steps: int | None = None) -> float:
        """Run H local steps. Returns the mean training loss over the round."""
        steps = steps or self.cfg.inner_steps
        self.model.train()
        total = 0.0
        self.last_round_tokens = 0
        for _ in range(steps):
            batch = next(self.data)
            self.last_round_tokens += batch[0].numel()
            loss = self.loss_fn(self.model, batch)
            self.inner_opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(_params(self.model), self.cfg.grad_clip)
            self.inner_opt.step()
            total += loss.item()
        self.tokens_trained += self.last_round_tokens
        return total / steps

    @torch.no_grad()
    def pseudo_gradient(self, global_params: Sequence[torch.Tensor]) -> list[Packet]:
        """Compress ``global - local`` (with error feedback) into wire packets."""
        local = _params(self.model)
        if self._residuals is None:
            self._residuals = [torch.zeros_like(g) for g in global_params]
        packets = []
        for i, (g, p) in enumerate(zip(global_params, local)):
            delta = g - p.detach()
            if self.cfg.error_feedback:
                delta = delta + self._residuals[i]
            pkt = self.compressor.compress(delta)
            if self.cfg.error_feedback:
                self._residuals[i] = delta - self.compressor.decompress(pkt)
            packets.append(pkt)
        self.bytes_sent += sum(pkt.nbytes() for pkt in packets)
        return packets


class Coordinator:
    """Holds the global model and applies the outer optimizer step."""

    def __init__(self, model: nn.Module, cfg: DiLoCoConfig, compressor: Compressor):
        self.model = model
        self.cfg = cfg
        self.compressor = compressor
        self.outer_opt = torch.optim.SGD(
            _params(model),
            lr=cfg.outer_lr,
            momentum=cfg.outer_momentum,
            nesterov=cfg.outer_momentum > 0,
        )
        self.round = 0

    @torch.no_grad()
    def global_params(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in _params(self.model)]

    def param_bytes(self) -> int:
        return sum(p.numel() * 4 for p in _params(self.model))

    @torch.no_grad()
    def apply(self, reports: Sequence[Sequence[Packet]]) -> bool:
        """Average pseudo-gradients from the workers that reported and take an outer step.

        Returns False (and leaves the model untouched) if fewer than ``min_quorum``
        workers reported.
        """
        if len(reports) < max(1, self.cfg.min_quorum):
            return False
        for i, p in enumerate(_params(self.model)):
            avg = torch.stack([self.compressor.decompress(r[i]) for r in reports]).mean(0)
            p.grad = avg.to(p.dtype)
        self.outer_opt.step()
        self.outer_opt.zero_grad(set_to_none=True)
        self.round += 1
        return True
