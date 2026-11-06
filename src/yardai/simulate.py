"""Simulate a swarm of unreliable devices training one model, in a single process.

Example:
    yardai-sim --workers 8 --rounds 30 --inner-steps 50 --dropout 0.2 --compression int8

Prints loss per round and compares bytes uploaded against a data-parallel baseline that
all-reduces fp32 gradients every step.
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass, field

import torch

from yardai.compression import get_compressor
from yardai.config import DiLoCoConfig
from yardai.diloco import Coordinator, Worker
from yardai.tiny import TinyGPT, arithmetic_corpus, batches, evaluate, lm_loss


@dataclass
class SimResult:
    eval_loss: list[float] = field(default_factory=list)
    participants: list[int] = field(default_factory=list)
    bytes_uploaded: int = 0
    ddp_bytes_equivalent: int = 0
    skipped_rounds: int = 0
    credited_tokens: dict[int, int] = field(default_factory=dict)
    """Per-worker training tokens from updates that were used in an outer step (RFC 0001)."""

    @property
    def reduction(self) -> float:
        return self.ddp_bytes_equivalent / max(1, self.bytes_uploaded)


def run(
    cfg: DiLoCoConfig,
    n_workers: int = 4,
    rounds: int = 10,
    dropout: float = 0.0,
    batch_size: int = 16,
    seq_len: int = 32,
    seed: int = 0,
    verbose: bool = False,
) -> SimResult:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    compressor = get_compressor(cfg.compression, cfg.topk_ratio)

    global_model = TinyGPT(seq_len=seq_len)
    coord = Coordinator(global_model, cfg, compressor)
    eval_data = arithmetic_corpus(2_000, seed=10_000)

    workers = [
        Worker(
            i,
            copy.deepcopy(global_model),
            batches(arithmetic_corpus(5_000, seed=seed * 1000 + i), batch_size, seq_len, seed + i),
            lm_loss,
            cfg,
            compressor,
        )
        for i in range(n_workers)
    ]

    res = SimResult(eval_loss=[evaluate(global_model, eval_data, seq_len)])
    for r in range(rounds):
        online = [w for w in workers if rng.random() >= dropout]
        g = coord.global_params()
        reports = []
        reporters = []
        for w in online:
            w.pull(g)
            w.train_round()
            # A device can also vanish mid-round, after doing the work but before reporting.
            if rng.random() < dropout / 2:
                continue
            reports.append(w.pseudo_gradient(g))
            reporters.append(w)
            res.ddp_bytes_equivalent += cfg.inner_steps * coord.param_bytes()
        if coord.apply(reports):
            for w in reporters:
                res.credited_tokens[w.worker_id] = (
                    res.credited_tokens.get(w.worker_id, 0) + w.last_round_tokens
                )
        else:
            res.skipped_rounds += 1
        res.participants.append(len(reports))
        res.eval_loss.append(evaluate(global_model, eval_data, seq_len))
        if verbose:
            print(
                f"round {r + 1:3d}  workers {len(reports):2d}/{n_workers}  "
                f"eval loss {res.eval_loss[-1]:.4f}"
            )

    res.bytes_uploaded = sum(w.bytes_sent for w in workers)
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--inner-steps", type=int, default=50)
    ap.add_argument(
        "--dropout", type=float, default=0.1, help="probability a device is offline for a round"
    )
    ap.add_argument("--compression", choices=["none", "int8", "topk"], default="int8")
    ap.add_argument("--topk-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = DiLoCoConfig(
        inner_steps=a.inner_steps, compression=a.compression, topk_ratio=a.topk_ratio
    )
    res = run(
        cfg, n_workers=a.workers, rounds=a.rounds, dropout=a.dropout, seed=a.seed, verbose=True
    )
    mb = 1024 * 1024
    print(f"\neval loss {res.eval_loss[0]:.4f} -> {res.eval_loss[-1]:.4f}")
    print(
        f"uploaded {res.bytes_uploaded / mb:.1f} MB vs data-parallel "
        f"{res.ddp_bytes_equivalent / mb:.1f} MB  ({res.reduction:.0f}x less)"
    )
    if res.skipped_rounds:
        print(f"{res.skipped_rounds} round(s) skipped for lack of quorum")


if __name__ == "__main__":
    main()
