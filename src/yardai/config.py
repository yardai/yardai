from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiLoCoConfig:
    """Hyperparameters for low-communication (DiLoCo-style) training.

    Defaults follow the DiLoCo paper (Douillard et al., 2023): AdamW inner optimizer,
    Nesterov-momentum SGD outer optimizer with lr 0.7 / momentum 0.9.
    """

    inner_steps: int = 50
    """Local optimizer steps per round (H). Communication happens once every H steps."""

    inner_lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    outer_lr: float = 0.7
    outer_momentum: float = 0.9

    min_quorum: int = 1
    """Minimum number of workers that must report for the coordinator to take an outer step."""

    compression: str = "none"
    """Pseudo-gradient compression: 'none' (fp32), 'int8', or 'topk'."""

    topk_ratio: float = 0.01
    """Fraction of entries kept per tensor when compression='topk'."""

    error_feedback: bool = True
    """Carry compression error into the next round so it is not lost."""
