"""Pseudo-gradient compression.

Every byte a worker uploads is a byte a home internet connection has to carry, so the
pseudo-gradients workers send at the end of each round are compressed. Each compressor
reports the exact number of bytes it would put on the wire, so experiments can compare
communication cost honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Packet:
    """A compressed tensor as it would be sent over the network."""

    shape: torch.Size
    values: torch.Tensor
    scale: torch.Tensor | None = None
    indices: torch.Tensor | None = None

    def nbytes(self) -> int:
        n = self.values.numel() * self.values.element_size()
        if self.scale is not None:
            n += 4  # one fp32 scale per tensor
        if self.indices is not None:
            n += self.indices.numel() * 4  # int32 indices on the wire
        return n


class Compressor:
    name = "base"

    def compress(self, t: torch.Tensor) -> Packet:
        raise NotImplementedError

    def decompress(self, p: Packet) -> torch.Tensor:
        raise NotImplementedError


class NoCompression(Compressor):
    """Plain fp32. The baseline."""

    name = "none"

    def compress(self, t: torch.Tensor) -> Packet:
        return Packet(shape=t.shape, values=t.detach().float().clone())

    def decompress(self, p: Packet) -> torch.Tensor:
        return p.values.view(p.shape)


class Int8Compression(Compressor):
    """Symmetric per-tensor int8 quantization (4x smaller than fp32)."""

    name = "int8"

    def compress(self, t: torch.Tensor) -> Packet:
        t = t.detach().float()
        scale = t.abs().max().clamp(min=1e-12) / 127.0
        q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)
        return Packet(shape=t.shape, values=q, scale=scale)

    def decompress(self, p: Packet) -> torch.Tensor:
        assert p.scale is not None
        return (p.values.float() * p.scale).view(p.shape)


class TopKCompression(Compressor):
    """Keep only the largest-magnitude fraction of entries; send them as fp16 + indices."""

    name = "topk"

    def __init__(self, ratio: float = 0.01):
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = ratio

    def compress(self, t: torch.Tensor) -> Packet:
        flat = t.detach().float().flatten()
        k = max(1, int(flat.numel() * self.ratio))
        _, idx = torch.topk(flat.abs(), k, sorted=False)
        return Packet(shape=t.shape, values=flat[idx].half(), indices=idx)

    def decompress(self, p: Packet) -> torch.Tensor:
        assert p.indices is not None
        out = torch.zeros(p.shape.numel(), dtype=torch.float32)
        out[p.indices] = p.values.float()
        return out.view(p.shape)


def get_compressor(name: str, topk_ratio: float = 0.01) -> Compressor:
    if name == "none":
        return NoCompression()
    if name == "int8":
        return Int8Compression()
    if name == "topk":
        return TopKCompression(topk_ratio)
    raise ValueError(f"unknown compression {name!r}; expected none, int8 or topk")
