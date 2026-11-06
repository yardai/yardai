import pytest
import torch

from yardai.compression import Int8Compression, NoCompression, TopKCompression, get_compressor


def test_none_roundtrip_exact():
    t = torch.randn(10, 7)
    c = NoCompression()
    assert torch.equal(c.decompress(c.compress(t)), t)
    assert c.compress(t).nbytes() == t.numel() * 4


def test_int8_roundtrip_close_and_4x_smaller():
    t = torch.randn(256, 64)
    c = Int8Compression()
    pkt = c.compress(t)
    err = (c.decompress(pkt) - t).abs().max().item()
    assert err <= t.abs().max().item() / 127 + 1e-6
    assert pkt.nbytes() == t.numel() + 4


def test_topk_keeps_largest():
    t = torch.zeros(1000)
    t[[3, 500, 999]] = torch.tensor([5.0, -7.0, 2.0])
    c = TopKCompression(ratio=0.003)
    out = c.decompress(c.compress(t))
    assert torch.allclose(out, t)


def test_unknown_compressor():
    with pytest.raises(ValueError):
        get_compressor("zip")
