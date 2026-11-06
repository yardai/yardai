import torch
from torch import nn

from yardai.compression import NoCompression
from yardai.config import DiLoCoConfig
from yardai.diloco import Coordinator
from yardai.simulate import run


def test_outer_step_with_no_momentum_moves_to_average():
    """With outer lr 1 and no momentum, DiLoCo reduces to plain parameter averaging."""
    cfg = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0)
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    coord = Coordinator(model, cfg, NoCompression())
    c = NoCompression()
    # Two workers ended at weights 1 and 3 -> pseudo-gradients -1 and -3 -> average 2.
    reports = [[c.compress(torch.full((1, 3), -1.0))], [c.compress(torch.full((1, 3), -3.0))]]
    assert coord.apply(reports)
    assert torch.allclose(model.weight, torch.full((1, 3), 2.0))


def test_quorum_skips_round():
    cfg = DiLoCoConfig(min_quorum=2)
    model = nn.Linear(2, 2)
    before = [p.clone() for p in model.parameters()]
    coord = Coordinator(model, cfg, NoCompression())
    assert not coord.apply([])
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_training_converges_with_dropout_and_int8():
    cfg = DiLoCoConfig(inner_steps=20, compression="int8")
    res = run(cfg, n_workers=4, rounds=6, dropout=0.25, seed=1)
    assert res.eval_loss[-1] < res.eval_loss[0] * 0.75
    # One int8 upload per round vs 20 fp32 all-reduces -> ~80x less.
    assert res.reduction > 50


def test_topk_with_error_feedback_still_learns():
    cfg = DiLoCoConfig(inner_steps=20, compression="topk", topk_ratio=0.05)
    res = run(cfg, n_workers=3, rounds=6, seed=2)
    assert res.eval_loss[-1] < res.eval_loss[0]


def test_credits_only_for_accepted_work():
    cfg = DiLoCoConfig(inner_steps=5, min_quorum=1)
    res = run(cfg, n_workers=4, rounds=4, dropout=0.4, batch_size=8, seq_len=16, seed=3)
    tokens_per_round = 5 * 8 * 16
    assert res.credited_tokens, "someone should have been credited"
    for credited in res.credited_tokens.values():
        assert credited % tokens_per_round == 0
    # Credited rounds never exceed rounds where a worker actually reported.
    assert sum(res.credited_tokens.values()) == sum(res.participants) * tokens_per_round
