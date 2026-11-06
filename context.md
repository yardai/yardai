# YardAI — project context

## What it is

YardAI is a fully open-source LLM project whose USP is **training across a large, global, unreliable swarm of volunteer devices** with as little communication ("chattiness") as possible. Devices come online and go offline at any time.

- **Founder:** Rahul Garg. Initial personal focus: learning LLMs deeply and optimising them for distributed training.
- **GitHub:** `github.com/yardai/yardai` (remote `git@github.com:yardai/yardai.git`).
- **License:** Apache-2.0. Contributions under DCO sign-off.
- **Ownership intent:** top 5 contributors each own 10% of the company. Statement of intent only — legal terms (entity, measurement, vesting, eligibility) not yet defined. See GOVERNANCE.md.
- **Serving/inference:** undecided — dedicated cloud or distributed like training.
- **Research focus (differentiators):** (1) truly consumer devices — Macs/laptops/gaming PCs, 8–16 GB memory, Apple Silicon (MPS/MLX), thermal/power-aware; (2) extreme churn — join/leave anytime incl. mid-round, no single point of failure. Low communication alone is not a differentiator: Prime Intellect, Templar (Bittensor) and Nous (Psyche) already ship DiLoCo-style training.
- **Device donor rewards:** compute credits redeemable for YardAI inference tokens — non-transferable, no cash value, not crypto, separate from equity. Draft design: `docs/rfcs/0001-compute-credits.md`.
- **Founder's learning plan:** `docs/learning-path.md` (Phase 0 basics → LLMs → memory → datacenter distributed → DiLoCo/compression → consumer devices → churn → real swarm).

## Strategy

- Start small with existing open-weight models; climb a ladder: TinyGPT (simulator) → SmolLM2-135M → SmolLM2-360M → Qwen2.5-0.5B → Qwen2.5-1.5B+ (all Apache-2.0; re-verify model cards before use).
- Core algorithm: DiLoCo (local AdamW inner steps, Nesterov-SGD outer step on averaged pseudo-gradients), plus pseudo-gradient compression with error feedback.
- Research directions: Streaming DiLoCo, async/deadline rounds, compressed downloads, robust aggregation, sharding for models larger than one device.

## Tech stack

- Python ≥ 3.10, PyTorch ≥ 2.2. Optional `hf` extra: transformers, datasets.
- Tooling: hatchling build, ruff, pytest, pre-commit. GitHub Actions CI (ruff, pytest on 3.10–3.12 with CPU torch, DCO sign-off check). Dependabot.

## Repo layout

```
src/yardai/
  config.py        DiLoCoConfig (inner_steps H, inner_lr, outer_lr=0.7, outer_momentum=0.9,
                   min_quorum, compression none|int8|topk, topk_ratio, error_feedback)
  compression.py   Packet (+nbytes), NoCompression, Int8Compression, TopKCompression, get_compressor
  diloco.py        Worker(pull, train_round, pseudo_gradient), Coordinator(global_params, apply)
  tiny.py          TinyGPT, lm_loss, arithmetic_corpus (synthetic "a+b=c"), batches, evaluate
  models.py        LADDER of candidate base models; load() via transformers
  simulate.py      run() -> SimResult; CLI `yardai-sim`
tests/             test_compression.py, test_diloco.py
docs/              architecture.md, learning-path.md, rfcs/0000-template.md
.github/           CI workflow, issue/PR templates, dependabot
```

## Key APIs

- `Worker.pull(global_params)` → copy global weights in.
- `Worker.train_round(steps=None)` → H local steps; returns mean loss. Inner optimizer state persists across rounds.
- `Worker.pseudo_gradient(global_params)` → list of `Packet` (Δ = global − local, + residual, compressed). Tracks `bytes_sent`.
- `Worker.tokens_trained` / `Worker.last_round_tokens` → training tokens processed (credit accounting input).
- `Coordinator.apply(reports)` → averages decompressed Δ over reporting workers, outer SGD step; returns False if below `min_quorum`.
- `simulate.run(cfg, n_workers, rounds, dropout, ...)` → `SimResult(eval_loss, participants, bytes_uploaded, ddp_bytes_equivalent, skipped_rounds, credited_tokens, reduction)`. `credited_tokens[worker_id]` counts only tokens from updates used in an outer step.
- DDP baseline accounting: `inner_steps × fp32 param bytes` per reporting worker per round (upload only).

## Feature status

| Feature | Status |
|---|---|
| DiLoCo reference implementation (in-process) | ✅ |
| int8 / top-k compression + error feedback | ✅ |
| Simulator with device dropout (before and mid-round) | ✅ |
| Networked coordinator / worker client | ⏳ not started |
| Robust aggregation / verification | ⏳ not started |
| Real base-model training (SmolLM2-135M) | ⏳ not started |
| Serving | ❓ undecided |
| Compute-credit accounting (simulator only) | ✅ |
| Compute credits ledger / redemption | 📝 RFC 0001 draft |

## Decisions

- Apache-2.0 for code (permissive, patent grant). DCO rather than CLA.
- Contributor Covenant 2.1 adopted by reference.
- Big changes via RFCs in `docs/rfcs/`.
- Synthetic arithmetic corpus for simulator (no dataset licensing issues).
- Device donors are rewarded with inference-token credits for accepted, verified work — not a crypto token, not cash, not equity (RFC 0001, draft).

## Known issues

- Rounds are synchronous; slow workers hold everyone up (no deadlines yet).
- Downloads are full fp32 weights (not compressed).
- Plain averaging — not robust to malicious workers.
- Code is kept Python 3.9-compatible syntax-wise (`from __future__ import annotations`) only incidentally; supported is ≥ 3.10.
- No CoC/security contact email yet; no CODEOWNERS.

## Changelog

- **2026-11-06** — Project bootstrapped: OSS scaffolding (Apache-2.0, CONTRIBUTING, CoC, SECURITY, GOVERNANCE with ownership intent, CI, templates), reference DiLoCo worker/coordinator, int8/top-k compression with error feedback, in-process swarm simulator (`yardai-sim`), tests, README, architecture doc, learning path, roadmap. Sample run: 8 workers, 15 rounds, H=50, 20% dropout, int8 → eval loss 2.67 → 1.46 with 200× less upload than data-parallel.
- **2026-11-06** — Pushed to `github.com/yardai/yardai` (`main`).
- **2026-11-06** — Research focus set to consumer devices + extreme churn; learning path rewritten from basics around those two frontiers. Draft RFC 0001 (compute credits → YardAI inference tokens for device donors). Simulator now tracks `tokens_trained` per worker and `SimResult.credited_tokens` for accepted updates; README updated.
