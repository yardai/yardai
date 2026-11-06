<div align="center">

# YardAI

**An open-source LLM, trained by everyone, on everyone's hardware.**

[![CI](https://github.com/yardai/yardai/actions/workflows/ci.yml/badge.svg)](https://github.com/yardai/yardai/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#status)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

Today, training a language model means a datacenter: thousands of GPUs in one building, wired together with interconnects that move hundreds of gigabytes per second. That puts frontier AI in the hands of a few companies.

Meanwhile, there is an enormous amount of compute sitting in gaming PCs, workstations, Macs and idle servers around the world — connected by ordinary internet.

**YardAI is building the training stack that turns that scattered compute into one model.** Devices join when they can, leave when they need to, and talk to the network as rarely as possible. Every line of code, every experiment and every training run happens in the open.

## Why it can work now

Standard data-parallel training synchronises gradients after *every* step. Over home internet, that's hopeless.

Recent research changed the picture. Methods like **DiLoCo** let each device train on its own for hundreds of steps, then exchange a single compact update. Combined with quantisation and sparsification, communication drops by orders of magnitude — and public projects have already trained multi-billion-parameter models across continents this way.

YardAI's focus is to push that further:

| | Datacenter training | YardAI |
|---|---|---|
| Sync frequency | every step | every *H* steps (hundreds) |
| What's sent | full fp32/bf16 gradients | compressed pseudo-gradients (int8, top-k, ...) |
| Hardware | identical GPUs, one building | whatever people have, anywhere |
| Node failure | stalls the job | expected; the swarm carries on |
| Who can join | the company | anyone |

**Less chatter is the whole game.** That's the problem we're optimising for.

## How it works

```
         ┌──────────────────────── round r ────────────────────────┐
         │                                                           │
 global  │   pull     train locally H steps     send compressed      │  outer step
 weights ├──► 🖥  ─────────────────────────────►  Δ = θ − θ_local  ──┼──► θ ← θ − η·avg(Δ)
   θ     ├──► 💻  ─────────────────────────────►  Δ (int8)         ──┤
         ├──► 🖥  ──────── ✗ went offline                            │  (average over
         └──► 🍎  ─────────────────────────────►  Δ (int8)         ──┘   whoever reported)
```

1. **Pull** the current global weights.
2. **Train locally** for *H* steps with an inner optimiser — zero network traffic.
3. **Send one compressed update** (the difference between where you started and where you ended).
4. **Aggregate** whatever arrived and take an outer optimiser step. Devices that dropped out are simply missing from the average.

Read more in [docs/architecture.md](docs/architecture.md).

## Try it in 60 seconds

The repo ships with a simulator: a swarm of unreliable devices training a tiny GPT in a single process on your laptop CPU.

```bash
git clone https://github.com/yardai/yardai.git && cd yardai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

yardai-sim --workers 8 --rounds 15 --inner-steps 50 --dropout 0.2 --compression int8
```

```
round  15  workers  6/8  eval loss 1.4611

eval loss 2.6742 -> 1.4611
uploaded 9.4 MB vs data-parallel 1881.2 MB  (200x less)
```

Eight devices, a fifth of them flaking out each round, **200× less upload than data-parallel training** — and the model still learns. Now imagine making that 1000×, at a billion parameters. That's the work.

## The model

We're starting small and climbing, rather than starting big and hoping:

| Stage | Base model | Goal |
|---|---|---|
| 0 | Tiny GPT (simulator) | Get the protocol right on a laptop |
| 1 | [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | First real run across volunteer devices |
| 2 | SmolLM2-360M | Scale the swarm, harden fault tolerance |
| 3 | Qwen2.5-0.5B | Multilingual, heterogeneous hardware |
| 4 | Qwen2.5-1.5B+ | Model sharding across consumer devices |

All candidates are open-weight under Apache-2.0. We'll continue pre-training and fine-tune existing models first, and move to training from scratch once the network is proven.

**Serving** is an open question: a dedicated cloud endpoint, a distributed swarm like training, or both. It'll be decided in the open via an [RFC](docs/rfcs/).

## Status

**Pre-alpha. Day zero.** What exists today:

- [x] Reference DiLoCo implementation with elastic participation
- [x] Pseudo-gradient compression (int8, top-k + error feedback) with exact byte accounting
- [x] Single-process swarm simulator with device dropout
- [ ] Real networked coordinator and worker client
- [ ] Byzantine-robust aggregation and update verification
- [ ] First volunteer training run (Stage 1)

The full roadmap is in [todo.md](todo.md).

## Open by default

- **Code:** Apache-2.0. Fork it, ship it, build on it.
- **Process:** decisions go through public [RFCs](docs/rfcs/); experiments are reproducible from a command line.
- **Training runs:** data sources, configs, logs and checkpoints will be published.

## Build it with us — and own part of it

YardAI is meant to belong to the people who build it. **The top 5 contributors will each own 10% of the company.** Contributions are tracked publicly and measured by impact, not commit count. The terms are still being formalised — read [GOVERNANCE.md](GOVERNANCE.md) for exactly what's decided and what isn't.

Good places to start:

- 🧪 **Research** — compression, sync schedules, async outer steps, robustness. Open a [research issue](https://github.com/yardai/yardai/issues/new?template=research.yml).
- 🛠 **Systems** — networking, NAT traversal, checkpointing, the worker client.
- 📚 **Learning in public** — we're writing a [learning path](docs/learning-path.md) for distributed LLM training. Improve it.

See [CONTRIBUTING.md](CONTRIBUTING.md) to get set up.

## Standing on shoulders

YardAI builds on ideas from DiLoCo and Streaming DiLoCo (Google DeepMind), OpenDiLoCo and INTELLECT-1 (Prime Intellect), DeMo / DisTrO (Nous Research), Hivemind, Petals and SWARM parallelism, and decades of federated and local-SGD research. Thank you to everyone who published their work openly.

## License

[Apache-2.0](LICENSE)
