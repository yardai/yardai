# Learning path: distributed LLM training

A reading and doing list for going deep on LLMs and how to train them across unreliable, far-apart devices. Roughly in order. PRs that improve it are very welcome.

## 1. How LLMs work

- Karpathy, *Neural Networks: Zero to Hero* video series, ending with "Let's build GPT" — then build nanoGPT yourself.
- Vaswani et al., 2017 — *Attention Is All You Need*.
- Radford et al., 2019 — GPT-2 paper; Brown et al., 2020 — GPT-3 paper.
- Touvron et al., 2023 — LLaMA (modern architecture choices: RMSNorm, RoPE, SwiGLU).
- Hoffmann et al., 2022 — *Chinchilla* scaling laws: how many tokens per parameter.
- Hugging Face SmolLM2 report — how a strong ~100M–1.7B model is trained (our Stage 1 base).

**Do:** train the simulator's `TinyGPT`, then fine-tune SmolLM2-135M on one machine.

## 2. How training is distributed in a datacenter

- Data parallelism and all-reduce (ring all-reduce).
- Rajbhandari et al., 2020 — *ZeRO*; Shoeybi et al., 2019 — *Megatron-LM* (tensor parallelism); GPipe (pipeline parallelism).
- Mixed precision (bf16), gradient checkpointing, gradient accumulation.

**Do:** run PyTorch DDP and FSDP across two processes; measure bytes sent per step.

## 3. Communicating less

- McMahan et al., 2017 — *FedAvg* (federated averaging).
- Stich, 2018 — *Local SGD Converges Fast and Communicates Little*.
- Douillard et al., 2023 — **DiLoCo** (the core of YardAI v0). Then *Streaming DiLoCo* (2025) and *DiPaCo*.
- Prime Intellect — OpenDiLoCo and the INTELLECT-1 technical report (DiLoCo across continents).
- Peng et al., 2024 — *DeMo: Decoupled Momentum Optimization* (Nous Research).

**Do:** reproduce DiLoCo's H-vs-loss curve in the simulator (`--inner-steps 10..500`).

## 4. Compressing what you send

- Alistarh et al., 2017 — *QSGD*; Lin et al., 2018 — *Deep Gradient Compression*.
- Vogels et al., 2019 — *PowerSGD* (low-rank compression).
- Karimireddy et al., 2019 — *Error Feedback Fixes SignSGD* (why residuals matter).

**Do:** add a new compressor to `src/yardai/compression.py` and compare bytes vs final loss.

## 5. Unreliable and untrusted devices

- Ryabinin et al., 2023 — *SWARM Parallelism* (training large models on preemptible, heterogeneous devices).
- Borzunov et al., 2023 — *Petals* (distributed inference of large models).
- The Hivemind library (decentralised averaging over a DHT).
- Blanchard et al., 2017 — *Krum*; Yin et al., 2018 — trimmed mean / median aggregation (Byzantine robustness).

**Do:** add a malicious worker to the simulator and make aggregation survive it.
