# Learning path: from zero to distributed LLM training on consumer devices

For a strong software engineer who hasn't worked with LLMs yet. The goal is to reach YardAI's two research frontiers:

1. **Consumer devices.** Train on Macs, laptops and gaming PCs, not rented H100s.
2. **Extreme churn.** Devices join and leave constantly, mid-round, without warning.

Every phase pairs **Learn** (read or watch) with **Do** (build something, ideally in this repo) and ends with a **Checkpoint**: a question you should be able to answer without notes. The time estimates assume part-time study. Skip ahead wherever you already know the material.

---

## Phase 0 — Just enough maths and PyTorch (1–2 weeks)

You don't need a maths degree. You need intuition for gradients and fluency with tensors.

**Learn**
- 3Blue1Brown, *Neural Networks* video series (chapters 1–4: what a network is, gradient descent, backpropagation).
- Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd*.
- PyTorch's official *Learn the Basics* tutorial (tensors, autograd, `nn.Module`, optimisers).

**Do**
- Build micrograd yourself, without peeking.
- Train a small MLP in PyTorch on your Mac using the `mps` device, then on `cpu`, and time both.

**Checkpoint:** What does `loss.backward()` compute, and what does `optimizer.step()` do with it?

---

## Phase 1 — How LLMs work (3–4 weeks)

**Learn**
- Karpathy, *Neural Networks: Zero to Hero*: makemore parts 1–5, then *Let's build GPT*, then *Let's build the GPT Tokenizer*.
- Jay Alammar, *The Illustrated Transformer*.
- Vaswani et al., 2017, *Attention Is All You Need*. Read it after the videos; it'll make sense then.
- Touvron et al., 2023, *LLaMA*. Covers modern choices: RMSNorm, RoPE, SwiGLU.
- Hoffmann et al., 2022, *Chinchilla*. Scaling laws: how many tokens a model of size N needs.
- Allal et al., 2025, the *SmolLM2* paper, which describes our Stage 1 base model.

**Do**
- Read `src/yardai/tiny.py` line by line, then train it: `yardai-sim --workers 1 --inner-steps 1000 --rounds 5`.
- Follow Karpathy's *Let's reproduce GPT-2 (124M)* at small scale on your own machine.
- Load SmolLM2-135M with `transformers`, generate text, and fine-tune it briefly on a small dataset on your laptop.

**Checkpoint:** Draw a transformer block from memory. Why does a causal mask exist? What is perplexity?

---

## Phase 2 — Where the memory and time go (2–3 weeks)

This is the foundation for consumer devices. Before you can train on a laptop, you have to know what fits.

**Learn**
- *Memory anatomy of training.* For a model with N parameters trained with fp32 AdamW, budget about 16 bytes per parameter: weights 4, gradients 4, two Adam moments 8. Activations come on top and scale with batch size × sequence length × layers.
  - SmolLM2-135M ≈ 2.2 GB before activations, so it fits almost any laptop.
  - SmolLM2-360M ≈ 5.8 GB, fine on a 16 GB Mac.
  - Qwen2.5-1.5B ≈ 24 GB, which won't fit most consumer devices without the tricks below.
- Mixed precision (bf16/fp16), gradient checkpointing, gradient accumulation.
- Memory-saving training methods:
  - Dettmers et al., 2021, *8-bit Optimizers via Block-wise Quantization*.
  - Hu et al., 2021, *LoRA*, and Dettmers et al., 2023, *QLoRA*.
  - Zhao et al., 2024, *GaLore* (low-rank gradient projection).
- Compute vs memory bandwidth: why FLOPs alone don't predict speed (the "roofline" model).
- Hugging Face, *The Ultra-Scale Playbook*: read the first chapters on single-GPU memory and throughput.

**Do**
- Write `scripts/bench_device.py`. Given a model size, it measures peak memory, tokens/sec and power draw where available (`powermetrics` on macOS) on `cpu`, `mps` and `cuda`. Put the results in a table in `docs/experiments/`. This becomes YardAI's device-capability probe.
- Measure how tokens/sec changes on a MacBook over 30 minutes of sustained training (thermal throttling).

**Checkpoint:** Given a laptop with 16 GB of unified memory, what's the largest model you can fully train, and what would you change to double it?

---

## Phase 3 — Distributed training the datacenter way (2 weeks)

Learn the standard methods first, so you know exactly which assumptions YardAI breaks.

**Learn**
- Data parallelism and ring all-reduce (Baidu's 2017 ring all-reduce write-up, or the Ultra-Scale Playbook's chapter).
- Rajbhandari et al., 2020, *ZeRO*; PyTorch FSDP docs.
- Tensor parallelism (Megatron-LM, Shoeybi et al., 2019) and pipeline parallelism (GPipe, Huang et al., 2019).
- `torch.distributed.elastic` / `torchrun`: how PyTorch handles workers restarting.

**Do**
- Run PyTorch DDP with two processes on your laptop using the `gloo` backend. Count the bytes sent per step and compare with `yardai-sim`'s data-parallel baseline.
- Kill one DDP process mid-run and watch what happens. That failure is the problem YardAI exists to solve.

**Checkpoint:** Why does synchronous data-parallel training stall when one worker is slow or dies?

---

## Phase 4 — Communicating less (3 weeks)

This is YardAI's core algorithm.

**Learn**
- McMahan et al., 2017, *FedAvg*; Stich, 2018, *Local SGD Converges Fast and Communicates Little*.
- Douillard et al., 2023, **DiLoCo**. Read it twice, then read `src/yardai/diloco.py`.
- Jaghouar et al., 2024, *OpenDiLoCo*, and Prime Intellect's *INTELLECT-1* technical report: DiLoCo across continents in practice.
- Charles et al., 2025, *Scaling Laws for DiLoCo*: how DiLoCo behaves as models grow.
- Douillard et al., 2025, *Streaming DiLoCo*: sync parts of the model at a time and overlap communication with compute.
- Peng et al., 2024, *DeMo: Decoupled Momentum Optimization* (Nous Research).
- Compression:
  - Alistarh et al., 2017, *QSGD*.
  - Lin et al., 2018, *Deep Gradient Compression*.
  - Vogels et al., 2019, *PowerSGD*.
  - Karimireddy et al., 2019, *Error Feedback Fixes SignSGD*.

**Do**
- Sweep `--inner-steps` from 10 to 500 in the simulator and plot final loss against bytes sent. Write it up as `docs/experiments/0001-h-sweep.md`.
- Add a new compressor to `src/yardai/compression.py`: sign + scale, or low-rank in the style of PowerSGD. Compare it with int8.

**Checkpoint:** Why does DiLoCo use a *different* optimiser for the outer step than for the inner steps?

---

## Phase 5 — Research frontier A: consumer devices (ongoing)

**Learn**
- PyTorch MPS backend docs: what works on Apple Silicon and what doesn't.
- Apple's **MLX** framework, including its distributed-communication docs. It may train faster than PyTorch MPS on Macs.
- Diskin et al., 2021, *Distributed Deep Learning in Open Collaborations* (DeDLOC). Volunteer training on heterogeneous hardware; probably the closest prior work to YardAI.
- Ryabinin et al., 2023, *SWARM Parallelism*: training models too big for any one device on preemptible, heterogeneous hardware.
- Pluralis Research's publications on splitting a model across nodes over the internet ("protocol learning").
- Volunteer computing history: BOINC and Folding@home. They solved scheduling, crediting and cheating for a decade before LLMs existed.

**Open research questions**
1. **Heterogeneous speed.** A desktop with a 4090 does 20× more steps than a MacBook Air in the same time. Should the slow device do fewer inner steps, train on smaller batches, or have its update down-weighted? How does DiLoCo convergence change?
2. **Memory ceilings.** Can a 16 GB Mac contribute to a 1.5B model? Options: LoRA-style low-rank training, GaLore, 8-bit optimiser states, training only a subset of layers per device, or pipeline sharding across nearby devices.
3. **Be a good guest.** Training must never make someone's laptop unusable. Run only on mains power, cap temperature and use idle time, and still contribute something useful.
4. **Asymmetric networks.** Home connections upload far slower than they download. Compress the upload hardest.

**Do**
- Run the simulator on MPS. Then add *heterogeneous* simulated workers that run different numbers of inner steps per round, and test weighting strategies.
- Prototype the same training loop in MLX and compare tokens/sec and memory with PyTorch MPS.

---

## Phase 6 — Research frontier B: extreme churn (ongoing)

**Learn**
- Ryabinin et al., 2021, *Moshpit SGD*: averaging in small, shifting groups of unreliable devices.
- Liu et al., 2024, *Asynchronous Local-SGD Training for Language Modeling*: what breaks when updates arrive late, and how to fix it.
- Hivemind library docs: decentralised averaging over a DHT, and how peers find each other.
- Templar's *Covenant-72B* write-ups: permissionless pretraining where any node can join or leave at any time. Read their reports critically.
- Nous Research's *Psyche* docs: their coordination and fault-tolerance design.
- Distributed-systems basics: leases and heartbeats, at-least-once vs exactly-once delivery, consistent hashing. *Designing Data-Intensive Applications* (Kleppmann), chapters 8–9.
- Robust aggregation:
  - Blanchard et al., 2017, *Krum*.
  - Yin et al., 2018, coordinate-wise median and trimmed mean.
  - Prime Intellect's *TOPLOC* (verifying untrusted computation in INTELLECT-2).

**Open research questions**
1. **No rounds at all.** Can the coordinator accept a pseudo-gradient whenever it arrives, correcting for staleness, instead of waiting for a round to close?
2. **Churn-rate limits.** At what dropout rate does training stop converging? Measure it, then push it higher.
3. **Cold joins.** A brand-new device has no optimiser state and may hold old weights. How quickly can it become useful? Can it get optimiser state or a weight diff from a peer instead of the coordinator?
4. **Mid-round partial work.** A device that leaves after 30 of 50 steps still did real work. Can it send a partial update that's still useful?
5. **No single coordinator.** Replicate the coordinator, or go fully peer-to-peer (gossip), so the coordinator isn't the one machine whose failure stops training.

**Do**
- Extend `yardai-sim` with realistic churn traces: sessions that follow a heavy-tailed distribution, a diurnal pattern (people's laptops sleep at night in their time zone), and correlated failures (a whole region going offline).
- Implement a deadline-based round, where the coordinator closes the round after T seconds with whoever reported, and measure the effect on loss.
- Add an adversarial worker that sends scaled or random updates, then add robust aggregation.

---

## Phase 7 — Make it real (when Phases 4–6 have results)

- Write the Stage 1 RFC (wire protocol, membership, checkpointing, credits).
- Build the networked coordinator and worker client.
- Train SmolLM2-135M with real volunteer devices, and publish everything.

---

### Habits that help

- Keep a research log in `docs/experiments/`: hypothesis, command, result, what you learned. It's useful for you, and it's how contributors will follow and extend your work.
- Reproduce before you innovate. Re-run a paper's key result in the simulator before trying to beat it.
- Measure on real consumer hardware early. Laptops behave differently from what you'd expect on paper.
