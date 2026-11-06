# Architecture

> Living document. Changes to anything in "Protocol" go through an RFC.

## Goals, in priority order

1. **Minimise communication.** Bytes on the wire and number of synchronisation points — the network, not compute, is the bottleneck for internet-scale training.
2. **Tolerate churn.** Devices join and leave at any time, including mid-round. Progress must never depend on a specific device.
3. **Tolerate heterogeneity.** A 4090, an M2 MacBook and a CPU server should all be able to contribute something useful.
4. **Tolerate adversaries.** Some workers will be buggy or malicious. Bad updates must not poison the model.
5. **Be transparent.** Every training run is reproducible and auditable.

## Components

```
                ┌────────────────────────────┐
                │        Coordinator         │  holds global weights θ, outer optimiser,
                │ (later: set of coordinators│  round schedule, membership, aggregation
                │  or fully peer-to-peer)    │
                └──────┬───────────────▲─────┘
           pull θ      │               │  compressed Δ
       (or diff of θ)  ▼               │
        ┌──────────────────────────────┴───────────────┐
        │  Workers (volunteer devices)                  │
        │  model replica · inner optimiser · data shard │
        │  compressor + error-feedback residuals        │
        └───────────────────────────────────────────────┘
```

| Module | File | Responsibility |
|---|---|---|
| Config | `src/yardai/config.py` | `DiLoCoConfig` — H, inner/outer optimiser, quorum, compression |
| Compression | `src/yardai/compression.py` | `NoCompression`, `Int8Compression`, `TopKCompression`; exact `nbytes()` |
| Training | `src/yardai/diloco.py` | `Worker` (local training, pseudo-gradients) and `Coordinator` (outer step) |
| Simulator | `src/yardai/simulate.py` | In-process swarm with dropout; compares bytes vs data-parallel |
| Tiny model | `src/yardai/tiny.py` | Tiny GPT + synthetic arithmetic corpus for CPU experiments |
| Base models | `src/yardai/models.py` | Candidate open-weight models and HF loader |

## Protocol (v0, reference)

Per round *r*:

1. Coordinator publishes θ_r.
2. Each online worker sets θ_local ← θ_r and runs H inner AdamW steps on its shard.
3. Worker computes Δ = θ_r − θ_local, adds its error-feedback residual, compresses, and uploads.
4. Coordinator decompresses and averages the Δs that arrived (if ≥ `min_quorum`), sets it as the gradient of θ and takes a Nesterov-SGD step (lr 0.7, momentum 0.9 by default).

Inner optimiser state stays on each worker across rounds.

## Known gaps / research directions

- **Synchronous rounds.** A round waits for everyone. Next: deadlines, asynchronous/stale updates, and Streaming DiLoCo (sync parameter fragments on a staggered schedule, overlapping communication with compute).
- **Download cost.** Workers currently pull full fp32 weights. Send compressed weight diffs instead.
- **Heterogeneous speed.** Fast devices should do more inner steps; weight their updates accordingly.
- **Model size.** Devices must currently hold the whole model. Larger stages need pipeline/expert sharding (SWARM-style) or smaller trainable subsets (e.g. low-rank adapters).
- **Trust.** Plain averaging is not robust. Need robust aggregation (trimmed mean, median, Krum-style), spot-check verification by recomputation, and reputation.
- **Data.** Licensed, documented, deduplicated open datasets with per-shard provenance.
- **Serving.** Undecided: dedicated cloud vs distributed (Petals-style) inference.
