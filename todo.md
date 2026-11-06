# Roadmap / backlog

## Now — Stage 0: protocol on a laptop

- [x] Reference DiLoCo worker + coordinator with elastic participation
- [x] int8 and top-k compression with error feedback and byte accounting
- [x] Single-process simulator with dropout
- [ ] Sweep H (inner steps) vs loss vs bytes; publish results in `docs/experiments/`
- [ ] Compress the download (weight diffs) as well as the upload
- [ ] Malicious-worker simulation + robust aggregation (trimmed mean / median)
- [ ] Heterogeneous-speed simulation (workers doing different numbers of inner steps)
- [ ] Deadline-based / asynchronous rounds; evaluate Streaming DiLoCo
- [ ] Write the Stage 1 RFC: wire protocol, membership, checkpointing

## Next — Stage 1: first real swarm (SmolLM2-135M)

- [ ] Networked coordinator (HTTP/gRPC) and worker client
- [ ] Device capability detection (CUDA / MPS / CPU, memory)
- [ ] NAT traversal / relay strategy
- [ ] Dataset choice with licence + provenance documentation
- [ ] Public training dashboard (loss, participants, bytes)
- [ ] Checkpoint publishing to Hugging Face under the `yardai` org

## Later

- [ ] Stage 2–4 model ladder (SmolLM2-360M, Qwen2.5-0.5B, 1.5B+)
- [ ] Model sharding for models that don't fit on one device
- [ ] Update verification (spot-check recomputation) and worker reputation
- [ ] Serving RFC: dedicated cloud vs distributed inference

## Project / org

- [ ] Push repo to `github.com/yardai/yardai`; enable private vulnerability reporting, branch protection, DCO check
- [ ] Set a contact email for Code of Conduct and security reports
- [ ] Add `CODEOWNERS` once maintainer GitHub handles are set
- [ ] Contribution ledger + scoring RFC (for the ownership program)
- [ ] Formalise the ownership program with legal counsel: entity, jurisdiction, measurement window, vesting, eligibility
- [ ] Community chat (Discord/Matrix) and link it from the README
