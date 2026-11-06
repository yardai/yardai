# Contributing to YardAI

Thanks for being here. YardAI is built entirely in the open, and every contribution — code, research, docs, benchmarks, compute, reviews — moves it forward.

## Ways to contribute

- **Code** — training loop, compression, networking, fault tolerance, tooling. Look for [`good first issue`](https://github.com/yardai/yardai/labels/good%20first%20issue).
- **Research** — run experiments with the simulator, reproduce papers, propose ideas via an RFC.
- **Docs** — explain things you had to figure out the hard way.
- **Review** — careful review of other people's PRs is a first-class contribution.
- **Compute** — once the worker client exists, donate device time to training runs.

## Development setup

```bash
git clone https://github.com/yardai/yardai.git
cd yardai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

Run the simulator to see low-communication training in action:

```bash
yardai-sim --workers 8 --rounds 30 --inner-steps 50 --dropout 0.2 --compression int8
```

## Workflow

1. **Open or pick an issue** before starting anything non-trivial, so work isn't duplicated.
2. **Big changes need an RFC.** Anything that changes the training protocol, wire format, model architecture or governance starts as a PR adding `docs/rfcs/NNNN-short-title.md` (copy [the template](docs/rfcs/0000-template.md)).
3. **Fork, branch, commit.** Keep PRs focused; one logical change per PR.
4. **Sign off your commits** (Developer Certificate of Origin): `git commit -s`. This certifies you wrote the code or have the right to submit it under Apache-2.0. CI rejects unsigned commits.
5. **Tests and lint pass**: `pytest` and `ruff check . && ruff format --check .`.
6. **Experiments include numbers.** If a PR claims a speedup, a communication reduction, or better loss, include the command and the results so anyone can reproduce them.

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `refactor:`, `research:`, `chore:`.

## Code style

- Python ≥ 3.10, formatted and linted by `ruff`.
- Type hints on public functions.
- Prefer small, readable modules over clever abstractions. Distributed systems are hard enough.
- Every protocol or algorithm change gets a test — the simulator makes most things testable on a laptop CPU.

## Recognition and ownership

Contributions are tracked publicly. See [GOVERNANCE.md](GOVERNANCE.md) for how contributions are recognised, including the founding-contributor ownership program.

## Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
