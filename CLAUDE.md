# CLAUDE.md

Read [context.md](context.md) at the start of every session — it describes the project, architecture, APIs, feature status, decisions and known issues.

**Always keep context.md up to date**: whenever a change affects code structure, APIs, the training protocol, dependencies, feature status, decisions or known issues, update the relevant section of context.md in the same change and add a dated entry to its Changelog.

Roadmap/backlog lives in [todo.md](todo.md). Deeper design lives in [docs/architecture.md](docs/architecture.md).

## Working rules

- Python ≥ 3.10, `ruff` for lint/format, `pytest` for tests. Run `pytest` and `ruff check .` before committing.
- Every algorithm or protocol change gets a test; the simulator (`yardai-sim`) should be able to demonstrate it on CPU.
- Claims about communication or loss must come with the command that produced the numbers.
- Protocol, architecture, licensing or governance changes need an RFC in `docs/rfcs/`.
- Commits use Conventional Commits and are signed off (`git commit -s`).
- Never commit weights, datasets or secrets (see `.gitignore`).
- Don't change the ownership/equity wording in GOVERNANCE.md or README.md without the founder's explicit instruction.
