# Security Policy

YardAI will run code on volunteers' devices and accept model updates from untrusted peers, so security is core to the project, not an afterthought.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Use GitHub's [private vulnerability reporting](https://github.com/yardai/yardai/security/advisories/new). Include steps to reproduce and the impact you expect. We aim to acknowledge reports within 72 hours.

## Areas we especially care about

- **Worker client safety** — anything that lets a coordinator or peer execute code or read files on a volunteer's device.
- **Poisoned updates** — ways a malicious worker can corrupt the shared model, bypass update validation, or claim credit for work it didn't do.
- **Supply chain** — dependencies, release artifacts and model checkpoints.

## Supported versions

YardAI is pre-alpha. Only the `main` branch is supported.
