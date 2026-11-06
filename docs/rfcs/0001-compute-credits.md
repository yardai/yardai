# RFC 0001: Compute credits — paying device donors in YardAI model tokens

- **Status:** Draft
- **Author(s):** Rahul Garg
- **Created:** 2026-11-06
- **Discussion:** _open a PR / issue_

## Summary

People who lend their devices for training earn **YardAI compute credits**. Credits are redeemed for **inference tokens** on YardAI models: you helped train it, so you get to use it. Credits are earned only for verified, useful work, recorded on a public ledger, and are separate from the founding-contributor equity program in [GOVERNANCE.md](../../GOVERNANCE.md).

## Motivation

YardAI needs a large, steady supply of volunteer compute. Goodwill alone (the BOINC / Folding@home model) works for some people. A direct, useful reward brings in more and keeps them around. Paying in *use of the thing you helped build* fits the project:

- it costs the project serving capacity, not cash it doesn't have;
- it's tied to YardAI's success, and the credits get more valuable as the models improve;
- it avoids the crypto-token approach (Bittensor/Templar, Psyche, Gensyn), which is one of the things that sets YardAI apart.

## Proposal

### 1. What earns credits

Credits measure **useful compute that was accepted**, not time online or electricity used.

```
work_units(update) = tokens_trained × model_flops_per_token
credits            = work_units × acceptance × reliability_bonus
```

- **tokens_trained** — training tokens the worker processed in the round (batch size × sequence length × inner steps). Counted by the worker, cross-checked by the coordinator.
- **model_flops_per_token** — a fixed constant per model stage (≈ 6 × parameter count). A token on a 1.5B model is worth more than a token on a 135M model.
- **acceptance** — 1 if the update was received *and* used in the outer step; 0 if it was late, failed validation, or arrived after the round closed. Partial credit for partial work is an open question (see the learning path, Phase 6).
- **reliability_bonus** — optional and small (e.g. up to +10%), for devices with a long record of accepted updates. It rewards steady participation without letting big operators dominate.

Credits are tracked per stage, so the rate can be revisited when the model changes.

### 2. Verification (no credit for fake work)

A worker could claim work it didn't do, or send garbage. Layered defences:

1. **Statistical checks on every update** — norm bounds, agreement with other workers' pseudo-gradients (cosine similarity), robust aggregation outlier flags.
2. **Spot-check recomputation** — the coordinator gives a small random fraction of rounds to a trusted re-computer with the same seed and data shard, and compares results within a tolerance.
3. **Penalties** — a failed spot-check voids that worker's credits for a window and lowers its reputation. Repeated failures ban the account.
4. **Sybil resistance** — one account per verified identity (GitHub OAuth to start). Device fingerprints are rate-limited per account.

### 3. The ledger

- A public, append-only log. For every round: round id, worker account, tokens trained, accepted yes/no, credits awarded.
- The coordinator **signs a receipt** for each accepted update and sends it to the worker. Workers keep their receipts, so anyone can audit the ledger against them.
- A hash of the ledger is published periodically (e.g. committed to the repo or a public timestamping service). This gives tamper-evidence without a blockchain.

### 4. Redemption

- Credits convert to inference tokens at a published rate, e.g. *1 training token at stage S = k inference tokens*. The rate is set by RFC and changed only by RFC.
- Redeemed through the YardAI API with an account key, once serving exists. This RFC doesn't depend on the serving decision. If serving is distributed, devices that serve inference can earn credits the same way.
- Credits from early, riskier stages may get a multiplier, to reward the people who showed up first.

### 5. What credits are not

To keep this simple, honest and legally low-risk, credits are:

- **not transferable or sellable** — tied to the account that earned them;
- **not convertible to cash, crypto, or equity**;
- **not an investment** and carry no ownership. Equity is only the founding-contributor program, which is about building YardAI, not donating compute (whether compute counts there is an open question in GOVERNANCE.md);
- **redeemable only for YardAI inference**, subject to published fair-use limits.

The project should still get legal review before launch.

## Evidence

- BOINC's credit system ran for over a decade across millions of volunteer devices, and it shows that public credit, leaderboards and anti-cheating measures matter.
- The simulator (`src/yardai/simulate.py`) already counts tokens trained per worker and credits only accepted updates (`SimResult.credited_tokens`). That's the core of this accounting, which makes it testable now.

## Alternatives considered

- **Crypto token.** Liquid and attractive to some contributors. But it brings speculation, regulatory exposure, and pressure to optimise for the token instead of the model. Rejected for now.
- **Cash payments.** The project has no revenue, and cash turns volunteers into contractors. Rejected.
- **Pure goodwill + leaderboard.** Free and simple, but weak at keeping contributors. Kept as a complement: the ledger doubles as a public leaderboard.
- **Paying for uptime.** Easy to game (idle devices earn). Rejected in favour of paying for accepted work.

## Risks and open questions

1. Exchange rate: how many inference tokens per training token, and how is it adjusted fairly?
2. Partial credit for devices that leave mid-round after doing real work.
3. Privacy: the public ledger exposes account names and activity. Should pseudonymous IDs be allowed?
4. Do credits expire? Expiry limits liability, but feels unfair to early donors.
5. Does serving capacity exist to honour credits when they're redeemed? Credits issued before serving launches are a promise.
6. Fairness across hardware: should a laptop's slower, smaller contribution earn less per hour, as it does under pure work-unit accounting, or should there be a floor?
