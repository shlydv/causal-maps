# Protocol: Computational chaining

*Causal Maps · 2026-07-13 · structural branch, post-`PROTOCOL_COMPATIBLE`*
*Status: **CLOSED — `CHAIN_INELICITABLE`.** G0 absolute stop (predicate native 0% @ L2). No donor fishing. Branch permanent.*

---

## Hypothesis (one sentence)

Independently extracted residual directions can form a **dataflow chain** — A writes a value, B selects among writes, C computes a function of the *selected* value — such that the final output is the composition \(C(B(A(\cdot)))\), not a sum of independent knobs and not mere pairwise readability.

---

## Existing-donor audit (Sahil ask: avoid a new C if possible)

Validated **early/strong** primitives in-hand:

| Donor | Evidence | Role |
|---|---|---|
| **Binding** \(\Delta_v\) | decompose / multislot / capacity / protocol | write value at slot |
| **Routing** \(\Delta_{\mathrm{route}}\) | select L2–14; protocol | select among slots |
| **Completion bit** \(\Delta_C\) | crossskill @ L2 | flip bit → action (own channel) |

Rejected as third link:

| Candidate | Why it fails the chain requirement |
|---|---|
| Bind → route → **emit value** | That is **protocol** (already `PROTOCOL_COMPATIBLE`): B reads A; no third *computation*. |
| Bind ∥ **Completion** | **Crossskill**: parallel channels; C’s action does **not** depend on the bound value. |
| Route → Completion | No dataflow: bit≠f(selected value). |
| Transform / Instruction / NOT / compare | Late/weak or typology-falsified — not “already-validated early/strong.” |

**Verdict: NO** — with only already-validated early/strong donors we cannot implement a genuine \(C(B(A))\) without collapsing into protocol or parallel composition.

Therefore we **keep one new donor (predicate)**, with **G0 as a hard stop** and **no donor fishing** after G0 fails.

---

## What we already have (do not re-claim)

| Result | Shows | Does **not** show |
|---|---|---|
| Multislot / capacity | Parallel slot-writes coexist | Sequential use of one write by another op |
| Crossskill | Bind ∥ Completion on separate channels | C’s target = f(B(A)) |
| **Protocol** (`PROTOCOL_COMPATIBLE`) | B **reads** A’s injected write | A **third** mechanism that *computes on* B’s selection |

Protocol = **communication**. This experiment = **computation over the communication**.

---

## Competing explanations (pre-register how we tell them apart)

| Account | Prediction for three-link chain |
|---|---|
| **Knobs** | Each Δ biases logits toward its own tokens; “chain” success is additive leakage. Flip/ablate of middle link barely matters once A and C are present. |
| **Protocol only** | B reads A (we already know). C reads *native* text or A directly, **skipping** B. Ablating B does not break C’s correct answer when A is present. |
| **Computational primitives (claim)** | Final answer tracks \(C(B(A))\). Ablating **any** link breaks the predicted output; flipping B flips C’s answer with A fixed; empty A ⇒ C has nothing coherent to classify. |

Load-bearing contrast: **protocol-only vs primitives** — does C require B’s selection, or can it bypass B?

---

## Experiment: BIND → ROUTE → PREDICATE

A, B = existing early/strong. **C = one new predicate donor** (standard diff-in-means; not Transform).

### Donors (extract independently, freeze)

1. **A — Binding** — \(\Delta_v\) at (L2, val_slot), Variable (existing).
2. **B — Routing** — \(\Delta_{\mathrm{route}}\) at (L2, flag digit), Select `value_of` (existing).
3. **C — Predicate (new, single design)** — frozen before any chain run:  
   *“Value: {v}. Is this an animal?”* + primer `Answer:` (trailing space; bare YES/NO or yes/no — freeze token ids in code PR).  
   \(\Delta_{\mathrm{pred}} = \mathrm{mean}\,h(\text{animal}) - \mathrm{mean}\,h(\text{non-animal})\) at **last position, L2 only** (no layer menu).

Value pool: single-token animals vs non-animals, partitioned offline. C trained on a held-out subset; chain test uses disjoint \(\{u,w\}\).

### Carrier (neutral; all three injected)

```
Let X = v0. Let Y = v0.
If flag=1, output the value of X. If flag=0, output the value of Y.
flag=0.
Is the answer an animal?
Answer: 
```

Surface flag=0 ⇒ native read Y. Inject \(\Delta_u@X\), \(\Delta_w@Y\), \(+\Delta_{\mathrm{route}}\) (→X), \(+\Delta_{\mathrm{pred}}\) at last pos.

**Pairing:** half trials \(u\) animal / \(w\) not; half reversed (balance).

### Primary signature — chain sensitivity

\(p = \mathrm{logit}(\mathrm{YES}) - \mathrm{logit}(\mathrm{NO})\) (frozen ids).

| Condition | Inject | Predicted \(p\) |
|---|---|---|
| **FULL** | A(u,w) + B(→X) + C | high if selected is animal |
| **FLIP** | A(u,w) + B(→Y) + C | low if selected is non-animal (or sign-flip when pairing reverses) |
| **CS** | \(p(\mathrm{FULL})-p(\mathrm{FLIP})\) oriented so animal-at-X ⇒ CS > 0 | > 0 vs same-norm null on B |

### Controls

| Control | Rules out |
|---|---|
| Ablate A@X | C hallucinating animal without write |
| **Ablate B** | C bypasses B (protocol-only) |
| Ablate C | Predicate dir doing the YES/NO work |
| Empty (B+C, no distinct binds) | Routing alone invents the class |
| C-only native “Value: u” | Donor sanity (feeds G0) |
| Null on B (and separately C) | Chance |

### Gates (frozen)

- **G0 — HARD STOP.** Before any chain condition: bind retrieve ≥80%; route native ≥80%; **predicate native ≥80% at L2**. If predicate fails G0 → verdict `CHAIN_INELICITABLE`, **stop**. No alternate predicate, no layer sweep, no Transform substitute, no Completion retrofit.
- **G1** CS > 0 vs null on B (p < 0.01).
- **G2** Ablate B: chain collapses (exact inequality frozen in code PR: e.g. \(p(\mathrm{no}\,B)\) within ε of FLIP / surface-Y baseline, or CS drops by ≥50%).
- **G3** Ablate A@X: \(p\) drops vs FULL.
- **G4** Empty: \(|p|\) small vs FULL.

### Verdicts

| Verdict | Meaning |
|---|---|
| `CHAIN_PRIMITIVE` | G0–G4 pass — C computes on B(A) |
| `CHAIN_PROTOCOL_ONLY` | G1 pass, **G2 fail** — communicate, C doesn’t require B |
| `CHAIN_KNOBS` | G1 fail |
| `CHAIN_INELICITABLE` | **G0 fail** — stop; no fishing |

---

## Novelty check (unchanged, honest)

Todd FV algebra / multi-steer = task or objective **superposition**. Our cut = heterogeneous **dataflow** + **ablate-B bypass**. Not novel if we only re-show B reads A or three vectors add.

**Risk owned:** new predicate may fail G0 (late/weak like Transform). That is an acceptable negative under hard stop — better than smuggling an unvalidated third link.

---

## Decision rule for coding

- Existing-only redesign: **rejected** (audit above).
- Novelty: **GO** with predicate C + hard G0.
- Code: one module `delta_chain.py`, one kernel, after Sahil signs this page.
- **Do not** open a predicate template menu after G0 fails.

---

## One-line stop discipline

G0 fail ⇒ `CHAIN_INELICITABLE`, stop. G2 fail after G1 ⇒ `CHAIN_PROTOCOL_ONLY`, stop. No donor fishing.
