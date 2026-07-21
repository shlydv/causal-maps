# Protocol: Distribution control for ROUTER_READS_RESIDUAL

*Causal Maps · 2026-07-13 · post-hoc validity check on `delta_router_read`*
*Status: **COMPLETE — `OOD_FAIL`.** Demote router-read to `ROUTER_READ_AMBIGUOUS`. Generator hyp not licensed.*

---

## Why

`ROUTER_READS_RESIDUAL` assumes keep_span / keep_res only remove the intended subspace. If keep_span produces a far more unnatural activation (norm collapse, huge displacement) than keep_res, router failure on SPAN is **OOD confound**, not evidence the router ignores \(\mathrm{span}\{\Delta\}\).

---

## Question

> On the same native protocol write sites, is keep_span distributionally harsher than keep_res relative to clean \(h\)?

---

## Metrics (at L2, X and Y val_slots; mean over trials)

For each op ∈ {span, res}, vs clean \(h\):

| Metric | Definition |
|---|---|
| `rel_norm` | \(\|h'\| / \|h\|\) |
| `cos` | \(\cos(h', h)\) |
| `rel_disp` | \(\|h'-h\| / \|h\|\) |
| `energy_frac` | for clean only: \(\|P_S h\|^2 / \|h\|^2\) (how much of native write sits in span) |

Aggregate: mean over trials and over {X,Y} slots.

---

## Gates (frozen)

Let \(n_s, n_r =\) mean `rel_norm` for span/res; \(c_s, c_r =\) mean cos; \(d_s, d_r =\) mean rel_disp.

**OOD_FAIL** (SPAN is unnaturally harsh) if **any**:
1. \(n_s < 0.5 \times n_r\) (span collapses norm much more), **or**
2. \(c_s < c_r - 0.30\) (span much less aligned with clean), **or**
3. \(d_s > d_r + 0.30\) (span displaces much more).

**OOD_PASS** otherwise.

Also report `energy_frac` (descriptive): if ≪ 0.1, span is a tiny slice of native \(h\) — keep_span is a near-zeroing op by construction (important context even if OOD_PASS).

---

## Verdicts

| Verdict | Meaning |
|---|---|
| `OOD_PASS` | Distribution control OK — `ROUTER_READS_RESIDUAL` interpretation remains in play |
| `OOD_FAIL` | SPAN condition confounded — demote main claim to `ROUTER_READ_AMBIGUOUS` |
| `OOD_INELICITABLE` | G0 / geometry fail |

**Hard stop:** one kernel. No re-running router-read. No new accounts. If OOD_PASS, next *why* question (Δ as generator) is design-only until Sahil OK.
