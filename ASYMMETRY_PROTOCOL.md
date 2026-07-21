# Protocol: Route–binding asymmetry (why route necessity ≫ bind)

*Causal Maps · 2026-07-13 · post-`ROUTE_NECESSARY_ONLY`*
*Status: **COMPLETE** — solid `ASYM_INCOMPLETE_BASIS` (¬A). C1 caveat (route knock invalid). Bus elevation not licensed. Line stopped.*

---

## Scientific question (not an interpretation)

**Why is routing necessary for native Select while binding is only weakly necessary for native Variable** (selective + utility-safe, but below magnitude T)?

This is the deepest empirical asymmetry the arc uncovered. We study it **before** naming a architecture (bus / ALU / etc.).

---

## What we do *not* claim yet

- Not “the residual stream is a memory+routing bus.”
- That phrase is an **interpretation** that becomes allowed only if a pre-registered redundancy / bottleneck pattern lands.

Descriptive hint from necessity (not a gate): `drop_span ≈ drop_true` (~8.0). That already pressures a naive “many extracted Δ_v writers” story — this experiment resolves competing accounts cleanly.

---

## Competing accounts (discriminate)

| Account | Prediction |
|---|---|
| **A. Extracted-span redundancy** | Ablating span{\(\Delta_u\)} hurts retrieve **much more** than ablating \(\Delta_v\) alone. |
| **B. Incomplete basis** | Span ≈ \(\Delta_v\) (both mild), but **site knockout** (destroy all info at val_slot) hurts **much more** than either — the write uses residual structure outside our extracted dirs. |
| **C. Soft write / hard route (bottleneck)** | At flag: \(\mathrm{drop}(\Delta_{\mathrm{route}}) / \mathrm{drop}(\mathrm{knockout})\) ≈ 1. At val_slot: \(\mathrm{drop}(\Delta_v) / \mathrm{drop}(\mathrm{knockout})\) ≪ 1. Route’s direction ≈ the whole site; bind’s direction is a fraction of the site. |

A and B are mutually exclusive on the bind site. C is the cross-skill asymmetry signature and can co-occur with B.

---

## Interventions (site-local only; L2)

Same sites as necessity: bind `(L2, val_slot)`, route `(L2, flag digit)`. Native prompts.

| Code | Intervention |
|---|---|
| `dir` | Project-out 1-D: \(\Delta_v\) (bind) or \(\Delta_{\mathrm{route}}\) (route) |
| `span` | Project-out QR span of all per-value \(\Delta_u\) (bind only) |
| `rand_k` | Project-out random orthonormal \(k\)-D, \(k=\#\)values (bind) or 1-D random (route) — null |
| `knock` | **Site knockout:** replace \(h_{\ell,p}\) with the **batch mean** at that position (removes example-specific content at the site; keeps average activation) |

No layer expansion. No weight edits. One kernel then stop.

---

## Primary signatures (frozen)

Let \(\mathrm{drop} = \mathrm{pref}_{clean} - \mathrm{pref}_{ablate}\) (bind) or gap/acc as in necessity (route).

**Bind redundancy ratio**
\[
R_{\mathrm{span}} = \frac{\mathrm{drop}(\mathrm{span})}{\mathrm{drop}(\mathrm{dir})}
\]
Pass **A** iff \(R_{\mathrm{span}} \ge 1.5\) and \(\mathrm{drop}(\mathrm{span})\) beats `rand_k` null (p < 0.01).

**Bind incompleteness ratio**
\[
R_{\mathrm{knock}} = \frac{\mathrm{drop}(\mathrm{knock})}{\max(\mathrm{drop}(\mathrm{dir}), \varepsilon)}
\]
Pass **B** iff \(R_{\mathrm{span}} < 1.5\) and \(R_{\mathrm{knock}} \ge 2.0\) and knock beats rand null.

**Bottleneck asymmetry (PRIMARY cross-skill)**
\[
\beta_{\mathrm{route}} = \frac{\mathrm{drop}(\Delta_{\mathrm{route}})}{\mathrm{drop}(\mathrm{knock}_{flag})},\quad
\beta_{\mathrm{bind}} = \frac{\mathrm{drop}(\Delta_v)}{\mathrm{drop}(\mathrm{knock}_{val})}
\]
Pass **C** iff \(\beta_{\mathrm{route}} \ge 0.7\) and \(\beta_{\mathrm{bind}} \le 0.4\) (route direction ≈ site; bind direction ≪ site).

Utility (primary guard, same spirit as necessity): knockout/project on the *other* skill’s site must not be required for headline — report cross drops; fail utility only if knock on target site also destroys Completion by ≥40pp (generic vandalism).

---

## Gates

- **G0:** Variable native ≥80%; Select `value_of` ≥80% both flags.
- **A1:** Account A (span redundancy) as above.
- **B1:** Account B (incomplete basis) as above.
- **C1:** Bottleneck asymmetry as above.
- **U:** Completion acc drop under bind-knock and under route-knock each ≤ 10pp **or** ≤ 0.25 × that arm’s knock drop (acc).

**Headline verdicts**

| Verdict | Meaning |
|---|---|
| `ASYM_BOTTLENECK` | C1 ∧ U — route is a site-bottleneck; bind is not |
| `ASYM_SPAN_REDUNDANCY` | A1 ∧ U — extracted binding dirs are redundant writers |
| `ASYM_INCOMPLETE_BASIS` | B1 ∧ U — our Δ_v miss most of the site’s write |
| `ASYM_BOTTLENECK_AND_INCOMPLETE` | C1 ∧ B1 ∧ U — strongest non-bus-named result |
| `ASYM_UNCLEAR` | G0 pass but no account clears |
| `ASYM_INELICITABLE` | G0 fail |

**Elevation rule (Sahil/ChatGPT sequence):** only if A1 or (C1∧B1) succeeds may a later writeup *interpret* this as a memory+routing bus. This kernel does **not** print “bus.”

---

## One kernel then stop

Module: `delta_asymmetry.py`. Reuse Variable Δ_v, Select `value_of` Δ_route, `forward_with_project`. Add mean-site knockout hook.

No fishing, no locus widen, no second kernel for “maybe more layers.”
