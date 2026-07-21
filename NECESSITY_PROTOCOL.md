# Protocol: Causal necessity of binding / routing directions

*Causal Maps · 2026-07-13 · post-`PROTOCOL_COMPATIBLE`; chain branch closed*
*Status: **COMPLETE — `ROUTE_NECESSARY_ONLY`.** Site-local stop. No layer expansion.*

---

## Hypothesis (one sentence)

The residual directions already shown **sufficient** (ADD) and **protocol-compatible** (bind→route) are also **necessary** for native store/select at their sufficiency sites: site-matched directional ablation selectively breaks Variable / Select while preserving unrelated micro-skill behavior.

---

## Progression

**Sufficiency → Compatibility → Necessity.**  
We do not expand breadth. This closes the triangle on the same primitives.

Calibrated claim ceiling: load-bearing at the site under directional ablation — **not** “unique natural circuit” (Makelov dormant pathways; Grant divergent latents).

---

## Novelty (scoped)

**Method not novel:** directional ablation / project-out (RepE; Arditi refusal; probe “silence”).

**Exact conjunction — run-worthy:** necessity for *the same* independently extracted micro-skill dirs that already passed sufficiency + protocol compatibility, with **selectivity/utility as primary evidence**, content-selective bind ablation (Δ_true vs Δ_foil), cross-skill preservation, and a pre-registered projection-strength sweep — all **site-local only**.

Do **not** claim first necessity ablation in the literature, nor unique-circuit ID.

---

## Intervention (frozen)

Directional ablation at the **sufficiency site only**:

\[
h_{\ell,p} \leftarrow h_{\ell,p} - \alpha\, P_S h_{\ell,p}
\]

with \(\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}\) (pre-registered strength sweep — cheap: same hook, five scalars).

| Arm | Site | Subspace \(S\) |
|---|---|---|
| **Bind** | (L2, `val_slot`) | Primary: 1-D span of true-value \(\Delta_v\). Support: full QR span of all per-value \(\Delta_u\). |
| **Route** | (L2, flag digit) | 1-D span of \(\Delta_{\mathrm{route}}\) (Select `value_of`, flag1−flag0). |

Surface = **native** prompts. Primary necessity test does **not** ADD our dirs.

**Hard stop on locus:** if site-local necessity fails → verdict `SUFFICIENT_ONLY` (or arm-specific partial). **No** layer expansion, all-position erase, weight orthogonalization, or post-hoc rescue.

---

## What distinguishes mechanism from generic damage (PRIMARY)

Selectivity / utility are **primary evidence**, not side controls.

A large target drop alone is **insufficient** for `NECESSARY` — that pattern is also consistent with nonspecific residual vandalism.

| Primary contrast | Pass means |
|---|---|
| **Content selectivity (bind)** | Ablating \(\Delta_v\) (true) hurts retrieve of \(v\); ablating \(\Delta_w\) (foil, \(w\neq v\)) does **not** (to frozen slack). |
| **Subspace selectivity (route)** | Ablating \(\Delta_{\mathrm{route}}\) at flag hurts Select; ablating a **binding** direction (or random 1-D) at the **same** flag site does **not**. |
| **Cross-skill utility** | Bind-ablate preserves Select native; route-ablate preserves Variable native; both preserve Completion native — within frozen slack. |
| **Null** | Target effects beat same-site random-direction project-out (p < 0.01). |

**Strength sweep (support, pre-registered):** target damage increases monotonically (Spearman ≥ 0.7 or each step non-decreasing within noise) in \(\alpha\); foil/utility stay flat. Non-monotonic target or utility collapsing with \(\alpha\) → treats as `PARTIAL` / generic damage, not clean necessity.

---

## Conditions

### Binding (native Variable)

`Let X = v. What is the value of X?` → gold `v`.

| Cond | \(\alpha=1\) expect |
|---|---|
| clean | high retrieve |
| ablate \(\Delta_v\) (true) | retrieve collapses |
| ablate \(\Delta_w\) (foil) | little/no collapse |
| ablate random 1-D | little/no collapse |
| ablate full span\(\{\Delta_u\}\) | collapse (support) |

Metric: logit preference for gold vs foils, and greedy accuracy.

### Routing (native Select `value_of`)

| Cond | \(\alpha=1\) expect |
|---|---|
| clean | ≥80% both flags |
| ablate \(\Delta_{\mathrm{route}}\) | flag sensitivity collapses |
| ablate random 1-D @ flag | little/no |
| ablate \(\Delta_v\) (binding dir) @ flag | little/no |

### Utility surfaces (PRIMARY)

- Completion native (own templates; in-repo).
- Cross: after bind-ablate, Select still OK; after route-ablate, Variable still OK.

---

## Gates (frozen)

Exact inequalities locked before code:

- **G0:** Variable native ≥ 80%; Select `value_of` ≥ 80% both flags. Else `NEC_INELICITABLE`, stop.

- **S1 (PRIMARY — bind selectivity):**  
  `drop_true − drop_foil ≥ 0.5 × drop_true` **and** `drop_true > 0`,  
  with `drop = pref_clean − pref_ablate` (or accuracy pp).  
  **and** `drop_true` beats random-dir null (p < 0.01).

- **S2 (PRIMARY — route selectivity):**  
  `drop_route − drop_wrongSub ≥ 0.5 × drop_route` **and** `drop_route > 0`,  
  wrongSub = max(drop_random, drop_bindDir@flag);  
  **and** null p < 0.01.

- **U (PRIMARY — utility):**  
  On Completion and on the cross-skill surface for that arm:  
  `drop_util ≤ 0.25 × drop_target` **OR** `drop_util ≤ 10` pp accuracy  
  (whichever is checked per metric family — freeze in code to the metric used for that surface).

- **T (target magnitude, secondary to S/U):**  
  Bind: `drop_true ≥ 0.5 × pref_clean` when pref_clean > 0 (else ≥ 40 pp accuracy).  
  Route: mean flag accuracy drop ≥ 40 pp **or** preference gap drop ≥ 50% of clean gap.

- **α-sweep (support):** Spearman(\(\alpha\), drop_true) ≥ 0.7 for bind and for route target; foil/util Spearman ≤ 0.3 in magnitude or flat. Failure demotes clean `NECESSARY` → `PARTIAL` if S/U/T otherwise pass.

**Headline `NECESSARY` ⟺ G0 ∧ S1 ∧ S2 ∧ U ∧ T.**  
α-sweep failure alone → `PARTIAL` (not a rescue path).

---

## Verdicts

| Verdict | Meaning |
|---|---|
| `NECESSARY` | Both arms: selective + useful-preserving + target magnitude |
| `BIND_NECESSARY_ONLY` / `ROUTE_NECESSARY_ONLY` | One arm passes S+U+T; other fails target or selectivity |
| `PARTIAL` | Target drops but selectivity/utility/α-sweep fail (generic damage pattern) |
| `SUFFICIENT_ONLY` | G0 pass; site-local necessity null — dirs install under ADD but native does not need them at site |
| `NEC_INELICITABLE` | G0 fail |

**If site-local fails → `SUFFICIENT_ONLY` (or arm FAIL). Stop. No layer expansion. No post-hoc rescue.**

---

## One kernel then stop

Module (when built): `delta_necessity.py` → stage `delta_necessity`.  
Reuse Variable Δ_v and Select `value_of` Δ_route extractors. New: project-out hook + α sweep.

No donor fishing, no locus widen, no second kernel for “maybe all layers.”

---

## Non-claims

- Not anti-steering (−αΔ ADD) as the primary necessity test.
- Not Todd-style head mediation / full circuit discovery.
- Not Arditi all-stream / weight orthogonalization (unless a *future* separately pre-registered study).
- Not unique binder/router identification.
