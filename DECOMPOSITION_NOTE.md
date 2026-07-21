# Value-Specific Directions Behind a "Generic" Binding Direction

*Follow-up to the frozen note ("Causal Site Maps Can Fail to Replicate While Residual
Directions Transfer"). Draft — internal. Qwen2.5-7B-Instruct, greedy, one skill, one
layer. All claims are sufficiency + steerability, not identification of a unique circuit.*

## Summary

In a variable-binding skill, the causal direction that installs a value at the value
token was previously flagged **GENERIC_BOOST**: a wrong-value direction transferred
almost as well as the matched one (cos 0.77 between their aggregates), which appeared to
block any "value-specific" reading. We show that this appearance is driven by
**averaging over values**, and that value-specific causal directions are in fact
recoverable:

1. **Per-value directions transfer.** Adding Δ_v to held-out prompts installs value *v*.
2. **They are value-selective.** Δ_v raises *v*'s logit and not the other values'
   (all 10 values, p = 0.0099 against a same-norm random-direction null).
3. **They are approximately orthogonal** across values (mean pairwise cos = 0.014).
4. **The "generic" aggregate is quantitatively consistent with being their centroid.**
   Each Δ_v aligns with the aggregate at cos ≈ 0.32 — exactly the value expected for the
   mean of ten orthogonal equal-norm vectors (√10/10 ≈ 0.316) — so the aggregate's
   value-agnostic appearance does not *require* a dominant shared mechanism.
5. **The value-specific effect survives removing the empirical generic direction, but
   that direction is not inert.** Projecting out the centroid leaves every Δ_v still
   transferring and selecting (p = 0.0099, all 10), yet costs a modest, value-heterogeneous
   share of selectivity (median ≈ 15%). The centroid added alone is non-selective.

We therefore do **not** claim the generic component is a pure artifact; we claim the
measurements are consistent with a centroid-of-approximately-orthogonal-directions
account, and that a genuine value-specific causal component exists beyond the centroid.

## Setup

- **Skill (Variable).** Prompt `Let {X} = {val}. What is the value of {X}?` with the
  assistant primed `{X} =`; the next token is the bound value. Minimal pair: clean value
  `v0` vs counterfactual `v1`. 5 variable names × 10 value pairs = 50 pairs (all survive
  tokenization; single-token values). Site: layer **L\*=2**, the value-token position
  (peak mean IE for this skill).
- **Per-value direction.** `Δ_v = mean over the 5 pairs with cf==v of (h_cf − h_clean)`
  at (L2, value slot). Ten cf-values: cat, blue, two, cold, grape, south, shut, right,
  beta, false.
- **Intervention.** ADD `α·d` to the residual at (L2, value slot) on **held-out** prompts
  (the 45 pairs with cf ≠ v), one forward each; read the change in every value token's
  logit.
  - `transfer(d,v) = mean_prompts Δlogit(v)`
  - `selectivity(d,v) = mean_prompts [ Δlogit(v) − mean_{w≠v} Δlogit(w) ]`
- **Null.** N = 100 random directions at matched norm; p-value floor 1/101 ≈ 0.0099.
- All runs Qwen2.5-7B-Instruct in 8-bit on one T4; pre-registered gates (see
  `CAUSAL_MAPS_LOG.md`, 2026-07-12).

## Results

### 1–4. Value-specific directions transfer, select, and are ≈orthogonal

Removing, per value, the subspace spanned by the *other* nine value directions leaves a
residual that transfers and selects for **all 10 values** (transfer +8.84, selectivity
+8.73; each p = 0.0099). The norm-matched generic component is weakly selective (+0.58)
while transferring (+1.91) — i.e. it boosts values roughly uniformly. A local geometric
cross-check (on the saved Δ vectors, no model):

- **mean pairwise cos(Δ_v, Δ_w) = 0.014** (range −0.09..+0.08) — at/below the
  √(9/3584) ≈ 0.05 chance baseline for a 9-d subspace in 3584-d.
- **cos(Δ_v, aggregate) = 0.21..0.53, mean ≈ 0.32.** For ten orthogonal equal-norm
  vectors the mean cosine with their centroid is √10/10 ≈ 0.316; the observed value
  matches, and the two above-baseline values (grape 0.52, shut 0.45) are exactly the
  larger-norm directions that dominate the mean. Consistent with "aggregate = centroid,"
  not a shared component.

*Caveat:* the residual **norm** is not evidence — projecting a 3584-d vector onto a 9-d
subspace removes little regardless of content (residual fraction ≈ 0.99 is largely
geometric). The load-bearing evidence is **selectivity**, which is content-directional,
not magnitude.

### 5. Centroid-removal control (the decisive test)

We remove the single explicit empirical generic direction — the centroid
`g = mean_v Δ_v` (the aggregate that produced GENERIC_BOOST) — from each value direction,
`d_v' = Δ_v − (Δ_v·ĝ)ĝ`, and re-measure.

| Quantity | Value |
|---|---|
| d′ transfers | +5.74, p = 0.0099 (all 10 values) |
| d′ selective | +5.88, p = 0.0099 (all 10 values) |
| selectivity retention (norm-matched, median) | **0.848** |
| centroid alone, selectivity | **+0.028** (≈ 0) |
| centroid alone, transfer | +2.31 (non-selective boost) |
| fraction of ‖Δ‖ removed (mean) | 5.9% |

**Reading.** (i) The value-specific effect is *not* merely the centroid — every d_v′ still
transfers and selects at p = 0.0099. (ii) The centroid is *not* inert — removing it costs
a median ≈ 15% of selectivity, **heterogeneously**: negligible for two/right/beta/false
(retention ≥ 0.91) but large for the directions that most overlap the centroid — grape
(cos 0.52 → retention 0.51) and shut (cos 0.45 → 0.72). (iii) The centroid *alone* selects
nothing (0.028). So value directions are largely-but-not-fully orthogonal to the centroid;
the centroid is non-selective on its own yet each value's projection onto it contributes
part of that value's selectivity (a nonlinear-response effect — the linear intuition
"non-selective ⇒ contributes nothing" does not hold empirically).

## Relation to prior work

- **Function/task vectors** (Todd et al. 2024; Hendel et al. 2023; Ilharco et al. 2023):
  established that a task is an addable activation direction and that task vectors compose
  (parallelogram). Those directions encode the **function**; ours encode the **content**
  (which value is bound). A recent *causal decomposition of function vectors* (2605.16591)
  decomposes an n-shot FV over **examples** (a weighted sum of demonstration sub-vectors);
  ours is a different decomposition — **generic-centroid vs value-specific residual**.
- **Binding** (Feng & Steinhardt binding-IDs; "LMs Use Lookbacks," ICLR 2026; "Mixing
  Mechanisms," 2510.06182): deeper accounts of *how* binding is implemented (binding-ID
  subspaces, lookback attention). We do not out-explain these; we characterize the
  **value code's geometry** (near-orthogonality) and its causal selectivity/transfer.
- **Steering (non-)identifiability** (2602.06801; 2505.22637; Opiełka, *Causality ≠
  Invariance*): a live constraint. Our added directions are *sufficient* to install a
  value selectively; this does not establish they are the model's unique natural mechanism.

## What this adds

Paired with the frozen note's site-map ↔ direction dissociation: not only does a direction
at a frozen site transfer where the full site-map fails to replicate — the direction space
**decomposes into approximately-orthogonal, value-selective content directions plus a
non-selective centroid**, and the value-specific effect **survives removal of that
centroid**. The "generic slot-update" appearance is quantitatively consistent with the
centroid of these directions rather than a separate dominant mechanism, though the centroid
is not fully separable from them.

## Limitations

1. One model (Qwen2.5-7B), one skill (variable binding), one layer (2), greedy decoding,
   value token only.
2. Sufficiency + steerability, not identification: added directions install values; we do
   not claim they are the unique natural circuit (cf. non-identifiability work).
3. The centroid is estimated in-sample; a held-out centroid is a natural robustness check.
4. Selectivity retention pre-registered at ≥ 0.90; observed median 0.848 → we report
   CENTROID_MATTERS rather than moving the threshold.
5. Residual-fraction (≈ 0.99) is partly high-dimensional geometry and is *not* used as
   evidence; selectivity is.

## Reproduce

`python kernel/run_kaggle.py run delta_decompose --config '{"quantization":"8bit"}'`
and `... run delta_centroid ...`. Artifacts: `runs/delta_decompose/`,
`runs/delta_centroid/`. Gate definitions and per-value numbers: `CAUSAL_MAPS_LOG.md`
(2026-07-12 entries).
