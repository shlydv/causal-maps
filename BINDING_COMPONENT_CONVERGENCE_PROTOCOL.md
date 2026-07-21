# Do Natural Binding Rewrites and Affine Operators Use the Same Components?

*Causal Maps · 2026-07-13*  
*Status: **PRE-REGISTERED — one component-level convergence test.***

## Question

`delta_operator_v11` established, at a residual-stream level, that the
affine operator \(z_b-z_a\) closely reproduces a natural textual value rewrite
in Qwen2.5-7B.  At L8, the natural and ADD states have equivalent causal patch
effects, and overwriting either state with CLEAN blocks its effect.

That result does **not** establish that the two conditions use the same
components.  This protocol asks a narrower question:

> At the frozen L8 mediator, do the natural textual rewrite and the affine ADD
> controller depend on the same small set of attention-head / MLP outputs at
> the final query position?

This is a convergence test, not a search for a uniquely responsible head and
not a claim about the full upstream circuit.

## Frozen setting

- Qwen2.5-7B-Instruct, exact pinned runtime/revision used by
  `delta_operator_v11`, 8-bit, seed 0.
- Existing 40 two-binding, content-specific source→target trials from
  `delta_operator`.
- Existing affine codebook: value prototypes from the four single-binding
  donor names \(X,Y,Z,W\); controller \(d_{a\to b}=z_b-z_a\) is added at
  L2 at the queried value slot.
- Mediator layer: **L8**, frozen from the completed operator protocol.
- Components at L8: all 28 attention-head outputs and the full MLP output,
  each intervened on only at the final query/readout token.  Head ablation
  zeros that head's input channel slice to `o_proj`; MLP ablation zeros its
  output at that token.  This is a well-defined output-side component test,
  not an attempt to identify all earlier writers.

## Conditions and estimand

For every source/target trial, form:

1. **CLEAN:** source value \(a\);
2. **NATURAL:** textual target value \(b\);
3. **ADD:** CLEAN plus \(d_{a\to b}\) at L2;
4. **ABLATE:** each of the above with a selected L8 component output zeroed.

For condition \(q\in\{\mathrm{NATURAL},\mathrm{ADD}\}\), component-set
loss is measured against an identically ablated CLEAN run:

\[
E_q(S)=\operatorname{LD}(q;S)-\operatorname{LD}(\mathrm{CLEAN};S),
\qquad
L_q(S)=\frac{E_q(\varnothing)-E_q(S)}{E_q(\varnothing)}.
\]

This subtraction is essential: an ablation's generic damage is not counted as
evidence that it selectively disrupts the rewrite.

## Held-out selection / test split

The forty deterministic trials are split within each source value and query
role: offsets 1/3 are discovery (20 trials; 10 per query role), and offsets
5/7 are held-out test (20 trials; 10 per query role).  No held-out output is
used to choose components.

On discovery trials, compute each candidate's \(L_{\rm NATURAL}\) and
\(L_{\rm ADD}\), and form:

- \(S_N\): top four candidates by natural loss;
- \(S_A\): top four candidates by ADD loss;
- \(S_\cap\): top four candidates by `min(natural loss, ADD loss)`.

All three sets are frozen before held-out evaluation.  Random controls are
100 seeded, size- and type-matched component sets: if a selected set contains
the single MLP candidate, every matched random set contains it and samples the
same number of heads.

## Gates

### G0 — valid baseline

On held-out trials, CLEAN and NATURAL greedy accuracy are each at least 80%;
the ADD effect is positive on at least 80% of rows; and ADD/NATURAL mean
logit-effect ratio lies in [0.70, 1.30].

### G1 — residual-level convergence re-confirmed

Before component claims, rerun the frozen L8 capture:

- L8 queried-slot ADD/NATURAL displacement cosine ≥0.80;
- L8 final-position displacement cosine ≥0.50.

### C1 — independently selected sets overlap

\(J(S_N,S_A)\geq0.50\).  This guards against using a deliberately constructed
shared score to conceal two unrelated candidate rankings.

### C2 — one held-out set is necessary for both effects

On held-out trials, \(S_\cap\) removes at least 50% of both the NATURAL and
ADD effect: \(L_N(S_\cap)\geq0.50\) and \(L_A(S_\cap)\geq0.50\).

### C3 — stronger than matched random sets

For each condition separately, no one of 100 matched random sets has loss
greater than or equal to the corresponding \(S_\cap\) loss (one-sided
permutation p = 1/101).

### C4 — position-matched damage control

Applying the exact same component ablations one token before the final
position removes at most 20% of either effect.  This is not a claim that the
preceding token is universally inert; it checks that C2 is not generic late
layer damage in this prompt.

## Verdicts

| Verdict | Rule |
|---|---|
| `SHARED_L8_COMPONENT_PATH` | G0 ∧ G1 ∧ C1 ∧ C2 ∧ C3 ∧ C4 |
| `OVERLAPPING_COMPONENTS_NOT_LOCALIZED` | G0 ∧ G1 ∧ C1, but C2 or C3 fails |
| `DIVERGENT_L8_COMPONENT_PATHS` | G0 ∧ G1; each own selected set is held-out necessary for its own condition, but C1 fails and the cross-set losses are materially smaller |
| `DISTRIBUTED_OR_REDUNDANT_L8` | G0 ∧ G1, but no selected four-component set is held-out necessary for its own condition |
| `COMPONENT_CONVERGENCE_INELICITABLE` | G0 or G1 fails |

## Interpretation limits

- A positive result supports a **shared L8 downstream component path**.  It
  does not prove that the full earlier circuit is shared or uniquely identified.
- A negative result does not show that no mechanism exists.  It may indicate a
  distributed, redundant, or component-granularity-mismatched mediator.
- No layer search, component-count change, source/template repair, or
  follow-up head hunt is licensed by any outcome of this kernel.
