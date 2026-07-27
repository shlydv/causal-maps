# Paper 2: context-geometry width screen

Status: **frozen before GPU output, 2026-07-27**.

This is a discovery screen, not a claim-establishing experiment. It decides
which mechanistic branch, if any, deserves a fresh prospective test.

## Competing explanations

1. **Smooth local geometry:** operation control is locally simple, but its
   response map changes with the computation.
2. **Discrete gating:** a compact task state switches the model between
   downstream regimes.
3. **Shared geometry:** local response maps are similar across computations,
   conflicting with the proposed context-specific explanation.
4. **Unstable/template-only control:** response maps do not replicate even
   within a computation; exact and mean templates work without a recoverable
   local law.

The screen uses four structurally different, previously eligible computation
families: private belief, two-hop pointer traversal, maximum-score comparison
and constraint elimination.

## Screen A: exact-residual dose curves

At the three answer-prefix positions after layer 21, intervene with

`h_origin + alpha * (h_opposite - h_origin)`

for `alpha = -0.50, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50`, in both operation
directions. Measure L24/L27 transport after subtracting the exact identity
carry-through of the injected final-position residual, as well as raw
transport and value-answer accuracy. The processed response, not the
tautological residual carry, controls adjudication.

A computation is smooth only when both directions:

- reach at least `0.25` exact-dose progress;
- have alpha/progress Spearman correlation at least `0.90`;
- have linear-fit R-squared at least `0.80` on alpha `0` through `1`;
- place no more than `0.55` of the full response in one adjacent dose step;
- preserve at least `80%` answer accuracy.
- retain a processed target contrast whose norm is at least `5%` of the raw
  target contrast, so division by a vanishing post-identity signal cannot
  manufacture a curve.

A gating candidate requires at least `0.25` exact progress, at least `0.80`
monotonicity, at least `0.65` of the response in one dose step and at least
`80%` answer accuracy. The overall smooth verdict requires three of four
families; the gating verdict requires two of four.

## Screen B: multi-output local response maps

Four calibration pairs, disjoint from eight screen pairs, form one natural
BELIEF-to-SEARCH mean displacement per family. Their uncentered rank-four SVD
basis is frozen without causal screening outcomes.

At every unchanged test state, centrally probe each basis direction with
matched `+/-` perturbations of `0.10` times the median family-template norm.
The response is the entire final-token residual vector at L24 and L27 after
subtracting the exact direct residual identity path from the patched final
answer-prefix position—not a scalar route-score derivative.

At the primary L27 checkpoint:

- stability requires median within-family map cosine at least `0.75`;
- context specificity requires within-family cosine to exceed cross-family
  cosine by at least `0.15`;
- a shared-map outcome requires cross-family cosine at least `0.75`;
- every probe arm must preserve at least `80%` value accuracy.
- the minimum processed-response norm must be at least `1e-4`.

## Frozen branch decision

- Smooth dose response plus stable context-specific maps:
  `LOCAL_GEOMETRY_PILOT_LICENSED`.
- Gated response plus stable maps:
  `SELECTOR_FACTORIAL_LICENSED`.
- Stable shared maps:
  `SHARED_RESPONSE_GEOMETRY_CONFLICT`.
- Otherwise:
  `NO_DEEPER_BRANCH_LICENSED`.

No prompt, row, alpha, basis-rank, perturbation-scale, layer, checkpoint or
threshold revision follows the output. Any licensed branch still requires a
new held-out, prospective causal experiment.
