# Paper 2: prospective held-out inverse-control pilot

Status: **frozen before GPU output, 2026-07-27**.

This pilot tests a prospective prediction of the candidate theory

`F_x(h + delta) - F_x(h) ~= M_x delta`,

where the model is globally nonlinear but admits a locally identifiable,
context-conditioned causal response map at the established L21 answer-prefix
interface. The experiment does not claim that the LLM is globally linear.

## Scientific prediction

Generic positive/negative probes around a held-out computation's starting
state can estimate its local response map well enough to synthesize an
intervention toward the opposite operation state, without observing that
opposite-operation activation before the prediction is frozen.

The experiment predicts a hidden-state transition. A positive result licenses
a separate route/behavior confirmation; it is not itself the final Paper 2
claim.

## Training and held-out computations

The four training families are:

- private belief;
- two-hop pointer traversal;
- maximum-score comparison;
- constraint elimination.

The two prospectively held-out prompt families are not used to construct the
probe basis, shared target, shared response map or any hyperparameter:

- **minimum score:** compare three numerical scores and return the label with
  the smallest score;
- **set intersection:** return the unique label appearing in both stated
  groups.

All answers use the same frozen eight color words. BELIEF and `X X SEARCH`
remain exactly length matched. Six training directed pairs are used. For each
direction, six held-out identification pairs and six further held-out causal
test pairs are mutually disjoint. Clean and counterfactual histories are kept
as separate examples.

## Frozen locus and representation

- source: output of layer 21;
- intervention positions: the three final answer-prefix command tokens;
- primary evaluation: final-token residual after layer 27;
- depth replication: final-token residual after layer 24;
- direct identity carry: subtract the patched final answer-prefix residual
  from every downstream state before constructing targets or measuring
  processed progress.

## Training-only construction

Natural L21 BELIEF-to-SEARCH displacements from the four training families
form an uncentered rank-four orthonormal basis `B`.

The shared downstream target `v` is constructed from training families only:
normalize each family's mean processed L27 BELIEF-to-SEARCH displacement,
average and renormalize those directions, then restore the median training
target norm. SEARCH-to-BELIEF uses `-v`.

At training-family BELIEF and SEARCH states, centrally probe every basis axis
with `+/- 0.10` times the median training controller norm. Averaging the
processed derivatives produces operation-specific shared response maps.

Training target coherence must have median pairwise cosine at least `0.20`.
If not, the run stops before held-out evaluation with
`TRAINING_TARGET_UNRESOLVED`.

## Target-blind local identification

For each held-out family and direction, observe only the origin operation on
six identification pairs. Centrally probe the four training-only basis axes
and average the resulting derivatives into one family-direction response map
`M_f`. No causal test row is probed.

The primary coefficient prediction is the ridge solution

`z = argmin ||M_f z - v_direction||^2 + lambda ||z||^2`,

where `lambda` is `0.01` times mean Gram-matrix diagonal. The corresponding
L21 intervention is `B^T z`. Its norm is capped at `1.25` times the median
training-controller norm.

Before any causal test opposite-operation batch is rendered or evaluated,
save and hash both JSON and NPZ prediction artifacts containing the basis,
shared target, identification response maps and all predicted coefficients.
Each frozen family-direction intervention is then applied unchanged to six
fresh test pairs in both histories.

## Frozen controls

Every control except the two explicit post-freeze oracles is also predicted
and hashed before target access:

- shared-map inverse;
- other-held-out-family response-map inverse, norm matched;
- same-held-out-family opposite-origin response-map inverse, norm matched;
- sign-reversed local inverse;
- three seeded random rank-four directions, norm matched.

After the prediction hash only:

- **basis-projection oracle:** project the exact held-out L21 operation
  displacement into the training basis;
- **target-informed inverse oracle:** invert the local map toward the exact
  held-out processed L27 target;
- **exact-state oracle:** transplant the exact held-out L21 target state.

The target-informed oracle diagnoses whether the local map and rank-four
control space have enough capacity. The exact-state oracle establishes assay
eligibility.

## Metrics and decision

For every family/direction cell, pool both histories (`12` examples):

- processed L27 progress toward the exact held-out target;
- processed L27 distance-to-target ratio;
- number of positive-progress examples;
- L24 processed progress;
- value-answer accuracy;
- recovery relative to the target-informed inverse oracle.

A cell passes prospectively when:

- exact-state oracle mean processed progress is at least `0.50`;
- target-informed inverse mean progress is at least `0.40`;
- local inverse mean progress is at least `0.25`;
- at least `9/12` examples have positive local progress;
- local median distance ratio is at most `0.95`;
- value-answer accuracy is at least `0.80`;
- local recovery of target-informed inverse is at least `0.60`;
- local mean progress exceeds the best shared, wrong-context,
  opposite-origin and random control by at least `0.10`;
- sign-reversed local control has non-positive mean progress.

The pilot passes only if at least three of four cells pass, with both held-out
families and both directions represented among the passing cells.

Verdicts distinguish:

- oracle/assay failure;
- local-map or rank-four capacity failure;
- failure of the training-only shared target;
- shared inverse sufficiency;
- control failure;
- `PROSPECTIVE_HELDOUT_INVERSE_STATE_CONTROL`.

Only the final verdict licenses route/behavior confirmation. No prompt,
family, row, layer, rank, probe scale, inverse regularization, cap, control or
threshold changes follow the output.
