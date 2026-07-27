# Paper 2 prospective causal-sensitivity protocol

Status: frozen before model execution  
Date: 2026-07-26  
Stage: `delta_prospective_causal_sensitivity`

Frozen protocol SHA-256:
`4CBE37E7719FF25144AE74FCCBBC674164D58358456CF615490C18E0270649D6`

## Hypothesis

Natural command contrasts span a compact causal control space. Different
domains depend on different coordinates in that space. A held-out domain's
functional coordinate is predicted by its downstream Jacobian, not by
activation variance.

This experiment is specifically allowed to reject that hypothesis. It does
not rotate axes, change layers, or choose a sensitivity statistic after seeing
target steering results.

## Domains and splits

The four frozen domains are color, material, animal, and shape state. Each
uses eight one-token candidate answers and 50 unique
source-target histories:

- histories 0--14: natural-controller discovery;
- histories 15--19: non-interventional target calibration;
- histories 20--34: prospective causal-core holdout;
- histories 35--49: untouched random-null confirmation holdout.

Tokenization/alignment failures are recorded. All four domains must remain
aligned and behaviorally eligible. The color prompt retains the previously
frozen irrelevant carrier sentence. The other three prompts are new.

## Donor-only basis

For each target domain \(t\), exclude it and use only discovery histories from
the other domains. For every donor domain \(d\), compute the content-cancelled
natural controller at the three answer-prefix positions after layer 21:

\[
D_d = \mathbb{E}_{i,h}
  [z^{B}_{d,i,h} - z^{S}_{d,i,h}],
\]

where \(i\) indexes discovery histories and \(h\) indexes the original and
counterfactual content states. Apply uncentered SVD to the matrix whose rows
are flattened \(D_d\). The right singular vectors \(u_k\) are unit-norm,
energy-ordered candidate coordinates. Their signs are oriented toward the
mean donor controller.

The target discovery controller is not included in this basis. It only fixes
the natural target-domain amplitude of each donor axis:

\[
v_{t,k} = \langle D_t,u_k\rangle u_k.
\]

No target causal result enters the basis, sign, or amplitude.

## Prospective selection statistic

On the five target calibration histories, define a unit layer-24 route readout
from the natural BELIEF-minus-SEARCH difference at the frozen query readout
position:

\[
w_t =
\frac{\mathbb{E}_{i,h}[y^B_{t,i,h}-y^S_{t,i,h}]}
{\|\mathbb{E}_{i,h}[y^B_{t,i,h}-y^S_{t,i,h}]\|_2}.
\]

Here \(y\) is the concatenated pre-output-projection attention state at layer
24. At each unchanged layer-21 answer-prefix state \(z\), differentiate
\(\langle y,w_t\rangle\) with respect to the three-position \(z\). Parameters
are frozen and the activation is not changed.

For every donor-derived coordinate:

\[
s_{t,k}^{B} = \langle\nabla_{z^B}\langle y,w_t\rangle,v_{t,k}\rangle,
\quad
s_{t,k}^{S} = \langle\nabla_{z^S}\langle y,w_t\rangle,v_{t,k}\rangle,
\]

\[
s_{t,k} = \min(s_{t,k}^{B},s_{t,k}^{S}).
\]

The selected axis is the largest \(s_{t,k}\), breaking ties toward the lower
SVD rank. The predicted BELIEF-to-SEARCH and SEARCH-to-BELIEF signs are the
signs of the two directional derivatives. The complete axis ranking and its
SHA-256 are written before any target answer-prefix causal intervention.

## Causal holdout arms

Only the 15 prospective-core histories are used here. Each fold tests:

- zero intervention/original behavior;
- exact natural answer-prefix interchange;
- every donor-basis target component \(v_{t,k}\);
- the complete target projection into the donor span;
- the norm-matched raw donor mean;
- the selected component at the three instruction positions;
- the selected component at three token-identical random positions;
- three independently generated, per-position norm-matched random smoke
  controls.

For a component \(v\), BELIEF receives \(-v\) and SEARCH receives \(+v\).
No post-hoc sign flip is allowed. Every arm measures the established
layer-24 mediation route score in both directions. Clean behavior, answer
identity, source-intervention sufficiency, and exact layer-21 source-state
preservation are reported.

## Baselines and secondary measurements

The prospective Jacobian selection competes against:

- highest SVD energy;
- lowest SVD energy;
- largest target coefficient/vector norm;
- greatest cosine similarity;
- greatest donor-sample accessibility
  \(|\operatorname{mean}(x u_k)|/\operatorname{sd}(x u_k)\);
- three core random smoke controls, followed by at least 19 confirmation
  directions only after a candidate result.

Report Pearson and Spearman correlation between the frozen sensitivity ranking
and measured axis effects, causal effect versus SVD energy, recovery relative
to the best measured axis, and the number of low-energy wins over PC1.

## Frozen gates

A fold is eligible only when untouched BELIEF and SEARCH retain at least 80%
clean and counterfactual answer accuracy, source intervention is sufficient,
and the original route gap is at least `0.03`.

The exact natural interchange must have functional bidirectional score at
least `0.025`.

The selected axis must:

- preserve behavioral and source gates;
- score at least `0.02`;
- move both aggregate directions correctly;
- have both predicted signs in at least 12/15 worlds;
- recover at least 75% of the best measured candidate-axis score;
- beat all three functional random smoke controls;
- preserve the layer-21 source state to maximum absolute error `1e-8`;
- keep instruction and random-position control scores below
  `max(0.01, 0.5 * selected_score)`.

When the selector chooses a non-PC1 axis, the fold additionally requires that
axis to beat PC1 by at least `0.005`. Thus a low-energy selection cannot count
as a primary success merely because it passes an absolute threshold.

Core verdicts are explicitly suffixed `CANDIDATE`: the core can validate the
prospective selection rule but cannot provide the final random-null
significance.

## Core verdict table

- `PROSPECTIVE_LOW_ENERGY_CAUSAL_GEOMETRY_CANDIDATE`: at least three folds
  pass, pooled sensitivity/effect Spearman correlation is at least `0.50`,
  and at least two passing non-PC1 axes beat PC1.
- `PROSPECTIVE_COMPACT_CAUSAL_GEOMETRY_CANDIDATE`: at least three folds and
  the pooled correlation pass, but the strict low-energy count does not.
- `UNIVERSAL_HIGH_ENERGY_AXIS_CANDIDATE`: the same conditions pass and every
  passing fold selects PC1.
- `CAUSAL_AXES_NOT_PROSPECTIVELY_PREDICTABLE`: at least three domains contain a
  causal candidate axis, but fewer than three prospective folds pass.
- `PROSPECTIVE_CAUSAL_GEOMETRY_PARTIAL`: exactly two folds pass without a
  stronger verdict.
- `NO_TRANSFERABLE_DONOR_CAUSAL_CONTROL`: fewer than two folds pass and no
  stronger failure category applies.
- Tokenization/alignment, behavioral, and natural-reference failures receive
  separate verdicts.

## Prespecified confirmation

Only if the core produces a candidate verdict will a second stage use histories
35--49. It will carry forward the already hashed prediction and test the
selected axis, PC1 when different, and at least 19 per-position norm-matched
random directions. No new axis selection, rotation, layer choice, amplitude,
or threshold is permitted. This stage supplies the empirical random-null
\(p\leq0.05\) required for a non-candidate verdict.

## Core cost estimate

The core evaluates 48 causal arms. A lean evaluator retains clean behavior,
source-intervention sufficiency, and the exact layer-24 route assay while
omitting an unused intermediate top-eight blockade. This requires
approximately 576 causal forward passes, plus 16 gradient
forwards/backwards and controller captures. Expected Tesla-T4 wall time is
approximately 45--75 minutes after model loading. The optional confirmation is
expected to require roughly another 60--90 minutes only after a core pass.
