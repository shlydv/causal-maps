# Paper 2: Controller-to-Circuit Epistasis and Rescue

Status: version 2 frozen after the version-1 row-construction failure and
before any controller-circuit activation or outcome was evaluated.

## Question

Does the confirmed layer-21 residual controller select the downstream
information-access route through a specific, causally necessary and sufficient
set of immediate layer-22 attention heads?

## Fixed causal graph

`L21 answer-prefix residual -> L22 gate heads -> frozen L22--L24 transport set`

The residual controller and the frozen transport set are inherited unchanged
from prior protocols. The transport set is:

`L23H11, L24H21, L22H1, L23H6, L22H25, L23H4, L23H13, L24H27`.

The experiment does not search over downstream outcomes or redefine the
transport circuit.

## Model and data

- Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Reconstruct the location, ownership, and color donor controllers exactly as
  before; construct the location/ownership PC1 and orthogonal color residual R.
- Reuse the 30 residual-only confirmation histories for activation-only
  discovery. They are already behaviorally validated, and no prior head-level
  activation or circuit outcome is used.
- Construct 30 causal-holdout histories absent from every preceding color set
  under the original, already behaviorally stable color prompt.
- No holdout activation or outcome enters head selection.

## Activation-only gate-head discovery

R is applied at the three layer-21 answer-prefix positions. Candidate gate
heads are all layer-22 attention heads except L22H1 and L22H25, which already
belong to the frozen transport set.

For each candidate head, define its source response as the concatenation of:

- forward-source-patched output minus clean output; and
- reverse-source-patched output minus natural output.

Let g be SEARCH response minus BELIEF response. Let u_B be the change caused
by applying -R to BELIEF, and u_S the change caused by applying +R to SEARCH.
For heads whose norm of g is at least the median candidate norm, score:

`min(dot(u_B,g)/||g||^2, dot(u_S,-g)/||g||^2)`.

Select the top four heads by score, breaking ties by head index. Selection uses
head activations only, never logits, mediation scores, or holdout data.

## Holdout calibration

On the 30 causal-holdout histories:

- all original and steered tasks must retain behavioral eligibility and
  source-intervention sufficiency;
- the original BELIEF-minus-SEARCH transport-mediation gap must be >= 0.03;
- -R on BELIEF and +R on SEARCH must each move the L24 frozen-transport
  mediation by >= 0.05 and by at least half the original gap;
- the layer-21 source-anchor state must remain unchanged to 1e-5.

Failure makes the circuit test unresolved.

## Same-prompt blockade

In each R-steered run, replace only the selected layer-22 head outputs with
their values from the corresponding unsteered run. Donors are matched by
operation, prompt, history state, and pass type (clean, natural, forward source
intervention, or reverse source intervention).

Blockade passes when, in both directions:

- at least 50% of the calibrated controller movement is removed; and
- at least 24/30 per-world blockade effects have the predicted sign, with an
  exact one-sided sign-test p <= 0.01.

## Same-prompt rescue

In each unsteered run, replace only the selected layer-22 head outputs with
their values from the corresponding R-steered run, again matched by operation,
prompt, history state, and pass type.

Rescue passes when, in both directions:

- at least 50% of the calibrated controller movement is recreated; and
- at least 24/30 per-world rescue effects have the predicted sign, with an
  exact one-sided sign-test p <= 0.01.

All selected blockade and rescue contexts must preserve behavioral eligibility
and source-intervention sufficiency.

## Head-set controls

Sample 19 seeded sets of four layer-22 heads from the candidate pool excluding
the selected heads. Apply the identical blockade and rescue procedure.

The selected bidirectional blockade fraction and rescue fraction must each
exceed their corresponding random-head null at empirical p <= 0.05. All
random contexts must remain behaviorally and source-intervention eligible.

## Verdicts

- `CONTROLLER_GATES_TRANSPORT_CIRCUIT`: calibration, source invariance,
  selected blockade, selected rescue, and both random-head specificity tests
  pass.
- `CONTROLLER_CIRCUIT_BLOCKADE_ONLY`: calibrated and specific blockade passes,
  but rescue does not.
- `CONTROLLER_CIRCUIT_RESCUE_ONLY`: calibrated and specific rescue passes, but
  blockade does not.
- `CONTROLLER_EFFECT_DISTRIBUTED_OR_NONSPECIFIC`: a selected causal effect is
  present but a random-head specificity gate fails.
- `CONTROLLER_CALIBRATION_FAILED`: the inherited controller does not reproduce
  on the causal holdout.
- `GATE_HEAD_DISCOVERY_UNSTABLE`: fewer than four eligible positive-scoring
  discovery heads exist.
- `CONTROLLER_CIRCUIT_UNRESOLVED`: otherwise.

Passing the top verdict would support a controller-to-circuit mechanism, not
merely a steerable activation direction. It would not by itself establish
cross-model generality.
