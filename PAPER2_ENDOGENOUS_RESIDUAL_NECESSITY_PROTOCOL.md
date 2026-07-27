# Paper 2: Endogenous Residual Necessity

Status: frozen while the residual-only confirmation is running and before any
evaluation on the carrier-template holdout.

## Question

Is the low-energy residual merely an effective injected steering direction, or
is its coordinate necessary for the model's naturally occurring
BELIEF/SEARCH route distinction?

## Frozen construction

- Model: Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Reconstruct the same location, ownership, and color donor controllers.
- Construct the location/ownership PC1, color projection P, and orthogonal
  color residual R exactly as in the preceding protocols.
- No evaluation activation or outcome enters P or R.

## Carrier-template holdout

Use 30 color rows under the same semantic task and answer contract, but append
one fixed irrelevant fact to every prompt: `A small wall clock in the room
showed noon throughout.` This makes every rendered prompt disjoint from prior
sets while preserving the state computation. Clean and natural histories are
unique within the set. All behavioral and source-sufficiency gates remain
mandatory.

Before adjudicating necessity, confirm that additive R alone still passes the
frozen 30-world route-switch gate on this carrier template. Failure of this
calibration makes the necessity test unresolved rather than negative.

## Midpoint-preserving endogenous equalization

For each history and each of the three answer-prefix positions, capture its
natural layer-21 BELIEF state B and SEARCH state S. Let d = B-S. Project d onto
the corresponding residual vector R. Replace only that projected contrast:

`B' = B - 0.5 proj_R(d)`

`S' = S + 0.5 proj_R(d)`

Thus B' and S' have identical R coordinates while their midpoint and every
component orthogonal to R are preserved. Coefficients are calculated from
activations only, independently per row and position; no outcome is fitted.

The original and equalized cumulative mediation curves are measured at
L22--L27. Necessity passes when:

- the original L24 BELIEF-minus-SEARCH mediation gap is at least 0.03;
- equalization reduces that gap by at least 0.04;
- the absolute remaining gap is at most 0.02;
- at least 27/30 per-world gap reductions are positive with exact one-sided
  sign-test p <= 0.01;
- all original and equalized behavioral, source-sufficiency, and depth gates
  pass;
- numerical midpoint preservation and residual-coordinate equalization hold
  to 1e-5.

## Controls

- Equalize the high-energy shared projection P at the answer prefix.
- Equalize R at the instruction-label positions.
- Equalize R at 19 seeded sets of matched identical-token positions.
- Equalize 19 seeded random directions at the answer prefix. Each is
  separately orthogonal to PC1 at every position and matched to R's
  per-position norm.

The residual necessity score must exceed both null families at empirical
p <= 0.05. Projection and instruction
equalization must each reduce the gap by less than half the selected residual
reduction.

## Verdicts

- `ENDOGENOUS_RESIDUAL_NECESSITY`: additive calibration passes; endogenous
  equalization passes; numerical invariants hold; and all controls are
  specific.
- `RESIDUAL_STEERING_WITHOUT_NECESSITY`: calibration passes but endogenous
  equalization does not.
- `NECESSITY_EFFECT_NONSPECIFIC`: equalization passes but a control fails.
- `CARRIER_TEMPLATE_UNRESOLVED`: original behavior or additive calibration
  fails.
- `ENDOGENOUS_NECESSITY_UNRESOLVED`: otherwise.

This protocol tests necessity of an endogenous coordinate. It does not by
itself establish architecture-level generality.
