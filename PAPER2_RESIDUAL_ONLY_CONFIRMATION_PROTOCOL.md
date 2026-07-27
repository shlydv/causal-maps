# Paper 2: Residual-Only Causal Controller Confirmation

Status: frozen while the shared-adapter control battery is still running and
before any evaluation on the third fresh color set.

## Confirmatory question

Does the low-energy color residual, by itself, provide a sign-specific and
position-specific causal controller on another unseen set, or was its earlier
success a random-direction, prompt-set, or intervention-magnitude artifact?

## Frozen construction

- Model: Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Reconstruct the location, ownership, and color controllers from their frozen
  15-row donor sets.
- Construct the location/ownership PC1 exactly as before.
- Let `P` be the orthogonal projection of the color controller `C` onto that
  PC1 and let `R = C - P`.
- Color evaluation activations are excluded from every construction step.
- Intervene with R alone at all three answer-prefix positions at residual
  layer 21. The high-energy projection P is never included in the selected
  arm or its null controls.

## Independent evaluation

Use a third set of 30 color histories. Its 30 clean and 30 natural rendered
histories must be unique and disjoint from every clean/natural color history
used by the cross-domain, controller-matrix, leave-color-out, and
shared-adapter experiments. Source-target shifts cycle through 1, 3, 5, and 7;
unused ordered distractor pairs are chosen deterministically.

The frozen arm gate is unchanged:

- both aggregate movements >= 0.05 and >= half the original gap;
- at least 27/30 movements have the predicted sign separately in each
  direction;
- both exact one-sided sign-test p-values <= 0.01;
- all original/patched behavioral, source-sufficiency, and depth gates pass.

## Signed dose response

Evaluate full cumulative L22--L27 curves for residual multipliers:

`-1.0, -0.5, +0.25, +0.5, +0.75, +1.0, +1.25`.

The selected arm is +1.0. Positive scores from +0.25 through +1.0 must be
nondecreasing within tolerance 0.005. Both movements at -0.5 and -1.0 must be
negative, establishing polarity rather than nonspecific disruption. The +1.25
arm is a frozen saturation probe and is not required to exceed +1.0.

## Residual-only specificity

- Patch R alone at the instruction-label positions.
- Patch R alone at 19 seeded sets of three matched identical-token positions.
- Patch 39 seeded random residuals alone at the answer-prefix positions. Each
  random residual is separately orthogonal to the shared PC1 at each position
  and matched exactly to R's norm at that position.

The selected +1.0 score must exceed the 39-direction null at empirical
p <= 0.025 and the 19-position null at p <= 0.05. The instruction-locus score
must be less than half the selected score.

## Verdicts

- `RESIDUAL_ONLY_CAUSAL_CONTROLLER`: the target and +1.0 arm pass; direction,
  position, and instruction controls are specific; positive doses are
  monotonic; and negative doses reverse both movements.
- `RESIDUAL_ONLY_EFFECT_NONSPECIFIC`: +1.0 passes but a specificity control
  fails.
- `RESIDUAL_DOSE_OR_POLARITY_FAILED`: +1.0 and controls pass but dose
  monotonicity or polarity fails.
- `COLOR_TARGET_UNRESOLVED`: original behavior is ineligible.
- `RESIDUAL_ONLY_NOT_REPLICATED`: otherwise.

This experiment establishes held-out sufficiency and intervention
specificity. It does not establish endogenous necessity; that remains the
separate next protocol.
