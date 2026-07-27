# Paper 2 endogenous-controller factorial protocol

Status: frozen before model execution  
Date: 2026-07-26  
Stage: `delta_endogenous_controller_factorial`

## Question

Is the independently validated low-energy color residual `R` a naturally used
answer-prefix route coordinate, or only an effective injected shortcut? Does
its causal efficacy depend non-additively on the dominant shared projection
`P`?

## Correction to the previous necessity run

The previous absolute reduction gate was impossible to satisfy when its
observed original route gap was smaller than the required reduction. This
protocol uses fractions of the original gap.

The previous equalization also fitted one coefficient independently at each of
the three answer-prefix positions. The validated additive controller uses one
scalar dose for the structured three-position vector. This protocol therefore
flattens the three positions and removes one sequence-level coordinate per
axis. `P` and `R` are globally orthogonal by construction, so their joint
equalization is a clean two-factor intervention.

## Frozen split

- Model: Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Donors: the existing 15 location, 15 ownership and 15 color donor rows.
- Evaluation: 30 exact color histories absent from all earlier color
  evaluations.
- Source-target pairs are unique. Each source and target color occurs three or
  four times.
- Irrelevant carrier sentence:
  `A sealed envelope rested on a nearby table throughout.`

No evaluation activation or model outcome is used to construct `P`, `R`, the
rows, thresholds or controls.

## Arms

At the three answer-prefix positions after layer 21:

1. untouched BELIEF and SEARCH;
2. additive `R` switch in both directions;
3. midpoint-preserving endogenous `P` equalization;
4. midpoint-preserving endogenous `R` equalization;
5. midpoint-preserving joint `P+R` equalization;
6. exact within-world transplantation of the natural opposite-command
   answer-prefix state.

The natural transplant is the on-manifold comparator. It is not used to choose
the controller.

## Measurements

- Existing source-intervention and cumulative transport mediation through
  layers 22--27.
- Per-world route movement and gap reduction.
- Fraction of the original BELIEF--SEARCH route gap removed.
- Factorial interaction:

  `joint reduction - P reduction - R reduction`.

- Post-block readout trajectories from layers 21--27:
  projection toward the natural opposite-command trajectory, normalized
  distance to that trajectory, and remaining BELIEF--SEARCH separation.
- Layer-21 source-state invariance.
- Clean answer accuracy and source-intervention sufficiency under every arm.

## Sanity controls

The full 39-direction and 19-position statistical null batteries already
passed in the independent residual-only confirmation and are not rerun.
This fresh template includes:

- residual equalization at the instruction occurrence;
- one frozen norm-matched orthogonal direction;
- residual equalization at one frozen matched identical-token position set.

## Frozen decision criteria

The full mechanism advances only if:

- untouched route gap is at least `0.03`;
- every arm retains clean behavioral eligibility and source sufficiency;
- additive `R` achieves at least half the bidirectional movement of exact
  natural answer-prefix interchange and passes the existing per-world sign
  test;
- endogenous `R` equalization removes at least half the original route gap and
  passes the existing 27/30-world sign test;
- at layer 24, at least 90% of worlds move toward the natural opposite
  trajectory in both directions and the median normalized distance is below
  one;
- endogenous `R` equalization reduces trajectory separation in at least 90%
  of worlds and its median remaining separation is below `0.65`;
- all intervention arms leave the layer-21 source state unchanged within
  `1e-5`;
- every sanity control removes less of the route gap than `R`;
- the absolute `P x R` interaction is at least `0.15` of the original gap,
  has the same sign in at least 24/30 worlds and has exact one-sided
  sign-test `p <= 0.01`.

Verdicts distinguish behavioral/source ineligibility, absent sufficiency,
absent necessity, non-natural trajectory change, nonspecificity and faithful
`R` without a stable interaction. Only
`ENDOGENOUS_CONTROLLER_INTERACTION` licenses the full controller-interaction
thesis.

## Interpretation

A pass would not by itself establish a universal controller. It would justify
the next, genuinely high-impact test: preregister a predictor of controller
success or reversal from the measured interaction/local downstream response
and prospectively test held-out contexts against magnitude, cosine, PCA-energy
and linear-accessibility baselines.

A failure of endogenous equalization or natural-trajectory convergence means
the residual should be reported as a powerful causal intervention that is not
identified with the model's native access mechanism.
