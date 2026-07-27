# Paper 2: Leave-Color-Out Shared-Component Protocol

Status: frozen before the new target evaluation.

## Question

Can a low-rank answer-prefix controller learned without any color-domain
activations causally reconfigure the BELIEF/SEARCH route on fresh color
histories?

This is a prospective test of the shared-component interpretation suggested by
the frozen controller matrix. It is not a post-hoc localization experiment.

## Frozen construction

- Model: Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Intervention locus: the three answer-prefix positions at residual layer 21.
- Donor data: the same 15 location and 15 ownership donor histories used by
  the controller-matrix experiment.
- The color controller and all color evaluation activations are excluded from
  the primary construction.
- Flatten the donor-mean location and ownership controllers into a 2 by
  (3*d_model) matrix and take its uncentered first right singular vector.
- Orient the vector so its mean projection on the two donor controllers is
  positive. Scale it by the mean of those two projections and reshape it to
  three answer-prefix vectors. This is the primary leave-color-out PC1
  controller.
- A simple mean of the two donor controllers is a frozen secondary comparator.
- Individual location and ownership controllers are frozen baselines.
- A color-domain controller learned on the earlier 15 color donor histories is
  an explicitly non-confirmatory oracle benchmark only. It is never used to
  construct or select the primary controller.

The donor-only geometry used to motivate the prediction is recorded before the
run: PC1 energy 0.9226676; PC1 cosine with the already frozen color controller
0.9012698. These numbers do not use the new evaluation histories.

## Fresh target set

Evaluate 30 new color histories. Source values cycle through the fixed
eight-color vocabulary and target shifts cycle through 1, 2, 3, and 4.
Distractor colors are chosen from the reverse end of the remaining cyclic
list. Therefore none of the exact histories appeared in either earlier color
experiment. No row is used for fitting.

## Readouts and arms

Measure cumulative causal mediation at layers 22--27 and report the first
passing prefix and layer-24 minimum mediation for:

- original BELIEF and SEARCH;
- location controller;
- ownership controller;
- donor mean;
- leave-color-out PC1 (primary);
- color oracle (benchmark).

Each controller is subtracted from BELIEF and added to SEARCH. Clean/natural
accuracy, source-intervention sufficiency, and answer preservation are required.

## Prospectively frozen statistical gate

The color target is eligible only if both original tasks are behaviorally
eligible, source-sufficient, depth-resolved, and have an original layer-24 gap
of at least 0.03.

An arm passes when:

- both aggregate movements are at least 0.05 and at least half the original
  gap;
- at least 27 of 30 worlds move in the predicted direction separately for
  BELIEF-to-SEARCH and SEARCH-to-BELIEF;
- each one-sided exact sign-test probability under p=0.5 is at most 0.01;
- all four original/patched tasks remain behaviorally eligible,
  source-sufficient, and depth-resolved.

This statistical 30-world rule was fixed before observing the new outcomes. It
does not revise the earlier controller-matrix verdict, which remains governed
by its frozen 15-of-15 rule.

## Specificity controls

For the primary PC1 controller:

- patch the instruction-label positions;
- test 19 seeded random directions matched separately to each position norm;
- test 19 seeded sets of three identical-token positions.

The primary effect must exceed both random nulls at empirical p <= 0.05, and
the instruction-locus score must be less than half the primary score.

## Verdicts

- `LEAVE_COLOR_OUT_SHARED_COMPONENT`: target eligible; oracle and PC1 pass;
  PC1 is specific; neither individual donor controller passes.
- `TARGET_EXCLUDED_TRANSFER`: target eligible; oracle and PC1 pass; PC1 is
  specific; at least one individual donor controller also passes.
- `MEAN_ONLY_TARGET_EXCLUDED_TRANSFER`: mean passes but PC1 does not.
- `COLOR_TARGET_UNRESOLVED`: the target or oracle benchmark fails.
- `NO_LEAVE_COLOR_OUT_TRANSFER`: otherwise.

All arms, rows, gates, null seeds, full curves, controller hashes, and runtime
provenance are retained regardless of outcome.
