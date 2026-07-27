# Paper 2: within-family conditional transport

Status: **frozen before GPU output, 2026-07-27**.

This is the final branch-closing test of the control-law hypothesis.

## Question

The cross-family source-only predictor failed even though exact matched state
transplantation succeeded in all eight computations. Does a low-complexity
predictor work when trained and tested inside one computation family?

For each family independently, fit the BELIEF-to-SEARCH and SEARCH-to-BELIEF
maps on 24 directed color pairs, select rank and ridge on eight different
pairs, and test on twelve histories formed from the remaining six directed
pairs with two distractor variants each.

All train, validation and test directed pairs are disjoint. The test
counterpart activation is unavailable when predictions are generated and
hashed.

## Fixed predictor and controls

The predictor, locus, ranks, ridge candidates, direct L24/L27 checkpoint
measurements and controls are identical to the preceding cross-family test.
Only the training scope changes:

- exact matched counterpart state;
- same-family conditional predictor;
- same-family mean displacement;
- same-family target-state centroid;
- same-family nearest neighbour;
- row-shuffled conditional prediction;
- instruction-position and matched identical-token controls.

Answer accuracy is part of every arm's functional gate. Route movement that
destroys the value answer cannot count as successful control.

## Frozen decision

A family passes only if the conditional predictor:

- passes both directions with at least 0.40 progress;
- recovers at least half of exact progress;
- moves at least 18/24 rows per direction;
- has median target-distance ratio at most 0.85;
- preserves at least 80% value accuracy;
- beats both same-family global templates and the row-shuffled prediction by
  at least 0.10;
- remains position-specific.

`FAMILY_SPECIFIC_CONTROL_LAWS` requires at least six of eight families.

- Six to eight: continue toward an operator-dictionary theory.
- Three to five: partial and insufficient for the proposed general mechanism.
- Fewer than three with valid exact references: close the learnable
  control-law branch.

No prompt, split, hyperparameter, threshold, locus, nonlinear-model or
architecture rescue follows a non-positive result.
