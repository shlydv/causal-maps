# Paper 2: shared-plus-context offline audit

Status: **frozen before decomposition output, 2026-07-27**.

This is an exploratory, zero-GPU analysis of the response maps saved by the
completed context-geometry width screen. It cannot retroactively change that
experiment's frozen `NO_DEEPER_BRANCH_LICENSED` verdict.

## Question

The width screen found smooth causal dose response but a mixed response-map
result: stable within-family maps, substantial cross-family similarity, and
insufficient separation for the predeclared context-specific claim. The
candidate descriptive model is therefore

`M(x) = M_shared + Delta_M_family(x) + epsilon_x`.

This audit asks whether `Delta_M_family` is reproducible on unseen rows and
improves prediction enough to motivate a new prospective hypothesis. A
tautological fit to the same maps is not sufficient.

## Data and split

Use the frozen L24/L27 response-map NPZ. Each family, operation and checkpoint
contains a tensor `[2 histories, 8 rows, 4 probe axes, 3584 outputs]`.

Enumerate all 70 choices of four training row IDs from eight. Both histories
for a row remain in the same split. No test row contributes to a predictor.
Report L27 as primary and L24 as a depth replication.

For every split, checkpoint and operation:

- `family_train[f]` is the mean map from family `f` on training rows;
- `shared_train[f]` is the mean of `family_train[g]` for `g != f`;
- `correction_train[f] = family_train[f] - shared_train[f]`;
- the shared-only prediction is `shared_train[f]`;
- the shared-plus-context prediction is
  `shared_train[f] + correction_train[f]`, equal to the training-only family
  template;
- the evaluation target is every individual map in the unseen rows.

The wrong-context control adds each of the three other families' correction
directions to the same shared predictor after norm matching it to the true
family correction. This holds correction magnitude fixed while changing its
identity.

## Frozen metrics

For every family and operation, aggregate across all outer splits:

- cosine between independently estimated train and test family residuals;
- relative squared prediction error of shared-only, shared-plus-context and
  wrong-context predictors;
- cosine prediction quality of those predictors;
- context-correction norm divided by shared-map norm.

Also report a variance decomposition on the full data:

- energy of the operation-specific grand mean;
- energy of between-family deviations;
- residual within-family energy;
- singular spectrum of the eight family-by-operation mean deviations.

## Decision rule

A prospective shared-plus-context experiment is justified only if, at L27:

1. median train/test context-residual cosine is at least `0.30`, with positive
   residual cosine in at least six of eight family-operation cells;
2. shared-plus-context reduces median unseen-row squared error by at least
   `10%` relative to shared-only, with improvement in at least six of eight
   cells;
3. shared-plus-context beats the mean norm-matched wrong-context control by
   at least `10%` in median squared error;
4. L24 has the same direction of effect for conditions 1 and 2.

If all conditions pass, this licenses designing—not claiming—a fresh,
preregistered held-out inverse-control experiment. Otherwise the
shared-plus-context geometry branch is closed.

No family, layer, operation, split, metric or threshold will be removed or
changed after seeing the output. All results, including failures, are logged.
