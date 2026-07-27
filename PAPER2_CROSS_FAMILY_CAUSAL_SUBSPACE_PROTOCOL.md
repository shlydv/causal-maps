# Paper 2 cross-family causal-subspace experiment

Status: frozen before model execution  
Date: 2026-07-26  
Stage: `delta_cross_family_causal_subspace`

Frozen protocol SHA-256:
`FDB053176FC5FCF4D059FAEEB562BAD7822FBB5FFA08DB55E3C3204091F98C2B`

## Scientific question

Do heterogeneous computations reuse a compact causal control subspace, with
task-specific coordinates and route orientation, or does each computation
require its own unrelated controller?

This is not a rerun of the heterogeneous eligibility screen and does not lower
its frozen threshold. The screen's first 15 source-target pairs are excluded.
All eight families enter this experiment; route response is a continuous
prospective outcome rather than a test-set eligibility criterion.

## Independent splits

For each family, the same frozen 50 unique directed color pairs are generated:

- pairs 1--15: prior screen, excluded;
- pairs 16--25: donor-controller estimation;
- pairs 26--35: held-out-family coordinate and route-orientation calibration;
- pairs 36--50: untouched causal test.

The families are private belief, latest update, key-value lookup, two-hop
pointer traversal, conditional selection, maximum-score comparison, constraint
elimination and temporal-slot retrieval.

## Geometry phase

For each held-out target family:

1. obtain layer-21 BELIEF-minus-SEARCH answer-prefix controllers from the
   other seven families only;
2. compute their uncentered SVD;
3. project the target's calibration controller into donor ranks 1, 2, 3 and
   7;
4. freeze every component, reconstruction statistic and basis artifact before
   any test-row intervention.

The causal phase runs only if median held-out rank-3 energy is at least 25%,
median rank-7 energy is at least 50%, and at least six families have rank-7
energy of at least 25%. A geometry-gate failure is a valid negative result.

## Prospective orientation

The sign of the calibration-row SEARCH-minus-BELIEF layer-24 route gap is
frozen for each family. Test interventions are scored by movement in this
predicted orientation. Thus positive and negative task orientations are both
valid predictions; no signed or absolute test-gap cutoff can make a family
ineligible.

## Causal arms

On 15 untouched rows per family:

- original;
- exact natural answer-prefix interchange;
- full within-family calibration controller;
- held-out donor-subspace reconstructions at ranks 1, 2, 3 and 7;
- norm-matched donor mean;
- target component orthogonal to the complete donor rank-7 span;
- rank-3 component placed at the instruction occurrence;
- three norm-matched random directions.

All arms retain answer accuracy, counterfactual accuracy and source
intervention checks. The 3-random core is only a smoke control; a 19-random
confirmation is reserved if the core earns it.

## Decision

A family's rank-3 arm passes only if:

- calibration orientation is at least `0.010` and has the same sign on test;
- exact interchange and the within-family controller establish a causal
  reference;
- rank 3 yields bidirectional movement of at least `0.015`;
- at least 10/15 worlds move in both predicted directions;
- rank 3 recovers at least 60% of the within-family score;
- it beats all three random directions and the instruction-position control;
- the rank-7-orthogonal residual is weaker;
- the earlier source state is unchanged.

Primary verdicts:

- `SHARED_LOW_RANK_CAUSAL_CONTROL`: rank 3 passes in at least 6/8 families and
  geometry predicts causal strength with pooled Spearman at least 0.50;
- `SHARED_HIGHER_RANK_CAUSAL_CONTROL`: rank 7, but not rank 3, passes in at
  least 6/8;
- `TASK_CONDITIONED_CAUSAL_RESIDUALS`: held-out residuals dominate in at least
  four families;
- `CAUSAL_ASSAY_OR_ORIENTATION_UNRESOLVED`,
  `LOCUS_OR_RANDOM_CONTROL_FAILURE`, or `NO_SHARED_CAUSAL_SUBSPACE`: the
  corresponding negative diagnosis.

## Interpretation

A low-rank pass would support a reusable causal control manifold rather than a
single universal PC1. A higher-rank pass would support shared infrastructure
without strong compression. Residual dominance would support genuinely
task-specific causal coordinates. Instruction or random-control failure would
favor a lexical or nonspecific perturbation account.

## Runtime

Expected Tesla-T4 runtime:

- geometry gate: approximately 4--7 minutes after model load;
- calibration plus full causal phase if the gate passes: approximately
  55--75 minutes total after model load.
