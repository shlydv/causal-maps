# Paper 2 exact-transplant locus diagnostic

Status: frozen before model execution  
Date: 2026-07-26  
Stage: `delta_exact_transplant_locus_diagnostic`

Frozen protocol SHA-256:
`C1D9FF28058F80EA51461794BB35BC974E2842CC010708FFC46AFE0A10E0DBEA`

## Purpose

This is the final diagnostic in the current Paper-2 experiment loop. It asks
why exact layer-21 answer-prefix interchange was bidirectional for
`maximum_score` but predominantly one-way for `two_hop_pointer`.

It distinguishes:

1. a computation-dependent causal layer or token locus;
2. control distributed beyond the answer prefix;
3. an incomplete layer-24 route assay;
4. intrinsically direction-asymmetric control;
5. failure of exact local state transport.

The result closes this loop. No prompt, threshold, layer or position revision
will follow.

## Data

The experiment uses 12 histories built from the six directed color pairs that
were absent from every previous 50-pair experiment. Each pair receives two
frozen distractor variants.

Families:

- `maximum_score`: prior bidirectional reference;
- `two_hop_pointer`: prior one-directional failure.

## Intervention grid

At layers `12, 16, 20, 21, 22, 23, 24, 26`, selected origin-operation states
are replaced by the exact opposite-operation states from the same row and
history. BELIEF-to-SEARCH and SEARCH-to-BELIEF are evaluated separately.

Position groups:

- three answer-prefix command positions;
- three instruction command positions;
- all six differing positions;
- three matched identical-token positions;
- six matched identical-token positions.

Identical controls are selected once with frozen seeds from tokens downstream
of the instruction command when possible. They control for transplanting an
equal number of contextualized hidden states.

## Outcomes

At strictly downstream checkpoints—layer 24 and/or layer 27—the experiment
measures:

- projected progress from the origin state toward the opposite-operation
  state;
- patched-to-target distance divided by original origin-to-target distance;
- value-token accuracy, which must remain at least 80%.

Same-layer checkpoint measurements are excluded to prevent a tautological
pass when the transplanted positions include the readout token.

A direction passes when:

- mean progress is at least `0.25`;
- median target-distance ratio is at most `0.90`;
- at least 18/24 clean-plus-counterfactual rows have positive progress;
- minimum value-token accuracy is at least `0.80`.

A cell passes only when both directions pass.

## Frozen interpretations

- `REFERENCE_FAMILY_NOT_REPRODUCED`: diagnostic assay did not reproduce the
  known positive family.
- `IDENTICAL_POSITION_CONTROL_FAILED`: transport is nonspecific.
- `L24_ASSAY_INCOMPLETE`: two-hop succeeds terminally but nowhere at layer 24.
- `DISTRIBUTED_CONTROL_BEYOND_ANSWER_PREFIX`: all differing positions rescue
  two-hop but answer-prefix positions do not.
- `COMPUTATION_DEPENDENT_CAUSAL_LOCUS`: two-hop succeeds at a different best
  layer or position group from the reference.
- `EXACT_TRANSPLANT_ROUTE_RESCUED`: both families share the same best locus.
- `DIRECTION_ASYMMETRIC_CONTROL`: no two-hop cell is bidirectional, but at
  least one cell passes exactly one direction.
- `NO_EXACT_TRANSPLANT_CAUSAL_ROUTE`: no coherent local transport is found.

## Runtime

Approximately 10--20 minutes on a Tesla T4 after model loading.
