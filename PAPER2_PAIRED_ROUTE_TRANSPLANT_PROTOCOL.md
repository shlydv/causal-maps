# Paper 2 discovery: exact paired lexical-route transplantation

Status: frozen before GPU output, 2026-07-24.

## Motivation

A donor-derived mean BELIEF-minus-SEARCH direction was causally necessary:
subtracting it from BELIEF changed L24 mediation from 0.736 to 0.650 and moved
the first passing depth from L24 to L26. Adding it to SEARCH was not
sufficient. This experiment distinguishes failure of the mean additive
approximation from failure of the L21 readout site itself.

## Frozen design

Use the same 15 held-out worlds, original belief question, `ac` field, and
exactly position-matched `BELIEF` and `X X SEARCH` contracts.

For every held-out world, capture the full output of block L21 at the final
teacher-forced readout position in four matched origins:

1. BELIEF clean;
2. BELIEF natural;
3. SEARCH clean;
4. SEARCH natural.

Evaluate:

- original BELIEF and SEARCH;
- SEARCH with its L21 readout state replaced by the paired BELIEF state from
  the same world and same clean/natural arm;
- BELIEF with its state replaced by paired SEARCH.

The state-marker source intervention is unchanged. Clean states are never
patched with natural states or vice versa, so no location content crosses
causal arms. Transformed baselines, source interventions, and cumulative
L22-L27 mediation curves are recomputed after transplantation.

## Frozen outcomes

Use the same original gap and continuous thresholds as the mean-vector switch:
each successful direction must move L24 minimum mediation by at least 0.05
and close at least 50% of the original BELIEF-minus-SEARCH gap.

- `BIDIRECTIONAL_PAIRED_ROUTE_TRANSPLANT`: all gates pass, both directions
  meet the continuous criteria, and at least one first-pass depth moves in the
  predicted direction.
- `CONTINUOUS_BIDIRECTIONAL_TRANSPLANT`: both continuous directions pass but
  neither categorical depth moves.
- `ASYMMETRIC_PAIRED_TRANSPLANT`: exactly one direction passes.
- `NO_PAIRED_ROUTE_TRANSPLANT`: all gates pass but neither direction passes.
- `ORIGINAL_GAP_ABSENT`, `BEHAVIORALLY_INELIGIBLE`,
  `SOURCE_SITE_INELIGIBLE`, and `DEPTH_UNRESOLVED` retain their previous
  meanings.

Bidirectional success would establish that the full label-conditioned L21
readout state is sufficient to select the late causal route while content is
held fixed. Failure of SEARCH-to-BELIEF would instead imply an upstream,
multi-position, or path-dependent control mechanism.

