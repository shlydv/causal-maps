# Multi-token causal-support locus curve

Status: frozen before first run, 2026-07-21. Paper 1 robustness control.

## Question

The Qwen-14B `STATECHECK` token is decodable within surface but a matched
single-token state swap moves at most 0.0003 of the natural effect. This
experiment asks whether the null is token-local while causal support is spread
across multiple pre-query positions. It does not retest the completed write,
probe, or checkpoint experiments.

## Frozen model and rows

- Qwen2.5-14B-Instruct-AWQ, official Kaggle checkpoint, dual Tesla T4 only.
- The same mechanically selected tokenizer-compatible structured-world census
  used by the headline run (up to 30 rows), query `belief_ac`.
- Clean changes Alice's cube belief from Paris to Rome only in the natural arm.
- Layers: 2, 4, 8, 12, 16, 20, 24, 26, 32, 36, 41, 46.

## Frozen loci

At each layer, patch matched natural activations into clean, and clean into
natural, at:

1. `marker_only`: final `STATECHECK` token;
2. `summary_span`: the full "Silently compute ... STATECHECK" span;
3. `source_anchors`: the six stated location-token positions;
4. `anchors_plus_summary`: union of 2 and 3;
5. `full_prequery`: every position through `STATECHECK`;
6. `full_matched_prefix`: every teacher-forced input position, including the
   question/readout prefix (exact intervention sanity bound);
7. six leave-one-anchor-out variants of locus 3;
8. three seeded, size-matched position sets disjoint from locus 4.

All positions are located mechanically and must be uniform across the retained
rows. No locus or layer may be added after results are seen.

## Endpoints and interpretation

Clean and natural accuracy must each be at least 0.80. For every locus/layer,
report forward natural-effect ratio, reverse recovery ratio, forward target
accuracy, reverse clean accuracy, and all row-level logit differences.

A locus is sufficient only when both ratios are in [0.60, 1.40] and both
directional accuracies are at least 0.80. The `full_matched_prefix` arm must
become sufficient or the intervention is invalid as an upper bound.

The result is a locus curve, not evidence that a compact world-state object
exists. A multi-token pass narrows the earlier claim to token-local
non-substitutability. Passing size-matched random loci weakens anatomical
specificity. `full_prequery`-only success supports distributed pre-query causal
support but is not itself a localized mechanism. If only the exact matched
prefix passes, query/readout-token state is required at the tested layer.
