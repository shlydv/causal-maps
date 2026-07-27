# Paper 2 diagnostic: position-matched output labels

Status: frozen before GPU output, 2026-07-24.

## Motivation

The neutral repeated-token ladder falsified a monotonic token-budget account
of causal handoff depth. The observed Qwen tokenization permits a stricter
control: `BELIEF`, `X TELL`, and `X X SEARCH` have the same total readout
position because each neutral `X` compensates for one token of label-length
difference in both the user formatting instruction and teacher-forced answer
prefix.

This experiment asks whether the output-label effect survives exact position
matching. It tests label conditioning, not yet lexical semantics.

## Frozen design

Model, 30 compatible worlds, narrative surface, L21 source interchange,
cumulative L22-L27 full-readout-attention clamps, and all existing behavior,
source, mediation, and endpoint gates remain unchanged.

Questions:

1. original belief question;
2. original tell/communication question;
3. original search/action question.

Position-matched output contracts:

1. `BELIEF`;
2. `X TELL`;
3. `X X SEARCH`.

The expected latent field is Alice's cube belief (`ac`) in all nine cells.
Within each question, all three readout positions must be exactly equal.

Primary outcome: mean L24 minimum bidirectional mediation for each label
across the three questions. Secondary outcome: the first passing cumulative
prefix for every cell.

## Frozen verdicts

- `POSITION_MATCHED_LABEL_EFFECT`: every cell is eligible and source
  sufficient; positions match exactly; the range across label means at L24 is
  at least 0.05; and at least one question has different first-pass depths
  across labels.
- `CONTINUOUS_POSITION_MATCHED_LABEL_EFFECT`: the L24 mean range is at least
  0.05 but all categorical first-pass depths are equal within every question.
- `NO_POSITION_MATCHED_LABEL_EFFECT`: all cells are eligible and resolved,
  but the frozen effect criteria fail.
- `POSITION_MATCH_INVALID`: readout positions differ within any question.
- `BEHAVIORALLY_INELIGIBLE`: any baseline fails.
- `SOURCE_SITE_INELIGIBLE`: baselines pass but any L21 source intervention
  fails.
- `DEPTH_UNRESOLVED`: any cell fails to pass through L27.

A positive result licenses "output-label-conditioned causal depth beyond
token length." It does not license "semantic label meaning" until identity
and meaning are independently manipulated.

