# Paper 2 diagnostic: neutral token-length ladder

Status: frozen before GPU output, 2026-07-24.

## Motivation

The question-command factorial found that continuous causal mediation curves
were much more stable within command labels than within semantic questions.
The labels also changed readout position: `TELL` was two tokens later than
`SEARCH`, and `BELIEF` was four tokens later. Because each command occurs in
the user formatting instruction and again in the teacher-forced answer prefix,
token budget is a candidate explanation for the apparent handoff hierarchy.

This experiment tests whether adding repeated, semantically inert `X` tokens
before a fixed terminal command shifts causal mediation toward earlier layers.
It does not yet distinguish user-instruction tokens from teacher-forced
answer-prefix tokens.

## Frozen design

Model, 30 compatible worlds, narrative surface, L21 source interchange,
cumulative L22-L27 full-readout-attention clamps, and existing behavior,
source-sufficiency, mediation, and endpoint gates are unchanged.

Two questions are tested independently:

1. original belief question;
2. original search question.

Five output contracts form the length ladder:

1. `SEARCH`;
2. `X SEARCH`;
3. `X X SEARCH`;
4. `X X X SEARCH`;
5. `X X X X SEARCH`.

The expected latent field remains Alice's cube belief (`ac`) in every cell.
The exact encoded readout position is recorded. It must increase strictly
with the registered filler count for both questions or the length
manipulation is invalid.

Primary continuous outcome: minimum bidirectional mediation at cumulative
L24. Secondary outcome: first passing cumulative prefix from L22 through L27.

## Frozen verdicts

- `TOKEN_LENGTH_DEPTH_SUBSTITUTION`: every cell is eligible and source
  sufficient; readout position increases strictly; L24 minimum mediation is
  nondecreasing with filler count for each question; and at least one question
  has both a strict L24 increase of at least 0.05 and an earlier first-pass
  depth at the longest prefix than at zero filler.
- `CONTINUOUS_LENGTH_EFFECT`: eligibility and position checks pass, both L24
  curves are nondecreasing, and at least one rises by 0.05, but the categorical
  first-pass depth does not move earlier.
- `NO_MONOTONE_LENGTH_EFFECT`: all cells are eligible and resolved but the
  frozen monotonic criteria fail.
- `TOKENIZATION_INVALID`: readout positions do not increase strictly with
  filler count in both question panels.
- `BEHAVIORALLY_INELIGIBLE`: any baseline fails.
- `SOURCE_SITE_INELIGIBLE`: baselines pass but any L21 source intervention
  fails.
- `DEPTH_UNRESOLVED`: any cell fails to pass through L27.

No semantic or autoregressive-time interpretation is licensed by this ladder
alone. A positive result requires a subsequent location-of-padding control.

