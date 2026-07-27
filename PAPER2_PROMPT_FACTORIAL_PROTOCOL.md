# Paper 2 diagnostic: crossed question-command factorial

Status: frozen before GPU output, 2026-07-24.

## Motivation

The preceding two-panel diagnostic found that all three original semantic
questions first passed at L26 when the answer command was fixed to `ANSWER`,
whereas changing only the command on the belief question moved `BELIEF` to
L24 while `TELL` and `SEARCH` remained at L26. This result contradicts a
surface-invariant semantic-operation ladder and nominates the teacher-forced
answer contract as a causal factor.

The present experiment tests that inference with a fully counterbalanced
3-by-3 design. It is confirmatory with respect to the command-prefix effect;
no prompts, layers, gates, or verdict rules may change after output.

## Frozen design

Model, 30 compatible worlds, narrative surface, L21 source interchange,
cumulative L22-L27 full-readout-attention clamps, and the existing 70%/80%
mediation gates are unchanged.

Questions (rows):

1. original belief question;
2. original tell/communication question;
3. original search/action question.

Teacher-forced answer commands (columns):

1. `BELIEF`;
2. `TELL`;
3. `SEARCH`.

All nine cells must pass CLEAN/NATURAL behavior and the bidirectional L21
source-sufficiency gate. Each cell's outcome is its first passing cumulative
prefix from L22 through L27. The complete mediation curves and endpoint
accuracies are retained.

## Frozen verdicts

- `COMMAND_INVARIANT_ACROSS_QUESTIONS`: every command column has one depth
  across all three questions, and at least two command columns differ.
- `QUESTION_INVARIANT_ACROSS_COMMANDS`: every question row has one depth
  across all three commands, and at least two question rows differ.
- `NO_DEPTH_VARIATION`: all nine cells have the same first-pass depth.
- `MIXED_OR_INTERACTION`: all cells are eligible and resolved, but neither
  exact invariance pattern holds.
- `BEHAVIORALLY_INELIGIBLE`: any baseline fails.
- `SOURCE_SITE_INELIGIBLE`: baselines pass but any L21 source intervention
  fails.
- `DEPTH_UNRESOLVED`: any cell fails to pass through L27.

Exact column invariance is deliberately stringent. A mixed verdict does not
erase the prior result; it means that question wording or a question-command
interaction also affects the causal depth. No semantic interpretation is
licensed merely by a row or column label.

