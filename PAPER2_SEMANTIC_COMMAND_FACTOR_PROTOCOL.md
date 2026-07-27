# Paper 2 diagnostic: semantic operation versus command prefix

Status: frozen before GPU output, 2026-07-24.

## Motivation

The original narrative contracts produced belief L24, tell L26, search L27,
but a ledger/paraphrase replication produced state L27, report L26, and an
ineligible action source intervention. This crossed diagnostic isolates two
prompt factors that were previously changed together.

## Frozen 2-panel design

Model, 30 worlds, narrative surface, L21 source interchange, cumulative
L22-L27 full-readout-attention clamps, and 70%/80% mediation gates are
unchanged.

Semantic panel — vary only the question, hold the answer command `ANSWER`
constant:

1. original belief question;
2. original tell/communication question;
3. original search/action question.

Command panel — hold the original belief question constant, vary only the
teacher-forced answer command:

1. `BELIEF`;
2. `TELL`;
3. `SEARCH`.

All six tasks must pass CLEAN/NATURAL behavior and the L21 source intervention.
No prompt, command, layer, or gate is selected after output.

## Verdict

- `SEMANTIC_OPERATION_EFFECT`: semantic depths are strictly ordered
  belief < tell < search, while all command-only depths are equal.
- `COMMAND_PREFIX_EFFECT`: semantic depths are not strictly ordered and the
  command-only depths differ.
- `MIXED_PROMPT_FACTORS`: all tasks are eligible but neither clean
  dissociation holds.
- `NO_DEPTH_VARIATION`: all six first-pass depths are equal.
- `BEHAVIORALLY_INELIGIBLE`: a baseline fails.
- `SOURCE_SITE_INELIGIBLE`: baselines pass but an L21 source intervention
  fails.
- `DEPTH_UNRESOLVED`: a curve never passes through L27.

This diagnostic adjudicates the original ordering; it does not by itself
establish a model-general mechanism.
