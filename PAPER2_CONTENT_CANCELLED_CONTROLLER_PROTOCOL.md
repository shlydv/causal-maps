# Paper 2 discovery gate: content-cancelled answer-prefix controller

Status: frozen before GPU output, 2026-07-25.

## Question

Does a world-averaged difference between BELIEF and SEARCH answer-prefix
representations causally reconfigure the route used to access an unchanged
state on held-out worlds?

This is a prerequisite for cross-domain transfer. A paired full-state
transplant can carry world content as well as control. Averaging the
BELIEF-minus-SEARCH difference across donor worlds is intended to cancel
world-specific content while retaining any shared control component.

## Frozen design

Use Qwen-2.5-7B-Instruct in 8-bit precision, the original belief question,
`ac` field, narrative surface, and the tokenizer-compatible 30-world set.

- Donor split: worlds 0-14.
- Evaluation split: worlds 15-29.
- Source layer: L21 block output.
- Intervention locus: the final three answer-prefix positions.
- Route measurement: cumulative late-attention mediation through L22-L27,
  with L24 as the preregistered scalar checkpoint.
- Commands: the position-matched `BELIEF` and `X X SEARCH` contracts already
  frozen in the distributed-label experiment.

For each donor world, capture the three L21 answer-prefix states under both
clean and natural histories. Define one fixed displacement:

`D = mean(BELIEF state - SEARCH state)`

where the mean is over the 15 donor worlds and both history arms. No
evaluation-world state, label, route measurement, or output is used to fit,
select, scale, rotate, or otherwise modify `D`.

On every evaluation world and history arm:

- patch `BELIEF - D` at its own three answer-prefix positions;
- patch `SEARCH + D` at its own three answer-prefix positions.

The visible prompt, tokens, world history, state marker, source state, and all
other activations remain unchanged. Report donor-row cosine alignment with
`D`, but do not use it for selection.

## Frozen controls

1. **Instruction locus.** Add/subtract the same `D`, row for row, at the
   position-matched three-token instruction occurrence.
2. **Norm-matched random directions.** Generate 19 seeded Gaussian
   three-position displacements. Independently normalize each position to the
   corresponding norm of `D`, then apply them at the answer-prefix locus.
3. **Matched random positions.** Generate 19 seeded sets of three
   token-identical post-marker positions and apply the unchanged `D` there.

All controls use the same evaluation worlds, both history arms, both
directions, source intervention, and L24 route measurement. The empirical
rank test is `(1 + exceedances) / 20`; the smallest possible p-value is 0.05.
For a random arm to reproduce the target effect it must remain behaviorally
eligible and source-site sufficient as well as match the bidirectional route
score. Raw scores and eligibility are retained for every arm.

## Frozen gates

The selected intervention is interpretable only if all original and selected
tasks satisfy:

- clean and natural answer accuracy at least 0.80;
- source intervention sufficient under the frozen Paper 2 gate;
- resolved cumulative handoff depth in L22-L27.

The original held-out L24 route gap must be at least 0.05. Define:

- BELIEF-to-SEARCH movement = original BELIEF L24 mediation minus patched
  BELIEF L24 mediation;
- SEARCH-to-BELIEF movement = patched SEARCH L24 mediation minus original
  SEARCH L24 mediation;
- primary score = the smaller of those two movements.

Each direction must move at least 0.05 and at least half the original gap.
Every one of the 15 worlds must move in the predicted direction in both
directions. The selected primary score must outrank all 19 random-direction
controls and all 19 matched-position controls (p = 0.05 for each family).
The instruction-locus primary score must be smaller than half the selected
primary score.

## Frozen verdicts

- `CONTENT_CANCELLED_PREFIX_CONTROLLER`: all gates pass and at least one
  patched route changes categorical first-passing depth.
- `CONTINUOUS_CONTENT_CANCELLED_PREFIX_CONTROLLER`: all gates pass without a
  categorical depth change.
- `ASYMMETRIC_CONTENT_CANCELLED_EFFECT`: exactly one direction passes.
- `NONUNIFORM_CONTENT_CANCELLED_EFFECT`: aggregate direction gates pass but
  at least one evaluation world moves against prediction.
- `NONSPECIFIC_RANDOM_DIRECTION` or `NONSPECIFIC_POSITION_EFFECT`: the
  corresponding empirical control fails.
- `INSTRUCTION_LOCUS_EFFECT`: the instruction control is too large.
- Explicit behavioral, source-site, depth, alignment, and absent-gap verdicts
  take precedence.
- Otherwise: `NO_CONTENT_CANCELLED_CONTROLLER`.

## Interpretation boundary and next action

A pass supports a low-dimensionality candidate only in the weak sense that a
single fixed three-position displacement generalizes across held-out worlds.
It does not yet establish a low-rank subspace, domain-general computation, or
architecture-general mechanism.

Only a pass unlocks the preregistered ownership, color-state, and key-value
cross-domain transfer run using this exact donor construction without
refitting. An asymmetric or null result stops that run and redirects analysis
to context dependence or nonlinear transport.
