# Paper 2 discovery: distributed label-position transplantation

Status: frozen before GPU output, 2026-07-24.

## Motivation

Both a mean additive control and the complete final-position L21 state were
necessary but insufficient to install BELIEF's early route in SEARCH. Because
the position-matched label occurs in both the user formatting instruction and
the teacher-forced answer prefix, its control state may be distributed across
multiple sequence positions.

## Frozen design

Use the same 15 held-out worlds, original belief question, `ac` field, and
position-matched `BELIEF` and `X X SEARCH` contracts.

For paired BELIEF and SEARCH batches:

1. verify identical sequence shape and identical state-marker position;
2. compute the frozen set of sequence indices at which input token IDs differ;
3. verify this difference mask is identical across all worlds and clean/
   natural arms;
4. partition differing indices into exactly two contiguous occurrences: the
   label in the user formatting instruction and the teacher-forced answer
   command prefix; abort if this frozen partition is not available;
5. capture block-L21 output states at exactly those positions;
6. transplant paired BELIEF states into SEARCH and paired SEARCH states into
   BELIEF in three preregistered arms:
   - instruction-label occurrence only;
   - answer command-prefix occurrence only;
   - all differing positions;
7. sample 19 fixed-seed matched random subsets from token-identical positions
   after the state marker, excluding the marker itself. Every subset has the
   same cardinality as the all-differing set. At each subset, transplant the
   paired opposite-label L21 states in both directions;
8. outside the chosen transplant set, leave every position untouched,
   including the world
   history, state marker, question content, and location target;
9. recompute transformed baselines, L21 source interventions, and cumulative
   L22-L27 mediation curves for the three preregistered arms. Random controls
   require the cumulative-L24 cell and all eligibility/source gates.

Clean and natural arms remain separate and patches are paired within world.
Random sampling uses NumPy seed 9143 and observes token identity only.

## Frozen outcomes

Use the same L24 movement thresholds: at least 0.05 absolute movement and at
least 50% closure of the original BELIEF-minus-SEARCH gap in each successful
direction. Define the bidirectional score as the minimum of the two signed
L24 movements. Compare the all-differing score with the 19 random-set scores
using the add-one empirical tail probability; specificity requires `p <=
0.05`.

- `SPECIFIC_DISTRIBUTED_LABEL_SWITCH`: all gates pass, the all-differing arm
  meets both continuous criteria, at least one first-pass depth moves as
  predicted, and its random-control p-value is at most 0.05.
- `CONTINUOUS_SPECIFIC_DISTRIBUTED_SWITCH`: both continuous directions and
  random specificity pass but neither categorical depth moves.
- `NONSPECIFIC_DISTRIBUTED_SWITCH`: the all-differing arm meets the
  bidirectional movement criteria but fails the random-position specificity
  test.
- `ASYMMETRIC_DISTRIBUTED_LABEL_SWITCH`: exactly one direction passes.
- `NO_DISTRIBUTED_LABEL_SWITCH`: neither direction passes.
- `TOKEN_ALIGNMENT_INVALID`: sequence shape, marker, or difference masks are
  not aligned as required.
- `RANDOM_CONTROL_INELIGIBLE`: any random-position arm fails a behavior or
  source-sufficiency gate, so the preregistered specificity comparison cannot
  be interpreted.
- Remaining eligibility, source, original-gap, and depth verdicts retain
  their previous definitions.

The instruction-only, answer-prefix-only, and all-position arms localize
whether one occurrence is sufficient or multiple lexical effects act jointly.
Even a specific all-position result does not establish one coherent control
representation. That stronger claim requires a later low-rank/subspace
reconstruction that reproduces the switch and generalizes across synonyms and
worlds. Continued asymmetry implies that the early route is committed before
L21 or depends on a trajectory that cannot be overwritten at L21.
