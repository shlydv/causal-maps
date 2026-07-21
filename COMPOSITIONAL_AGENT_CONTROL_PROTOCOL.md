# Compositional agent-control registers

Status: preregistered frozen discovery and confirmation, 2026-07-14.

## General hypothesis

An agent's next behavior is governed by at least two separable latent control
variables: its workflow phase (`act` versus `answer`) and its answer evidence
policy (`internal` versus `external`). At a common final decision marker, these
states can be recovered from natural prompt contrasts, edited independently,
and composed. The claim is about repeatable causal functions, not a universal
literal vector or a complete decomposition of thought.

## One-load program

One frozen Mistral-7B-Instruct-v0.3 8-bit kernel runs four declared analyses:

1. database-workflow discovery;
2. calculator-workflow local replication with independently learned controls;
3. literal database-to-calculator transfer, reported separately from local
   functional replication;
4. supplemental transfer to untouched original zero-result transcripts.

There is no layer or alpha sweep. Seed is zero. Directions are extracted after
L2 at the common final assistant decision marker and applied at alpha one.
Normalized trajectory checkpoints are L2/L9/L16/L23/L30; L23 is the sole
mediator. Donors and held-out rows are disjoint.

## Factorial task

Every prompt contains the same two explicitly defined registers:

- phase A: emit the exact listed tool call;
- phase B: emit `ANSWER` plus the selected value;
- source I: select the stated internal candidate;
- source E: select the executed tool result.

The four natural cells are A/I, A/E, B/I, and B/E. Apart from the two register
values, each row's text is identical. Database workflows use actually
executable database calls. Calculator workflows use actually executable
addition calls, with a distinct stated internal rival. All evaluated evidence
rows have different internal and external values.

Each family must achieve at least 87.5% exact generation in all four natural
cells before its causal claims are eligible. Families unlock independently so
one failed screen does not waste the loaded model; literal transfer requires
both families to be eligible. Tokenization must show one unique
phase-token change and one unique source-token change, uniform within family.

## Directions and held-out interventions

From donor rows only:

- phase direction = marker(B/I) minus marker(A/I);
- evidence direction = marker(B/E) minus marker(B/I).

On held-out A/I prompts:

- no edit must emit the tool call;
- phase only must emit `ANSWER <internal>`;
- evidence only must remain in the tool-call phase;
- phase plus evidence must emit `ANSWER <external>`.

On held-out B/I prompts, evidence only must emit `ANSWER <external>`. On A/E,
phase only must emit `ANSWER <external>`. These cross-conditioned tests make
non-interference and composition load-bearing rather than descriptive.

Database discovery and calculator local replication each require at least
87.5% exact accuracy on every stated intervention, L23 cosine at least .80 and
relative trajectory error at most .60 for phase, evidence, and composition.
The composed edit must match both natural phase and teacher-forced evidence
effects within 0.60--1.40 with at least 75% positive rows.

For database discovery only, L23 natural/induced patch effects must match
within 0.70--1.30, CLEAN overwrite must remove at least 70% of both natural
and composed phase effects with block-fraction gap at most .20, and at most
1/100 norm-matched random composed directions may exceed the learned combined
phase-plus-evidence logit effect. Applying both directions eight tokens away
must fail exact B/E generation. The marker token is identical in all cells,
so its raw embedding difference is exactly zero by construction.

## Transfer claims

Literal database directions are also applied unchanged to held-out calculator
rows. Passing all exact intervention accuracies at 87.5% yields literal
cross-workflow transfer. Failure does not invalidate calculator local
replication; it establishes that the functions recur in locally rebuilt
coordinates rather than one portable vector.

Finally, each locally learned evidence direction is applied separately at the
final decision marker of untouched original lookup transcripts with key D and
executed result zero. This bridge is supplemental and cannot determine the
factorial verdict. It is reported without choosing the better direction after
seeing outcomes.

## Verdicts

- `COMPOSITIONAL_AGENT_CONTROL`: database discovery and calculator local
  replication both pass.
- `PORTABLE_COMPOSITIONAL_AGENT_CONTROL`: the above plus literal cross-family
  transfer passes.
- `AGENT_CONTROL_NOT_DECOMPOSED`: an eligible causal core fails.
- `SINGLE_FAMILY_COMPOSITIONAL_CONTROL`: exactly one eligible family passes;
  useful discovery evidence, but not the required cross-family confirmation.
- `COMPOSITIONAL_CONTROL_BEHAVIORALLY_INELIGIBLE`: a required natural grid
  fails before any family can be causally tested.

Even the strongest verdict establishes a controlled two-variable agent-state
decomposition, not arbitrary thought reading, universal coordinates, or
general planning control.
