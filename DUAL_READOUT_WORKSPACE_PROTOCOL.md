# Cross-fitted dual-readout workspace protocol

Status: preregistered Qwen discovery. This file freezes the primary gates before
the first GPU run.

## Question

Can one direction learned at a shared pre-query arithmetic checkpoint on donor
problems edit held-out computational state strongly enough to produce two
different correct consequences: the exact sum and its parity?

This is deliberately stronger than matched activation patching. Row-matched
patches may contain the entire answer and are only used to select a candidate
workspace layer on donor rows. The held-out intervention is one donor-mean
direction, never a held-out natural activation.

## Data and split

- CLEAN contains two one-digit operands.
- NATURAL increases the first operand by exactly one.
- The prompt orders a `Computation checkpoint: READY` before the question.
- Each problem has two suffixes: exact sum and even/odd.
- Donor second operands are 1 or 2. Held-out second operands are 3, 4, or 5.
- No held-out activation, behavior, or result may influence layer selection.

## Gates

1. **Behavior:** CLEAN and NATURAL greedy accuracy must each be at least 7/8
   for both readouts on both splits. Otherwise stop.
2. **Donor localization:** choose the earliest of L4, L8, L12, L16, L20, L24,
   and L26 where row-matched checkpoint exchange is bidirectionally sufficient
   for both donor readouts. Sufficiency requires at least 6/8 target/source
   accuracy, positive effect on at least 6/8 rows, and 0.60--1.40 natural-effect
   matching. If no layer passes, stop.
3. **Cross-fitted edit:** at the frozen layer, average NATURAL-minus-CLEAN READY
   states across donor rows. Add this single vector to every held-out CLEAN row.
   It must meet the same accuracy, sign, and effect-match thresholds for both
   exact sum and parity.
4. **Controls:** reverse direction and raw operand-embedding difference must
   fail the primary gate for each readout. Row-matched CLEAN overwrite of the
   held-out NATURAL checkpoint must restore the CLEAN answer on at least 6/8
   rows for both readouts.
5. **Random control:** 20 preregistered norm-matched isotropic directions. The
   real minimum effect ratio across readouts must have add-one p <= .05. This is
   a discovery null; frozen cross-model confirmation will use 100 draws.

## Interpretation

A full pass supports a reusable, causally necessary internal state edit with
multiple coherent consequences. It does not prove a universal arithmetic
register or a general theory of reasoning. A behaviorally eligible failure is
a meaningful boundary: the prior operand-address edit propagates computation,
but its later state is not captured by one transferable affine direction under
this protocol.

After a pass, freeze the selected normalized layer and all thresholds for one
Mistral confirmation. Only after cross-model confirmation should a planning
task be attempted.
