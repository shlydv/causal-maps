# Paper 2 confirmation: Qwen-14B distributed answer-prefix replication

Status: frozen before GPU output, 2026-07-24.

## Claim under test

The causal route from an unchanged stored state can be reconfigured by
transplanting distributed answer-prefix representations. This is a candidate
mechanism; storage-versus-access remains an interpretation.

## Frozen model and sites

Replicate on Qwen-2.5-14B-Instruct-AWQ, using the existing 30-world compatible
set and held-out rows 15-29. Layer 32 is the source/transplant layer because it
was the latest sufficient source-anchor layer in the frozen Paper-1 Qwen-14B
closeout; layer 36 had already failed that gate. Evaluate cumulative
full-readout-attention mediation at checkpoints
`34, 36, 38, 40, 41, 42, 44, 46, 47`, clamping every layer from 33 through the
checkpoint. Layer 41 is the preregistered decision checkpoint, matching
Qwen-7B L24 by normalized depth.

## Frozen design

Use the identical belief question, `ac` field, and position-matched `BELIEF`
and `X X SEARCH` contracts from the Qwen-7B discovery.

1. Require identical sequence shape, state-marker position, and clean/natural
   difference masks.
2. Require exactly two contiguous differing-token groups.
3. Capture block-L32 output at all sequence positions.
4. Transplant paired opposite-contract states in both directions at:
   instruction-label positions only, answer-prefix positions only, and all
   differing positions.
5. Preserve the frozen 19-set matched random control: subsets with the same
   cardinality as the answer-prefix occurrence, sampled from token-identical
   post-marker positions with seed 9143, with paired transplants in both
   directions.
6. Require 100% clean/natural behavior and bidirectional sufficiency of the
   unchanged state-marker intervention in every selected and random arm.

## Frozen primary outcome

At cumulative checkpoint L41, first require an original BELIEF-minus-SEARCH
minimum-mediation gap of at least 0.05. A direction succeeds when it moves at
least 0.05 and closes at least half this gap. Primary replication requires:

- both answer-prefix-only directions succeed;
- at least one direction changes first-passing checkpoint as predicted;
- the answer-prefix bidirectional score exceeds all 19 random scores
  (add-one empirical `p = 0.05`);
- all behavior and source-sufficiency gates pass.

The all-differing arm is reported as a secondary replication. Instruction-only
must be reported regardless of outcome. Failure of the original route gap is a
failed/absent replication, not a license to select a different checkpoint.

## Interpretation

A positive result is `CROSS_SCALE_DISTRIBUTED_ROUTE_REPLICATION`, evidence that
the causal interface generalizes from Qwen-7B to Qwen-14B. It is not
cross-architecture replication and does not establish a unified low-rank
controller. A negative result bounds the phenomenon to the tested model/scale
and protocol.
