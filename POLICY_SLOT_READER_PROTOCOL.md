# Causal policy-slot reader test

*Causal Maps · 2026-07-13*
*Status: **PRE-REGISTERED — one mechanistic test.***

## Motivation

The same early L2 controller changes both tool selection and answer
integration, but the corresponding L20 decision directions are not
interchangeable. A concrete explanation is that the policy label token acts as
an upstream memory slot and different downstream pathways read that slot at
different workflow stages.

## Question

Does causal access to the L2-steered policy-label token flow through distinct
sets of transformer layers during tool selection and answer integration?

This test localizes **read-layer pathways**, not individual attention heads.

## Frozen setup

- Qwen2.5-7B-Instruct, pinned revision, 8-bit runtime, seed 0.
- Template B and its fixed policy-label position 103.
- Even rows are donors; odd rows are held-out tests.
- Learn the `blue`−`red` residual direction at L2 from donor rows.
- Call stage: red prompts ending in `CALL`; score lookup versus calculator.
- Answer stage: red prompts containing a correct database call/result; score
  database result versus calculator sum on the eight non-collision rows.
- Candidate reader layers are L3–L27. The controller is added after L2.

## Causal edge intervention

At a selected layer, modify only its attention mask: every token after the
policy label is forbidden from attending directly to the policy-label key.
All other attention edges and model activations remain unchanged.

Every masked steered run is compared with an identically masked unsteered run.
The estimand is therefore the controller effect under that mask, not the
mask's total effect on the naturally present red policy token.

As a position-matched damage control, apply the identical mask to position 102,
the immediately preceding token. Both interventions begin masking at query
position 104, so neither changes processing at the policy token itself.

## Frozen discovery and test

1. On donor rows, measure each layer's reduction of the controller-induced
   target-minus-source logit effect.
2. Independently select the top six call layers and top six answer layers.
3. Freeze those sets.
4. On held-out rows, evaluate each stage with:
   - its own selected layer set;
   - the other stage's selected layer set;
   - its own layer set masking control position 102.
5. Compare each own-set loss with 100 seeded random six-layer sets.

No layer, cardinality, position, coefficient, or threshold search.

## Gates

- G0: on donor and held-out rows the unblocked L2 controller has a positive
  mean causal logit effect, and on held-out rows achieves ≥90% target call
  decisions and ≥87.5% target answer decisions.
- R0: each own six-layer set removes ≥50% of its stage's mean controller
  logit effect.
- R1: none of 100 random six-layer sets matches or exceeds each own-set loss.
- R2: call/answer reader-set Jaccard overlap ≤0.33, and in each stage own-set
  loss exceeds cross-set loss by ≥0.20.
- C0: masking position 102 with the own layer set changes the controller
  effect by no more than 20% in either direction in each stage.

## Verdicts

- all gates: `STAGE_SPECIFIC_POLICY_SLOT_READERS`;
- G0 fails: `POLICY_SLOT_READER_DIAGNOSTIC_INVALID`;
- G0, R0, R1, and C0 pass but R2 fails:
  `POLICY_SLOT_READERS_NOT_STAGE_SPECIFIC`;
- otherwise: `NO_LOCALIZED_POLICY_SLOT_READERS`.

A positive result would support a shared upstream policy slot with distinct
stage-local read pathways. It would not establish a universal agent policy,
cross-template transfer, or a complete attention-head circuit.

## Result

- Controller target accuracy: 100% at call and 87.5% at answer.
- Own-set effect removal: 43.6% at call and 89.1% at answer.
- The two six-layer sets shared four layers (Jaccard 0.50).
- Cross-stage sets were about as disruptive as own-stage sets.
- G0 and C0 passed; R0, R1, and R2 failed.

Answer integration had a strongly localized direct-read pathway, but tool
selection was more distributed and the identified pathways were substantially
shared rather than stage-specific.
