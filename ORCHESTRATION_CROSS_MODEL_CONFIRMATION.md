# Cross-model confirmation of local orchestration control

Status: preregistered frozen confirmation, 2026-07-14.

## Revised hypothesis

The general object is a locally encoded policy-state edit with repeatable
causal function, not one literal vector shared across models or prompt
coordinate systems. Because Qwen and Mistral have different residual spaces,
the Mistral direction is learned from Mistral donor rows; no Qwen vector is
transported.

## Frozen setup

- Model: Mistral-7B-Instruct-v0.3, 32 layers, 8-bit.
- Exact original template-A calculator/database content and 20 payloads.
- Original even-index donor and odd-index held-out split.
- One Mistral donor-mean lookup-minus-calculate direction at the sole changed
  mode token after L2, alpha 1.
- Frozen normalized trajectory checkpoints L2/L9/L16/L23/L30, corresponding
  to Qwen L2/L8/L14/L20/L26; sole mediator L23.
- Original 100 norm-matched random nulls and all original thresholds.
- Token positions may differ by tokenizer but must be unique and uniform; no
  position search is permitted. The teacher-forced decision point is the
  longest token prefix shared by the literal continuations `CALL calculator`
  and `CALL database`; logits compare their first differentiating token. This
  reduces exactly to the original post-`CALL` Qwen measurement.

## Hard behavioral gate

Native calculate and lookup workflows must each reach at least 90% on every
original metric: exact calls, correct row-specific arguments, final answers,
agreement with actual tool execution, and same-row end-to-end success. If G0
fails, stop before activation extraction or intervention. Such an outcome is
behavioral ineligibility, not failed causal replication.

## Confirmation gates

The unchanged Qwen gates apply: local natural-state equivalence, target-tool
logit matching, exact generated target workflows, reverse workflows, natural
L23 trajectory equivalence, matched-state mediation, CLEAN-overwrite
necessity, raw embedding rejection, and at most 1/100 random-null exceedances.

All core gates plus the baseline gate yield
`LATENT_ORCHESTRATION_CONTROLLER`; within this stage that verdict means the
causal function replicated in a second model family. A pass does not imply
literal cross-model vector identity, cross-template transfer, or persistent
control across independent turns.

If it passes, the next experiment is a three-workflow battery with separately
learned local controllers and common structural metrics. If it fails after G0,
inspect the preregistered failed gate and do not rescue layers or alpha.
