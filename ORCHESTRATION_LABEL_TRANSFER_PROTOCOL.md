# Gate 1: Does the controller transfer across labels and template?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — CONTROLLER_NOT_REPLICATED.***

## Question

Does the original template-A `calculate`→`lookup` L2 controller transfer,
without refitting, to a structurally different template B whose tool policies
are labeled `red` and `blue`?

## Frozen setup

- Same Qwen7B commit/runtime and 20 payloads.
- Same even-index donor / odd-index test split.
- Template A direction: mean `lookup`−`calculate` L2 mode-state displacement.
- Template B:
  - `red` routes to calculator;
  - `blue` routes to database;
  - reordered packet/policy language;
  - 119 tokens, sole red/blue difference at token 103.
- Inject the A direction at B's red token, α=1.
- Reverse: inject −A direction at B's blue token.
- No fitting or scaling on B test activations.

## B-specific positive reference

Independently learn mean B `blue`−`red` direction on B donor payloads and test
it on the same B test split. This distinguishes failed cross-template transfer
from a template that does not support the controller mechanism at all.

## Per-direction gates

- native red/blue workflows each ≥90%;
- tool-logit natural-effect ratio [.70,1.30], positive on ≥80%, ≤1/100
  norm-matched random exceedances;
- exact target calls and end-to-end workflows ≥80%;
- reverse exact/end-to-end workflows ≥80%;
- L20 teacher-forced `CALL` displacement cosine ≥.80, error ≤.60.

## Verdict

- A direction and B-specific reference pass:
  `CROSS_LABEL_TEMPLATE_TRANSFER`;
- only B-specific reference passes:
  `TEMPLATE_SPECIFIC_CONTROLLER`;
- A transfer passes but the B-specific reference fails:
  `CROSS_TRANSFER_REFERENCE_INVALID` (fail closed);
- neither passes: `CONTROLLER_NOT_REPLICATED`;
- native B behavior fails: `TEMPLATE_TRANSFER_INELICITABLE`.

This is one held-out template, not universal abstraction. A pass unlocks the
multi-workflow battery.

## Result

Native template-B behavior was eligible (calculator 100% end-to-end; database
90%). The frozen template-A direction did not transfer: A/B direction cosine
.021, tool-effect ratio .108, 0% forward/reverse workflow switches, and L20
cosine .079.

The B-specific positive reference switched all calls and reproduced the
natural tool-choice trajectory (ratio .998, 100% calls, reverse 100%, L20
cosine .981), but only 60% of forward runs used the executed database result
in the final answer. It therefore failed the preregistered ≥80% end-to-end
gate.

Verdict: `CONTROLLER_NOT_REPLICATED`. The planned workflow/model expansion is
not unlocked. The supported claim remains a prompt-bound latent controller,
not an abstract orchestration controller.
