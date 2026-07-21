# Original-context evidence bridge

Status: preregistered frozen discovery, 2026-07-14.

## Question

Mistral selected and executed the database correctly but rejected two correct
zero-valued results in the original multi-turn workflow. Is this merely an
under-strength version of the original prompt's native `lookup` policy state?

This is not a search for a universal trust vector. The candidate coordinate is
learned only from the original agent prompt. Because the edited mode token
precedes the tool result, it cannot encode that result; the causal question is
whether strengthening this upstream policy coordinate is sufficient to alter
later evidence use.

## Frozen setup

- Mistral-7B-Instruct-v0.3, 8-bit, seed 0, 32 layers.
- Original calculator/database agent template and even/odd donor/test split.
- Twenty rows all use database key `D`, whose executed value is zero; payload
  sums vary and therefore conflict with the external result.
- One donor-mean L2 `lookup` minus `calculate` direction at the original mode
  token, alpha 1, with no layer or scale sweep.
- Ten held-out rows.
- Normalized trajectory layers L2/L9/L16/L23/L30 and sole mediator L23.
- Controls: successful clean arbitration direction, norm-matched lexical
  direction, wrong address, and 100 norm-matched random directions.
- An explicit authoritative-result reminder is the natural upper bound. It is
  a deliberately different suffix and is not a matched prompt.

## Eligibility gate

Before any activation extraction:

- calculate-policy conflict uses the internal sum on at least 80% of rows;
- authoritative-result condition uses zero on at least 90%;
- unmodified lookup-policy conflict uses zero on at most 50%, reproducing the
  failure rather than testing already-correct behavior.

Failure yields `MULTITURN_BRIDGE_DIAGNOSTIC_INELIGIBLE` and stops.

## Causal pass

Native amplification must:

- generate zero on at least 80% of held-out rows and improve by at least 50
  percentage points over unmodified lookup;
- produce 0.70--1.30 of the authoritative upper-bound logit effect, with at
  least 80% positive row effects and at most 1/100 random exceedances;
- reproduce the held-out local native prompt displacement (cosine at least
  .80, relative error at most .60);
- converge toward the authoritative final-decision trajectory at L23 (cosine
  at least .80, error at most .60);
- have L23 state patching reproduce 0.70--1.30 of the authoritative patch
  effect, while overwriting amplified L23 with baseline removes at least 70%
  of its output effect;
- beat all three named controls, each below 80% target generation accuracy.

All gates yield `CONTEXT_ALIGNED_MULTITURN_EVIDENCE_RESCUE`. Otherwise the
frozen conclusion is `MULTITURN_EVIDENCE_NOT_SIMPLE_AMPLIFICATION`: the
original failure is not explained by the same native upstream policy
coordinate merely being too weak. This does not establish a global trust
variable. The unmatched authoritative suffix limits trajectory and effect
comparisons to a natural upper-bound interpretation.
