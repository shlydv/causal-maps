# Screen: Which nontrivial reasoning control is behaviorally eligible?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `ELIGIBLE_ARITHMETIC_PROGRAM`.***

## Purpose

The graph controller kernel stopped at G0 because Qwen7B did not solve the
native task. Before another causal protocol, screen fixed task families using
native behavior only. No activations are extracted or inspected.

## Frozen families

1. **Kinship depth (priority):** mode one asks for a named person's parent;
   mode two asks for their grandparent. Each example contains a two-link parent
   chain plus a distractor fact. Names and fact order vary.
2. **Arithmetic program (fallback):** mode add computes \(a+b\); mode subtract
   computes \(a-b\), with held-out operand pairs and exact integer answers.

Both families use equal-length mode pairs differing at exactly one token.
Answers are validated in exact continuation context.

## Data

- Qwen2.5-7B-Instruct, 8-bit, seed 0.
- 20 examples per family.
- Kinship middle and grandparent answers balanced over A–J.
- Arithmetic pairs are fixed and disjoint.
- Save row-level greedy predictions.

## Eligibility and selection

A family is eligible iff greedy accuracy is ≥90% in both modes.

- Select kinship if eligible.
- Otherwise select arithmetic if eligible.
- Otherwise return `NO_REASONING_TASK_ELIGIBLE`.

Selection uses behavior only. The chosen task receives a new preregistered
causal kernel; this screen cannot support a causal claim.

## Result

- Kinship depth: 40% mode-one / 55% mode-two; ineligible.
- Arithmetic program: 100% add / 100% subtract; eligible.

Per the frozen priority rule, arithmetic was selected because kinship failed.
Verdict: **`ELIGIBLE_ARITHMETIC_PROGRAM`**.

No activations were extracted during selection.
