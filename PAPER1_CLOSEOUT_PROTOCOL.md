# Paper 1 experimental closeout

Status: frozen before first run, 2026-07-22. No rescue redesign is permitted
after the tokenizer preflight or full result is observed.

## Purpose

Close the two remaining robustness questions without rerunning completed
batteries or adding models:

1. Is the single edited Alice-cube (`ac`) source position itself sufficient
   under an exact matched-state swap, rather than only a six-anchor set?
2. Does the anchor/checkpoint/readout dissociation survive longer, less
   templated prose and a held-out value transition?

## Frozen model and data

- Official Qwen2.5-14B-Instruct-AWQ on two Tesla T4s.
- Exact single-anchor arm: the original 30 Paris→Rome headline worlds.
- Naturalized arm: 30 distinct Tokyo→Delhi worlds at `ac`, divided by index
  modulo three across three fixed prose styles (case note, witness transcript,
  and curator narrative). This transition was not used by the structured
  headline census.
- Queries: Alice belief and Alice report; Bob belief is the unrelated-state
  specificity control.
- Depths: 2, 4, 8, 12, 16, 20, 24, 26, 32, 36, 41, 46.

## Interventions

### Exact `ac`-only matched swap

At every frozen depth, replace only the edited source token's residual state
between row-matched clean and natural prompts, in both directions. Report
row-level effects, effect ratios, and target/recovery accuracy. Sufficiency
requires both ratios in [0.60, 1.40] and both accuracies at least 0.80.

### Naturalized surface

- Write the held-out Tokyo→Delhi neutral-carrier difference at the naturalized
  `ac` source token at layer 2.
- Require Alice-belief and Alice-report ratios in [0.60, 1.40], target accuracy
  and positive-row fraction at least 0.80, with clean/natural behavior at least
  0.80.
- Require Bob's unrelated cube belief to remain at least 0.80 accurate.
- Compare against 20 seeded norm-matched random directions; exact one-sided
  p-value must be below 0.05.
- At `STATECHECK` and the final readout prefix, perform row-matched swaps in
  both directions over all frozen depths. `STATECHECK` must remain below 0.30
  maximum absolute effect ratio; at least one readout layer must satisfy the
  bidirectional sufficiency gate above.

## Interpretation and stopping

Passing closes Paper 1 experimentation. Failure is reported as a surface or
minimal-support boundary and cannot be replaced by another prompt/model run.
The exact matched-prefix sanity control already passed at every depth in the
frozen locus experiment and is not rerun. No official Llama, larger model, or
additional task family is required for Paper 1 after this closeout.
