# Causal evidence arbitration after tool execution

Status: v2 preregistered Mistral discovery, 2026-07-14. V1 was behaviorally
ineligible and ran no intervention.

## Question

What post-tool computation determines whether an executed external result
overrides the model's internally available arithmetic answer? The experiment
first tests one low-dimensional addressed-state hypothesis, but a failure may
indicate distributed or context-dependent arbitration rather than absence of
such a computation.

## Frozen design

- Mistral-7B-Instruct-v0.3, 32 layers, 8-bit.
- The original 20 calculator/database payloads and even-donor/odd-test split.
- One clean user turn states the payload, the model's internal arithmetic
  result, and the successfully executed external database result. It contains
  no tool-call-generation instruction.
- The original mode token is the sole changed policy address:
  `calculate` means use the internal payload sum; `lookup` means use the
  executed database result.
- Both rules are stated symmetrically and the active mode value appears once.
  The sole source/target token difference must remain unique and uniform.
- Donor mean lookup-minus-calculate state at L2, alpha 1.
- Frozen layers L2/L9/L16/L23/L30 and sole mediator L23.

## Gates

1. Before activation extraction, donor and held-out native internal/external
   conditions must each answer at least 87.5% of non-collision conflicts
   correctly. Otherwise stop.
2. On held-out internal-policy conflicts, the learned edit must reach at least
   87.5% external-result use, improve by 25 points, reproduce 0.70--1.30 of the
   natural output effect, be positive on 75%, and exceed at most 1/100
   norm-matched random directions.
3. The reverse edit must restore internal answers on at least 87.5%.
4. L23 answer-decision state must match the natural external-policy trajectory
   (cosine >=.80, relative error <=.60). Matched L23 patching and CLEAN
   overwrite must pass the original mediation and necessity thresholds.
5. Norm-matched raw lexical and wrong-address controls must each fail the
   target behavioral gate.

## Untouched-failure bridge

Only after the primary held-out test, add the learned direction at the literal
mode-label address in the original multi-turn lookup transcripts. Report
whether it rescues the two
previous `D -> 0` failures. This is supplemental because the failure rows are
already known; it cannot make the primary verdict pass.

## Interpretation

All gates yield `CAUSAL_EVIDENCE_ARBITRATION_STATE`: a locally encoded policy
state causally controls whether external evidence or an internal answer wins,
and recruits the natural downstream answer pathway. This does not prove a
universal trust vector. A behaviorally eligible causal null points toward a
distributed arbitration computation and ends affine-vector searches here.
