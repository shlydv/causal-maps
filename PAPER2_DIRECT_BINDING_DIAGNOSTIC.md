# Paper 2 diagnostic: is DeepSeek spillover intervention-specific?

Status: frozen before GPU output, 2026-07-23. This is a kill test, not a paper
claim.

## Motivation

The first Paper 2 pilot showed that DeepSeek's structured belief world is not
address-correct under the textual counterfactual itself. Therefore that world
cannot distinguish a synthetic-edit routing failure from an existing binding
capability failure.

This diagnostic removes belief inference while preserving the exact collision
structure: three addresses hold the source value and a fourth holds the target
value. It asks whether DeepSeek can bind direct textual records correctly and,
conditional on that ability, whether the same validated synthetic content
write uniquely destroys address specificity.

## Frozen design

- DeepSeek-R1-Distill-Llama-8B, 8-bit, dual Tesla T4.
- Thirty distinct ordered source→target location pairs.
- Four direct records: Alice/cube, Bob/cube, Alice/sphere all hold the source;
  Bob/sphere holds the target.
- NATURAL changes only Alice/cube from source to target.
- SYNTHETIC leaves text unchanged and adds the row-matched neutral-carrier
  target-minus-source direction at Alice/cube's value token at L2.
- Evaluate all four direct lookups.
- Wrong-address positive control applies the same write at Bob/cube and must
  change Bob/cube while preserving Alice/cube.
- No layer, coefficient, prompt, value, row, or null search.

## Gates and verdicts

Behavioral eligibility requires >=80% CLEAN and NATURAL accuracy for all four
lookups. Conditional on eligibility:

- intended Alice/cube synthetic target accuracy >=80%, positive-row fraction
  >=80%, and effect ratio versus NATURAL in [0.60, 1.40];
- Bob/cube and Alice/sphere must remain at the source >=80%;
- Bob/sphere must retain the legitimate target >=80%;
- wrong-address write must pass the same effect gates on Bob/cube and preserve
  Alice/cube >=80%.

Verdicts:

- `DIRECT_BINDING_SPECIFIC`: all gates pass. The structured-world anomaly is a
  task/capability boundary, not an established synthetic-only routing failure.
- `INTERVENTION_SPECIFIC_DIVERGENCE`: textual behavior is fully eligible and
  the intended write works, but an invariant address fails. This licenses a
  natural-minus-synthetic causal-component study on this direct task.
- `SYNTHETIC_CONTENT_WRITE_FAILED`: behavior is eligible but the intended
  synthetic write fails.
- `BEHAVIORALLY_INELIGIBLE`: textual direct binding itself fails.
