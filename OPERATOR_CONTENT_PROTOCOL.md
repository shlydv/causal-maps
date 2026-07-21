# Protocol: Does the affine operator select its intended value?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR`.***

## Motivation

`delta_operator` passed all frozen state, trajectory, role, and shared-mediator
gates. However, a wrong-target operator still increased the frozen
target-vs-source logit contrast by about half the correct effect. That scalar
metric can improve merely because both operators suppress the source value.

This control asks whether correct and wrong operators each reproduce the
**multiclass output state of their own natural textual counterfactual**.

## Frozen setup

Reuse exactly:

- Qwen2.5-7B-Instruct;
- ten values;
- four-context prototype codebook;
- forty balanced two-binding trials;
- L2 queried-slot operators \(d_{a\rightarrow b}=z_b-z_a\).

For every trial compare:

1. CLEAN source \(a\);
2. natural target rewrite \(a\rightarrow b\);
3. ADD target operator \(d_{a\rightarrow b}\);
4. natural wrong rewrite \(a\rightarrow w\);
5. ADD wrong operator \(d_{a\rightarrow w}\).

No layer, α, value, or template change.

## Measurements

Read the full ten-value candidate-logit vector and global greedy token.

For target and wrong conditions separately:

- global greedy intended-value accuracy;
- cosine and normalized error between the centered ten-logit displacement
  caused by ADD and its matching natural textual rewrite;
- Jensen–Shannon divergence between candidate distributions.

Also report:

- intended-target logit gain;
- source-logit suppression;
- the original target gain under the wrong operator;
- fraction where the intended value beats both source and alternate target.

All per-trial values must be saved.

## Gates

### G0 — natural outputs

Natural target and natural wrong rewrites each greedily emit their intended
value on ≥90% of trials.

### C1 — operator content accuracy

ADD target emits \(b\) and ADD wrong emits \(w\), each on ≥90% of trials.

### C2 — natural output-state equivalence

For both target and wrong operators:

- mean centered ten-logit displacement cosine ≥0.95;
- mean normalized error ≤0.25.

### C3 — condition discrimination

- under target ADD, \(logit(b)>\max(logit(a),logit(w))\) on ≥90%;
- under wrong ADD, \(logit(w)>\max(logit(a),logit(b))\) on ≥90%.

## Verdict

- `CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR` iff G0∧C1∧C2∧C3.
- Otherwise `OUTPUT_EQUIVALENCE_FAILED`.

Only the positive verdict licenses the cross-skill prompt-reachability atlas.
No rescue run.

## Result

All frozen gates passed on 40 trials:

- natural target accuracy 100%; natural wrong accuracy 95%;
- ADD target accuracy 100%; ADD wrong accuracy 95%;
- target ADD vs natural target logit displacement: cosine 0.99982,
  normalized error 0.0193;
- wrong ADD vs natural wrong logit displacement: cosine 0.99979,
  normalized error 0.0212;
- target and wrong discrimination each 100%;
- wrong operator gain on the original target was only +0.12, while its
  intended-value gain was +37.70.

The earlier +37.81 wrong-operator binary contrast was source suppression
(mean 37.69), not accidental selection of the original target.

**Verdict: `CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR`.**
