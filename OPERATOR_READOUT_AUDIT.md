# Audit: Is the affine operator target-specific?

*Causal Maps · 2026-07-13*
*Status: **SUPERSEDED BEFORE RUN** by the stricter
`OPERATOR_CONTENT_PROTOCOL.md` (90% gates + natural-wrong logit-vector
equivalence).*

`delta_operator` passed every frozen state, trajectory, role, and shared
mediation gate. One diagnostic was concerning: a WRONG target operator still
raised the frozen `logit(target) − logit(source)` contrast by +37.81 versus
+75.78 for the correct operator.

That contrast is not target-specific. Replacing source \(a\) with any other
value can suppress `logit(a)`, thereby raising `logit(b)-logit(a)` even when the
model actually selects wrong value \(w\).

This audit changes no prior gate. It determines whether the output claim is a
true target-specific rewrite or only source suppression.

## Frozen rerun

- Identical Qwen7B model, donor prototypes, 40 balanced multi-binding trials,
  L2 sites, and deterministic correct/wrong operators from
  `COUNTERFACTUAL_OPERATOR_PROTOCOL.md`.
- Conditions only: CLEAN, natural CF, correct ADD \(z_b-z_a\), wrong ADD
  \(z_w-z_a\).
- No random nulls, trajectory recapture, layer/α search, or new prompts.

## Metrics

For each condition save:

1. full-vocabulary greedy token;
2. greedy accuracy for its intended value;
3. margin of intended value over the strongest of the other nine value tokens;
4. for WRONG ADD, accuracy/margin for both \(w\) and the original target \(b\).

## Gates

- **T1 correct target:** correct ADD emits \(b\) greedily on ≥80% of trials and
  its mean ten-value margin is ≥70% of the natural CF margin.
- **T2 wrong target:** wrong ADD emits \(w\) greedily on ≥80% of trials, emits
  \(b\) on ≤20%, and has positive mean margin for \(w\).
- **T3 separation:** correct ADD’s mean margin for \(b\) exceeds WRONG ADD’s
  mean margin for \(b\).

## Verdict

- `TARGET_SPECIFIC_OPERATOR` iff T1∧T2∧T3.
- Otherwise `NONSPECIFIC_REPLACEMENT`; retain state-equivalence findings but do
  not elevate the affine operator as a content-specific causal rewrite.

Only `TARGET_SPECIFIC_OPERATOR` clears cross-skill/model scaling.
