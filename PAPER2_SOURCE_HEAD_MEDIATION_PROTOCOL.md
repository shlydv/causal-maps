# Paper 2 discovery: source-to-head causal mediation

Status: frozen before GPU output, 2026-07-24.

## Motivation

The sparse-transport discovery and locked confirmation support a frozen
cross-layer set of late attention-head outputs. The top eight transfer belief
state across agent and object addresses and also support `tell_ac`, but do not
fully recover `search_ac`. This experiment asks whether those heads actually
mediate a causal path from the source anchor, and whether the path boundary is
operation-specific.

The claim is not that attention weights are explanations. The intervention
tests a source-state-to-head-output causal path.

## Frozen model, sites, and tasks

- Qwen2.5-7B-Instruct, validated 8-bit Tesla T4 recipe.
- All 30 tokenizer-compatible structured worlds.
- CLEAN Alice/cube belief: Paris. NATURAL changes only that anchor to Rome.
- Upstream source site: Alice/cube location-token residual after L21, directly
  before the discovered L22-L24 transport set.
- Frozen top eight:
  `L23H11, L24H21, L22H1, L23H6, L22H25, L23H4, L23H13, L24H27`.
- Queries: `belief_ac`, `tell_ac`, and `search_ac`.
- No layer, head, row, prompt, coefficient, or gate is reselected.

## Interventions

For every query and both causal directions:

1. Capture matched CLEAN and NATURAL L21 source-anchor states and frozen head
   outputs at the final readout.
2. Source sufficiency/reversal:
   - patch NATURAL L21 source state into CLEAN;
   - patch CLEAN L21 source state into NATURAL.
3. Selected-path blockade:
   - repeat the forward source patch while clamping all frozen head outputs to
     their matched CLEAN values;
   - repeat the reverse source patch while clamping them to matched NATURAL
     values.

The source intervention passes when its forward and reverse natural-effect
ratios are each in [0.60, 1.40] and both destination accuracies are at least
80%. The frozen heads mediate a direction when clamping removes at least 70%
of the source intervention effect and restores the originating endpoint with
at least 80% accuracy. Both directions must pass.

## Specificity controls

- On `belief_ac`, compare the selected blockade with 39 seeded size-matched
  random eight-site clamps drawn from the original L21-L24 candidate space
  outside the frozen top eight. Report the empirical tail probability
  `(1 + #random >= selected) / 40` for the minimum bidirectional mediation
  fraction and the number satisfying the full mediation gate.
- Patch the matched Bob/cube Paris-to-Rome L21 state at Bob's anchor while
  querying Alice/cube. Alice's answer must remain CLEAN with at least 80%
  accuracy and absolute target-drift ratio at most 0.20.

## Frozen verdicts

- `OPERATION_SPECIFIC_HEAD_MEDIATION`: source interventions pass all three
  queries; selected-head mediation passes `belief_ac` and `tell_ac` but not
  `search_ac`; no random clamp passes; empirical p <= 0.025; and the
  wrong-address control passes.
- `SHARED_HEAD_MEDIATION`: the same conditions hold and selected mediation
  passes all three queries.
- `NONSPECIFIC_MEDIATION`: a random clamp passes or empirical p > 0.025.
- `SOURCE_SITE_INELIGIBLE`: the L21 source intervention fails on any query.
- `MIXED_MEDIATION`: all other eligible outcomes.

An operation-specific positive licenses direct K/V-edge localization,
necessity, predicted-error, and rescue. It does not by itself identify an
individual attention edge.
