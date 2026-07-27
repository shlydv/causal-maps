# Paper 2 discovery: bidirectional lexical-route switching

Status: frozen before GPU output, 2026-07-24.

## Hypothesis

A pretrained lexical control state at the answer readout dynamically selects
how a query-independent L21 world state is routed through late attention.
Transplanting that control state should change the route without changing the
stored location.

## Frozen design

Model and worlds are the established Qwen-2.5-7B setup. Use the original
belief question and Alice-cube belief field (`ac`) with the exactly
position-matched contracts:

- epistemic: `BELIEF`;
- search/action: `X X SEARCH`.

The 30 compatible worlds are split deterministically: indices 0-14 are donors
and indices 15-29 are held-out tests.

At the output of block L21 and at the final teacher-forced readout position,
capture paired BELIEF and SEARCH states for both clean and natural donor
histories. Define the frozen control vector as the mean of all 30 paired
differences:

`delta_control = mean(state_BELIEF - state_SEARCH)`.

On held-out worlds:

1. evaluate original BELIEF and SEARCH contexts;
2. add `delta_control` to the SEARCH readout state at block L21;
3. subtract `delta_control` from the BELIEF readout state at block L21;
4. leave the L21 state-marker intervention unchanged;
5. recompute transformed CLEAN/NATURAL baselines, source interventions, and
   cumulative L22-L27 full-attention mediation curves.

The control vector is learned only from donor worlds. No scale, layer, prompt,
or threshold is selected after held-out output.

## Frozen outcomes

Primary continuous outcome is cumulative-L24 minimum bidirectional mediation.
Let:

- `B` be original BELIEF L24 mediation;
- `S` be original SEARCH L24 mediation;
- `S+` be SEARCH after adding the control vector;
- `B-` be BELIEF after subtracting it.

The original gap is `B - S` and must be positive.

- `BIDIRECTIONAL_ROUTE_SWITCH`: all original and transformed cells pass
  behavior and L21 source-sufficiency gates; `S+ - S >= 0.05`;
  `B - B- >= 0.05`; each change closes at least 50% of the original gap; and
  at least one transformed first-pass depth moves in the predicted direction.
- `CONTINUOUS_BIDIRECTIONAL_SWITCH`: the same continuous criteria pass but
  neither categorical first-pass depth moves.
- `ASYMMETRIC_ROUTE_SWITCH`: exactly one direction improves by at least 0.05
  and closes at least 50% of the original gap.
- `NO_CAUSAL_ROUTE_SWITCH`: transformed cells remain eligible but the effect
  criteria fail.
- `ORIGINAL_GAP_ABSENT`: original `B - S < 0.05`.
- `BEHAVIORALLY_INELIGIBLE`: any original or transformed baseline fails.
- `SOURCE_SITE_INELIGIBLE`: baselines pass but any original or transformed
  L21 source intervention fails.
- `DEPTH_UNRESOLVED`: any original or transformed curve fails through L27.

A positive result is a discovery result, not final confirmation. It must be
followed by norm-matched random/sign-flip controls, wrong-label controls,
additional tasks, and another model before a general mechanism claim.

