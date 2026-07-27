# Paper 2 discovery: operation-dependent handoff depth

Status: frozen before GPU output, 2026-07-24.

## Question

The L22-L24 sparse path plus a small shared complement closes belief and tell
mediation but not search. Does search transport the same L21 source state into
the readout at a later depth?

## Frozen design

- Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- All 30 compatible worlds.
- Same matched L21 Alice/cube source-state intervention.
- Queries: `belief_ac`, `tell_ac`, `search_ac`.
- Candidate readout-attention layers: every remaining layer L22-L27.

At the final readout position, clamp all 28 pre-output-projection attention
head slices to their matched originating trajectory. Test both causal
directions for:

1. each layer individually;
2. cumulative prefixes L22 through L;
3. cumulative suffixes L through L27.

Every cell reports effect removal and restoration accuracy using the existing
70%/80% mediation gate. The first passing cumulative prefix is the frozen
handoff-depth statistic. No layer is selected from output.

## Verdict

- `OPERATION_DEPENDENT_HANDOFF_DEPTH`: all three cumulative-prefix curves
  pass, and search first passes at a strictly later layer than both belief and
  tell.
- `SHARED_HANDOFF_DEPTH`: all three pass first at the same layer.
- `SEARCH_ROUTE_OUTSIDE_LATE_READOUT_ATTENTION`: belief and tell pass a
  prefix, but search does not pass even through L27.
- `PARTIAL_DEPTH_ORDERING`: another mixture of eligible prefix outcomes.
- `SOURCE_SITE_INELIGIBLE`: an L21 source intervention fails.

A depth-ordering positive would motivate head/edge localization at the
operation-specific transition layer. Failure through L27 would redirect the
search mechanism to earlier query positions, MLP outputs, or the residual
bypass.
