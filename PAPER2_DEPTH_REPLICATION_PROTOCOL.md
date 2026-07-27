# Paper 2 locked replication: operation-depth ordering

Status: frozen before GPU output, 2026-07-24.

## Claim under test

On the narrative surface, cumulative late-readout attention mediation first
passed for belief at L24, communication at L26, and search/action at L27. This
replication tests the ordering under a new history surface and new operation
contracts without changing the source intervention, layer grid, or gates.

## Frozen shift

- Model: Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Worlds: the same exhaustive 30 compatible latent worlds.
- History surface: structured private-event ledger rather than narrative.
- New query/command contracts:
  - direct state: “Consult Alice's private record for the cube. State its
    location.” Reply `STATE <location>`.
  - communication: “Alice must communicate her private cube record to a
    teammate. Which location does she communicate?” Reply
    `REPORT <location>`.
  - action: “Alice must act to recover the cube using only her private
    record. Which location does she visit?” Reply `GO <location>`.
- Same matched Alice/cube CLEAN/NATURAL source-state interchange after L21.
- Same cumulative full-readout-attention prefixes L22 through each of
  L22–L27.
- Same 70% effect-removal and 80% originating-endpoint gates.

No prompt, command, layer, row, coefficient, or gate may be changed after
output. Each query must pass CLEAN and NATURAL behavior and the L21 source
intervention before contributing to the ordering.

## Verdict

- `DEPTH_ORDERING_REPLICATED`: all three prefix curves pass and the first
  passing depths satisfy direct state < communication < action.
- `MONOTONE_WITH_TIE`: all pass and direct state <= communication <= action,
  with at least one tie.
- `ORDERING_NOT_REPLICATED`: all are eligible but the ordering is reversed,
  mixed, or incomplete.
- `BEHAVIORALLY_INELIGIBLE`: a CLEAN/NATURAL baseline fails.
- `SOURCE_SITE_INELIGIBLE`: a baseline passes but an L21 source intervention
  fails.

This confirms an operation-depth relationship only within Qwen2.5-7B. A
second-model or second-scale confirmation remains required for a general
architectural claim.
