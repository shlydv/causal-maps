# Does the Shared Value-Slot State Reach Readout Through Direct Attention?

*Causal Maps · 2026-07-13*  
*Status: **PRE-REGISTERED — one fresh-trial directed-edge transport test.***

## Motivation

The binding causal-state timeline established that NATURAL and ADD are fully
interchangeable at the queried value slot from L2 through L20. The slot then
ceases to be necessary by L26, while the final readout becomes necessary but
remains context-dependent. This suggests communication of a shared slot state
into a distributed answer context.

This test asks one narrow causal question:

> Does the final readout obtain the shared binding state through direct
> attention from the queried value slot in the L20–L26 transition window?

## Fresh confirmation setting

- Qwen2.5-7B-Instruct, 8-bit, seed 0; same donor codebook and two-binding
  prompt form as the completed operator.
- **Fresh source→target mappings only:** offsets 2/4/6/8 (40 rows). Prior
  binding operator, component, subspace, and timeline kernels used offsets
  1/3/5/7 only.
- NATURAL is textual target rewrite; ADD is CLEAN plus `z_target-z_source` at
  L2 queried value slot.
- Fixed all-head attention-edge window: layers 20, 21, 22, 23, 24, 25, 26.

## Interventions

At one fixed layer at a time, block **all query heads** from the final readout
token to the queried value-slot key. The blocked edge is only
`final-readout query → own value-slot key`; it does not alter writers at the
source slot or any other final-token keys.

For every own-slot edge intervention, use a matched control at the exact same
layer/query but with the key changed to the *other variable slot*. This
controls for generic disruption of final-token attention.

Also block the queried-slot edge in the full fixed L20–L26 window
simultaneously. The cumulative intervention distinguishes a redundant,
multi-layer transport path from a single-layer bottleneck.

For each edge condition, rerun CLEAN, NATURAL, and ADD under the **same** edge
mask. Effects are NATURAL/CLEAN and ADD/CLEAN differences under that matched
mask, so generic damage is not treated as binding-specific loss.

## Gates

### G0 — fresh behavioral/operator confirmation

On the fresh rows: CLEAN and NATURAL greedy accuracy ≥80%, ADD positive on
≥80% of rows, and ADD/NATURAL effect ratio in [0.70, 1.30].

### Shared direct transport at a fixed layer

At one fixed layer, own-slot edge block must:

- remove ≥50% of both NATURAL and ADD effects; and
- exceed the matched other-slot loss by ≥25 percentage points for both.

### Shared distributed attention transport

No individual layer qualifies, but the fixed cumulative L20–L26 own-slot block
meets the same two conditions.

## Verdicts

| Verdict | Rule |
|---|---|
| `SHARED_DIRECT_SLOT_ATTENTION_PATH` | G0 and ≥1 individual shared direct layer |
| `SHARED_DISTRIBUTED_SLOT_ATTENTION_PATH` | G0, no individual layer, cumulative block qualifies |
| `DIVERGENT_SLOT_ATTENTION_TRANSPORT` | G0 and the own-vs-control criterion holds for only NATURAL or only ADD |
| `NONDIRECT_OR_UNRESOLVED_SLOT_TRANSPORT` | G0 and neither individual nor cumulative own edge qualifies |
| `SLOT_TRANSPORT_INELICITABLE` | G0 fails |

## Limits

- Passing shows one direct all-head attention route is causally required; it
  does not identify which head carries it.
- Failing does not disprove attention-mediated transport. Information may have
  reached readout earlier, through intermediate tokens, or via redundant paths.
- No post-hoc layer/window/head expansion is licensed by this kernel.
