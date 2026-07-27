# Paper 2 heterogeneous-family eligibility screen

Status: v1.1 correction of a signed-gap implementation error in v1  
Date: 2026-07-26  
Stage: `delta_heterogeneous_family_screen`

Frozen protocol SHA-256:
`1A7E511A42D104D91A1414E95E50F4DFCCA9F41BB4CA3BA5C3F4CF5215816A62`

## Purpose

The previous prospective experiment used near-isomorphic private-record
prompts. All four controllers collapsed into PC1, so it could not distinguish
a universal control channel from template similarity.

This screen identifies structurally different computations that satisfy the
existing behavioral, source, and route-gap gates before another causal
experiment is attempted. It is a selection screen, not evidence for the final
mechanistic claim.

## Frozen families

All families use the same eight color answer tokens, the same 15 unique
source-target pairs, and the same `BELIEF` versus `X X SEARCH` output contract.
Only computational structure changes:

1. private belief after an unobserved change;
2. chronological last-write state update;
3. exact key-value lookup;
4. two-hop pointer traversal;
5. conditional branch selection;
6. maximum-score comparison;
7. constraint elimination;
8. temporal-slot retrieval.

Each prompt contains the counterfactual answer anchor exactly once. Changing
that token constructs an aligned counterfactual history without altering
prompt length. Prompts, family order, values, rows, layer 21 source site, and
layer 24 route assay are frozen.

## Measurements

For untouched BELIEF and SEARCH in every family:

- clean and counterfactual answer accuracy;
- bidirectional layer-21 source-intervention sufficiency;
- layer-24 mediation route score;
- signed BELIEF-minus-SEARCH route gap and its absolute magnitude;
- token and answer-prefix alignment.

No answer-prefix steering, SVD, axis selection, or prompt revision occurs.

## Eligibility and selection

A family passes only when:

- clean and counterfactual answer accuracy are each at least 80%;
- source intervention is sufficient in both operations;
- the absolute magnitude of the original layer-24 route gap is at least
  `0.03`.

The sign indicates which operation has the larger route score; it is not an
eligibility requirement. The first implementation accidentally compared the
signed difference with `+0.03`, falsely rejecting strong contrasts with the
opposite orientation. Version 1.1 corrects only that classifier. The raw route
measurements from the original screen remain valid and can be re-scored
without another model run.

Every failure and sub-gate is reported. The downstream experiment receives the
first four passing families in the frozen order. A ready verdict requires at
least four passing families. Prompts will not be edited after seeing this
screen.

## Interpretation

- `HETEROGENEOUS_FAMILY_SET_READY`: at least four families are safe to use in
  the prospective breakthrough test.
- `INSUFFICIENT_HETEROGENEOUS_FAMILIES`: the route contrast is not stable
  enough across distinct computations; do not run the expensive test.
- `TOKENIZATION_OR_ALIGNMENT_FAILURE`: at least one frozen family could not be
  compared under the exact-position contract.

## Runtime

The screen performs eight original-arm route assays, approximately 96 model
forwards total. Expected Tesla-T4 runtime is 10--20 minutes after model
loading.
