# Paper 2 locked confirmation: sparse late attention transport

Status: frozen before confirmation GPU output, 2026-07-24.

## Claim under test

The discovery pilot selected four Qwen2.5-7B attention-output sites using only
`belief_ac` discovery rows:

`L23H11, L24H21, L22H1, L23H6`.

This confirmation does not re-rank or replace them. It asks whether the frozen
set transports the same Alice/cube counterfactual under two held-out query
contracts that were never used for head selection: `tell_ac` and `search_ac`.
Passing would establish a reusable causal transport set across consequence
readouts. It would not yet identify the source edges read by those heads.

## Frozen tests

- Model: Qwen2.5-7B-Instruct, validated 8-bit T4 recipe.
- Worlds: all 30 tokenizer-compatible structured worlds.
- Counterfactual: Alice/cube changes from Paris to Rome.
- Confirmation tasks: `tell_ac` and `search_ac`.
- Frozen nested sets:
  - top four: `L23H11, L24H21, L22H1, L23H6`;
  - top eight adds `L22H25, L23H4, L23H13, L24H27`.
- Directional tests:
  - NATURAL head slices patched into CLEAN;
  - CLEAN head slices patched into NATURAL.
- A cell passes when both effect ratios are in [0.60, 1.40] and both endpoint
  accuracies are at least 80%, matching the discovery protocol.

For `tell_ac`, compare the frozen top-four set with 99 seeded, size-matched
random sets drawn without replacement from the 104 sites outside the frozen
top eight. The frozen scalar statistic is
`min(forward_ratio, reverse_ratio)`. Report the finite-sample empirical tail
probability `(1 + #random >= observed) / 100`, random pass count, and maximum
random statistic.

Patch the selected sites grouped by layer as mechanistic controls:
`L22H1`, `L23H11+L23H6`, and `L24H21`. These are not substitutes for the
cross-layer set and do not affect the verdict.

## Exploratory generalization

Without changing the frozen heads, probe `belief_as` after an Alice/sphere
counterfactual and `belief_bc` after a Bob/cube counterfactual. These address
transfers are reported behind their own behavioral eligibility gates and do
not affect the confirmatory verdict. Success would support shared transport;
failure would motivate address-specific head-set mapping.

## Verdict

- `LOCKED_SPARSE_TRANSPORT_CONFIRMED`: both held-out query contracts are
  behaviorally eligible, the frozen top-four set passes bidirectionally on
  both, no random `tell_ac` set passes, and the empirical tail probability is
  at most 0.01.
- `QUERY_LIMITED_TRANSPORT`: the behavior gates pass but the frozen set fails
  one or both held-out query contracts.
- `NONSPECIFIC_TRANSPORT`: a matched random set passes or the empirical tail
  probability exceeds 0.01.
- `BEHAVIORALLY_INELIGIBLE`: either held-out query contract fails its clean or
  natural 80% baseline gate.

No result from the exploratory address probes can upgrade the verdict.
