# Head-level policy-routing circuit

*Causal Maps · 2026-07-13*
*Status: **PRE-REGISTERED — fresh confirmation pending.***

## Motivation

The L2 policy controller affects both tool selection and answer integration.
Layer-level direct-read pathways overlap substantially, but different attention
heads can coexist inside the same layer. This experiment tests whether
stage-specific routing appears at head resolution.

## Frozen setup

- Qwen2.5-7B-Instruct, pinned revision, 8-bit runtime, seed 0.
- Transformers SDPA attention with 28 query heads and 4 key/value heads.
- Template B, policy-label position 103, controller added after L2.
- The controller direction and head discovery use only the original even-index
  donor rows.
- Candidate layers are frozen from the completed layer experiment:
  L4, L5, L7, L10, L11, L15, L17, and L18.
- All 28 query heads in each candidate layer are screened.
- Call and answer stages use the same definitions as the layer experiment.

## Fresh confirmation rows

The ten confirmation rows are frozen below. Every number pair has `a < b` and
therefore did not occur in the original row battery. Every key is paired with a
new payload, and database results differ from calculator sums on all rows.

1. `(1,2,A→1)`
2. `(1,3,C→7)`
3. `(2,3,D→0)`
4. `(1,4,E→3)`
5. `(2,4,G→9)`
6. `(3,4,H→2)`
7. `(1,5,I→5)`
8. `(2,5,J→8)`
9. `(3,5,B→4)`
10. `(1,6,F→6)`

These rows are used only after head selection is frozen.

## Causal head intervention

For one query head at one layer, block attention from every query after the
policy label to the policy-label key. All other heads and edges remain intact.
Every steered run is compared with an identically masked unsteered run.

The position control blocks the same selected heads and query range from key
position 102 instead of the policy key.

## Discovery

1. Measure each of the 224 candidate heads independently on donor rows.
2. Rank by reduction of the controller-induced target-minus-source logit
   effect, separately for call and answer.
3. Freeze the top eight call heads and top eight answer heads.

No head, layer, cardinality, position, coefficient, or threshold search.

## Fresh-row conditions

- unblocked controller;
- own-stage top-eight heads blocked;
- other-stage top-eight heads blocked;
- intersection heads blocked;
- call-unique and answer-unique heads blocked;
- own-stage heads blocking control position 102;
- 100 seeded random head sets, each matching the own set's exact per-layer
  head counts.

## Gates

- G0: native source and target call accuracy ≥90%; native source and target
  answer accuracy ≥80%; unblocked controller target accuracy ≥90% at call and
  ≥80% at answer; donor, fresh, and natural target logit effects are positive.
  The fresh controller effect must be positive on ≥80% of rows and remain
  within 0.70–1.30× the natural target effect. Native source answers use a
  calculator call/result; controller tests use a database call/result.
- H0: each own top-eight set removes ≥50% of its fresh controller effect.
- H1: no matched random set equals or exceeds the own-set loss in either stage.
- H2: call/answer head-set Jaccard overlap ≤0.25 and own-set loss exceeds
  cross-set loss by ≥0.20 in each stage.
- C0: the position-control mask changes controller effect by at most 20% in
  either direction in each stage.

## Verdicts

- all gates: `STAGE_SPECIFIC_POLICY_ROUTING_HEADS`;
- G0 fails: `POLICY_HEAD_ROUTING_DIAGNOSTIC_INVALID`;
- G0, H0, H1, and C0 pass but H2 fails:
  `SHARED_OR_OVERLAPPING_POLICY_ROUTING_HEADS`;
- otherwise: `NO_LOCALIZED_POLICY_ROUTING_HEADS`.

A positive stage-specific verdict would establish head-level routing inside
overlapping layers. A shared verdict would move the stage-specific computation
downstream of a common router. A null would motivate multi-hop relay-token path
patching rather than further direct-edge searches.
