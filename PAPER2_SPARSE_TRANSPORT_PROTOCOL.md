# Paper 2 discovery pilot: sparse late attention transport

Status: frozen before GPU output, 2026-07-24.

## Question

Paper 1 established a robust temporal handoff: source anchors are causally
sufficient early/mid-depth, the query-independent checkpoint is inert, and a
late query-specific readout becomes sufficient. This pilot asks a new
component-level question:

> Is that late handoff carried by a sparse, reproducible set of attention-head
> outputs at the readout position?

Success identifies a causal transport bottleneck. It does not yet prove which
source edge each head reads; edge necessity/rescue is the licensed follow-up.

## Frozen model and split

- Qwen2.5-7B-Instruct, the validated 8-bit T4 recipe.
- The same 30 tokenizer-compatible structured worlds and `belief_ac` query.
- First 15 worlds are discovery rows; final 15 are untouched evaluation rows.
- CLEAN has Alice/cube belief Paris; NATURAL changes only that source anchor
  to Rome.
- Candidate layers L21, L22, L23, L24, spanning the observed readout handoff
  (near-zero at L21 and 0.828× with 100% bidirectional accuracy at L24).

## Frozen discovery and confirmation

At each candidate layer, intercept the input to the attention output
projection at the final readout position and treat each concatenated
attention-head slice as one site.

1. On discovery rows only, patch one matched NATURAL head slice into CLEAN and
   rank all layer×head sites by the forward target-logit effect.
2. Freeze nested top-K sets for K in {1,2,4,8}.
3. On evaluation rows, test both NATURAL→CLEAN sufficiency and CLEAN→NATURAL
   reverse recovery for each top-K set.
4. Compare with five seeded, size-matched random site sets at every K, drawn
   without replacement from the sites outside the frozen top-8 set.
5. Report full-layer attention-output patches as intervention upper bounds.

No head, layer, K, coefficient, row, or prompt is chosen from evaluation
output.

## Gates and verdicts

Behavioral eligibility requires CLEAN and NATURAL target accuracy >=80%.
A set is bidirectionally sufficient when forward and reverse effects are each
in [0.60,1.40] of the natural effect and both directional accuracies are
>=80%.

- `SPARSE_TRANSPORT_PATH`: a top-K set with K<=8 is bidirectionally sufficient
  and no matched random set at that K is sufficient.
- `DISTRIBUTED_ATTENTION_TRANSPORT`: a full-layer attention patch is
  sufficient but no top-K set passes.
- `ATTENTION_OUTPUT_NOT_SUFFICIENT`: no full-layer attention patch passes,
  despite the previously validated full residual-state readout upper bound.
- `NONSPECIFIC_HEAD_SET`: a top-K and at least one matched random set pass.
- `BEHAVIORALLY_INELIGIBLE`: the held-out behavior gate fails.

A sparse positive licenses source-edge ablation, predicted error patterns,
rescue, held-out query/address generalization, and Qwen-14B confirmation.
