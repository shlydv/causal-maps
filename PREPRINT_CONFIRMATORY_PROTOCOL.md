# Token-Anchored State: confirmatory protocol

Status: frozen before the first `delta_preprint_battery_v2` launch, 2026-07-15.

## Claim hierarchy

The confirmatory runs test three claims separately. Failure of a higher tier
does not erase a lower-tier result.

1. **Writable value state.** A neutral-carrier residual difference written at
   the value token produces the same downstream change as a textual value
   substitution.
2. **Addressed world-state editing.** The same write changes the edited
   entity's direct and two-hop consequences, while preserving unrelated
   entities and facts.
3. **Anchor/checkpoint dissociation.** In the structured belief world, the
   textual anchor is causally writable, while a designated query-independent
   `STATECHECK` token carries at most 0.30 of the matched natural effect.

The paper will not infer that no distributed state exists. The negative claim
is restricted to the tested token positions, layers, models, and prompt family.

## Confirmatory units

- Workspace matrix: 30 distinct ordered value transitions per cell and three
  independently generated row sets (seeds 0, 1, 2).
- Entity matrix: 30 unique worlds per family and three independently generated
  row sets. Repeated worlds are forbidden.
- Anchor write: one exhaustive 30-world test set when the tokenizer preserves
  one aligned intervention position. If a model segments otherwise, use the
  largest mechanically defined equal-length/equal-anchor bucket from those 30
  worlds and report its size (never hand-select rows). Repeating that fixed
  bucket under three random seeds would not constitute data replication.
- Random-direction control: 50 draws for workspace cells, 30 for entity cells,
  and 99 for the single anchor census. Random generators are explicitly seeded.
- Checkpoint: the same 30 structured worlds, all pre-registered depth points,
  and three sites: `STATECHECK` (primary), final question token, and final
  readout-prefix token (trajectory controls). The checkpoint verdict depends
  only on `STATECHECK`; later sites cannot rescue or invalidate it.

## Primary endpoints

- Workspace: compute-cell natural-effect ratio, target accuracy, wrong-value
  accuracy, and random-direction p-value. The family-level endpoint requires
  retrieval plus at least three of four compute cells.
- Entity: the city two-hop cell is primary. Retrieval is an instrument check;
  other-entity shift is the specificity control. The keys family is a
  pre-registered conceptual replication.
- Anchor: `belief_ac` natural-effect ratio and target accuracy are primary.
  `tell_ac`, four invariants, wrong-address editing, and random directions are
  controls.
- Checkpoint: maximum absolute `STATECHECK` effect ratio over the full layer
  sweep. A value below 0.30 is the frozen inertness criterion, conditional on
  clean and natural behavioral accuracy of at least 0.80.

All row-level natural and intervention effects are retained. Confidence
intervals are computed from rows, not from random-direction draws. Exact
randomization p-values use the +1 correction. No failed behavioral gate is
interpreted as mechanistic evidence.

## Model plan

1. Qwen2.5-7B-Instruct: full battery, instrument and cross-scale bridge.
2. Mistral-7B-Instruct: identical full battery, independent-family test.
3. Qwen2.5-14B-Instruct-AWQ: anchor/checkpoint/probe emphasis, reusing one load.
4. Llama-3.1-8B-Instruct or Gemma-2-9B-Instruct: one frozen independent-family
   confirmation after a tokenizer-only preflight. Ineligible cells are
   reported rather than redesigned post hoc.

## Analysis and stopping

The original small runs are discovery pilots. Confirmatory results are never
pooled with them. We stop adding task families after the model plan above. A
new task or altered prompt is exploratory and cannot replace a failed frozen
cell in the preprint's headline table.
