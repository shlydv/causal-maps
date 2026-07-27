# Paper 2 discovery pilot: token instances or content-equivalence classes?

Status: frozen before GPU output, 2026-07-24.

## Hypothesis

DeepSeek's repeated-value failures suggest that a synthetic value write may
act on a content-equivalence class rather than only the edited token instance.
This is a candidate causal storage organization, not an assumed mechanism.

The decisive prediction is a collision-load curve. If register A is edited
from source to target:

- with one source-valued register, no other register should change;
- with two source-valued registers, exactly the second matching register
  should change;
- with three source-valued registers, exactly the two other matching registers
  should change;
- registers holding distinct filler values must remain unchanged;
- register D, which already holds the target, must remain correct.

Global target broadcast predicts changes to filler registers as well.
Address-specific storage predicts no changes to matching registers. Generic
confidence or target suppression predicts failure at A or D.

## Frozen design

- DeepSeek-R1-Distill-Llama-8B, Paper-1 8-bit dual-T4 recipe.
- Thirty ordered source→target location transitions at each collision load
  `k ∈ {1,2,3}`.
- Four minimal registers `A`, `B`, `C`, `D`; A always holds source and D always
  holds target. B/C contain source or distinct fillers according to `k`.
- NATURAL changes only A from source to target.
- SYNTHETIC keeps CLEAN text and adds the row-matched neutral-carrier
  target-minus-source direction at A's value token at L2.
- Direct lookup of every register; no layer, coefficient, surface, row, value,
  prompt, or null search.

## Gates

The pilot is eligible when CLEAN lookup is >=80% for all 12 load×register
cells and NATURAL A target accuracy is >=80% at every load. The synthetic A
write must pass the existing target-accuracy, positive-row, and natural-effect
ratio gates.

`CONTENT_EQUIVALENCE_ALIASING` requires the exact preregistered synthetic
pattern above at >=80% in every cell. It is annotated
`BEHAVIORAL_AND_CAUSAL` if NATURAL text shows the same content-linked failures,
or `INTERVENTION_SPECIFIC` if NATURAL preserves addresses.

Other verdicts are `ADDRESS_SPECIFIC`, `GLOBAL_TARGET_BROADCAST`,
`MIXED_COLLISION_EFFECT`, `SYNTHETIC_CONTENT_WRITE_FAILED`, and
`BEHAVIORALLY_INELIGIBLE`.

A positive exact curve licenses locked Qwen/Gemma controls, held-out register
labels/surfaces, minimal-support tests, and causal de-aliasing rescue. Any
other branch stops this hypothesis before those expensive runs.
