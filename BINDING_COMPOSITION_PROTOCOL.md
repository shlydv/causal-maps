# Simultaneous compositional binding-controller test

Version: 2026-07-14-v1. This follows the two-model literal cross-surface
replication. It asks whether two independently specified residual writes can
be installed at two addresses at the same time without preventing either
address from being read.

Use the three-binding mapping prompt `X maps to ...; Y maps to ...; Z maps to
...`, with a direct `X =` or `Y =` completion. Build raw L2 value prototypes
only from original single-binding `Let` donors. In each fresh row, change both
X and Y values, leave Z unchanged, and apply the two corresponding raw
source-to-target deltas simultaneously at the X and Y value slots. Evaluate
the same physical double-write context under an X query and a Y query.

The controls are each binding's own-only write and a swapped-address double
write (X's direction at Y and Y's direction at X). No scale, alignment, layer,
prompt, or value selection is permitted. Qwen runs first; Mistral runs only if
Qwen passes. Confirmation requires behavior accuracy >= .80, positive joint
effects in >= .80 rows, joint/natural and joint/own-only ratios in [.70, 1.30],
mean joint-minus-own-only cross-talk no larger than 15% of natural effect, and
joint effect at least .10 above swapped-address joint effect.
