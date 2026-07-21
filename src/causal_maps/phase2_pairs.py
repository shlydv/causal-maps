"""Phase 2 pair libraries (computational variables).

Skills (Sahil redesign 2026-07-12; Instruction dropped after hand-10 fail):
  1. Completion state   — one boolean bit (+ implicit family ablation)
  2. Variable substitution — one bound value
"""
from . import completion_pairs, variable_pairs

make_completion_pairs = completion_pairs.make_completion_pairs
make_variable_pairs = variable_pairs.make_variable_pairs

COMPLETION_HAND10 = completion_pairs.HAND10
VARIABLE_HAND10 = variable_pairs.HAND10
FAMILY_IDS = completion_pairs.FAMILY_IDS
