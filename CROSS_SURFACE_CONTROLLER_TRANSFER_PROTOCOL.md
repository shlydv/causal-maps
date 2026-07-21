# Literal cross-surface controller-transfer test

Version: 2026-07-14-v1. This is a strict follow-up to the two-model mapping
surface replication. It asks whether a controller is reusable across surface
forms, rather than merely whether the same phenomenon can be re-derived in
each form.

The source family is the original direct grammar:

`Let X = VALUE. What is the value of X?` followed by `X =`.

The target family is the behaviorally validated mapping v2 grammar:

`In the table, X maps to VALUE_X; Y maps to VALUE_Y. What does X map to?`
followed by `X =` (or its Y counterpart).

At fixed L2, build one prototype per value using only source-family donor
prompts with names X/Y/Z/W. On the mapping family held-out offsets 5/7, add
the raw `prototype_let(target) - prototype_let(source)` at the queried value
slot. There is no scale fitting, whitening, rotation, alignment, layer choice,
or target-family direction selection. The pre-existing mapping-native direction
is evaluated concurrently only as a sanity baseline; it cannot modify the
literal transfer direction.

The controls are the literal source-family wrong-value direction at the queried
slot and the literal target direction at the other binding slot. Values are
restricted to model-tokenizer single-token values; fewer than eight values is
ineligible.

The Qwen2.5-7B-Instruct run happens first. Mistral-7B-Instruct-v0.3 is run
only if Qwen meets all confirmation gates. Confirmation requires: mapping
baseline behavior (CLEAN/NATURAL accuracy >= .80 and native ADD/NATURAL ratio
in [.70, 1.30]); positive literal transfer in >= .80 of rows; literal
ADD/NATURAL and literal/native-ADD ratios each in [.70, 1.30]; and literal ADD
at least .10 above both controls. A failure is evidence against raw literal
controller reuse under this measurement, not evidence against the previously
confirmed per-surface operator phenomenon.
