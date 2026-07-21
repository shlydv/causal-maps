# Mistral binding-backup causal port

Version: 2026-07-14-v1. This is one constrained cross-model causal test,
conditioned on the completed Mistral affine-operator replication.

Model: `mistralai/Mistral-7B-Instruct-v0.3`, 32 layers, 8-bit. Use the same
two-binding prompt, tokenizer-valid value set, and held-out offsets 5/7 as the
Mistral operator gate. The pre-established Mistral L2 ADD is
`prototype(target)-prototype(source)`.

The Qwen index geometry is mapped once from a 28-layer model's final index 27
to Mistral's final index 31: post-layer-L20 overwrite maps to L23; Qwen
post-overwrite reads L21-L26 map to L24-L30; Qwen early formation L3-L8 maps
to L3-L9. These are the only causal windows tested.

First verify the direct Mistral mask baseline matches the unmasked NATURAL and
ADD effect within 5%. Then replace the queried slot's post-L23 residual state
with its matched CLEAN state (`P`). Block later-query attention to that slot
over L24-L30 (`L`). The port is ineligible unless `E(P+L)-E(P) >= 5` for both
NATURAL and ADD.

The single formation test has four matched conditions: `P+E(own)`,
`P+L+E(own)`, `P+E(other)`, and `P+L+E(other)`, where E is L3-L9. It replicates
the early backup formation only if both NATURAL and ADD have control recovery
>=5, nonnegative own recovery, and prevent at least 50% of controlled recovery
in the difference-in-differences. A failure does not authorize a later-window,
head, or receiver search.

This evaluates whether the Qwen recoverable-backup phenotype travels to
Mistral. It cannot demonstrate shared heads, a shared circuit, or a literal
memory copy.
