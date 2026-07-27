# Paper 2 confirmation: causal answer-prefix transfer to unseen synonyms

Status: frozen before GPU output, 2026-07-25.

## Question

Does the answer-prefix mechanism transfer beyond the exact `BELIEF` and
`SEARCH` prompts on which it was discovered?

## Frozen design

Use Qwen-2.5-7B-Instruct in 8-bit precision, the original belief question,
`ac` field, narrative surface, and compatible worlds 15-29. Reuse the
tokenizer-only position-matching plan frozen in the lexical-class experiment:

- anchor donors: `BELIEF` and `X X SEARCH`;
- held-out epistemic recipients: `X X THINK`, `X X KNOW`;
- held-out search recipients: `X X FIND`, `X X LOOK`.

No synonym state is used to construct or fit a controller. For each held-out
world and clean/natural arm, capture complete L21 states from the two anchor
donors. Patch exactly the final three answer-prefix positions:

- SEARCH-anchor states into THINK and KNOW;
- BELIEF-anchor states into FIND and LOOK.

This is paired within world and arm. The visible recipient prompt, answer
tokens, question, world history, state marker, and every non-prefix state
remain unchanged.

## Controls

1. Within-class anchor patches: BELIEF into THINK/KNOW and SEARCH into
   FIND/LOOK.
2. Instruction-window patches: apply the opposite anchor at a tokenizer-only
   three-position window ending at the differing instruction-label token,
   while leaving the answer prefix untouched.
3. Nineteen fixed-seed sets of three token-identical post-marker positions,
   excluding the state marker, answer-prefix positions, and instruction
   windows. Apply the same opposite-anchor patches to all four recipients.

Random seed: 27191. Random sampling observes token identity and position only.

## Measurements and gates

For each original and cross-class recipient, recompute:

- clean/natural behavior;
- bidirectional L21 state-marker sufficiency;
- cumulative full-readout-attention mediation at L22-L27;
- L24 minimum mediation and first-passing checkpoint.

Within-class, instruction, and random controls require the same behavior and
source gates and are evaluated at cumulative L24.

Define signed transfer movements:

- epistemic movement: original minus SEARCH-anchor-patched L24 mediation;
- search movement: BELIEF-anchor-patched minus original L24 mediation.

The primary score is the smaller of the mean epistemic and mean search
movements.

## Frozen outcomes

First require:

- every behavior and source gate passes;
- all original and cross-class curves resolve by L27;
- original held-out epistemic mean exceeds search mean by at least 0.03.

`CROSS_SYNONYM_ROUTE_TRANSFER` requires:

- mean epistemic and search movement are each at least 0.05;
- all four individual movements have the predicted sign;
- at least one epistemic recipient moves to a later first-passing checkpoint
  and at least one search recipient moves earlier;
- zero of 19 matched random sets reaches the primary score (add-one empirical
  `p = 0.05`).

`CONTINUOUS_CROSS_SYNONYM_TRANSFER` meets the continuous and specificity
criteria but not both categorical criteria.

`PARTIAL_CROSS_SYNONYM_TRANSFER` has exactly one class mean reach 0.05.
`NONSPECIFIC_SYNONYM_TRANSFER` meets both class movement criteria but fails
the random-position test. `NO_CROSS_SYNONYM_TRANSFER` meets neither.
Eligibility, source, depth, alignment, and original-gap failures are reported
directly.

A positive result shows that anchor-derived answer-prefix states causally
reconfigure unseen lexical contexts. It does not yet establish a compact or
domain-general controller; those require low-rank and cross-domain transfer.
