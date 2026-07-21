# Agent-workspace context eligibility screen

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — WORKSPACE_CONTEXTS_INELICITABLE.***

## Purpose

Qualify six fixed prompt contexts for a later causal test of whether
context-specific tool instructions become a shared downstream action
representation.

This is not a claim that Jacobians or context-aware steering are new.
Anthropic's J-lens/J-space work already establishes verbalizable global
workspace representations and cross-context concept swaps. Steering Vector
Fields already adapts steering directions to hidden-state context. Our
distinct target is the causal entry and control of structured, emulated tool
workflows: tool identity, variable arguments, parsed-call execution, and use
of the returned result. This screen itself makes no causal novelty claim.

## Frozen contexts

All contexts express the same calculator-versus-database workflow but differ
in syntax, ordering, and active labels:

- A: `calculate` / `lookup`;
- B: `red` / `blue`;
- C: `north` / `south`;
- D: `left` / `right`;
- E: `alpha` / `beta`;
- F: `open` / `close`.

Each pair is tokenizer-aligned: equal length and exactly one changed token.
The existing 20 payloads are reused without modification.

## Eligibility

For both modes in every context, require aggregate accuracy over 20 rows:

- exact tool call ≥90%;
- correct action and arguments ≥90%;
- final answer ≥90%;
- answer equals actual tool result ≥90%;
- end-to-end workflow ≥90%.

All six contexts must pass. There is no template replacement or repair.

## Frozen runtime and provenance

- Qwen2.5-7B-Instruct revision
  `a09a35458c702b33eeacc393d103063234e8bc28`;
- Torch `2.10.0+cu128`, Transformers `5.0.0`, bitsandbytes `0.49.2`;
- 8-bit model loading, seed 0;
- protocol version `2026-07-13-v1`;
- runtime output records hashes of the context specification, payload rows,
  and shipped screen/helper source files.

The Kaggle launcher uses its ambient image rather than `requirements.txt`; the
kernel aborts unless the asserted versions above are present.

## Verdicts

- `WORKSPACE_CONTEXTS_ELIGIBLE`: unlock one causal canonicalization test.
- `WORKSPACE_CONTEXTS_INELICITABLE`: stop this program.

## Result

Contexts A, C, D, and E passed. Context B failed because its lookup workflow
reached only 80% end-to-end accuracy. Context F produced 0% valid workflows in
both modes. Because all six contexts were required, the verdict is
`WORKSPACE_CONTEXTS_INELICITABLE`.

Per protocol, no template replacement or repair is allowed, and the causal
canonicalization test is not unlocked.
