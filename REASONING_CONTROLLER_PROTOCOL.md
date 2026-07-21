# Protocol: Can one latent command reroute unseen multi-step reasoning?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `REASONING_INELICITABLE`.***

## Question

Can a content-independent activation operator switch which branch of an unseen
two-step graph the model follows, causing the hidden trajectory to reproduce a
natural instruction edit?

This is stronger than steering a fixed answer. Every graph has a different
source and target endpoint, and endpoint labels are balanced in the donor set.
The operator therefore cannot succeed by carrying one answer token.

## Literature boundary

Prior steering work changes reasoning styles, depth, truthfulness, or aggregate
benchmark accuracy. Function-vector work can induce task behavior. This kernel
does not claim novelty for steering reasoning in general. Its test is narrower:
a single held-out-answer branch operator must reproduce the multiclass output
displacement, full downstream trajectory, and shared causal mediator of the
corresponding natural instruction counterfactual.

## Frozen task

Each prompt defines a five-node graph:

- start →red→ red-middle →green→ red-end;
- start →blue→ blue-middle →green→ blue-end.

The clean instruction says “follow red, then green.” The natural
counterfactual changes exactly one instruction token, red→blue. The answer is
the reached node.

- Node vocabulary: A–J, each verified as one output token.
- 20 balanced donor graphs and 20 disjoint balanced test graphs.
- Dataset seed: 0.
- Every node appears equally often as the blue endpoint in each split.
- Red endpoints are also balanced and always differ from blue endpoints.
- Train/test graphs, generation seeds, prompt, and split are fixed before
  model execution.

## Operator and intervention

- Model: Qwen2.5-7B-Instruct, 8-bit.
- Donor operator:

\[
\Delta_{\text{blue-red}} =
\operatorname{mean}_{train}
\left(h^{blue}_{L8,last}-h^{red}_{L8,last}\right).
\]

- Inject α=1 at L8 final/query position of held-out red prompts.
- Reverse control: inject \(-\Delta\) into held-out blue prompts.
- No layer or strength sweep.

The held-out final/query trajectory is captured at L8, L12, L16, L20, and L26.
L20 is the frozen mediator checkpoint.

## Controls

1. **Embedding baseline:** inject raw embedding(`blue`)−embedding(`red`) at the
   same L8 query site, with no rescaling.
2. **Shared random null:** 100 draws. Each draw is one random vector shared
   across every test graph and norm-matched to Δ.
3. **Reverse operator:** blue→red using −Δ.
4. **Multiclass readout:** all A–J logits, not only target-vs-source.

## Gates

### G0 — elicitation and alignment

- clean and natural-counterfactual greedy accuracy each ≥80%;
- every clean/CF prompt pair has equal token length and exactly one changed
  token;
- all candidate answers are single tokens.

Failure stops before intervention: `REASONING_INELICITABLE`.

### A1 — held-out natural operator at injection

- mean L8 displacement cosine ≥.50;
- normalized error ≤.80;
- cosine exceeds all but at most 1/100 shared random draws.

### O1 — answer-changing reasoning

- ADD greedy target accuracy ≥80%;
- positive target-vs-source effect on ≥80% of test graphs;
- ADD/natural mean output-effect ratio in [.70, 1.30];
- effect exceeds all but at most 1/100 shared random draws.

### C1 — content-specific output state

Across A–J candidate logits, ADD vs natural displacement:

- mean row cosine ≥.80;
- mean normalized error ≤.60.

### R1 — bidirectionality

Applying −Δ to blue prompts recovers the red endpoint with ≥80% accuracy.

### Q1 — natural downstream trajectory

At L20 final/query position:

- ADD/natural displacement cosine ≥.80;
- normalized error ≤.60.

L12/L16/L26 are reported descriptively; they cannot replace L20.

### M1/M2 — shared L20 mediator

- ADD-state/natural-state patch effects into clean are positive and have ratio
  [.70, 1.30];
- overwriting L20 query state with CLEAN blocks ≥70% of both ADD and natural
  effects;
- block-fraction gap ≤.20.

### B1 — beyond raw lexical displacement

The raw embedding baseline does not itself satisfy O1, C1, and Q1 together.

## Verdicts

- `LATENT_REASONING_CONTROLLER` if G0/A1/O1/C1/R1/Q1/M1/M2/B1 all pass.
- `LEXICAL_REPLAY_EQUIVALENT` if every core gate except B1 passes.
- `REASONING_OUTPUT_ONLY` if O1 passes but any of C1/Q1/M1/M2 fails.
- `REASONING_OPERATOR_AMBIGUOUS` if O1 and all downstream-equivalence gates
  pass but A1 or R1 fails.
- `REASONING_CONTROL_NULL` if G0 passes and O1 fails.
- `REASONING_INELICITABLE` if G0 fails.

One result ends the kernel. No prompt, layer, α, or donor repair.

## Result

All mechanical preflights passed:

- 20 balanced donor and 20 disjoint balanced test graphs;
- uniform 88-token prompts;
- every red/blue pair differed only at token position 66;
- exact continuation answer IDs were valid single tokens.

Native behavior failed the frozen ≥80% G0:

- clean/red accuracy: 35%;
- natural-blue accuracy: 20%.

The kernel correctly stopped before donor extraction, intervention, or causal
measurement. Verdict: **`REASONING_INELICITABLE`**.

This says nothing about whether a latent reasoning controller exists. Qwen7B
did not reliably perform this preregistered direct-answer graph construct, so
the causal hypothesis was never tested. Per protocol, no prompt repair or
rerun.
