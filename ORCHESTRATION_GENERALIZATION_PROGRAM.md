# Program: Generality of latent orchestration controllers

*Causal Maps · 2026-07-13*
*Status: **CLOSED AT GATE 1 — generalization not established.***

## Candidate claim

Low-dimensional latent controllers for orchestration are distinct from scalar
lexical semantics and induce native downstream execution trajectories.

The existing calculate→lookup result is one positive instance, not yet a
general phenomenon.

## Gate 1 — held-out labels and template

On Qwen2.5-7B:

- learn the original `calculate`→`lookup` controller in template A;
- apply it without fitting to template B, where arbitrary labels `red` and
  `blue` select the same tools;
- require exact variable actions, actual execution, final answers, reverse
  workflows, and natural L20 convergence;
- compare against a B-specific positive-reference controller.

Interpretation:

- cross-template direction passes: abstract label/template transfer;
- only B-specific direction passes: template-specific latent controller;
- neither passes: non-replicated mechanism.

## Gate 2 — orchestration battery

Only after Gate 1, test 3–5 fixed, behavior-qualified tool decisions. Every
cell uses:

- donor/test payload split;
- variable arguments and tool results;
- full forward/reverse workflows;
- next-tool trajectory and L20 mediator;
- raw, norm-matched, and donor-optimal lexical controls;
- shared nulls and frozen verdicts.

No post-result prompt/layer/α repair.

## Gate 3 — model replication

Only after at least three Qwen7B workflow cells pass:

1. replicate unchanged on another open Qwen size;
2. replicate unchanged on one open non-Qwen decoder model.

Model eligibility is behavior-only. A model/task cell that fails native
behavior is `INELICITABLE`, not repaired.

## Elevation threshold

Elevate to a general paper claim only if:

- held-out labels/templates transfer;
- at least three distinct workflows pass;
- at least one second model family passes;
- delayed natural convergence and lexical-control rejection recur.

Otherwise report the strongest narrower boundary.

## Outcome

Gate 1 returned `CONTROLLER_NOT_REPLICATED`. The original template-A
direction did not transfer to held-out `red`/`blue` labels and template B.
The B-specific direction controlled tool calls and converged naturally, but
failed the frozen forward end-to-end threshold (60%).

Per the gated design, the multi-workflow and cross-model battery is stopped.
