# Does orchestration require answer-turn latent control?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — ANSWER_TURN_LATENT_CONTROL.***

## Question

The template-B controller switched `red` calculator prompts to the correct
database call on 10/10 test rows, but only 6/10 final answers used the executed
database result. The answer turn was a fresh forward pass with no steering and
the unchanged `red` instruction.

Does reapplying the independently learned B `blue`−`red` L2 controller at the
separate answer turn causally restore target-result use?

Answers are one token, so this identifies stage-specific reapplication at the
answer decision. It does not establish persistent control across a long,
multi-token trajectory.

## Frozen setup

- Same pinned Qwen2.5-7B-Instruct runtime and 8-bit loading.
- Same template B, mode token position 103, L2 intervention, α=1.
- Learn B `blue`−`red` on the original 10 even-index donor payloads.
- Evaluate the 10 odd-index test payloads.
- Primary analysis uses the eight diagnostic rows where calculator and
  database results differ; the two result collisions are reported separately.
- Calls are fixed to their correct strings and truly parsed/executed before
  answer generation. This isolates result integration from tool selection.

## Conditions

1. `natural_target`: blue prompt + database call/result, no intervention.
2. `natural_source`: red prompt + calculator call/result, no intervention.
3. `source_target_unsteered`: red prompt + database call/result, no
   intervention. This must reproduce the conflict.
4. `source_target_reapplied`: condition 3 with the B controller reapplied at
   the red token for the answer decision.
5. `source_target_lexical`: condition 3 with the raw blue−red embedding
   direction scaled to the B controller norm.

## Metrics and gates

- G0: natural target and source each ≥87.5% exact on the eight diagnostic
  rows.
- R0: unsteered conflict target accuracy ≤62.5%, reproducing the prior result.
- C1 behavior: reapplied controller target accuracy ≥87.5% and improves over
  unsteered by ≥25 percentage points.
- C2 output: target-vs-source next-token effect ratio to the natural blue
  counterfactual is [.70, 1.30], positive on ≥75% of diagnostic rows, and
  exceeded by at most 1/100 norm-matched random directions.
- B1: norm-matched lexical control fails either C1 or C2.

No α sweep, layer change, prompt repair, or row replacement.

## Verdicts

- all gates and B1: `ANSWER_TURN_LATENT_CONTROL`;
- controller passes but lexical also passes:
  `ANSWER_TURN_CONTROL_LEXICAL`;
- G0 or R0 fails: `ANSWER_TURN_DIAGNOSTIC_INVALID`;
- otherwise: `ANSWER_TURN_CONTROL_NULL`.

This diagnostic does not reopen the failed static cross-template-transfer
claim. It tests whether a template-specific controller must be reapplied at a
separate workflow turn.

## Result

All gates passed:

- natural blue target use: 7/8 diagnostic rows;
- natural red source use: 8/8;
- contradictory red prompt + database transcript, unsteered: 5/8 target;
- same transcript with B controller reapplied: 7/8 target (+25 points);
- learned output effect +10.42 vs natural blue +9.16, ratio 1.137,
  positive on 8/8 rows, 0/100 null exceedances;
- norm-matched blue−red embedding: 6/8 target, ratio .240, positive on
  5/8 rows, 27/100 null exceedances.

Verdict: `ANSWER_TURN_LATENT_CONTROL`.

The controller does not permanently alter a later independent model call.
Reapplying the same template-specific policy direction at the answer turn
restores result use to the natural-blue behavioral ceiling. This establishes
stage-specific reuse, not multi-token persistence or cross-template
generality.
