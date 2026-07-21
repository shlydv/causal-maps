# Protocol: Can one latent control switch an entire tool workflow?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `LATENT_ORCHESTRATION_CONTROLLER`.***

## Question

Can one calculate→lookup activation operator, learned on 10 payloads, switch
10 operator-held-out agent workflows from:

`calculator call → numeric result → final answer`

to:

`database call → key-specific result → final answer`?

The key and answer vary by row, so the operator cannot carry a fixed action
argument or answer.

## Frozen setup

- Same Qwen commit/runtime contract and 20 behavior-qualified payloads from
  `ORCHESTRATION_ELIGIBILITY_SCREEN.md`: torch 2.10.0+cu128, transformers
  5.0.0, bitsandbytes 0.49.2. Abort on mismatch.
- Donors: even-indexed 10 rows; tests: odd-indexed 10 rows.
- Direction: mean lookup−calculate residual displacement at the mode token,
  post-layer L2.
- Inject α=1 at the calculate mode token.
- No layer, α, prompt, or donor search.

For trajectory and mediator measurements, teacher-force the common first action
token `CALL` and measure calculator-vs-database tool logits at the next token.
Capture the final `CALL` position at L2/L8/L14/L20/L26. L20 is the mediator.

## End-to-end generation

Greedily regenerate the complete action while recomputing the prompt with the
same single L2 mode-site intervention. Execute the parsed action, then ask the
model for the final answer using the generated action and actual tool result.

Measure:

- exact database call;
- correct key and dummy argument;
- final answer from actual execution;
- final answer must match both the actual executed result and the target task
  answer;
- same-row end-to-end success;
- reverse \(-\Delta\): lookup→calculator.

## Controls

- raw embedding(`lookup`)−embedding(`calculate`) at the same site;
- separate 100-draw shared random nulls norm-matched to the learned and raw
  embedding vectors;
- exact full-continuation parser;
- all states, logits, generations, and row metrics saved.

## Gates

- **G0:** native calculate/lookup full workflows each ≥90%.
- **A1:** L2 local displacement cosine ≥.80, error ≤.60, ≤1 null exceedance.
- **O1:** tool-logit effect ratio [.70,1.30], positive on ≥80%, ≤1 null
  exceedance; generated exact lookup calls ≥80%.
- **W1:** correct steered actions, final answers, and same-row end-to-end
  workflows each ≥80%.
- **R1:** reverse exact calculator workflows ≥80%.
- **Q1:** L20 `CALL`-position displacement cosine ≥.80, error ≤.60.
- **M1:** ADD/natural L20 patch effects positive, ratio [.70,1.30].
- **M2:** clean overwrite blocks ≥70% of both effects, gap ≤.20.
- **B1:** raw embedding baseline does not pass O1/W1/Q1 together.

## Verdict

- all core gates + B1: `LATENT_ORCHESTRATION_CONTROLLER`;
- all core gates except B1: `LEXICAL_ORCHESTRATION_REPLAY`;
- O1/W1 with failed Q1/M1/M2: `ORCHESTRATION_ALTERNATE_PATH`;
- O1/W1 and downstream-equivalence gates with failed A1 or R1:
  `ORCHESTRATION_OPERATOR_AMBIGUOUS`;
- failed O1 or W1 after G0: `ORCHESTRATION_CONTROL_NULL`;
- failed G0: `ORCHESTRATION_INELICITABLE`.

One kernel, no rescue.

## Result

All frozen gates passed:

- native calculate/lookup workflows: 100% / 100%;
- L2 local state: cosine .9994, error .0356, zero null exceedances;
- tool-logit effect: +47.58 vs natural +48.16 (ratio .988), positive
  on 10/10 rows, zero null exceedances;
- steered database calls/actions: 100%; final/end-to-end: 90%;
- reverse calculator workflows: 100%;
- L20 `CALL` state: cosine .9800, error .2000;
- L20 patch ratio .9957;
- clean overwrite blocked 100.96% / 94.33% of ADD/natural effects
  (gap .0663);
- raw embedding baseline failed O1-like/W1/Q1.

Verdict: **`LATENT_ORCHESTRATION_CONTROLLER`**.

Safe scope: in one fixed Qwen7B tool template, an L2 mode-site direction
learned on 10 payloads switches all 10 held-out actions to row-specific
database calls and reverses all 10 lookup prompts to calculators. The
teacher-forced next-tool trajectory becomes natural-like by L20.

One steered row called the correct database key and received the correct tool
result but returned the original calculator answer, so complete workflow
control was 9/10 rather than 10/10.

Important unresolved lexical control: the raw embedding baseline norm was
1.33 versus 27.61 for the learned operator. Its failure does not exclude a
scaled or transformed lexical mode replacement. The frozen verdict is
mechanically correct, but an abstract/non-lexical control claim requires a
scale-matched or held-out-label control. Scope is one template, model, layer,
and two tools; L20 mediation concerns the next-tool decision, not the complete
autonomous workflow.
