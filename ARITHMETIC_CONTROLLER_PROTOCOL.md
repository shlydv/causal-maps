# Protocol: Can one latent operator switch arithmetic programs?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `ARITHMETIC_CONTROL_NULL`.***

## Question

Can a single content-independent activation operator change operator-held-out problems
from addition to subtraction and reproduce the internal trajectory of naturally
changing the operation instruction?

The answer varies with both operands. A fixed operator cannot carry one answer.

## Qualification and frozen data

`REASONING_ELIGIBILITY_SCREEN.md` selected this family using behavior only:
Qwen2.5-7B-Instruct scored 100% on all 20 add and subtract prompts.

- Same fixed 20 operand pairs, all answers single digits.
- Donors: even-indexed 10 pairs.
- Tests: odd-indexed 10 pairs.
- Prompt pairs have equal length and differ only at add/subtract token 56.
- Model 7B, 8-bit; seed 0; no task, layer, or α search.

## Operator

\[
\Delta_{\text{sub-add}} =
\operatorname{mean}_{donor}
(h^{sub}_{L8,last}-h^{add}_{L8,last}).
\]

Inject α=1 at L8 final/query position of held-out addition prompts. Capture
L8/L12/L16/L20/L26. L20 is the frozen mediator.

## Controls

- reverse: inject \(-\Delta\) into subtraction prompts;
- raw embedding(`subtract`)−embedding(`add`) at the same L8 site;
- 100 learned-operator-norm random vectors and a separate 100
  embedding-baseline-norm random vectors, one shared vector per draw;
- full digit 0–9 logit displacement;
- all row-level data saved.

## Gates

- **G0:** held-out add/subtract native accuracy each ≥90%.
- **A1:** L8 ADD/natural displacement cosine ≥.50, error ≤.80, and
  ≤1/100 null exceedances.
- **O1:** ADD target accuracy ≥80%; positive target-vs-source effect on ≥80%;
  ADD/natural effect ratio [.70,1.30]; ≤1/100 null exceedances.
- **C1:** digit-logit displacement cosine ≥.80 and error ≤.60.
- **R1:** reverse subtraction→addition accuracy ≥80%.
- **Q1:** L20 query displacement cosine ≥.80 and error ≤.60.
- **M1:** ADD/natural L20 patch effects positive, ratio [.70,1.30].
- **M2:** clean overwrite blocks ≥70% of both effects, gap ≤.20.
- **B1:** raw embedding baseline does not pass O1/C1/Q1 together.

## Verdicts

- all core gates + B1: `LATENT_ARITHMETIC_CONTROLLER`;
- all core gates except B1: `LEXICAL_ARITHMETIC_REPLAY`;
- O1 with failed C1/Q1/M1/M2: `ARITHMETIC_OUTPUT_ONLY`;
- O1 and downstream equivalence but failed A1/R1:
  `ARITHMETIC_OPERATOR_AMBIGUOUS`;
- failed O1 after G0: `ARITHMETIC_CONTROL_NULL`;
- failed G0: `ARITHMETIC_INELICITABLE`.

One kernel, no rescue.

## Result

Native behavior passed at 100% add / 100% subtract, but the operator had no
causal effect:

- steered target accuracy: 0%;
- reverse accuracy: 0%;
- natural output effect: +36.37;
- steered effect: −0.41 (ratio −.011; p=.485);
- L8 displacement cosine .166, error 1.050;
- L20 cosine .068, error .999;
- multiclass digit-logit cosine −.291, error 1.013.

The raw embedding baseline was also null. Only G0 and the baseline-failure
control passed; A1/O1/C1/R1/Q1/M1/M2 all failed.

Verdict: **`ARITHMETIC_CONTROL_NULL`**.

This rules out the preregistered content-independent mean L8 query operator for
switching these held-out addition problems to subtraction. It does not rule
out layer-specific, nonlinear, instance-dependent, or explicit
plan/orchestration controls. Per protocol, no layer or α rescue.
