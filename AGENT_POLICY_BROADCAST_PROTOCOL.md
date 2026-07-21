# Is one latent policy state broadcast across agent workflow stages?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — NO_SHARED_POLICY_INTERCHANGE.***

## Question

The same template-B L2 controller can switch the initial tool decision and,
when reapplied in a fresh pass, restore later database-result use. Do the tool
decision and answer decision consume a shared downstream policy
representation, or distinct stage-local representations?

## Frozen setup

- Same pinned Qwen2.5-7B-Instruct runtime, template B, payload split, and
  eight non-collision test rows.
- No prompt, layer, row, or strength search.
- Fixed downstream layer L20.
- Independently extract on even-index donors:
  - `d_call`: natural blue−red displacement at the teacher-forced `CALL`
    decision token;
  - `d_answer`: natural blue−red displacement at the final answer decision
    token, with the same correct database call/result in both transcripts.
- Test on odd-index payloads only.
- For cross-stage interchange, rescale the donor-stage direction to the
  recipient-stage reference norm using donor data only. This removes
  stage-gain as a confound without fitting α on test behavior.

## Tests

1. Geometry: cosine(`d_call`, `d_answer`) ≥.80.
2. Same-stage positive references:
   - `d_call` switches red `CALL` decisions to exact database calls;
   - `d_answer` restores database-result answers in red transcripts.
3. Cross-stage interchange:
   - add `d_call` at the red answer-decision token;
   - add `d_answer` at the red `CALL` decision token.
4. For each cross-stage intervention require:
   - target behavior ≥87.5% on diagnostic rows;
   - target-vs-source output-effect ratio [.70,1.30];
   - positive effect on ≥75% of rows;
   - at most 1/100 norm-matched random-null exceedances.

The call-generation test applies the L20 edit only when predicting the tool
token after `CALL`; subsequent argument tokens are generated unsteered.

## Verdicts

- all gates: `SHARED_AGENT_POLICY_STATE`;
- same-stage references pass but geometry/interchange fails:
  `NO_SHARED_POLICY_INTERCHANGE`;
- a same-stage reference fails: `POLICY_BROADCAST_DIAGNOSTIC_INVALID`.

This tests a shared causal policy code across two stages in one emulated
workflow. It does not establish cross-template or cross-model generality.

## Result

The same-stage references passed, but the cross-stage hypothesis failed:

- call/answer direction cosine: .345;
- call-specific direction: 100% exact calls in its own stage;
- answer-specific direction: 7/8 target answers in its own stage;
- answer→call: 0% target calls, output ratio .169;
- call→answer: 4/8 target answers, output ratio −.133.

Verdict: `NO_SHARED_POLICY_INTERCHANGE`.

The reusable L2 policy controller does not become one interchangeable L20
vector across stages. Tool selection and result integration transform the
upstream policy into distinct stage-local decision representations.
