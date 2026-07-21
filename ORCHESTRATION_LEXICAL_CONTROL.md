# Control: Is orchestration just scale-matched lexical replay?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `BEYOND_SCALE_MATCHED_EMBEDDING`.***

## Motivation

`delta_orchestration_controller` passed every frozen gate, but the learned L2
operator norm was 27.61 while raw embedding(`lookup`)−embedding(`calculate`)
had norm 1.33. The α=1 embedding baseline was therefore underpowered.

## Frozen test

Recreate the same donor/test split, L2 mode-token site, teacher-forced `CALL`
measurement, full generated workflow, and L20 trajectory.

Evaluate:

1. **Positive reference:** the original learned mean L2 operator.
2. **Norm-matched lexical baseline:**

\[
\Delta_E^{norm} =
(e_{lookup}-e_{calculate})
\frac{\|\Delta\|}{\|e_{lookup}-e_{calculate}\|}.
\]

3. **Donor-optimal lexical baseline:** scale the same embedding direction by
the least-squares scalar fitted only against the donor learned operator:

\[
\alpha^* =
\frac{\langle e,\Delta\rangle}{\langle e,e\rangle}.
\]

No test-state fitting, layer search, or α sweep.

For each lexical baseline test:

- calculate→lookup exact actions, arguments, actual execution, and final answer;
- reverse lookup→calculate workflows;
- next-tool logit effect and natural-effect ratio;
- L20 `CALL`-position cosine/error;
- 100 shared random vectors norm-matched to that baseline.

## Replay gate

A lexical baseline explains the result iff all pass:

- tool-logit effect ratio [.70,1.30], positive on ≥80%, ≤1 null exceedance;
- exact target calls and same-row end-to-end workflows ≥80%;
- reverse exact/end-to-end workflows ≥80%;
- L20 cosine ≥.80 and error ≤.60.

## Verdict

- native behavior or the learned positive reference fails to reproduce:
  `LEXICAL_CONTROL_INVALID`;
- either lexical baseline passes: `SCALE_MATCHED_LEXICAL_REPLAY`;
- workflow/output passes but L20 fails:
  `LEXICAL_WORKFLOW_REPLAY_WITHOUT_STATE_EQUIVALENCE`;
- neither baseline passes output/workflow:
  `BEYOND_SCALE_MATCHED_EMBEDDING`.

The last verdict excludes these two scalar lexical baselines only. It does not
prove the operator is abstract or non-lexical; nonlinear transformed lexical
representations would remain possible.

## Result

The learned positive reference reproduced exactly:

- tool-effect ratio .988;
- exact target actions 100%, end-to-end 90%;
- reverse workflows 100%;
- L20 cosine .980/error .200.

The lexical geometry was weakly aligned:

- learned norm 27.61;
- embedding norm 1.33;
- learned/embedding cosine .150;
- norm-match scale 20.77;
- donor-optimal scale 3.11.

Neither lexical baseline approached the active result:

- norm-matched: output ratio .025, exact/end-to-end/reverse 0%, L20 cosine
  .187/error .986;
- donor-optimal: ratio −.016, exact/end-to-end/reverse 0%, L20 cosine
  .133/error .992.

Both failed every replay gate. Verdict:
**`BEYOND_SCALE_MATCHED_EMBEDDING`**.

Safe claim: the orchestration result is not explained by either raw magnitude
or the donor-optimal scalar multiple of the input embedding contrast. This
strengthens the latent-controller interpretation, but does not exclude a
nonlinearly transformed lexical representation. Abstractness still requires
held-out labels/templates and model replication.
