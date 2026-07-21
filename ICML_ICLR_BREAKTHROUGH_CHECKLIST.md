# Addressed-state editing: ICML/ICLR breakthrough checklist

Status: live research checklist, 2026-07-14. A checked item means the claim has
passed a controlled experiment; it does not mean the broader theory is proven.

## Evidence already established

- [x] Value-state substitution matches a natural prompt rewrite on held-out
  substitutions.
- [x] Correct-direction, wrong-direction, and wrong-address controls.
- [x] Literal transfer across two binding grammars.
- [x] Replication in Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3.
- [x] Two simultaneous writes at distinct addresses, independently readable
  with negligible cross-talk in both models.
- [x] Neutral-carrier rival test: the early vector is portable lexical/value
  state, not binding grammar.
- [x] Qwen arithmetic discovery: a neutral processed digit-state write at the
  operand token reproduces the naturally recomputed answer; raw embedding
  difference fails.
- [x] Qwen matched workspace localization: final-query state is not
  interchangeable at L4-L20 and becomes bidirectionally interchangeable at
  L26.
- [x] Negative localization result: no small reader-head/component account.
- [x] Negative universality result: Qwen's early backup-formation timeline does
  not replicate in the frozen Mistral window.

## Required before a strong general claim

- [ ] **Cross-model arithmetic consequence.** Freeze the Qwen operand-edit
  result on Mistral, including the normalized late workspace checkpoint.
  First frozen attempt was behaviorally ineligible (0%/0% native accuracy), so
  this remains open rather than counting as a causal replication failure.
- [ ] **Reusable latent-state edit.** Learn a workspace intervention on donor
  problems and apply it to held-out problems. Matched row-wise patching is only
  an upper bound and does not satisfy this item.
- [ ] **Multiple downstream consequences.** One latent edit must coherently
  change at least two readouts of the same state, so success cannot be reduced
  to steering one output token.
- [ ] **Different computational family.** Replicate beyond direct binding and
  one-step addition, preferably relational inference or rule-based state
  transition.
- [ ] **Planning consequence.** Edit current-state or goal state and obtain a
  valid changed multi-action plan, not merely a changed first action.
- [ ] **Cross-model confirmation of the strongest latent result.** Discovery on
  Qwen; frozen normalized-layer confirmation on Mistral without rescue.
- [ ] **Capacity and failure boundary.** Report when writes stop composing as
  addresses, simultaneous interventions, or reasoning depth increase.

## Mandatory controls for latent/reasoning claims

- [ ] Native CLEAN and NATURAL behavior gate before interventions.
- [ ] Donor/test split by problem instance, values, and surface form where
  possible.
- [ ] Natural rewrite upper bound and matched-state upper bound.
- [ ] Raw embedding, wrong content, wrong address, reverse-sign, and
  norm-matched random controls.
- [ ] Multiple output/readout measurements, including full continuation or
  separate queries where appropriate.
- [ ] Necessity test: overwriting the proposed workspace with CLEAN state must
  block both natural and induced counterfactual consequences.
- [ ] Confirmation layers and thresholds frozen from discovery; no rescue
  sweep after a failed confirmation.
- [ ] Row-level results, exclusions, prompts, token positions, and code saved.

## Next experiment: dual-readout latent arithmetic state

**Superseded as the immediate frontier on 2026-07-14 by the structured
workspace experiment.** Anthropic's J-space paper already demonstrates causal
swaps of silently computed intermediate concepts, including downstream
reasoning. The open differentiator is no longer simply "an endogenous state can
be edited"; it is whether that state has bound relational structure. The
frozen next stage is `delta_structured_workspace`, documented in
`STRUCTURED_WORKSPACE_PROTOCOL.md`. The arithmetic protocol remains a useful
secondary replication, not the highest-value next run.

Implementation status: preregistered in `DUAL_READOUT_WORKSPACE_PROTOCOL.md`
and dispatched as stage `delta_dual_readout_workspace`. Protocol v1 stopped at
G0: sum was 100% accurate, but parity missed the frozen behavioral threshold.
No causal intervention ran. A behavior-only second-readout screen is required
before a new, separately frozen discovery protocol.

Use a shared state-marker position after two operands and before the question.
For every underlying problem create two readouts:

1. the numeric sum;
2. the parity of the sum.

Change the first operand by +1 so both readouts change. On donor rows, identify
the earliest state-marker layer where matched NATURAL/CLEAN exchange is
bidirectionally sufficient for both readouts. Freeze that layer, learn a single
NATURAL-minus-CLEAN workspace direction from donors, and inject it into held-out
CLEAN rows for both readouts.

Primary pass requires the same learned workspace edit to reproduce natural
effects and target accuracy for both sum and parity. Reverse-sign, wrong-state,
embedding-at-workspace, random, and CLEAN-overwrite controls are required. A
pass is evidence for a reusable computational-state edit; a failure after
behavioral eligibility is a real boundary.

## Planning follow-up, conditional on latent arithmetic

Use a deterministic small graph with a state marker before the query. Edit the
goal or current-location workspace state on held-out graphs. Score the entire
teacher-forced action sequence and validate that the resulting path reaches the
counterfactual goal without collisions. Require a wrong-goal control and at
least two changed actions. Run Qwen discovery first and only a frozen Mistral
confirmation after a pass.

## Paper-level claim ladder

1. **Current:** processed lexical/value states can be written at addresses,
   composed, and consumed by native binding and arithmetic computation.
2. **After dual-readout latent pass:** transformers expose reusable, causally
   writable computational state, not merely input-token substitution.
3. **After planning replication:** addressed-state editing can control coherent
   multi-step computation across models and task families.

Level 1 is a strong specialized mechanistic result. Level 2 is a credible
ICML/ICLR-scale central claim. Level 3 is the high-ceiling result.

## Agent/tool-use return path

- [x] Qwen template-A held-out latent orchestration controller.
- [x] Variable tool arguments, actual execution, reverse workflows, and
  downstream natural-trajectory/necessity tests.
- [x] Raw, norm-matched, and donor-optimal scalar lexical controls rejected.
- [x] Boundary: literal controller did not transfer to a new label/template.
- [x] Boundary: call and answer stages did not share one interchangeable L20
  representation; answer-turn reapplication nevertheless succeeded.
- [ ] Frozen second-model replication of the local causal function. Protocol:
  `ORCHESTRATION_CROSS_MODEL_CONFIRMATION.md`. The first Mistral attempt passed
  exact tool calls but was G0-ineligible because lookup result integration was
  8/10; no causal intervention ran.
- [ ] Three distinct behavior-qualified workflows with locally learned
  controllers and the same causal-function gates.
- [ ] Persistent multi-turn state edit or an explicitly refreshed controller
  that changes at least two sequential agent decisions coherently.
- [x] Causal evidence arbitration: a local answer-turn policy
  state controls external-result override and rescues Mistral's zero-result
  conflicts in the clean held-out task with full controls, natural-trajectory
  convergence, and necessity. Protocol: `EVIDENCE_ARBITRATION_PROTOCOL.md`.
  Boundary: the same literal direction did not rescue the original multi-turn
  zero-result failures, so cross-context arbitration transport remains open.
- [ ] Original-context evidence bridge. Frozen v1 reproduced zero-result
  rejection (lookup used zero on 1/10) but was behaviorally ineligible because
  the authoritative upper bound answered zero on only 6/10 and otherwise
  repeated the tool call. No causal intervention ran. A separately frozen,
  behavior-qualified final-answer interface is required.
- [ ] Compositional agent control: independently edit workflow phase and
  evidence policy at a shared decision marker, require non-interference and
  factorial composition, then locally replicate in a second tool family. V1
  was behaviorally ineligible before interventions: database was broadly
  unstable; calculator donors were 100% but held-out cells fell to 75% on two
  sum-9/internal-rival-1 rows. A separately frozen balanced screen is needed.
  Protocol: `COMPOSITIONAL_AGENT_CONTROL_PROTOCOL.md`.
