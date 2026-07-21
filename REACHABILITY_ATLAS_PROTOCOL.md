# Protocol: Which steering vectors are natural counterfactual operators?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — printed `MIXED_REACHABILITY`; scientific interpretation
demoted to `REACHABILITY_ATLAS_AMBIGUOUS` after instruction-confound audit.***

## Central question

The binding result establishes one approximately prompt-reachable activation
operator: ADD \(z_b-z_a\) closely reproduces the natural textual rewrite.

The broader question is:

> Is approximate natural reachability a general class of steering mechanism,
> or a special property of binding memory?

This is not exact mathematical surjectivity. Equality to every decimal is
neither expected nor claimed. We test whether a cross-fitted intervention
reproduces the matching natural counterfactual state, downstream trajectory,
output, and causal mediator.

## Literature boundary

- Linear representations, function vectors, binding IDs, and steering are
  established.
- Non-identifiability and divergent-intervention work shows that behavioral
  equivalence does not identify a natural mechanism.
- Recent non-surjectivity work argues that generic steering need not have a
  textual preimage.

The candidate contribution is an empirical mechanistic taxonomy of
**approximately natural-reachable operators versus alternate-path controls**.
The Qwen atlas is a gate, not a generality claim.

## Frozen cells

### 0. STORE — positive reference, no rerun

Use the completed audited results:

- `AFFINE_COUNTERFACTUAL_OPERATOR`;
- `CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR`.

### 1. SELECT — natural route flip

- Existing `value_of` template from `delta_select`.
- Existing seed-0 deterministic 8 train / 8 test value pairs.
- Natural counterfactual: flag 0→1, answer B→A.
- Direction: mean paired L2 flag-token displacement on train.
- Injection: L2 flag token on held-out flag-0 prompts.
- Checkpoint: L8 flag token; query state: L8 final position.

### 2. TRANSFORM — natural arithmetic increment

- Existing `direct_sum` template.
- Generate all single-digit pairs with \(3\le a+b\le8\) and \(b<8\).
- Natural counterfactual: \((a,b)\rightarrow(a,b+1)\), so answer \(d\to d+1\).
- Deterministic seed-0 shuffle; first two-thirds train, final third test.
- Direction: mean paired L20 final-position displacement on train.
- Injection: L20 final position on held-out source computations.
- Checkpoint/query state: L26 final position.

### 3. INSTRUCTION — natural data→instruction reframing

- Existing v2 isolated directive/data prompts and seed-0 8/8 payload split.
- Natural counterfactual: quoted-data prompt→live instruction prompt for the
  same payload \(W\), answer `Output`→\(W\).
- Direction: mean paired L20 final-position displacement on train.
- Injection: L20 final position on held-out data prompts.
- Checkpoint/query state: L26 final position.
- Different sequence lengths are an explicit limitation; only semantically
  corresponding final readout states are compared.

No template search, layer search, α sweep, or new elicitation.

## Shared conditions

For each new cell:

1. CLEAN;
2. natural textual CF;
3. ADD cross-fitted direction;
4. ANTI (negative direction);
5. 100 same-norm random directions.

Capture injection layer/site, checkpoint site, and checkpoint final/readout
state. Save every row and all aggregate null draws.

At the checkpoint:

- patch ADD state into CLEAN;
- patch natural CF state into CLEAN;
- overwrite ADD checkpoint with CLEAN state;
- overwrite natural-CF checkpoint with CLEAN state.

## Shared metrics

- displacement cosine and normalized error at injection and query states;
- ADD output effect and ratio to natural-CF effect;
- fraction of held-out rows moving toward the natural target;
- ADD/natural patch-effect ratio;
- ADD and natural block fractions;
- activation norm ratio (coarse guard only);
- random-direction empirical p-values.

## Cell gates

### G0 — elicitation

- SELECT and TRANSFORM: CLEAN and CF greedy accuracy each ≥80%.
- INSTRUCTION: preserve its prior frozen ≥70% gate on data and instruction.

Fail → `INELICITABLE`; no causal analysis for that cell.

### A1 — injection-state reachability

- mean cosine ≥0.80;
- mean normalized error ≤0.60;
- cosine beats random null, p<0.01;
- cosine exceeds ANTI.

### Q1 — downstream/query reachability

- mean cosine ≥0.50;
- mean normalized error ≤0.80;
- cosine beats random null, p<0.01.

### O1 — output equivalence

- ADD effect >0 and p<0.01 vs random;
- ADD/natural-CF effect ratio in [0.70, 1.30];
- ≥80% of rows move toward the natural target.

### M1 — equivalent mediator

ADD-state and natural-state patch effects are positive, with effect ratio in
[0.70, 1.30].

### M2 — shared necessity

CLEAN overwrite removes ≥70% of both ADD and natural-CF effects, and the two
block fractions differ by ≤0.20.

### D1 — coarse distribution guard

Mean ADD/natural activation norm ratio at injection and checkpoint is within
[0.8, 1.2]. A1/Q1 remain load-bearing; D1 alone is not manifold evidence.

## Cell verdicts

- `NATURAL_REACHABLE` iff G0∧A1∧Q1∧O1∧M1∧M2∧D1.
- `OUTPUT_EQUIVALENT_ONLY` iff G0∧O1 but the full conjunction fails.
- `CONTROL_NULL` iff G0 passes and O1 fails.
- `INELICITABLE` iff G0 fails.

## Atlas verdict

- `COUNTERFACTUAL_OPERATORS_GENERALIZE` if at least two of
  SELECT/TRANSFORM/INSTRUCTION are `NATURAL_REACHABLE`.
- `BINDING_ROUTING_REACHABILITY` if SELECT alone is reachable.
- `BINDING_SPECIFIC_REACHABILITY` if none are reachable.
- `MIXED_REACHABILITY` for any other one-cell pattern.

Only `COUNTERFACTUAL_OPERATORS_GENERALIZE` licenses immediate multi-family
replication. Other verdicts require interpreting the boundary, not adding
templates or tuning thresholds.

## Result

- SELECT: `OUTPUT_EQUIVALENT_ONLY`. Injection state cosine 0.998/error
  0.063; output ratio 0.991; patch ratio 0.998; shared blocking 100.8%/99.8%.
  Q1 failed: L8 final-position cosine 0.413/error 1.020, p=.832.
- TRANSFORM: `CONTROL_NULL`. The generic arithmetic +1 direction had injection
  cosine −0.032 and output ratio −0.006.
- INSTRUCTION: `NATURAL_REACHABLE`. Injection cosine 0.952/error 0.300; L26
  query cosine 0.892/error 0.455; output ratio 0.760; patch ratio 0.772;
  shared blocking 100.5%/96.0%. All gates passed.

Only INSTRUCTION was fully reachable. Atlas verdict: `MIXED_REACHABILITY`.

Scope caution: TRANSFORM tested a single source-invariant increment operator,
whereas prior transform success used answer-specific late directions. Its null
rules out this preregistered +1 operator, not every possible computed-value
operator. SELECT's failure is specifically full query-trajectory equivalence;
its local state, output, and causal mediator were essentially natural.

## Post-result instruction audit

The printed `MIXED_REACHABILITY` verdict and all gate arithmetic are
mechanically correct. The abstract interpretation of INSTRUCTION is **demoted
to ambiguous**:

- instruction and data prompts are 34 vs 49 tokens, so the learned direction
  includes a fixed 15-token template/position displacement;
- L20 and the exact payload split were reused from the prior instruction run;
- L26 patch/block evidence is weak because the final residual is a mandatory
  cut-set near unembedding;
- random null draws used independent per-row vectors rather than one shared
  operator per draw.

Safe claim: for this fixed template pair, the train mean L20 displacement
transfers across held-out payload words and mimics the corresponding natural
final-state/output shift. It is not yet a template-independent instruction
operator.

Therefore the scientific atlas status is `REACHABILITY_ATLAS_AMBIGUOUS`
pending one equal-length, template-held-out validation. Model scaling remains
blocked.
