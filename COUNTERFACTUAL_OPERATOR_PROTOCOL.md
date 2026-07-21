# Protocol: Are binding vectors affine counterfactual operators?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `AFFINE_COUNTERFACTUAL_OPERATOR`.***

---

## Why this is the next question

`delta_trajectory` showed that a cross-fitted source→target direction produces
an L2 write state nearly identical to the natural textual counterfactual and a
matching downstream query trajectory. But each target had one fixed source
(for example dog→cat), and folds varied only the variable name.

The narrow result could therefore be:

1. an **affine value code**: transitions factor as \(z_b-z_a\), independent of
   source and binding context;
2. a **pair-specific replacement**: dog→cat transfers, but the same geometry
   does not support arbitrary source/target rewrites;
3. a **single-template shortcut**: the match disappears in multi-binding
   memory.

This protocol tests those accounts. It is not a rescue of the failed
“downstream generator” gate; the state must be native-like immediately.

---

## Novelty boundary

- Feng & Steinhardt (ICLR 2024) already establish additive, transferable
  binding-ID vectors. A broad “binding is linear” claim is unavailable.
- Linear representations and counterfactual steering are established.
- Recent non-identifiability, divergent-intervention, and non-surjectivity work
  shows that behavioral steering usually does not establish a natural,
  prompt-reachable mechanism.

The potentially new conjunction is:

> A cross-context latent operator, constructed compositionally as \(z_b-z_a\),
> reproduces the natural counterfactual **internal trajectory and causal
> mediator** for unseen source×target transitions in role-specific memory.

One model and one synthetic task cannot establish the general claim. This
kernel is only the validity gate for a later multi-skill, multi-model program.

---

## Donor codebook

- Model: Qwen2.5-7B-Instruct, 8-bit; residual stream fp16.
- Layer/site: L2, value token.
- Values: the ten existing single-token target values.
- Donor contexts: four single-variable prompts using variable names
  \(X,Y,Z,W\), each instantiated separately with every value \(v\).
- Capture \(h_c(v)\) at the value token.

Define a context-centered prototype:

\[
z_v = \frac{1}{|C|}\sum_c \left[h_c(v)-\frac{1}{|V|}\sum_u h_c(u)\right].
\]

For any ordered source/target pair \(a\ne b\), construct:

\[
d_{a\rightarrow b}=z_b-z_a.
\]

No natural \(a\rightarrow b\) displacement from the test context is used to
construct the operator.

---

## Frozen primary test: multi-binding memory

Forty deterministic trials, balanced across values:

`Let X = {a}. Let Y = {d}. What is the value of {Q}?`

- \(Q\in\{X,Y\}\), balanced.
- The queried slot changes naturally from source \(a\) to target \(b\).
- The other slot contains an independently sampled distractor \(d\).
- CLEAN and CF differ only at the queried value token.
- ADD runs CLEAN and adds \(d_{a\rightarrow b}\) to the queried value slot.
- WRONG adds \(d_{a\rightarrow w}\), \(w\ne b\).
- OTHER-SLOT adds the correct operator at the unqueried value slot.
- RAND uses 100 per-row same-norm random operators.

Secondary support: the same constructed operators on the held-out
single-variable name \(K\). It cannot rescue failure on multi-binding.

`S1` (secondary support) passes iff its L2 mean cosine is ≥0.80, L2 mean
normalized error is ≤0.60, L8 final-position cosine is ≥0.50, and its mean
ADD/natural-CF output-effect ratio is ≥0.70. No secondary p-value or alternate
threshold is used.

Capture L2 and L8 at the queried value slot and final query/readout position.

---

## Metrics

For natural and induced displacements:

\[
D_{\text{nat}}=h_{\text{CF}}-h_{\text{CLEAN}},\qquad
D_{\text{add}}=h_{\text{ADD}}-h_{\text{CLEAN}}.
\]

Report:

- cosine \(\cos(D_{\text{add}},D_{\text{nat}})\);
- normalized error
  \(\|D_{\text{add}}-D_{\text{nat}}\|/\|D_{\text{nat}}\|\);
- fraction of individual trials with cosine ≥0.50;
- ADD output effect and ratio to the natural CF output effect;
- wrong-target and other-slot effects;
- same-norm random nulls.

At L8 queried value slot:

1. patch the ADD-generated state into CLEAN;
2. patch the natural CF state into CLEAN;
3. run ADD at L2, then overwrite L8 with the CLEAN state.
4. run the natural textual CF, then overwrite its L8 queried-slot state with
   the CLEAN state.

All per-trial metrics and all 100 aggregate null draws must be saved, not only
their means, so heterogeneity and p-values can be independently recomputed.

---

## Frozen gates

### G0 — behavior

CLEAN source and natural CF target greedy accuracy are each ≥80%.
Fail → `OPERATOR_INELICITABLE`.

### A1 — affine write-state equivalence

At L2 queried slot:

- mean cosine ≥0.80;
- mean normalized error ≤0.60;
- ≥80% of individual trials have cosine ≥0.50;
- mean cosine exceeds WRONG.

### Q1 — natural query trajectory

At L8 final position:

- mean cosine ≥0.50;
- ≥80% of trials have cosine ≥0.25;
- beats same-norm random null, p<0.01.

### O1 — causal rewrite

- mean ADD output effect >0 and p<0.01 vs RAND;
- mean ADD/natural-CF effect ratio ≥0.70;
- ≥80% of trials move toward the target.

### R1 — role specificity

The absolute OTHER-SLOT output effect is ≤20% of the queried-slot ADD effect.

### M1 — equivalent downstream mediator

L8 ADD-state patch and natural-state patch are both positive, with
ADD/native patch-effect ratio in [0.70, 1.30].

### M2 — mediator necessity

Overwriting the L8 queried-slot state with CLEAN removes ≥70% of both:

- the ADD effect; and
- the natural textual-CF effect.

The two block fractions must differ by ≤0.20. This is the shared-path
necessity test; ADD-only blocking is insufficient.

### D1 — distribution guard

Mean \(\|h_{\mathrm{ADD}}\|/\|h_{\mathrm{CF}}\|\) at L2 and L8 queried slot is
within [0.8, 1.2].

---

## Verdicts

| Verdict | Rule |
|---|---|
| `AFFINE_COUNTERFACTUAL_OPERATOR` | G0 ∧ A1 ∧ Q1 ∧ O1 ∧ R1 ∧ M1 ∧ M2 ∧ D1 |
| `STATE_EQUIVALENT_NOT_ROLE_CLEAN` | G0 ∧ A1 ∧ Q1 ∧ O1 ∧ M1 ∧ M2 ∧ D1, but R1 fails |
| `PAIR_OR_CONTEXT_SPECIFIC` | held-out single-variable support passes but primary multi-binding A1/Q1/O1 fails |
| `NOT_AFFINE` | G0 passes but neither positive rule holds |
| `OPERATOR_INELICITABLE` | G0 fails |

---

## Decision

- Only `AFFINE_COUNTERFACTUAL_OPERATOR` licenses a broader
  prompt-reachability study across existing Store/Select/Transform directions
  and multiple model families.
- Any weaker verdict closes the affine-operator route without source/template
  repair, layer search, or α tuning.

---

## Result

Audited v1.1 completed on all 40 balanced multi-binding trials:

- behavior: CLEAN 97.5%, natural CF 100%;
- L2 queried-slot equivalence: cosine **0.9928**, error **0.1125**;
- L8 queried-slot equivalence: cosine **0.9952**, error **0.0952**;
- L8 final/query trajectory: cosine **0.7977** vs random **0.3758**,
  p=.0099;
- ADD output effect **+75.78** vs natural CF **+75.57**, ratio **1.003**;
- all 40 ADD effects moved toward the target;
- unqueried-slot effect **+3.05** (4.0% of queried-slot effect);
- ADD-state / natural-state patch-effect ratio **1.002**;
- CLEAN-state block removed **97.69%** of ADD and **97.35%** of the natural
  textual-CF effect; gap **0.0033**.

All frozen gates G0/A1/Q1/O1/R1/M1/M2/D1 passed. Secondary held-out
single-variable S1 also passed. Verdict:
`AFFINE_COUNTERFACTUAL_OPERATOR`.

Safe scope: in this Qwen7B synthetic variable-memory system, value prototypes
learned in four single-binding contexts compose as \(z_b-z_a\) to reproduce an
unseen multi-binding natural counterfactual at the write state, downstream
query trajectory, output, role specificity, and shared L8 mediator. This is not
yet a model-general or skill-general law.

Standing caveat before broader elevation: the wrong-target operator increased
the frozen target-vs-source logit contrast by +37.81, about half the correct
effect, despite much lower natural-target state alignment (L2 cosine 0.497 vs
0.993). Replacing the source can suppress the source candidate without
selecting the intended target. State equivalence and role specificity stand;
target-specific output interpretation needs a multiclass readout audit.
