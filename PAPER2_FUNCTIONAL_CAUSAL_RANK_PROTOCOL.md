# Paper 2: functional causal-rank experiment

## Status

Prospectively frozen before the first experimental output.

## Why this experiment is distinct

The causal-atlas experiment rejected a PCA/Procrustes map between full hidden
states. That does not test whether the *downstream effect* of an exact state
difference is compressible.

Earlier experiments constrain the new question:

- energy-ranked controller SVD already showed that causal importance can
  concentrate in a low-energy axis;
- a rank-four generic local inverse already failed;
- an endogenous low-energy coordinate changed the route while producing only
  about ten percent full-state convergence;
- exact matched-state transport remains broadly sufficient;
- the earlier SELECT quotient test found delayed natural convergence in one
  different early-layer selection task.

Therefore this experiment neither repeats ordinary PCA nor claims that
geometrically divergent states are already equivalent. It asks whether exact
operation-state differences contain a small component selected by their local
downstream response.

## Question and mathematics

Let the established L21 answer-prefix state be \(h\), and let the processed
L27 downstream state be \(F_c(h)\) in context \(c\). Ordinary PCA orders an
activation basis \(U\) by variance in exact differences \(\Delta h\).

The local functional metric is the pullback

\[
G_c = J_c^\top J_c,
\qquad
J_c = \frac{\partial F_c}{\partial h}.
\]

The experiment estimates this metric inside the exact-difference span. For
each RMS-standardized activation axis, apply equal positive and negative
interventions and concatenate the target-normalized odd L27 responses. The
Gram matrix of those responses supplies a causal ordering \(V\).

For an exact held-out difference with standardized coordinates \(q\), its
rank-\(k\) causal projection is

\[
\widehat{\Delta h}_k =
U^\top S V_k V_k^\top q,
\]

where \(S\) restores the training RMS coordinate scales.

The question is whether a small \(k\) reproduces the downstream effect of
\(\Delta h\), even when equal-rank activation PCA does not.

## Frozen scope

- Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Source: layer 21, final three position-matched answer-command tokens.
- Outcomes: processed layer 24 and layer 27 final states, with direct
  final-position identity removed.
- Families: `maximum_score` and `two_hop_pointer`.
  These are the two families used in the pre-existing exact-transplant locus
  diagnostic and bracket strong/weak causal-atlas outcomes.
- Directions: epistemic-to-search and search-to-epistemic.
- Basis/metric lexical panel: `BELIEF/SEARCH`.
- Rank-selection lexical panel: `THINK/FIND`.
- Held-out test lexical panel: `KNOW/LOOK`.
- World splits:
  - 24 directed pairs for the exact-difference basis;
  - four disjoint pairs for central-difference metric calibration;
  - four disjoint pairs for rank selection;
  - twelve held-out histories from disjoint directed pairs for final test.

The uncentered exact SEARCH-minus-BELIEF differences form an orthonormal
activation basis of at most rank 47. Candidate cumulative ranks are
`1, 2, 4, 8, 16, 32, 47`, truncated to available rank.

Rank is chosen independently for each family and direction as the smallest
causal rank recovering at least 80% of full train-span progress on the
`THINK/FIND` selection panel. If none reaches that gate, the best-progress
rank is frozen. No test result can change rank.

## Interpretation boundary

The final test projection may use the exact held-out target difference after
the basis, causal ordering, and selected rank are frozen. This is intentional:
the experiment measures **functional compression capacity**, not source-only
synthesis or practical steering.

Training-mean and causal-projected-training-mean arms are reported separately.
Only those arms are target-blind.

## Final controls

- exact matched target state;
- full train-difference-span projection;
- selected causal-order projection;
- equal-rank activation-PCA projection;
- training mean difference;
- training mean projected into selected causal modes;
- 19 seeded equal-rank random subspaces in standardized coordinates;
- selected projection at instruction positions;
- selected projection at matched identical-token positions.

Central differences provide an explicit nonlinearity diagnostic: the median
even-response norm divided by odd-response norm must not exceed `0.35`.

## Gates

Each of four family/direction cells requires:

- exact progress at least `0.45`, at least 20/24 positive rows, and minimum
  answer accuracy `80%`;
- full-span progress at least `0.35` and at least `55%` recovery of exact;
- selected rank at most `16`;
- selected causal progress at least `0.35`;
- at least `80%` recovery of full-span and `50%` recovery of exact;
- at least 20/24 positive rows and minimum answer accuracy `80%`;
- margin at least `0.08` over equal-rank PCA, raw training mean, and
  causal-projected training mean;
- instruction and identical-position progress below
  `max(0.10, 0.5 × selected progress)`;
- passing central-difference linearity.

Across all four cells, the minimum selected score must beat all 19 random
subspaces under the add-one test (`p = 0.05`).

## Verdicts

- `LOW_RANK_CAUSAL_EFFECT_SUBSPACE`;
- `CONTEXT_DEPENDENT_FUNCTIONAL_COMPRESSION`;
- `NO_CAUSAL_ORDER_ADVANTAGE`;
- `HIGH_DIMENSIONAL_FUNCTIONAL_CONTROL`;
- `TRAIN_DIFFERENCE_SPAN_INSUFFICIENT`;
- `ASSAY_INELIGIBLE`.

## Stopping rule

No prompt, family, panel, split, layer, rank, scale, metric, threshold, random
seed, or control rescue follows the result. A full pass licenses a separate
source-only synthesis experiment. Any other verdict is interpreted as written.

Implementation:
`src/causal_maps/delta_functional_causal_rank.py`.

The final T4-only mathematical guard passed under protocol SHA-256
`C7A452A587846D7A16AA624F52E65A6740402CD4122D1510DCE1199EA4F85897`.
The uploaded source ZIP SHA-256 is
`AC0A1588F5C171A9DFE918480551FC4C361A936A7ABB7C0706C13DAFC5F34194`.
