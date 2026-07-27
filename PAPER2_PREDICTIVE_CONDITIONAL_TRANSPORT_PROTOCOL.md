# Paper 2: predictive conditional transport

Status: **frozen before GPU output, 2026-07-27**.

Protocol hash is computed from the complete machine-readable `PROTOCOL` object
in `delta_predictive_conditional_transport.py` and printed before evaluation.

## Question

Can an operation-state displacement be predicted from the current hidden state
alone, or does exact transplantation work only because it copies the
counterfactual target activation?

The hypothesis is:

> BELIEF/SEARCH control is locally linear but globally state-conditioned. A
> low-complexity function of the origin state predicts the appropriate causal
> displacement for an unseen computation.

## Prospective separation

For each of eight computation families, hold that family out completely.
Fit on the other seven families using 24 directed color pairs and select
rank/ridge only on eight different donor-validation pairs. Test on twelve
histories formed from the six directed pairs disjoint from both donor splits.

The predictor input is only the origin-operation layer-21 state at the three
answer-prefix tokens. It predicts the opposite-minus-origin displacement.
There is no counterpart-state argument in the prediction interface.

Target-family BELIEF-to-SEARCH and SEARCH-to-BELIEF predictions are made
sequentially and hashed before counterpart test states are recaptured and
before any causal intervention.

## Predictor and controls

The predictor is reduced-rank linear regression on principal coordinates of
the origin state. Candidate ranks are 1, 3, 8, 16 and 32; ridge values are
0.01, 0.1 and 1.0. Minimum donor-validation normalized displacement MSE selects
the model, with smaller rank and ridge as deterministic tie-breakers. The
selected hyperparameters are then refit on donor train plus validation rows;
the target family remains entirely excluded.

At the fixed layer-21 answer-prefix locus, compare:

1. exact counterpart state (oracle upper bound);
2. selected conditional prediction;
3. global mean displacement;
4. global target-state centroid;
5. nearest donor in predictor feature space;
6. a deterministic row-derangement of conditional predictions;
7. the conditional displacement at instruction positions;
8. the same displacement at matched identical-token positions.

Checkpoint-state convergence is measured directly at layers 24 and 27.
Layer 27 is primary. Value-token accuracy must remain intact.

## Frozen adjudication

An individual family passes only when:

- exact transport passes in both directions;
- predicted progress is at least 0.40 in both directions;
- predicted progress recovers at least half of exact transport;
- at least 18/24 rows move toward the counterpart in each direction;
- median target-distance ratio is at most 0.85;
- value accuracy is at least 80%;
- the conditional bidirectional score exceeds both global baselines and the
  row-shuffled control by at least 0.10;
- instruction and identical-position scores are each below
  `max(0.10, 0.5 × conditional score)`.

`PREDICTABLE_STATE_CONDITIONED_TRANSPORT` requires exact reference transport
and predictive success in at least six of eight held-out families.

Partial, nearest-neighbour-only, global-template, nonspecific, oracle-only and
reference-failure outcomes are reported separately. No prompt, split, rank,
ridge, threshold or locus rescue follows this run.
