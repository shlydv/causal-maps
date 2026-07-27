# Paper 2: causal atlas and parallel transport

## Status

Prospectively frozen before the first experimental output. This is a
high-risk structural test, not a rescue of the closed universal-controller
thread.

## Scientific question

Prior experiments establish two facts that a useful theory must explain:

1. an exact matched L21 answer-prefix state can reconfigure the downstream
   computation across eight heterogeneous prompt families; and
2. one fixed shared vector does not preserve breadth, synonym transfer,
   specificity, and downstream consequences.

The candidate explanation is a **causal atlas**. A model need not use one
global controller vector. It may encode related readout frames as local,
context-conditioned coordinate charts over shared content. Moving between
frames would then require a chart-specific transport map.

The preregistered hypothesis is:

> Within each heterogeneous computation, position-matched pretrained
> readout frames are local coordinate charts over a shared content state. A
> low-rank atlas learned from two lexical panels should predict and causally
> transport a third held-out lexical panel, while approximately satisfying
> inverse, composition, and content-equivariance laws.

This is deliberately narrower than saying that an LLM is globally linear.
The transformer remains nonlinear; the experiment asks whether a local,
low-dimensional transition law is recoverable at one established causal
locus.

## Frozen design

- Model: Qwen2.5-7B-Instruct, 8-bit.
- Hardware: Tesla T4 only.
- Source: layer 21, final three position-matched answer-command tokens.
- Readouts: processed layer 24 and layer 27 states, with the direct final
  source-position contribution removed.
- Families: private belief, latest update, key-value lookup, two-hop pointer,
  conditional selection, maximum score, constraint elimination, and temporal
  slot.
- Frames:
  - epistemic;
  - communication;
  - search.
- Training lexical panels:
  - `BELIEF / TELL / SEARCH`;
  - `THINK / REPORT / FIND`.
- Held-out lexical panel:
  - `KNOW / SAY / LOOK`.
- World split:
  - 24 directed pairs for fitting;
  - 8 disjoint directed pairs for rank selection;
  - 12 disjoint histories for the final test.
- All three lexical frames are position matched. The held-out panel and final
  worlds never participate in fitting or rank selection.

## Atlas construction

For each computation family and frame, fit a PCA chart on the flattened
three-token L21 state. Candidate ranks are `2, 4, 8, 16, 32`. Align the three
charts to the epistemic chart using paired orthogonal Procrustes. Select rank
only by mean normalized transition error on validation rows, then refit that
rank on training plus validation rows.

For source frame \(a\) and target frame \(b\), transport a state by:

1. expressing its on-chart component in chart \(a\);
2. rotating that coordinate into the common chart;
3. rotating from the common coordinate into chart \(b\);
4. reconstructing it around the target mean; and
5. preserving the source off-chart residual.

No test target state is an argument to this prediction. Source-only
predictions are written, hashed, reloaded, and verified before any target
baseline is captured.

## Primary tests

The six ordered frame transitions are tested in both clean and natural
histories. The primary intervention inserts the predicted L21 target state
into the source prompt and measures downstream movement toward the target
prompt.

Three structural laws are tested:

- inverse: \(T_{ba}(T_{ab}(h)) \approx h\);
- composition: \(T_{bc}(T_{ab}(h)) \approx T_{ac}(h)\);
- content equivariance: the natural-minus-clean content update is preserved
  after transport into the target frame.

## Frozen controls

- exact matched target state;
- independently selected pairwise reduced-rank predictor;
- transition mean displacement;
- wrong target-frame transport;
- 19 row-shuffled atlas displacements with add-one empirical \(p\);
- atlas displacement at instruction positions;
- atlas displacement at matched identical-token positions.

Three frozen row-shuffled controls are also run causally. All 19 are scored
offline. Exact matched target states are evaluation-only oracles and never
enter model selection.

## Gates and verdicts

A family is exact-eligible only if its exact oracle has mean processed-L27
progress at least `0.45`, all six transitions are positive, and minimum value
accuracy is at least `80%`.

Causal atlas transport additionally requires:

- mean progress at least `0.35`;
- at least five of six positive transitions;
- at least `55%` recovery of exact progress;
- minimum value accuracy at least `80%`;
- margin at least `0.08` over both pairwise and mean baselines;
- margin at least `0.08` over wrong-target and causal-random controls;
- instruction and identical-position effects below
  `max(0.10, 0.5 × atlas progress)`.

The algebraic gate requires:

- composed causal progress at least `0.30`;
- five of six positive composed transitions;
- minimum value accuracy at least `80%`;
- median direct-versus-composed disagreement at most `0.35`;
- median inverse-loop error at most `0.50`;
- content-equivariance cosine at least `0.60`;
- content-equivariance error at most `0.85`.

The full headline requires at least six of eight families to pass exact,
causal-transport, and algebra gates, and the across-family atlas breadth score
must beat all 19 row-shuffled controls (`p = 0.05`).

Frozen verdicts:

- `CAUSAL_ATLAS_WITH_COMPOSITION`;
- `CAUSAL_TRANSPORT_WITHOUT_ALGEBRA`;
- `ALGEBRAIC_FIT_WITHOUT_CAUSAL_CONTROL`;
- `NO_LOW_COMPLEXITY_CAUSAL_ATLAS`;
- `ASSAY_INELIGIBLE`.

## Stopping rule

No prompt, split, panel, family, rank, metric, threshold, position, control,
seed, or model-class rescue follows the output. A full pass licenses an
architecture replication. A causal-only pass supports transport but not the
atlas algebra. Any other verdict is interpreted as written and this specific
low-rank atlas is closed.

Implementation:
`src/causal_maps/delta_causal_atlas.py`.

The T4-only pre-run self-test passed under protocol SHA-256
`46477A04C5D9BF650E1942167AA2A4CAD7D83FBC4EE7D56031C89672E4E1FCA0`.
The uploaded source ZIP SHA-256 is
`77FEF163539EA40085098B34A7787F7C87346C7EAE793AB4CC16F8AE2274F974`.
