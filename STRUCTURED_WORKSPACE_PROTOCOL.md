# Structured workspace interchange protocol

Status: frozen discovery protocol, 2026-07-14. Stage:
`delta_structured_workspace`. The first GPU run has not occurred.

## Scientific question

Does a transformer expose a query-independent internal state that contains
**bound relations**, rather than only an unordered bag of active concepts?

Anthropic's 2026 Jacobian-lens work establishes a verbalizable global
workspace and causally swaps silent concepts used in reasoning. Its stated
open limitation is relational structure: a readout containing `spider`,
`legs`, and `eight` does not reveal how those concepts are bound, assigned
roles, or organized. This protocol tests that missing structure directly.

The proposed contribution is not "another internally computed concept can be
steered." It is:

> A single bound role-filler relation in a silently computed multi-relation
> world state can be edited, with all and only its consequences changing; two
> relation edits remain independent and compose.

## World and latent variables

Each history determines six variables:

- `belief(Alice, cube)` (`ac`)
- `belief(Bob, cube)` (`bc`)
- `belief(Alice, sphere)` (`as`)
- `belief(Bob, sphere)` (`bs`)
- `truth(cube)` (`tc`)
- `truth(sphere)` (`ts`)

An observer's belief is the last placement that observer personally saw.
Later unseen moves alter truth but not that belief. The final states are never
listed as a table; the model must derive them from the event history.

The literal token `STATECHECK` occurs after the event history and before the
question. A causal transformer's activation at that position cannot depend on
which later question will be asked. All readouts therefore test the same
pre-query state.

Two disjoint surfaces are used:

- donor: compact private-event ledger;
- held-out: narrative event history.

Each split contains five histories. Truth locations vary across histories.

## Primary edit and address control

The primary edit is:

`belief(Alice, cube): Paris -> Rome`

In every clean world, two additional belief registers still contain `Paris`
and another register already contains `Rome`. Consequently, a global
Paris-to-Rome concept replacement is incorrect. A valid relational edit must:

- change Alice's belief report about the cube;
- change where Alice searches for the cube;
- change what Alice tells a teammate about the cube;
- preserve Alice's belief about the sphere;
- preserve Bob's belief about the cube;
- preserve both true locations.

Separate donor directions edit `belief(Alice, sphere)` and
`belief(Bob, cube)` from Paris to Rome. Each must change its own relation while
preserving `belief(Alice, cube)`. These are positive wrong-address controls,
not merely null directions.

## Frozen gates

### G0: behavior

Before interventions, candidate-token accuracy must be at least 80% in every
required clean and natural cell, on donor and held-out surfaces. Required cells
include belief reports, search actions, communication, invariant relations,
truth, and the joint counterfactual.

Failure means `STRUCTURED_WORKSPACE_BEHAVIORALLY_INELIGIBLE`. No activation
intervention is interpreted.

### G1: native bidirectional interchange

Candidate checkpoint layers are frozen to L8, L12, L16, L20, L24, and L26,
subject only to model depth. On donor histories, row-matched natural checkpoint
states are patched into clean runs, and clean states into natural runs, for:

- Alice/cube belief and search;
- Bob/sphere belief and search.

For every required readout at a layer:

- forward target accuracy >= 80%;
- forward positive-row fraction >= 80%;
- forward effect ratio in [0.60, 1.40] relative to the natural history;
- reverse clean accuracy >= 80%.

The earliest passing layer is frozen for every subsequent analysis. If no
layer passes, the verdict is `NO_BIDIRECTIONAL_STRUCTURED_CHECKPOINT`. A
one-way steering result is not accepted as native state.

### G2: donor-mean transfer

At the frozen layer, the intervention is the mean donor checkpoint difference
`mean(h_natural - h_clean)`. It is applied unchanged to held-out histories and
the narrative surface. Row-matched held-out states are never used to construct
the intervention.

The primary relation must pass belief, search, and communication readouts.
The same procedure is repeated for Tokyo-to-Delhi and Cairo-to-Oslo. At least
two of the three value transitions must pass all their readouts.

### G3: specificity

All four primary invariant readouts must retain at least 80% accuracy. Both
wrong-address directions must change their own relation and preserve the
primary relation at the same thresholds.

### G4: joint composition

On new worlds, independently learned directions perform:

- `belief(Alice, cube): Paris -> Rome`;
- `belief(Bob, sphere): Tokyo -> Delhi`.

Each single edit must change its own belief and search readout while preserving
the other. Their vector sum must match the natural joint counterfactual on all
four target readouts and preserve the four unrelated belief/truth readouts.

This is a superposition test. We do not call the order of two additions a
commutativity result, because addition at one residual-stream site is
algebraically commutative by construction.

### G5: rival directions and full output

- Thirty norm-matched random directions form a frozen null; the learned mean
  normalized effect must have permutation `p < .04`.
- A processed, unbound Paris-to-Rome concept direction is applied at the same
  layer and checkpoint. This is explicitly a bag-of-concepts rival, **not** an
  exact reproduction of Anthropic's Jacobian lens. Its target and collateral
  effects are reported without moving the primary thresholds.
- Greedy full continuations, not only the first discriminating token, must be
  at least 80% exact for primary clean, natural, and edited readouts and for
  natural versus edited joint composition.

## Verdicts

- `STRUCTURED_WORKSPACE_BEHAVIORALLY_INELIGIBLE`: underlying task failed.
- `NO_BIDIRECTIONAL_STRUCTURED_CHECKPOINT`: no native interchangeable state in
  the frozen layer set.
- `STRUCTURED_WORKSPACE_PARTIAL_OR_NULL`: checkpoint exists, but one or more
  transfer, specificity, composition, null, or generation gates failed.
- `FACTORIZED_RELATIONAL_WORKSPACE`: every frozen primary gate passed.

The last verdict would justify a claim about structured relational state in
Qwen-7B. It would not yet establish a cross-model law. The next step would be
an untouched-threshold confirmation on Mistral and Qwen-14B, followed by an
exact released-J-lens baseline if the unbound-concept rival remains ambiguous.

## Efficiency and stopping rules

One Qwen-7B model load executes all gates. Expensive nulls and full generation
run only after G0 and G1 pass. No prompt repair, coefficient sweep, rescue
layer, or threshold change is permitted after seeing the GPU result. A failed
gate determines the next scientific question; it does not license rerunning
nearby variants.
