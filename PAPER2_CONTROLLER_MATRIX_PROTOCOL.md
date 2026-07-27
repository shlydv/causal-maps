# Paper 2 discovery: donor-only controller coordinate matrix

Status: frozen before GPU output, 2026-07-25.

## Question

The unchanged location controller failed cleanly in color state, while its
ownership effect was suggestive but behaviorally ineligible. Is access control
absent outside the location task, or is the same kind of causal computation
expressed in domain-specific activation coordinates?

## Frozen hypotheses

- **Coordinate hypothesis:** separately estimated domain controllers work on
  held-out worlds within their own domains, while raw cross-domain addition
  fails reciprocally.
- **Universal-direction hypothesis:** independently estimated controllers are
  aligned and cross-apply successfully.
- **Location-specific hypothesis:** a color-derived controller fails even
  within held-out color worlds.

No alignment, rotation, projection, scaling, or controller combination is fit
in this experiment. It measures the raw controller matrix and its geometry.

## Frozen domains and splits

Use Qwen-2.5-7B-Instruct in 8-bit precision, L21 three-token answer-prefix
states, and the same position-matched `BELIEF` / `X X SEARCH` contracts.

### Location

Reuse the exact original 30 compatible location worlds:

- donor worlds 0-14;
- evaluation worlds 15-29.

This reconstructs the already confirmed location controller and provides the
location row and column of the matrix.

### Ownership and color state

Reuse the exact ownership and color prompts and eight-value vocabularies from
`PAPER2_CROSS_DOMAIN_CONTROLLER_PROTOCOL.md`, but use 30 fresh ordered pairs
that were absent from that experiment.

For row `i` in 0-29:

- source index = `i mod 8`;
- rows 0-7 use cyclic target shift 3;
- rows 8-15 use shift 4;
- rows 16-23 use shift 5;
- rows 24-29 use shift 6.

Thus none overlaps the prior shift-1/shift-2 pairs. Rows 0-14 are donors and
rows 15-29 are held-out evaluation worlds. Distractors are the first two
cyclic successors unequal to source and target.

Key-value memory is excluded before this experiment because its prerequisite
BELIEF-minus-SEARCH route ordering was absent in the frozen cross-domain run.

## Frozen controller construction

For each domain independently:

`D_domain = mean(BELIEF - SEARCH)`

using only the 15 donor worlds and both clean/natural histories at the final
three L21 answer-prefix positions. No evaluation activation, behavior, route
measurement, or output is used to fit any controller.

Report:

- controller norms and hashes;
- flattened and per-position cosine matrix;
- singular values and cumulative energy of the three flattened controllers;
- donor-row cosine alignment with each mean.

Geometry is descriptive and cannot select an intervention or verdict.

## Frozen causal matrix

For every target domain, evaluate the original BELIEF and SEARCH routes on its
15 held-out worlds. Then apply every source-domain controller:

- `BELIEF - D_source`;
- `SEARCH + D_source`.

Measure cumulative L22-L27 mediation and preregister L24 as the scalar
checkpoint. Preserve clean/natural behavior and source-site gates.

A matrix cell passes when:

- all original and patched contexts have accuracy at least 0.80 and sufficient
  L21 source intervention;
- cumulative handoff depth is resolved;
- the original target-domain L24 gap is at least 0.03;
- each direction moves at least 0.05 and at least half the original gap;
- all 15 held-out worlds move in the predicted direction.

Categorical and continuous passes are distinguished by first-passing depth.

## Frozen within-domain specificity controls

Controls run only for target domains whose **original, unpatched** contexts
pass behavior, source-site, depth, and original-gap gates. This conditional
rule is fixed before interventions and cannot depend on controller outcomes.

For each eligible target:

1. apply its own controller at the instruction locus;
2. apply 19 seeded per-position norm-matched Gaussian directions at the
   answer-prefix locus;
3. apply its own unchanged controller at 19 seeded matched
   token-identical three-position loci.

All controls use both causal directions and L24 mediation. A control cell is
functional only if patched behavior and source sufficiency remain eligible.
The within-domain controller must outrank all 19 nulls in each family
(add-one p = 0.05), and the instruction score must be less than half the
selected score.

## Frozen adjudication

The principal location-color test is reciprocal:

- location controller passes within held-out location;
- color controller passes within held-out color with both specificity
  families;
- location controller fails in color;
- color controller fails in location.

Verdicts:

- `DOMAIN_SPECIFIC_CONTROLLER_COORDINATES`: the reciprocal pattern passes.
- `SHARED_RAW_CONTROLLER_DIRECTION`: both within-domain and both reciprocal
  cross-domain cells pass.
- `ASYMMETRIC_CONTROLLER_COORDINATES`: both within-domain cells pass but
  exactly one reciprocal cross cell passes.
- `LOCATION_SPECIFIC_CONTROLLER`: location passes within but color fails
  within despite an eligible original color contrast.
- explicit behavioral/source/depth/gap/control verdicts take precedence.

Ownership is a preregistered third row/column. If behaviorally eligible, it is
adjudicated identically and can strengthen or complicate the principal
result. If ineligible, its raw matrix effects are reported but do not affect
the principal location-color verdict.

## Interpretation boundary and next action

`DOMAIN_SPECIFIC_CONTROLLER_COORDINATES` would establish that access control
is causally reproducible outside location but is not represented by a
universal raw direction. It would motivate a subsequent, separately frozen
test of whether a low-rank donor-only alignment learned across domains
predicts a held-out domain.

It would not itself prove that the domain controllers implement an identical
algorithm or that a shared aligned subspace exists.
