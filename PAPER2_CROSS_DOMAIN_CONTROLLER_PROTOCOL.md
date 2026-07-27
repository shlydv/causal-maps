# Paper 2 confirmation: cross-domain causal route controller

Status: frozen before GPU output, 2026-07-25.

## Question

Does the content-cancelled answer-prefix controller discovered in the
location-belief task reconfigure causal access routes in state spaces with
different entities, relations, prose, and answer vocabularies?

This is a transfer test, not a new controller search. Target-domain
activations, route measurements, outputs, and labels may not be used to fit,
select, rotate, scale, or otherwise modify the controller.

## Frozen source controller

Use Qwen-2.5-7B-Instruct in 8-bit precision. Reconstruct the source controller
exactly as frozen in `PAPER2_CONTENT_CANCELLED_CONTROLLER_PROTOCOL.md`:

- original location-belief narrative and compatible worlds 0-14 only;
- position-matched `BELIEF` and `X X SEARCH` contracts;
- L21 block outputs at the final three answer-prefix positions;
- `D = mean(BELIEF - SEARCH)` across 15 donor worlds and both clean/natural
  histories.

The evaluation worlds 15-29 and every target domain are excluded from this
construction. The reconstructed controller hash, norms, and donor alignment
are reported. The prior controller artifact SHA-256 is
`C59EA1539F4F3E63B953470A0EE94CEDCDC84D288C8C1DBB49EB2C19B5C91B71`;
bitwise identity is reported but is not a gate because GPU/runtime revisions
can change floating-point bytes without changing the frozen construction.

## Frozen target domains

Each domain contains 15 deterministic ordered source-target worlds. For row
`i`, the source is vocabulary item `i mod 8`; rows 0-7 use the next item
(cyclic shift 1) as target and rows 8-14 use cyclic shift 2. This gives 15
unique pairs and balanced source coverage. Two distractor values are the first
two cyclic successors not equal to source or target.
The clean and natural prompts differ at exactly one witnessed/cached state
token. The target value already appears as a candidate but is not used to
choose or fit an intervention.

1. **Ownership**
   - values: `Alice, Bob, Carol, David, Emma, Frank, Grace, Henry`;
   - Rina sees a sapphire badge assigned to the state owner, then leaves;
   - an unseen reassignment and unrelated badge assignment are distractors;
   - question: who does Rina believe owns the sapphire badge?
2. **Color state**
   - values: `red, blue, green, black, white, brown, pink, gray`;
   - Rina sees a status lamp in the state color, then leaves;
   - an unseen color change and a second lamp are distractors;
   - question: what color does Rina believe the status lamp is?
3. **Key-value memory**
   - values: `one, two, three, four, five, six, seven, eight`;
   - Rina reads the value of key ALPHA, then disconnects;
   - an unseen server update and key BETA are distractors;
   - question: what value does Rina remember for key ALPHA?

The private-record rule is stated explicitly in each prompt. Every prompt
contains `STATECHECK` before the question. The two route conditions use the
same question and differ only in the position-matched reply contract:
`BELIEF` versus `X X SEARCH`.

No vocabulary item or world may be removed based on model behavior. A domain
with nonuniform tokenization, a non-single-token clean/natural difference, or
a non-three-token answer-prefix locus is reported as alignment-ineligible.

## Frozen intervention and measurement

For every aligned domain, capture its own clean and natural L21 states. Apply
the unchanged location-derived controller at that domain's final three
answer-prefix positions:

- `BELIEF - D`;
- `SEARCH + D`.

All visible tokens, questions, state content, source markers, and non-prefix
activations remain unchanged. Measure cumulative late-attention mediation
through L22-L27, with L24 as the preregistered scalar checkpoint.

For each original and selected context require:

- clean and natural answer accuracy at least 0.80;
- sufficient bidirectional L21 source intervention under the frozen gate;
- resolved cumulative handoff depth in L22-L27.

For each domain, the original BELIEF-minus-SEARCH L24 gap must be at least
0.03. Each causal direction must move at least 0.05 and at least half that
domain's original gap. Every one of 15 worlds must move in the predicted
direction in both arms.

## Frozen controls

1. **Instruction locus.** Apply the unchanged `D` at the three-token
   instruction occurrence in every domain.
2. **Norm-matched random directions.** Generate 19 seeded Gaussian
   three-position directions. Normalize every position independently to the
   corresponding norm of `D`, and apply each shared direction to all domains
   at their answer-prefix loci.
3. **Matched random positions.** For each domain, generate 19 seeded
   three-position sets from positions whose tokens are identical across
   BELIEF/SEARCH and clean/natural prompts. Apply the unchanged `D` there.

Controls use the same 15 worlds, both history arms, both causal directions,
source intervention, and L24 route measurement. A control-domain cell is
functional only if its behavior and source-site gates pass; otherwise its
generalization score is set below all functional scores while its raw effect
is retained.

The multi-domain statistic is the second-largest functional bidirectional
score across the three domains: a direction must therefore work in at least
two domains. The selected statistic must outrank all 19 random-direction and
all 19 matched-position statistics, giving add-one p = 0.05 in each family.
The instruction-locus statistic must be less than half the selected statistic.

## Frozen verdicts

Per-domain:

- `CROSS_DOMAIN_ROUTE_SWITCH`: all gates pass and a categorical handoff depth
  changes;
- `CONTINUOUS_CROSS_DOMAIN_TRANSFER`: all gates pass without a categorical
  depth change;
- explicit alignment, behavioral, source-site, depth, absent-gap,
  asymmetric, nonuniform, or null verdict otherwise.

Overall:

- `UNIVERSAL_CROSS_DOMAIN_ROUTE_CONTROLLER`: all three domains pass plus both
  specificity families and the instruction control;
- `MULTI_DOMAIN_ROUTE_CONTROLLER`: exactly two domains pass plus all controls;
- `SINGLE_DOMAIN_TRANSFER` or `NO_CROSS_DOMAIN_TRANSFER`: fewer than two pass;
- `NONSPECIFIC_RANDOM_DIRECTION`, `NONSPECIFIC_POSITION_EFFECT`, or
  `INSTRUCTION_LOCUS_EFFECT`: the corresponding control fails;
- `CROSS_DOMAIN_BEHAVIORALLY_INELIGIBLE`: fewer than two domains reach the
  preregistered behavioral/source/depth/original-gap evaluation gate.

## Interpretation boundary and next action

A multi-domain pass supports a domain-general causal control component: a
location-derived displacement reconfigures access routes over different
relations and answer vocabularies without target-domain fitting. It does not
yet establish a low-rank basis or architecture generality.

Only a multi-domain pass unlocks donor-only SVD/low-rank reconstruction.
Failure redirects the program toward domain-conditioned or nonlinear
controllers rather than post-hoc prompt tuning.
