# Paper 2 discovery log

## 2026-07-23 — shared causal component of spillover, preregistered

Paper 1's unexplained observation is the starting point: Qwen/Gemma synthetic
content writes preserve address specificity, whereas
DeepSeek-R1-Distill-Llama-8B transfers the intended content but also changes an
unrelated same-valued belief. This is a model-family mechanistic divergence;
it is not yet licensed as an architecture-only effect because training and
model-family differences remain confounded.

The first Paper 2 hypothesis is deliberately interpretation-neutral:
the natural-minus-synthetic state difference may contain a low-dimensional,
donor-learned causal component that controls spillover on held-out worlds.
It is not called routing in advance.

Alternatives frozen before GPU output:

1. confidence/calibration or generic target-token suppression;
2. a distributed context difference unrelated to address specificity;
3. an off-manifold side effect of synthetic activation addition;
4. a high-dimensional or row-specific difference with no shared component;
5. a true causal component that rescues behavior but is not necessary;
6. a causal address-specificity component, with routing as a later
   interpretation requiring path evidence.

The decisive controls are held-out donor/evaluation separation, simultaneous
preservation of the intended Rome answer and a second legitimate Rome answer,
reverse induction by subtracting the component from natural runs, and
size-matched random pre-query loci. The frozen design and verdicts are in
`PAPER2_SHARED_COMPONENT_PROTOCOL.md`.

### Result

Kernel `cm-paper2-deepseek-shared-component-d1-t4` completed and correctly
stopped at `BEHAVIORALLY_INELIGIBLE` before any layer/locus screen.

- `belief_ac`: CLEAN, NATURAL, and SYNTHETIC target accuracy were all 100%.
- `belief_bc`: CLEAN address accuracy was 93.3%, but NATURAL address accuracy
  was 0%; both NATURAL and SYNTHETIC favored Rome on all evaluation rows.
- `belief_bs`: the legitimate Rome control was already 0% correct in CLEAN and
  NATURAL (SYNTHETIC 6.7%).

Immutable artifact SHA-256:
`2E633EDFC8C2E0496097BE7C57A22CB2991AE76CD54D473E125D3561BFC4EBB9`.

This falsifies the assumed contrast on the structured belief world: the
textual counterfactual is not address-correct, so the natural-minus-synthetic
difference cannot yet be interpreted as the missing cause of synthetic
spillover. Paper 1's calibrated observation—content transfer without strict
address specificity—stands, but the failure is behaviorally present under
text too.

Next decision: run one cheap direct-binding diagnostic with the same
same-valued collision structure. It must establish fully address-correct
textual behavior before comparing the synthetic write. A synthetic-only
failure licenses mechanism work; synthetic specificity closes this anomaly as
a structured-belief capability boundary.

## 2026-07-24 — direct-binding diagnostic complete; collision hypothesis

Kernel `cm-paper2-deepseek-direct-binding-diagnostic-t4` completed with
`BEHAVIORALLY_INELIGIBLE`.

- CLEAN/NATURAL address accuracy: Alice/cube 100%/90%, Bob/cube
  96.7%/36.7%, Alice/sphere 80%/30%, Bob/sphere 40%/70%.
- The Alice synthetic write itself was strong: target 100%, positive rows
  100%, ratio 1.214.
- Under that write, the two other source-valued records were preserved only
  6.7% and 0%, while the already-target-valued record remained correct 100%.
- Wrong-address own-target accuracy was 63.3% (positive rows 100%, ratio
  1.083), while Alice/cube was preserved 96.7%.

Artifact SHA-256:
`7A724B3D94CBA8E66BD95821DBE3A0ADFA3A77D985C535C793E6BDBE2A93C503`.

The original natural-versus-synthetic residual story is closed on both the
belief and direct-ledger surfaces. A sharper candidate emerged from the error
structure: content-equivalence aliasing. The next frozen pilot varies the
number of registers sharing the source value while retaining a target-valued
control. It tests whether synthetic spill follows content identity exactly,
rather than globally broadcasting the target or following position.

## 2026-07-24 — collision-load pilot ineligible; divergence branch closed

Kernel `cm-paper2-deepseek-content-aliasing-d1-t4` completed with
`BEHAVIORALLY_INELIGIBLE`. The intended A write passed at every load (100%
target accuracy, ratios 0.988–1.041), and source-sharing B/C cells often
favored the target at 83–87%. However the minimal register surface failed its
lookup control: A was 100% clean, while B/C/D clean accuracy ranged from
13.3% to 96.7%. The model largely defaulted to A, so the apparent collision
curve is not identifiable as content aliasing.

Artifact SHA-256:
`162C98DFEAECE477B36933C2CB2C0BE8A5E74FA3CB52809CF81E1F2B1B13C87E`.

Decision: stop prompt/surface rescue and close the DeepSeek divergence branch.
The robust Paper-2 target returns to Paper 1's cross-model handoff:
query-independent source support becomes a late, query-specific causal
readout. The next pilot asks whether a sparse set of late attention-head
outputs explains that handoff with held-out selection, bidirectional causal
patches, and matched random-head controls.

## 2026-07-24 — sparse transport discovery positive

Kernel `cm-paper2-qwen7b-sparse-transport-d1-t4` completed with the
preregistered verdict `SPARSE_TRANSPORT_PATH`.

The 112 candidate layer×head sites at L21–L24 were ranked only on the first
15 worlds. On the untouched final 15 worlds, the frozen top-four set
`L23H11, L24H21, L22H1, L23H6`:

- reproduced 0.772× of the natural forward effect and 0.736× in reverse;
- achieved 100% target accuracy forward and 100% clean-source recovery
  backward;
- passed on every evaluation row with large, same-sign logit effects.

The nested top-eight set reached 0.870× forward and 0.776× reverse, also with
100% endpoint accuracy. None of five seeded size-matched random sets at any K
passed; at K=4 their largest bidirectional minimum ratio was 0.017. No
single-layer all-head patch passed: the strongest was L23 at 0.481× forward,
0.362× reverse.

Artifact SHA-256:
`1B17C70E03D1BE0308195907314AE59217921C3D60C57B9DD1E55B76CC0A1586`.

Interpretation is deliberately limited. This is the first evidence for a
sparse, cross-layer causal transport bottleneck at the query readout, not yet
proof of a source-to-query circuit or of individual-head necessity. The
non-monotonic comparison with whole-layer patches suggests that unselected
heads can counteract transport, but this is only a hypothesis because
cross-layer sparse patches and single-layer full patches are different
interventions.

Decision: promote the finding to locked confirmation. Increase the random-set
null enough to support an empirical tail probability, repeat on a fresh
world/query/address split without reselecting heads, and add matched
same-layer and ranking-perturbation controls. Only after that passes should
source-edge localization, necessity, predicted-error, and rescue tests begin.

## 2026-07-24 — locked cross-query confirmation: real but query-limited

The corrected T4 kernel completed with the preregistered verdict
`QUERY_LIMITED_TRANSPORT`. The frozen discovery heads were never reselected.

- `tell_ac` was behaviorally eligible at 100%/100%. The top four produced
  0.743× forward and 0.5999× reverse with 100%/100% endpoint accuracy. It
  therefore failed the literal 0.60 reverse-ratio gate by 0.0001; this gate is
  not moved. The frozen top eight passed at 0.838×/0.676× and 100%/100%.
- `search_ac` was eligible at 83.3%/100%. Top four produced
  0.676×/0.549× with 66.7%/43.3% endpoint accuracy. Top eight improved to
  0.796×/0.609× and 93.3%/66.7%, but still failed reverse accuracy.
- The 99-set top-four null on `tell_ac` was decisive despite the formal gate
  miss: zero random sets passed or exceeded the observed bidirectional
  statistic, empirical p=0.01, maximum random statistic 0.026.
- No selected same-layer subset was sufficient. L22, L23, and L24 achieved
  only 0.104–0.396× forward and 0.090–0.275× reverse.

The preregistered exploratory address probes separate address from operation:

- `belief_as`: top four and top eight both passed; top four reached
  0.790×/0.741× with 100%/96.7% endpoint accuracy.
- `belief_bc`: top eight passed at 0.753×/0.818× with 100%/90%; top four
  narrowly missed reverse accuracy at 76.7% despite 0.678×/0.709× effects.

Artifact SHA-256:
`6BDDFAEB3A6448DA43B3434C96079162CB3CED98626207086F8D829E30ED4BCC`.

Interpretation: the sparse late pathway is not tied to the original
Alice/cube address. It transfers across object and agent addresses for the
same belief-readout operation, and the larger frozen set also supports the
`tell` consequence. It is not a universal query-output pathway: `search`
retains a substantial logit effect but fails behavioral reverse recovery.
This supports a sharper candidate factorization between a shared transported
state and operation-specific readout machinery. That factorization is a new
hypothesis, not yet a claim.

Decision: do not rerun this confirmation or relax its gates. Next localize the
source-token attention edges of the frozen heads separately for belief, tell,
and search; test whether edge patterns predict the query boundary; then run
edge necessity and matched rescue. This directly asks whether a shared
transport backbone is composed with operation-specific routing.

## 2026-07-24 — source-to-head mediation is large, specific, and incomplete

The frozen mediation kernel completed with the preregistered verdict
`MIXED_MEDIATION`. No gate was moved.

The L21 source-anchor interchange was fully sufficient for every operation:

- `belief_ac`: 1.158× forward, 1.183× reverse, 100%/100% accuracy;
- `tell_ac`: 1.112× forward, 1.008× reverse, 100%/100%;
- `search_ac`: 1.119× forward, 1.060× reverse, 100%/100%.

Clamping the frozen top eight to the originating trajectory removed a large
but incomplete fraction of those source effects:

- `belief_ac`: 69.55% forward, 78.07% reverse, with both originating
  endpoints restored at 100%. The forward fraction missed the frozen 70% gate
  by 0.45 percentage points.
- `tell_ac`: 63.08% forward, 81.00% reverse, with both endpoints restored at
  100%.
- `search_ac`: 58.92% forward, 71.57% reverse. Reverse restored NATURAL at
  93.3%, but forward did not restore CLEAN (0%); the residual route still
  selected the target.

The selected belief mediation statistic was 0.6955. None of 39 random
eight-site clamps passed or exceeded it (empirical p=0.025); the maximum
random statistic was only 0.0154. The wrong-address Bob/cube L21 patch
preserved Alice/cube at 100% with target-drift ratio 0.060, passing the frozen
control.

Artifact SHA-256:
`5967FEC72E0F3B9A88FBEC0D9C04EF089851C6F02E9F488D588EB5C10EB998FF`.

Interpretation: L21 exposes an address-specific, query-independent source
state that is sufficient for belief, tell, and search. The frozen sparse heads
are genuine major mediators, not generic disruptive sites, but they are not
the entire causal route. The asymmetry is graded rather than binary:
belief/tell retain behaviorally weak residual margins, while search retains a
behaviorally decisive forward backup route.

Decision: postpone individual K/V-edge claims until the missing route is
mapped. Run a held-out conditional screen: clamp the frozen top eight, rank
remaining L22-L24 heads by how much each removes the residual source effect,
freeze small complements, and test them bidirectionally plus cross-query. A
sparse shared complement would complete the backbone; query-specific
complements would establish operation-dependent redundant routing.

## 2026-07-24 — conditional backup screen: sparse belief/tell, unresolved search

The held-out conditional screen completed with
`PARTIAL_BACKUP_LOCALIZATION`.

Discovery rankings were strongly overlapping. `L24H23`, `L22H15`,
`L22H24`, `L22H19`, and `L23H5` appeared near the top for all three
operations, arguing against three unrelated same-depth backup sets.

On the untouched 15 evaluation worlds:

- `belief_ac`: adding only `L24H23` to the frozen top eight passed mediation
  at 70.9%/80.0% with 100%/100% endpoint restoration. K=8 reached
  76.5%/86.1%.
- `tell_ac`: nested complements improved monotonically; K=8 passed at
  70.8%/88.3% with 100%/100% restoration.
- `search_ac`: K=1 through K=8 improved monotonically from 60.1%/74.1% to
  66.8%/80.5%, but forward CLEAN restoration rose only from 0% to 33.3%.
  The query-selected complement therefore did not close the search route.

No matched random complement passed. The 3×3 transfer matrix showed that the
tell-selected and search-selected K=8 complements both passed belief and tell,
but none of the three complements passed search. This is not evidence for
distinct sparse same-depth query backups; it is evidence for a shared
belief/tell completion plus a search route outside the screened L22-L24
readout-head sites.

Artifact SHA-256:
`F40D78EA31263FE152DE89B6CD6CF14C654E68658BF22FDD517CCD16C2C694D4`.

Decision: test operation-dependent handoff depth before invoking an MLP or
distributed non-attention explanation. Clamp full readout attention outputs
individually and cumulatively across L22-L27 during the same L21 source
intervention. A later search transition would establish operation-specific
temporal routing; failure even through L27 would redirect localization to
earlier query positions, MLPs, or the residual bypass.

## 2026-07-24 — operation-dependent causal consolidation depth

The frozen full-attention depth diagnostic completed with
`OPERATION_DEPENDENT_HANDOFF_DEPTH`.

All L21 source interventions again passed bidirectionally with 100% endpoint
accuracy. The first cumulative prefix of full readout-attention clamps meeting
the unchanged 70%/80% mediation gate was:

- `belief_ac`: through L24;
- `tell_ac`: through L26;
- `search_ac`: through L27.

The curves were graded and monotonic, not isolated single-layer switches:

- belief prefix minimum mediation: 0.108 (L22), 0.513 (L23), 0.730 (L24),
  0.735 (L25), 0.872 (L26), 0.999 (L27);
- tell: 0.101, 0.453, 0.643, 0.656, 0.829, 0.999;
- search: 0.125, 0.448, 0.633, 0.657, 0.779, 0.999.

At L26, search already showed 77.9%/82.7% effect removal but restored the
CLEAN forward endpoint on only 63.3% of rows; adding L27 raised restoration
to the clean baseline ceiling of 83.3% and passed. Belief and tell restored
both endpoints at 100% earlier.

No individual full-layer clamp passed for any query. Suffix L23-L27 passed
belief and tell but not search; the full L22-L27 suffix passed all three.
Thus the result is a cumulative-path hierarchy, not evidence that one layer
alone stores an operation.

Artifact SHA-256:
`43EA058557857D51D512344BDFCDA3E5E8A502182F081619AF967502D7F5164F`.

Interpretation: the same address-specific L21 source state is progressively
converted into query-dependent behavior over late attention pathways. The
depth needed for causal endpoint control increases from direct belief readout
to communication and then action/search. This is best described as
operation-dependent causal consolidation depth, not as three hard handoff
layers. Baseline margin and task difficulty remain alternatives until the
ordering is confirmed on new operations/addresses or another model.

Decision: freeze this as the central Paper-2 discovery candidate. The next
work should prioritize a locked replication of the depth ordering and
source-edge necessity/rescue at the transition layers, rather than additional
unstructured head screens.

## 2026-07-24 — strict depth ordering did not replicate across surface

The locked ledger/paraphrase replication completed with
`SOURCE_SITE_INELIGIBLE`; the strict direct-state < communication < action
ordering was not confirmed.

- `STATE`: behavior 86.7%/100%, source intervention passed at
  1.108×/1.242×, first cumulative prefix pass L27.
- `REPORT`: behavior 100%/100%, source intervention passed at
  1.148×/1.134×, first pass L26.
- `GO`: behavior 100%/100%, but the source intervention exceeded the frozen
  ratio gate at 1.420×/1.634×. Full L22-L27 clamping removed 99.9% and
  restored both endpoints, but cannot upgrade the ineligible source arm.

Artifact SHA-256:
`AC6B1AD9BD758C6F9328444F56C4B8F244CE0A3F920240209EECDA84F6CA14ED`.

The result falsifies a surface-invariant cognitive-complexity ladder. It does
not falsify query-dependent consolidation: the curves still differ and the
full late-attention path controls every endpoint. The leading alternatives
are prompt/command-prefix effects and answer-margin calibration rather than
operation semantics alone.

Decision: do not describe belief<Ltell<search as general. Run a crossed
semantic-versus-command diagnostic: hold the teacher-forced answer command
constant while varying belief/tell/search questions, then hold the question
constant while varying BELIEF/TELL/SEARCH command prefixes. This directly
tests whether the original depth ordering was semantic or lexically induced.

## 2026-07-24 - semantic/command diagnostic isolates a command-prefix effect

The frozen two-panel diagnostic completed with `COMMAND_PREFIX_EFFECT`. All
six tasks passed the behavior and bidirectional L21 source-sufficiency gates.

When the teacher-forced answer command was fixed to `ANSWER`, all three
questions first passed at the same cumulative prefix:

- belief question: L26;
- tell question: L26;
- search question: L26.

When the belief question was fixed and only the answer command changed:

- `BELIEF`: L24;
- `TELL`: L26;
- `SEARCH`: L26.

Artifact SHA-256:
`9910AE53B4889AB963626DB95C0D56AFE6F16B33835961C806E14C959F1E5CEE`.

This decisively rejects the interpretation that the original
belief < tell < search ordering measured semantic operation complexity. The
early belief transition follows the teacher-forced answer contract in this
diagnostic. The supported description is output-contract-conditioned causal
consolidation depth. This is also a methodological warning: readout-patching
results can look operation-specific when the manipulated factor is the
forced answer prefix.

Decision: freeze and run a complete 3-by-3 crossing of the three questions
with the three `BELIEF`/`TELL`/`SEARCH` commands. Column-wise invariance would
show that the command controls depth across semantic questions; deviations
would quantify question-command interaction. Do not restore the semantic
complexity claim.

## 2026-07-24 - crossed factorial reveals a structured readout interaction

The frozen 3-by-3 question-command factorial completed with
`MIXED_OR_INTERACTION`. All nine cells passed the behavior and bidirectional
L21 source-sufficiency gates. First passing cumulative prefixes were:

| question | `BELIEF` | `TELL` | `SEARCH` |
|---|---:|---:|---:|
| belief | L24 | L26 | L26 |
| tell | L26 | L26 | L26 |
| search | L24 | L26 | L27 |

Artifact SHA-256:
`45458167C3D885A1A9200909D33E0A7A033709E3501F6BBB86575F0F1DAF088B`.

The categorical interaction is real under the frozen gates, but much of it
is threshold-driven. At L24, the mean minimum mediated fraction by command
was 0.720 for `BELIEF`, 0.634 for `TELL`, and 0.624 for `SEARCH`; the ranges
across questions were only 0.690-0.739, 0.621-0.643, and 0.612-0.633.
Conversely, mean L24 mediation by question, averaging commands, was nearly
flat: 0.659 belief, 0.653 tell, and 0.665 search. The search-search L27 label
arose because L26 endpoint restoration remained below gate despite 0.779/0.827
effect mediation.

The command labels also change token budget systematically. Within every
question, the readout position for `TELL` is two tokens later than `SEARCH`,
and `BELIEF` is four tokens later than `SEARCH`. Because the command appears
in both the user formatting instruction and the teacher-forced answer prefix,
the original L24/L26/L27 diagonal may reflect extra token-position computation
rather than semantic operation depth.

Decision: freeze a neutral repeated-token length ladder while holding the
question and terminal command fixed. Test whether added `X` prefix tokens
monotonically increase early-layer mediation and move the first passing depth
earlier. This is a test of token-budget/depth substitution, not yet a claim
about autoregressive answer-time computation.

## 2026-07-24 - neutral length ladder falsifies token-budget substitution

The frozen repeated-`X` ladder completed with `NO_MONOTONE_LENGTH_EFFECT`.
All ten cells passed the behavior and L21 source-sufficiency gates, and each
added `X` increased the readout position by exactly two tokens as intended.

For the belief question, L24 minimum mediation across zero through four
fillers was 0.612, 0.638, 0.621, 0.718, and 0.693; first-pass depths were
L26, L26, L26, L24, and L25. For the search question, L24 mediation was
0.633, 0.631, 0.649, 0.628, and 0.619; first-pass depths were L27, L27,
L26, L26, and L27.

Artifact SHA-256:
`3F3EC4DBB9DC044BDA7EE3DED2981E59F2F6E0646B0DE9EDBE0FB7C26AE97B50`.

The search curve is essentially flat and the belief curve is nonmonotonic.
The categorical changes follow endpoint calibration rather than a graded
token-count effect. Extra neutral token budget therefore does not explain the
original command-conditioned curves.

The tokenizer geometry now enables an exact position-matched comparison:
`BELIEF`, `X TELL`, and `X X SEARCH` terminate at the same readout position.
Existing separate runs already show `BELIEF` at L24 versus `X X SEARCH` at
L26 for the belief question despite identical readout position, but this was
not a preregistered within-run contrast.

Decision: freeze a full question-by-position-matched-label factorial using
those three contracts. A surviving label effect establishes readout-label
conditioning beyond sequence length; it does not yet distinguish lexical
identity from learned label meaning.

## 2026-07-24 - output-label effect survives exact position matching

The frozen position-matched factorial completed with
`POSITION_MATCHED_LABEL_EFFECT`. All nine cells had 100% CLEAN/NATURAL
behavior, passed bidirectional L21 source sufficiency, and matched readout
positions exactly within each question.

First passing prefixes were:

| question | `BELIEF` | `X TELL` | `X X SEARCH` |
|---|---:|---:|---:|
| belief | L24 | L26 | L26 |
| tell | L26 | L26 | L26 |
| search | L24 | L26 | L26 |

Mean L24 minimum mediation across questions was 0.720 for `BELIEF`, 0.666 for
`X TELL`, and 0.617 for `X X SEARCH`. The monotone label ordering survives
the exact token-position control and is visible in the continuous causal
measure, not only thresholded first-pass depths.

Artifact SHA-256:
`8FDB985FCDAA60B5B29CDBDBAFF4B4053FE6068E8A9C325F264324EC8E710441`.

Supported conclusion: causal consolidation at the answer readout is
conditioned by the output label beyond question semantics and sequence
length. The label can change how much of an L21 world-state intervention is
mediated through late readout attention, even when the requested latent value,
behavior, and readout position are fixed.

Open alternative: the effect may reflect arbitrary token identity rather than
learned label meaning. Decision: run a position-matched 3-by-3 codebook
factorial with arbitrary terminal labels `ALPHA`/`BETA`/`GAMMA` independently
crossed with definitions `BELIEF`/`TELL`/`SEARCH`. Compare meaning and surface
main effects before using semantic language.

## 2026-07-24 - in-context code definitions do not transfer routing

The frozen arbitrary-label codebook completed with `NO_CODE_FACTOR_EFFECT`.
All nine cells passed behavior and bidirectional L21 source-sufficiency gates,
and tokenizer-only padding matched every readout position at 210.

Mean L24 mediation by explicitly defined meaning was:

- `BELIEF`: 0.690;
- `TELL`: 0.679;
- `SEARCH`: 0.660.

The meaning range was 0.030, below the frozen 0.05 criterion. Mean mediation
by arbitrary surface label was 0.669 for `ALPHA`, 0.682 for `BETA`, and 0.678
for `GAMMA`, a range of only 0.013. First-pass depths were almost uniformly
L26, with two BETA cells at L25.

Artifact SHA-256:
`80906DED3896C9FC69B609D4FB86D3362B6C40635C70CA6A2AD43EA47345159B`.

The positive position-matched effect therefore does not arise from any label
assigned a BELIEF/TELL/SEARCH meaning in one prompt. It is tied to properties
of the pretrained lexical labels themselves. The remaining distinction is
between a semantic lexical class and idiosyncratic token identity.

Decision: run a position-matched lexical generalization panel with three
pretrained labels per class: BELIEF/THINK/KNOW, TELL/REPORT/SAY, and
SEARCH/FIND/LOOK. Require the ordering to survive after excluding the three
original anchor words before claiming semantic-class generalization.

## 2026-07-24 - pretrained lexical routing generalizes to held-out synonyms

The frozen synonym panel completed with `LEXICAL_CLASS_GENERALIZATION`. All
nine tasks passed behavior and bidirectional L21 source-sufficiency gates, and
all readout positions were matched exactly at 197.

Mean L24 minimum mediation by class was:

- epistemic (BELIEF/THINK/KNOW): 0.716;
- communication (TELL/REPORT/SAY): 0.702;
- search/action (SEARCH/FIND/LOOK): 0.648.

The epistemic-minus-search difference was 0.068. After excluding the original
anchor labels BELIEF/TELL/SEARCH, held-out epistemic synonyms averaged 0.709
and held-out search synonyms averaged 0.661, preserving a 0.048 difference
above the frozen 0.03 gate. Individually, KNOW reproduced BELIEF's L24 pass;
REPORT and SAY also passed at L24; all three search/action labels first passed
at L26.

Artifact SHA-256:
`43E4DABD89C81A7E5B4DF23F53FEF3BB69BA1118AE580BD7CFFF8914D6E1AE84`.

The supported phenomenon is now a pretrained lexical-class modulation of
causal readout, not a single-token, position-length, question-semantic, or
in-context-codebook artifact. The clearest separation is epistemic/
communicative reporting versus search/action, rather than three sharply
separated classes.

Decision: attempt a held-out bidirectional causal switch. Estimate the mean
L21 readout-state contrast BELIEF minus position-matched SEARCH on 15 donor
worlds (clean and natural states), then add it to SEARCH and subtract it from
BELIEF on 15 disjoint worlds. Require continuous L24 mediation to move at
least halfway across the original gap in both directions while preserving
behavior and source sufficiency. A positive discovery must then receive a
sign-flip/null confirmation battery.

## 2026-07-24 - mean lexical control is necessary but not sufficient

The frozen held-out switch completed with `ASYMMETRIC_ROUTE_SWITCH`. All four
original/transformed contexts retained 100% CLEAN/NATURAL accuracy and passed
bidirectional L21 source sufficiency.

On the 15 held-out worlds:

- original BELIEF: L24 mediation 0.736, first pass L24;
- original SEARCH: 0.620, first pass L26;
- BELIEF minus the donor control: 0.650, first pass L26;
- SEARCH plus the donor control: 0.628, first pass L26.

Subtracting the control removed 0.087 mediation, closed about 75% of the
original 0.116 gap, and changed the categorical route exactly as predicted.
Adding the same vector to SEARCH increased mediation by only 0.007 and did
not change depth.

The donor contrast itself was highly stable: mean row-to-control cosine 0.986,
minimum 0.970, across 30 clean/natural donor differences. Its L2 norm was
97.73 versus a mean row-difference norm of 99.10.

Artifact SHA-256:
`F5BA037E4A911B9EE42B59ADA3329AE67093BE314C8A33245CA4A2A6E22A3902`.

Supported conclusion: the pretrained lexical component is causally necessary
for BELIEF's early route, but a single mean additive vector is not sufficient
to install that route in SEARCH. The asymmetry is consistent with nonlinear
or distributed control, an interaction with the remaining SEARCH context, or
routing established before the patched L21 readout site.

Decision: perform exact paired L21 readout-state transplantation on held-out
worlds. Each SEARCH state receives the BELIEF state from the same world and
same clean/natural arm, and vice versa. This preserves content while replacing
the full label-conditioned state. Bidirectional success establishes local
sufficiency; another asymmetric result localizes the missing cause upstream
or outside the single readout position.

## 2026-07-24 - full final-position state remains only necessary

The exact paired transplant completed with
`ASYMMETRIC_PAIRED_TRANSPLANT`. All original and transformed contexts retained
100% CLEAN/NATURAL behavior and passed bidirectional L21 source sufficiency.

- original BELIEF: L24 mediation 0.736, first pass L24;
- original SEARCH: 0.620, first pass L26;
- BELIEF receiving paired SEARCH state: 0.669, first pass L26;
- SEARCH receiving paired BELIEF state: 0.640, first pass L26.

Replacing the complete final-position L21 state removed 0.068 from BELIEF,
closed 58% of the original gap, and changed its route to L26. The reverse
transplant gained only 0.019 and did not install the early route.

Artifact SHA-256:
`02B038383B4FBE4B1722FCCABC8E284D1119797B6C31DAF4D1FC9C94855620BE`.

This rules out failure of the mean-vector approximation as the sole cause of
asymmetry. The final L21 readout state is necessary for maintaining the early
epistemic route but is not sufficient to create it in a SEARCH trajectory.
The missing positive cause must be established earlier, distributed across
other label-token positions, or path-dependent.

Decision: at L21, transplant the full residual states only at every sequence
position whose input token differs between the exactly length-matched BELIEF
and SEARCH prompts. Preserve all identical world, state-marker, question, and
content positions. Bidirectional success localizes a distributed lexical
control state; continued asymmetry redirects to an earlier-layer commitment
sweep.

Protocol refinement before GPU output: evaluate the instruction-label
occurrence, answer-prefix occurrence, and all differing positions separately.
Add 19 fixed-seed same-cardinality random subsets of token-identical
post-marker positions and require an add-one empirical p-value at most 0.05
for the all-position bidirectional score. Do not call a positive result a
unified control state; reserve that claim for low-rank reconstruction and
cross-synonym/world generalization.

## 2026-07-24 - specific bidirectional switch localizes to answer prefix

The controlled distributed-position experiment completed with
`SPECIFIC_DISTRIBUTED_LABEL_SWITCH`.

The six BELIEF-versus-SEARCH token differences formed exactly two contiguous
L21 position groups:

- user formatting instruction: positions 172-174;
- teacher-forced answer prefix: positions 195-197.

All original and transformed contexts retained 100% CLEAN/NATURAL behavior
and passed bidirectional L21 source sufficiency.

Original routes:

- BELIEF: L24 mediation 0.736, first pass L24;
- SEARCH: 0.620, first pass L26.

Instruction-label positions alone did not switch either direction:

- BELIEF receiving SEARCH instruction states: 0.740, L24;
- SEARCH receiving BELIEF instruction states: 0.601, L26.

Answer-prefix positions alone produced a complete bidirectional switch:

- BELIEF receiving SEARCH prefix states: 0.604, L26;
- SEARCH receiving BELIEF prefix states: 0.724, L24.

All six differing positions also switched bidirectionally:

- BELIEF to SEARCH: 0.631, L26;
- SEARCH to BELIEF: 0.721, L24.

The all-position bidirectional score was 0.100. Across 19 fixed-seed,
same-cardinality random subsets of token-identical post-marker positions,
scores ranged from -0.011 to 0.002 (mean -0.004). No null exceeded the
selected score; add-one empirical p = 0.05.

Artifact SHA-256:
`2307B2C71FED5FB6DD2818BE10E0D93AD7E7A6985BC142D89E5C59D70D322E73`.

Supported discovery: while the L21 state marker preserves the same
address-specific world content, the L21 states over the generated answer
prefix form a specific bidirectional causal interface that selects whether
late attention follows an epistemic or search/action readout route. The user
instruction occurrence is not sufficient; the route is selected on the
answer-prefix trajectory itself.

This result establishes a specific distributed lexical effect over three
prefix positions, not yet one coherent low-dimensional control
representation. Decision: learn a donor-only PCA/SVD basis of paired
BELIEF-minus-SEARCH answer-prefix differences and test held-out
reconstructions at fixed ranks. Then require cross-synonym and cross-model
generalization before a broad mechanism claim.

## 2026-07-24 - Qwen-14B shows the predicted route signal but fails eligibility

The preregistered cross-scale run completed with
`BEHAVIORALLY_INELIGIBLE`, so it is not a formal replication. The untouched
Qwen-14B SEARCH baseline retained 100% clean accuracy but only 46.7% natural
accuracy (7/15); its L32 source intervention was therefore also insufficient.
No post-hoc world filtering will be used to relabel this run confirmatory.

Conditional on the continuous route statistic, the predicted pattern was
nevertheless unusually exact:

- original BELIEF: L41 mediation 0.756, first pass L41;
- original SEARCH: L41 mediation 0.605, no eligible passing checkpoint;
- BELIEF receiving SEARCH answer-prefix states: 0.662, first pass L44;
- SEARCH receiving BELIEF answer-prefix states: 0.719, first pass L42.

Thus the answer-prefix transplant moved both directions by 0.095 and 0.113,
closing more than half of the preregistered 0.151 gap in both directions.
Zero of 19 cardinality-matched random position sets reached its bidirectional
score of 0.095 (add-one empirical p = 0.05). Instruction-only states moved
neither direction. The answer-prefix SEARCH-to-BELIEF transplant also raised
the otherwise failing SEARCH natural behavior from 46.7% to 100%, which is
interesting but makes competence rescue an alternative explanation rather
than evidence to ignore.

Artifact SHA-256:
`389B9CAE014D53B00CF6D51CBA3ED64D3B2FFD2D6B2A87CF2AC908C63688940A`.

Supported conclusion: Qwen-14B contains an exploratory, specific
answer-prefix route-switching signal consistent with Qwen-7B, but the frozen
behavior gate prevents a cross-scale mechanism claim. This remains a
breakthrough candidate, not a breakthrough.

Decision: do not spend the next run rescuing Qwen-14B post hoc. Attempt a
preregistered cross-architecture replication on the already available
Mistral-7B model, retaining the full behavior/source gates and 19 matched
random controls. Synonym transfer and low-rank compression remain downstream
of a behaviorally eligible second-model result.

## 2026-07-24 - Qwen-14B ineligibility is prefix-conditioned instability

The frozen behavior-only screen reproduced the first run exactly on two
disjoint 15-world halves:

- BELIEF clean: 30/30;
- BELIEF natural: 30/30;
- SEARCH-prefix clean: 30/30;
- SEARCH-prefix natural: 14/30, exactly 7/15 in each half.

This is not a failure of the natural belief question. The diagnostic keeps
that question fixed and teacher-forces an artificial, position-matched
`X X SEARCH` answer prefix before the location decision. Mean gold-versus-best
other location margins were +18.28 for BELIEF clean, +14.16 for BELIEF
natural, +16.72 for SEARCH-prefix clean, and -0.06 for SEARCH-prefix natural.
Every one of the 16 SEARCH-prefix natural failures predicted the stale clean
answer `Paris` instead of the counterfactual answer `Rome`.

The failure also depended on causally irrelevant truth context: all five
worlds with truth-cube `Tokyo` and all five with `Miami` passed; all five each
with `Cairo`, `Delhi`, and `Oslo` failed; `Lima` was mixed. This is evidence
that the forced SEARCH prefix exposes a fragile, distractor-sensitive access
trajectory, not that Qwen-14B lacks the underlying belief state.

Only seven fresh candidate rows passed all four frozen behavioral cells,
below the preregistered minimum of eight. Verdict:
`ELIGIBLE_BUCKET_TOO_SMALL`. The threshold will not be lowered and those seven
will not be relabeled a confirmation.

Behavior-screen artifact SHA-256:
`6A25939A56CF9481BA315368ECC66CA36121EB5A3E07B6FF058A0652A11E6E1A`.

Interpretive consequence: the answer-prefix transplant's restoration of
SEARCH-prefix natural behavior to 100% is consistent with installing the
BELIEF access trajectory, but it is also a competence rescue. The first
Qwen-14B run therefore remains strong exploratory mechanism evidence rather
than an eligible replication.

## 2026-07-25 - transfer-first program

The Paper-2 program is now ordered by the central falsification question:
does the candidate mechanism transfer beyond the exact prompt and task on
which it was discovered?

Frozen priority:

1. causal synonym transfer;
2. cross-domain transfer to a new state family;
3. donor-only low-rank reconstruction;
4. one additional architecture, only after a behavior-only eligibility gate.

The first test patches complete L21 answer-prefix states from anchor BELIEF
and SEARCH contexts into held-out THINK/KNOW and FIND/LOOK contexts. It uses
worlds 15-29, same-class and instruction-window controls, and 19
three-position token-identical random controls. No synonym state is used to
fit the transferred controller. Protocol SHA-256:
`077C94073F835670606A2A31FD0CCC05EFC3CEE711772BC1F350F6B764E5CCB5`.

## 2026-07-25 - causal route control transfers to unseen synonyms

The preregistered synonym experiment completed with
`CROSS_SYNONYM_ROUTE_TRANSFER`. Every original, selected, same-class,
instruction, and random context retained 100% clean/natural behavior and
passed bidirectional L21 state-marker sufficiency.

Opposite-class anchor answer-prefix states moved all four unseen labels in the
predicted direction:

- THINK receiving SEARCH states: L24 0.684 -> 0.607, movement 0.077,
  first pass L26 -> L26;
- KNOW receiving SEARCH states: 0.729 -> 0.610, movement 0.119,
  L24 -> L26;
- FIND receiving BELIEF states: 0.646 -> 0.725, movement 0.079,
  L26 -> L24;
- LOOK receiving BELIEF states: 0.683 -> 0.716, movement 0.033,
  L26 -> L24.

The epistemic mean movement was 0.098 and search mean movement 0.056; the
primary bidirectional score was 0.0558. All four label movements had the
predicted sign in all 15/15 worlds. Per-world movement ranges were:

- THINK: 0.037 to 0.120;
- KNOW: 0.085 to 0.201;
- FIND: 0.027 to 0.103;
- LOOK: 0.007 to 0.061.

The answer-prefix intervention ranked first of 20. Across 19 matched
three-position random sets, the largest primary score was 0.0051 and the mean
was -0.0020; zero exceeded the selected score, add-one p = 0.05.

Same-class anchors did not mimic the switch (primary score -0.069), and
instruction-window opposite anchors also failed in the predicted direction
(-0.039). This localizes transfer to the answer-prefix state rather than
generic donor-state mass or instruction semantics.

Artifact SHA-256:
`777012D2928CCD316EB51145385E03E8E9A2105967BECBE5DE0AECC16346781C`.

Supported conclusion: complete answer-prefix states derived from the original
BELIEF/SEARCH anchors causally reconfigure unseen THINK/KNOW/FIND/LOOK
contexts. The candidate mechanism therefore transfers across pretrained
lexical realizations and is not specific to the discovery labels. It remains
unproven whether the transferable state is domain-general or compressible.

Decision: next test cross-domain transfer on a newly frozen state family.
Only after domain transfer should donor-only low-rank reconstruction test for
a compact controller.

## 2026-07-25 - a content-cancelled controller switches held-out routes

The preregistered donor-averaged experiment completed with
`CONTENT_CANCELLED_PREFIX_CONTROLLER`. The controller was constructed only
from worlds 0-14 as the mean L21 three-position answer-prefix difference
`BELIEF - SEARCH`, pooling clean and natural histories. It was then applied
without fitting, scaling, or selection to worlds 15-29.

All four original and selected contexts retained 100% clean/natural answer
accuracy and passed the frozen bidirectional L21 source-site sufficiency
gate. At the preregistered L24 checkpoint:

- original BELIEF route mediation: 0.7363, first pass L24;
- original SEARCH route mediation: 0.6204, first pass L26;
- BELIEF minus the fixed controller: 0.5919, first pass L26;
- SEARCH plus the fixed controller: 0.7157, first pass L24.

The BELIEF-to-SEARCH movement was 0.1444 and the SEARCH-to-BELIEF movement
was 0.0953, for a primary bidirectional score of 0.0953. Both categorical
handoff depths switched. Every one of 15/15 held-out worlds moved in the
predicted direction in both arms; per-world ranges were 0.1247-0.1687 and
0.0618-0.1486 respectively.

The instruction-locus control failed bidirectionally (primary -0.0122).
All 19 norm-matched random directions failed to move SEARCH toward BELIEF;
their largest raw bidirectional score was -0.0307. Seventeen remained fully
functional and two disrupted a behavioral/source gate. The selected
controller ranked first of 20 under the frozen functional statistic,
add-one p = 0.05.

All 19 matched three-position random loci remained functional. Their largest
bidirectional score was -0.00063 and none exceeded the selected controller,
add-one p = 0.05.

The 30 donor arm-world differences were exceptionally aligned with their
mean: mean cosine 0.9890, minimum 0.9840, positive fraction 1.0. This was
reported but not used for fitting or selection. The fixed controller had
Frobenius norm 183.20 and per-position norms 108.44, 110.68, and 97.73.

Protocol SHA-256:
`2CA54C50FBAAF94B2A4F63AB81CD81F28637E6347896F0726BEF9F7FC3CCDDFB`.

Result artifact SHA-256:
`43C0FC4B5EA7927C16F4FF000DFB28E8518788E787B8C522D902698A657E3F7B`.

Controller artifact SHA-256:
`C59EA1539F4F3E63B953470A0EE94CEDCDC84D288C8C1DBB49EB2C19B5C91B71`.

Supported conclusion: the paired switch was not merely carrying
world-specific content. A single fixed, donor-only answer-prefix displacement
causally and bidirectionally reconfigures the route used to access unchanged
state on held-out worlds, with categorical handoff-depth changes and two
independent specificity controls.

Interpretation boundary: this establishes a content-cancelled, transferable
controller candidate across worlds and lexical realizations. It does not yet
establish domain generality or a low-rank subspace.

Decision: the cross-domain ownership/color/key-value experiment is unlocked.
It must reuse this exact donor construction without refitting from any target
domain.

## 2026-07-25 - raw controller does not generalize universally across domains

The preregistered cross-domain experiment completed with
`CROSS_DOMAIN_BEHAVIORALLY_INELIGIBLE`. This is a scientific negative, not an
implementation or provenance failure:

- all three target domains passed token alignment;
- the location-derived controller reconstructed bit-for-bit identically to
  the prior artifact;
- controller SHA-256:
  `C59EA1539F4F3E63B953470A0EE94CEDCDC84D288C8C1DBB49EB2C19B5C91B71`;
- no target-domain activation or output was used to fit or modify it.

### Ownership

The ownership task had a strong original L24 route gap (0.1128). BELIEF was
fully behaviorally eligible, but the forced SEARCH natural baseline was only
10/15 correct (66.7%) and its source intervention was therefore insufficient.
The domain was formally `BEHAVIORALLY_INELIGIBLE`.

The intervention remains mechanistically suggestive: the unchanged location
controller moved BELIEF toward SEARCH by 0.1126 and SEARCH toward BELIEF by
0.0421, with predicted signs in all 15/15 worlds. It changed first-passing
depth from L27 to L27 in the first direction and from unresolved to L27 in the
second. Both patched contexts reached 100% clean/natural accuracy and passed
source sufficiency. This is a competence rescue inside an ineligible original
contrast, not a confirmation.

### Color state

Color state was the cleanest falsification. All four original and patched
contexts retained 100% clean/natural accuracy and passed source sufficiency.
The original L24 gap was 0.0593. The controller moved BELIEF toward SEARCH by
only 0.0336 and moved SEARCH in the wrong direction by -0.0297. Only 4/15
worlds moved bidirectionally as predicted; first-passing depth stayed L26 in
all four contexts. Verdict: `NO_CROSS_DOMAIN_TRANSFER`.

### Key-value memory

All four key-value contexts retained 100% behavior and source sufficiency, but
the original route ordering was absent: BELIEF L24 mediation was 0.8475 and
SEARCH was 0.8869, a gap of -0.0394, with both first passing at L23. The
controller moved both relevant comparisons in the wrong direction
(-0.0106 and -0.1159). Verdict: `ORIGINAL_GAP_ABSENT`.

Because fewer than two domains had an eligible original route contrast, the
pooled random-control p-values are not inferentially meaningful and will not
be used to characterize specificity.

Protocol SHA-256:
`D72BE5C785C0E8169B98B5ACCB2CD155621EA3E45F82F18592CFB83B2F34112D`.

Result artifact SHA-256:
`A10087848E6754AEBF206CDB49BAED1078C850DFEA7F988A1AA93B1066563D82`.

Supported conclusion: the answer-prefix controller is not a universal raw
direction that can be added unchanged across arbitrary state and answer
spaces. The synonym result and held-out-world result remain valid, but their
scope is now bounded.

New mechanistic hypothesis: access control may be a shared computation
expressed in domain- or answer-manifold-specific coordinates. The ownership
rescue, combined with clean color failure and absent key-value route gap,
predicts that separately estimated domain controllers may work within their
own domains while failing under raw cross-domain addition. A donor-only
cross-controller matrix is the next kill test; low-rank compression of the
location controller alone is deferred until this coordinate-frame hypothesis
is tested.

## 2026-07-25 - controller matrix rejects separate coordinate frames

The preregistered donor-only controller matrix completed with the formal
verdict `LOCATION_SPECIFIC_CONTROLLER`. The coordinate-frame hypothesis did
not pass its strict all-world gate, but the full result reveals a stronger
shared-component hypothesis.

All controllers used fresh donor-only construction. Ownership/color used 30
shift-3 through shift-6 source-target pairs disjoint from the earlier
shift-1/shift-2 experiment; rows 0-14 were donors and 15-29 evaluation. The
location controller reproduced bit-for-bit with SHA-256
`C59EA1539F4F3E63B953470A0EE94CEDCDC84D288C8C1DBB49EB2C19B5C91B71`.

Original evaluation gates:

- location: eligible;
- color state: eligible;
- ownership: behaviorally ineligible because the original forced SEARCH
  context remained 73.3% clean and 66.7% natural.

### Within-domain location

The confirmed location controller replicated exactly:

- BELIEF-to-SEARCH movement 0.1444;
- SEARCH-to-BELIEF movement 0.0953;
- categorical depths L24/L26 -> L26/L24;
- predicted signs in 15/15 worlds;
- instruction primary -0.0122;
- 19 norm-matched directions: zero exceedances, p = 0.05;
- 19 matched positions: zero exceedances, p = 0.05.

### Within-domain color

The independently estimated color controller produced a strong near-pass:

- original L24 gap 0.0625;
- BELIEF-to-SEARCH movement 0.0813;
- SEARCH-to-BELIEF movement 0.0524;
- 100% clean/natural behavior and source sufficiency in all contexts;
- instruction primary -0.0321;
- 19 norm-matched directions: zero exceedances, p = 0.05;
- 19 matched positions: zero exceedances, p = 0.05.

Fourteen of 15 held-out worlds moved bidirectionally as predicted. The sole
failure was one SEARCH-direction movement of -0.00279; its BELIEF direction
remained positive. Because the frozen gate required 15/15, the cell verdict
was `NONUNIFORM_CROSS_DOMAIN_EFFECT` and formal specificity was not awarded.
No threshold was changed after inspection.

### Cross-domain asymmetry

The location controller again failed in held-out color:

- movements +0.0319 and -0.0333;
- verdict `NO_CROSS_DOMAIN_TRANSFER`.

Unexpectedly, the color-derived controller transferred back to held-out
location:

- movements +0.1538 and +0.0599;
- predicted signs in 15/15 worlds;
- verdict `CROSS_DOMAIN_ROUTE_SWITCH`.

Thus the data do not support independent rotated coordinate frames. Transfer
is asymmetric: the color controller contains a component sufficient for the
location route, while the location controller lacks something required for
color.

### Controller geometry

The three flattened donor controllers were highly aligned:

- location-ownership cosine 0.845;
- location-color cosine 0.855;
- ownership-color cosine 0.877.

Per-position location-color cosines were 0.933, 0.835, and 0.781. Singular
values were 299.19, 72.66, and 63.28; one component explained 90.6% of total
controller energy, two explained 95.9%.

Protocol SHA-256:
`CE32CBE7FBCBEA73570A34892523267FBD82FBE7C6C04CE708C4269B0CD6B3E2`.

Result artifact SHA-256:
`5BE6ED4284061154F3DF47C6B07C42B5D127B47DB7A22598769BF8BF96F5D800`.

Controller archive SHA-256:
`9C858A0CEEE74ECD2F0AC38D0AA2138B212C047BF2F44BB7A10146EE7D52DF55`.

Supported conclusion: there is no evidence for three unrelated domain
coordinate systems. Instead, donor controllers share a dominant direction,
and causal transfer is target-asymmetric. Color provides a controlled
near-replication of within-domain access control plus successful transfer
into location.

Next prediction: a donor-only shared component constructed without the target
domain should outperform an individual location controller on held-out color.
The clean test is leave-one-domain-out rank-one reconstruction or controller
averaging, with no evaluation fitting. This is now higher priority than
compressing the location controller alone.

## 2026-07-25: leave-color-out shared component — clean falsification

The prospective leave-color-out experiment completed successfully on a Tesla
T4. The primary PC1 was constructed only from the frozen location and ownership
donor controllers. Color activations were excluded. Evaluation used 30 new
color rows whose 30 clean and 30 natural rendered histories were all unique
and absent from the earlier color runs.

Formal verdict: `NO_LEAVE_COLOR_OUT_TRANSFER`.

The result is cleanly interpretable:

- target original gate: `ELIGIBLE`;
- original BELIEF/SEARCH gap: 0.06092;
- all original and patched tasks retained 100% clean/natural accuracy and
  source-intervention sufficiency;
- the independently learned color oracle passed on the fresh rows:
  movements +0.08631/+0.05226, predicted signs 30/30 and 27/30,
  exact one-sided sign p = 9.31e-10 and 4.22e-6;
- the leave-color-out PC1 failed:
  movements +0.01034/-0.03397, signs 17/30 and 6/30;
- the simple donor mean failed similarly:
  movements +0.00936/-0.03241, signs 17/30 and 5/30;
- location alone failed at +0.03455/-0.03197, signs 24/30 and 5/30;
- ownership alone failed at +0.00229/-0.03808, signs 16/30 and 4/30.

The failure therefore cannot be attributed to an unstable color task, weak
behavior, a missing natural effect, or the earlier 15-world all-sign gate. It
falsifies the prediction that the dominant cross-domain PC1 is itself a
domain-general causal route controller.

Geometry explains why similarity was misleading. The donor-only PC1 explains
92.27% of location/ownership energy and has cosine 0.9013 with the color
oracle, yet it is not causally substitutable. Projecting the color oracle onto
that PC1 leaves an orthogonal residual with norm 79.05: only 18.77% of color
controller energy, but 31.1%, 47.1%, and 53.2% of the three position norms.
The residual is essentially orthogonal to the donor mean (cosine 0.00044).

This sharpens the working mechanism: a high-energy shared backbone is
insufficient; a lower-energy domain-specific component may gate or orient its
causal use, especially at the later answer-prefix positions. The next decisive
test is a held-out causal decomposition on another fresh color set:
shared projection, residual alone, graded shared-plus-residual reconstruction,
per-position residual additions, and matched orthogonal residual controls.

Protocol SHA-256:
`ACE78DBD4FBB8BFBF9C070571D40B977AF59B7139E80ECB54B5AA1B2D819AA37`.

Result artifact SHA-256:
`BEB208250AFAF817037C1E25E4111B959CC68D26CFE93BDC74A0F1A936B3E7A2`.

Controller archive SHA-256:
`CD3044BB619D5C80B3040E986E2C8344B89E37439AC187FD900DF6566E9899E2`.

## 2026-07-25: shared-adapter decomposition — residual-only surprise

The second fresh color evaluation completed on a Tesla T4. The frozen
top-level verdict was `COLOR_TARGET_UNRESOLVED`: the exact reconstructed color
controller missed the frozen SEARCH-to-BELIEF aggregate threshold by 0.00325
(movement 0.04675 versus 0.05). This verdict is retained unchanged.

The preregistered component arms nevertheless yielded a sharp result:

- shared PC1 failed: movements +0.01060/-0.03449, signs 18/30 and 8/30;
- color projection failed: +0.01290/-0.02747, signs 17/30 and 10/30;
- the orthogonal color residual alone passed: +0.09006/+0.05376,
  signs 30/30 in both directions;
- adding residual doses to the projection produced monotonic scores
  -0.02747, -0.00335, +0.01559, +0.03278, +0.04675.

The residual alone was stronger than the complete color controller. Thus the
frozen compositional shared-backbone-plus-adapter hypothesis was not supported.
Instead, the data suggest that the low-energy residual carries the causal
control while the high-energy shared projection is ineffective or mildly
antagonistic.

The frozen composition controls were specific (19 orthogonal residual and 19
position nulls each had zero exceedances, empirical p = 0.05; instruction
score -0.03455), but they were designed around projection-plus-residual rather
than residual alone. They therefore do not replace the independent
residual-only confirmation.

Position subsets localized the useful residual sharply:

- offsets 0 alone and 1 alone failed;
- offset 2 alone produced +0.07430/+0.04449;
- pairs containing offset 2 produced approximately +0.078--0.080 and
  +0.045--0.048;
- the pair without offset 2 failed.

The third answer-prefix position is therefore the leading causal locus, though
it did not independently cross the frozen 0.05 bidirectional gate.

Protocol SHA-256:
`4FB9E6BAA956CADD68F11F6C05BC88A56ECD17CAED3905BA038C2D508AEB13CB`.

Result artifact SHA-256:
`45BC2EF7BE4445D75CA763C351D6CD7B18A4DF936977B5CC4EF91F71402A8E79`.

Controller archive SHA-256:
`BBAC46AFEFD238578517D9BCF3CA2D5ADE69FD82D4D469BC131D68D570BD5E8C`.

Next: the already frozen residual-only confirmation tests another disjoint
30-world set, signed doses, 39 residual-only direction nulls, 19 residual-only
position nulls, and an instruction control. Its protocol hash is
`74708FBD05FF1A528C07999C7243CAD9AAC0A3FC05A8752BB163BC9578A36C71`.

## 2026-07-25: independent residual-only confirmation — passed

The independent 30-world confirmation completed on a Tesla T4 with formal
verdict `RESIDUAL_ONLY_CAUSAL_CONTROLLER`. The color target was behaviorally
eligible, and the selected residual-only intervention passed every frozen
causal and specificity gate.

The signed dose response was coherent:

- dose -1.00: movements -0.04266/-0.07972, predicted signs 0/30 and 0/30;
- dose -0.50: -0.02747/-0.03494, signs 0/30 and 0/30;
- dose +0.25: +0.01762/+0.01537, signs 30/30 and 30/30;
- dose +0.50: +0.03833/+0.03258, signs 30/30 and 30/30;
- dose +0.75: +0.06162/+0.04618, signs 30/30 and 30/30;
- dose +1.00: +0.08758/+0.05381, signs 30/30 and 29/30, `PASS`;
- dose +1.25: +0.11306/+0.05564, signs 30/30 and 28/30, `PASS`.

The positive-dose bidirectional score was monotonic within the frozen
tolerance, and both negative doses reversed the effect. Specificity also
passed:

- instruction-position score -0.01863;
- 39 orthogonal, norm-matched residual directions: zero exceedances,
  empirical p = 0.025; strongest null score 0.04954 versus selected 0.05381;
- 19 matched identical-token position controls: zero exceedances,
  empirical p = 0.05; strongest null score 0.00145.

This independently confirms that the low-energy component orthogonal to the
dominant cross-domain geometry is a sign-, dose-, and position-specific causal
controller. It does not yet prove that the unmodified model endogenously uses
that coordinate; the next frozen experiment tests midpoint-preserving
equalization of the naturally occurring BELIEF/SEARCH residual contrast.

Protocol SHA-256:
`74708FBD05FF1A528C07999C7243CAD9AAC0A3FC05A8752BB163BC9578A36C71`.

Result artifact SHA-256:
`1CC258D1013F832247BE3AC580D5326C634C5126297B0D6DB5FD8D4EFD16663C`.

Controller archive SHA-256:
`73F0CB1453A3AABCA70DC0A18D7BD0E0DBE27932BA76D0E28960BB080837EF15`.

## 2026-07-25: endogenous residual necessity — frozen before evaluation

The next experiment is frozen before any carrier-template evaluation. It asks
whether the confirmed residual is merely an effective injected steering
direction or an endogenous coordinate the model uses naturally.

On a disjoint carrier template, the experiment first requires additive
residual calibration to replicate. It then equalizes only the naturally
occurring BELIEF/SEARCH contrast projected onto the residual at the three
layer-21 answer-prefix positions. The construction preserves each pair's
midpoint and every orthogonal component. Necessity requires a gap reduction of
at least 0.04, an absolute remaining gap at most 0.02, at least 27/30 positive
per-world reductions, intact behavioral/source/depth gates, and numerical
invariants at 1e-5.

Controls equalize the high-energy projection, the residual at instruction
positions, 19 orthogonal norm-matched directions, and 19 matched
identical-token position sets. Both empirical null gates are p <= 0.05. The
null count was reduced from the draft's 39 to 19 before hashing and before any
new evaluation, solely to reduce runtime while retaining a valid prospective
specificity test.

Protocol SHA-256:
`DC81755570515423FABB3E346C714811B26AACBBB1A93CAB3EF6238935B3B3DF`.

## 2026-07-26: endogenous residual necessity — strong unresolved result

The carrier-template run completed on a Tesla T4. The frozen formal verdict is
`CARRIER_TEMPLATE_UNRESOLVED`, and it must not be reported as a preregistered
necessity pass.

The carrier itself remained behaviorally eligible, with intact source and
depth gates. Additive residual calibration was broad but missed the absolute
bidirectional movement gate:

- BELIEF-to-SEARCH movement +0.07328, signs 30/30;
- SEARCH-to-BELIEF movement +0.04031, signs 29/30;
- frozen calibration verdict `AGGREGATE_MOVEMENT_FAILED` because the second
  arm was below 0.05.

The endogenous equalization result was nevertheless large and highly uniform:

- original BELIEF-minus-SEARCH L24 mediation gap 0.03185;
- equalized gap 0.00681;
- gap reduction 0.02505, or 78.6% of the original gap;
- positive reductions in 29/30 worlds, exact one-sided sign
  p = 2.8871e-8;
- all behavioral, source-sufficiency, and depth gates passed;
- midpoint, residual-coordinate, and orthogonal-component numerical
  invariants all passed at 1e-5.

It failed the frozen aggregate necessity gate because reduction was below the
absolute 0.04 threshold. In hindsight, that absolute criterion is poorly
scaled to an original gap of 0.03185: exact collapse to zero would still
produce only 0.03185 reduction. This design issue is documented rather than
repaired post hoc.

Specificity was mixed:

- 19 orthogonal norm-matched directions: zero exceedances, p = 0.05;
  maximum gap reduction 0.00172;
- 19 matched-position controls: zero exceedances, p = 0.05;
  maximum gap reduction 0.00271;
- instruction-position equalization reduced the gap only 0.00027;
- high-energy projection equalization reduced the gap 0.01474, or 46.3% of
  the original gap and 58.9% of the residual reduction.

Because the projection control exceeded half of the residual reduction, the
frozen `specific` flag is false. The result therefore supports but does not
establish endogenous residual necessity. It suggests a sharper mechanism:
the low-energy residual is sufficient on its own and carries most endogenous
route contrast, while the high-energy shared projection may carry a smaller,
non-null endogenous component despite being ineffective as an injected
controller.

The clean next test must be prospective and scale-free on a new held-out
template or domain. It should quantify fraction of gap removed, compare
residual, projection, and joint equalization in a 2x2 causal decomposition,
and retain direction/position controls. No threshold from this run may be
retroactively changed.

Result artifact SHA-256:
`C91DA6FF371A92BA11996F19A9BEA2196C4667546B56FF0BD4ABE3EC00F9C609`.

Controller archive SHA-256:
`8A994A2243BED14BBA22C1E9C083C8DB1FC31F2C92699321B8EDAA24166757F1`.

## 2026-07-26: controller-to-circuit epistasis — frozen, not launched

The next experiment moves beyond controller confirmation to a mechanistic
controller-to-circuit test. Its fixed causal graph is:

`L21 answer-prefix residual -> L22 gate heads -> frozen L22--L24 transport set`.

Sixty histories absent from all preceding color sets are split before
evaluation into 30 activation-only discovery histories and 30 causal-holdout
histories. On discovery only, candidate layer-22 heads are ranked by whether
the residual makes their source-response resemble the opposite natural route.
No logits, mediation outcomes, or holdout activations enter selection. The top
four positive heads above the median natural-gap norm are frozen.

On holdout, same-prompt blockade replaces those heads in R-steered runs with
their unsteered values. Same-prompt rescue replaces them in unsteered runs
with their R-steered values. Donors are matched by operation, prompt, history
state, and pass type. Each direction must block or recreate at least 50% of
the calibrated residual movement and show the predicted sign in at least
24/30 worlds. Nineteen matched random four-head sets provide separate
blockade and rescue nulls at empirical p <= 0.05.

The inherited residual controller, previously frozen transport set, behavioral
and source gates, and exact layer/token sites are unchanged. A passing result
would establish a controller-to-circuit mechanism rather than merely another
steerable activation direction.

Protocol SHA-256:
`AE975A7F9CC4F6684AF25DF2605D9EBD061063CD44E56426742B781443F697BE`.

## 2026-07-26: controller-circuit protocol v2 after construction failure

Version 1 failed after deterministic controller construction and before any
controller-circuit activation or causal outcome was evaluated. The failure was
purely combinatorial: the protocol requested 60 new histories, requiring 120
unused `(state, d1, d2)` prompt signatures, but the finite eight-color design
did not retain that many signatures after all prior color sets.

The exact v1 protocol is preserved as
`PAPER2_CONTROLLER_CIRCUIT_EPISTASIS_PROTOCOL_V1.md`, SHA-256
`AE975A7F9CC4F6684AF25DF2605D9EBD061063CD44E56426742B781443F697BE`.

Version 2 reuses the already validated residual-only confirmation rows for
activation-only discovery. No prior head activation or circuit endpoint
exists for those rows. The causal holdout is constructed from a deterministic
matching over the remaining unused prompt signatures. Static verification
found:

- 30 discovery and 30 causal-holdout rows;
- 120/120 signatures unique within the experiment;
- 60/60 holdout signatures absent from every preceding color set;
- source, target, first-distractor, and second-distractor marginals each
  balanced to counts of three or four across the eight colors.

No model outcome informed this revision. All controller, head-selection,
blockade, rescue, effect-size, sign, and random-null gates are unchanged.

Version-2 protocol SHA-256:
`B6B0F793E0117C92A0D1D636EDDD0E663F0AE6D00AD23B54ECB2B5FBA5E6B9D2`.

## 2026-07-26: endogenous-controller factorial — frozen, not launched

The four-head epistasis result closed the fixed-head circuit hypothesis. The
next decision experiment returns to the stronger unresolved issue: whether the
low-energy color residual is part of the model's natural route computation or
only an effective injected shortcut.

The new protocol corrects two weaknesses in the earlier necessity test. Its
primary effects are fractions of the observed route gap, avoiding an impossible
absolute reduction threshold. It also treats the three answer-prefix positions
as one structured sequence-level coordinate, matching the scalar dose used by
the validated additive controller; the earlier test independently fitted three
position coefficients.

On 30 newly frozen color histories, the experiment compares additive residual
switching, endogenous projection equalization, endogenous residual
equalization, joint projection-plus-residual equalization and exact natural
answer-prefix interchange. It measures the `P x R` causal interaction and
layer-21--27 convergence toward the natural opposite-command trajectory while
checking source-state and answer invariance.

The complete frozen specification and stop/go rules are in
`PAPER2_ENDOGENOUS_CONTROLLER_FACTORIAL_PROTOCOL.md`. The next stage after a
pass is not another localization experiment: it is a preregistered prospective
predictor of controller success or reversal on held-out contexts, compared
against magnitude, cosine, PCA-energy and linear-accessibility baselines.

## 2026-07-26: endogenous-controller factorial — completed

Kaggle completed successfully on a Tesla T4. The frozen top-level verdict was
`ROUTE_CHANGE_WITHOUT_NATURAL_TRAJECTORY_CONVERGENCE`.

This is a genuine mixed result, not a wording failure.

### Passed

- Untouched behavioral/source gate: `ELIGIBLE`.
- Every intervention preserved behavior and source-intervention sufficiency.
- Additive residual sufficiency passed in both directions, with route movements
  `+0.08643` and `+0.06134`, 30/30 predicted signs in each direction.
- Relative to exact natural answer-prefix interchange, additive `R` produced
  `1.406x` and `1.806x` as much route movement.
- Sequence-level endogenous `R` equalization reversed the natural route gap:
  `+0.03588 -> -0.03511`, a reduction of `0.070996` or `1.978x` the original
  gap, with 30/30 predicted signs (`p=9.31e-10`).
- Joint `P+R` equalization also reversed the gap:
  `+0.03588 -> -0.02658`, 29/30 signs.
- Layer-21 source states were unchanged exactly.
- Instruction-position, orthogonal-direction, and matched-position controls
  changed the gap by only `-0.00069`, `-0.00186`, and `-0.00120`; all moved in
  the wrong aggregate direction relative to `R`.
- All numerical midpoint and coordinate-removal invariants passed below
  `1e-8` coordinate error.

### Failed

- Full residual-stream trajectory convergence failed. At layer 24, the
  additive `R` intervention moved only about `0.10` of the way toward the
  natural opposite-command state. Median distance ratios were approximately
  `0.99` for BELIEF->SEARCH and `1.06` for SEARCH->BELIEF.
- Endogenous `R` equalization strongly reversed the causal route score but did
  not shrink full-state BELIEF--SEARCH distance: the mean layer-24 median
  separation ratio was about `1.017`.
- The aggregate `P x R` interaction was `+0.00703`, or `19.6%` of the original
  gap, but its sign was consistent in only 18/30 worlds
  (`p=0.181`). It therefore failed the frozen interaction criterion.

### Mechanistic reading

The cleanest supported statement is:

> A low-energy, sequence-level answer-prefix coordinate is sufficient and
> endogenously necessary for the tested internal route distinction, while
> remaining largely invisible in full residual-stream trajectory distance.

The high-energy projection is not a second stable controller. Removing `P`
alone enlarged the original gap by `43.4%`, whereas removing `R` reversed it.
The dominant geometry therefore appears antagonistic or buffering on average,
but the non-additive interaction is not stable across individual worlds.

This establishes a stronger endogenous causal coordinate result than the
previous necessity experiment, but it does not establish that `R` recreates
the model's complete natural computation. Full-state Euclidean convergence was
the wrong signature for the successful causal effect, or the intervention uses
a compact/off-manifold shortcut. Those alternatives remain open.

The highest-value next direction is a prospective causal-sensitivity
prediction: use the local downstream response of the route score to predict
which low-energy coordinates will control routing on unseen contexts, then
compare against norm, cosine, PCA energy and linear-accessibility baselines.
This directly tests whether causal geometry, rather than representational
energy, explains controller success.

Result SHA-256:
`5558A16C573B0C3BBE8DDBB9763C6082C93943B68C199DB87EFD207B2F302B27`.

## 2026-07-26: causal-rank spectrum — frozen, not launched

The factorial result established that the color residual is sufficient and
endogenously necessary for the tested route score, but it did not show whether
that residual is unique or belongs to a larger control space.

The new experiment constructs eleven natural, donor-only answer-prefix
controllers: all nine epistemic/search lexical pairings plus ownership and
color. Their uncentered SVD supplies an energy-ordered orthogonal basis without
using any causal evaluation outcome. Held-out location and color worlds then
test cumulative ranks `1,2,3,4,6,11`; color additionally tests axes `1--6`
separately and axes `7--11` as a tail group.

The frozen outcomes distinguish a single shared causal switch, a low-rank
structured causal subspace, and high-rank/domain-specific control. All
directions, coefficients, ranks, thresholds, held-out splits and random
controls are fixed before launch in
`PAPER2_CAUSAL_RANK_SPECTRUM_PROTOCOL.md`.

## 2026-07-26: causal-rank spectrum — completed

Kaggle completed successfully on a Tesla T4. The frozen top-level verdict was
`BEHAVIOR_OR_SOURCE_INELIGIBLE`, but this was caused solely by the third
location random-direction control damaging BELIEF baseline behavior
(`g0_clean=0.533`). It does not invalidate the preregistered natural,
reconstruction, or color arms. The color target and every color arm remained
behaviorally and source eligible.

### Main result

- The eleven natural controllers have only seven numerically nonzero SVD axes.
  Axis 1 contains `66.6%` of representational energy; ranks 1--6 contain
  `98.2%`.
- On held-out location worlds, axis 1 alone recovered `96.7%` of the full
  controller effect (`0.09252` versus `0.09567`), with the predicted
  bidirectional sign in 15/15 worlds.
- On held-out color worlds, the same axis recovered only `2.3%`
  (`0.00106`). Ranks 1--6 together recovered only `35.2%`.
- The axes 7--11 tail recovered `91.7%` of the full color effect
  (`0.04193` versus `0.04571`) with predicted signs in 30/30 worlds. Axes
  8--11 have effectively zero singular value, so this tail effect is
  numerically axis 7.
- The exact natural color transplant scored `0.03396`; the reconstructed
  color controller scored `0.04571`, with predicted signs in 28/30 worlds.

### Mechanistic reading

This rejects a single universal causal-switch interpretation. The dominant
high-energy axis is sufficient for location routing but nearly inert for color,
whereas a low-energy domain-separating axis carries almost all causal control.
The result therefore exposes a dissociation between representational variance
and causal importance: control appears sparse and structured across multiple
axes, with different domains reading different coordinates.

This is a breakthrough candidate, not yet a general theorem. The color
controller participated in constructing the donor-only basis, although all
causal tests used held-out histories. The decisive next test should prospectively
predict the active axis for wholly held-out domains and then intervene on that
axis, comparing causal sensitivity against SVD energy, cosine similarity and
random norm-matched directions. A successful leave-one-domain-out prediction
would turn the present geometry into a transferable mechanism.

Result SHA-256:
`3CE993358B183CA0AE2FDB4C9E874B4BD84E923D570035A020794845705F6352`.

## 2026-07-26: prospective causal sensitivity — frozen, not launched

The causal-rank result ruled out a single universal energy-ranked switch:
location recovered 96.7% of its effect from PC1, while color recovered only
2.3% from PC1 and 91.7% from the numerically axis-7 tail. That result was
retrospective with respect to causal efficacy because color participated in
constructing the basis.

The next experiment is a strict leave-one-domain-out prediction. For each of
four domains, its candidate basis is constructed from natural controllers in
the other three domains only. On five separate target calibration histories,
the selected coordinate is the donor axis with the largest bidirectional
directional derivative of an unchanged layer-24 route readout with respect to
the unchanged layer-21 answer-prefix state. Target steering outcomes are not
used for basis construction or selection.

The complete prediction ranking and SHA-256 are written before target causal
evaluation begins. Fifteen new histories then test the selected coordinate,
every other donor axis, PC1, the lowest-energy axis, the full donor-span
projection, raw donor mean, instruction and matched-position controls, and
three random smoke directions. The experiment separately gates behavior,
natural reference strength, world-level signs, recovery of the best measured
axis, locus specificity and pooled prospective rank correlation.

To avoid a four-to-six-hour screening run, 15 additional histories per domain
are frozen but untouched. Only after a prospective candidate verdict will a
second stage reuse the hashed prediction on those histories with at least 19
norm-matched random directions, supplying the final empirical null. The core
is expected to take 45--75 minutes on T4; confirmation should add 60--90
minutes only when warranted.

The exact mathematics, fixed prompts, splits, thresholds, verdict table and
runtime estimate are in
`PAPER2_PROSPECTIVE_CAUSAL_SENSITIVITY_PROTOCOL.md`. No Kaggle run should be
launched until local static validation and protocol hashing are complete.

Frozen core protocol SHA-256:
`4CBE37E7719FF25144AE74FCCBBC674164D58358456CF615490C18E0270649D6`.

## 2026-07-26: prospective causal sensitivity core — completed

Kaggle completed successfully on a Tesla T4 in approximately 43 minutes. The
frozen verdict was `BEHAVIORALLY_INELIGIBLE`.

This was not a tokenizer, answer-accuracy, source-intervention, Jacobian, or
implementation failure. Every domain had 100% clean/counterfactual answer
accuracy and sufficient bidirectional source interventions. The only original
eligibility failure was the frozen `0.03` route-gap threshold:

- material: `0.023386`;
- animal: `0.029685`, only `0.000315` below threshold.

Color (`0.032212`) and shape (`0.074354`) were eligible. Exact natural
interchange passed in all four domains.

### Prospective result

The non-interventional Jacobian selected PC1 before target steering in all
four leave-one-domain-out folds. PC1 was also the best causally measured axis
in every fold:

- color: score `0.04203`, 14/15 signs;
- material: score `0.02422`, 6/15 signs;
- animal: score `0.04103`, 15/15 signs;
- shape: score `0.03895`, 14/15 signs.

The pooled frozen-sensitivity/measured-effect correlation was Spearman
`0.7832` and Pearson `0.9367`. Every random smoke control was weaker than the
selected axis; instruction and matched-position controls were also specific.
The animal result was substantively strong despite formal original-gap
ineligibility. Material was genuinely weak and nonuniform.

### Mechanistic reading

The result does not support the preregistered low-energy/domain-specific
version of the hypothesis. In every fold, PC1 carried 94.5--95.4% of donor
activation energy, and all coefficient, norm, cosine, probe and Jacobian
baselines selected that same axis. The sensitivity rule therefore predicted
effects accurately but did not outperform ordinary variance.

This does not contradict the earlier location-versus-color axis dissociation.
It shows that the apparent energy rank is basis-ensemble dependent: among four
near-isomorphic private-record templates, the common controller becomes PC1.
Consequently, this core supports a shared controller within that narrow prompt
family, not yet a general causal-control geometry.

The prespecified 19-direction confirmation should not be launched because the
core did not earn a candidate verdict. The next experiment must first secure
behaviorally eligible, structurally heterogeneous computation families through
a cheap frozen screen. Repeating vocabulary substitutions or lowering the
`0.03` gate after seeing the result would be post-hoc and is disallowed.

Result SHA-256:
`F5166316B0EC0AFB26ED77BAA650ECCF7259DA93500DEA41CA85EDC990D49D11`.

## 2026-07-26: heterogeneous-family eligibility screen — frozen

The homogeneous prospective core showed that vocabulary-swapped private-record
tasks all collapse into PC1. Before another causal run, a cheap frozen screen
will test eight different computational structures: private belief,
chronological last-write update, key-value lookup, two-hop pointer traversal,
conditional selection, maximum-score comparison, constraint elimination and
temporal-slot retrieval.

All families share answer vocabulary and counterfactual construction so that
computational structure, rather than output tokens, differs. The screen uses
untouched behavior only: clean/counterfactual answer accuracy, layer-21 source
intervention and the layer-24 route gap. It performs no answer-prefix steering
or axis selection.

The first four passing families in the frozen order will be used downstream.
Every candidate and failure is reported, and prompts cannot be revised after
the screen. Expected T4 runtime is 10--20 minutes after model loading.

Frozen protocol SHA-256:
`8CCDB669C21C686A712E10027E4541DDAFFEB77319CA1BFCD5EC79D6672AB9EB`.

## 2026-07-26: heterogeneous-family screen v1.1 — completed

The first implementation mistakenly compared the signed BELIEF-minus-SEARCH
route difference with `+0.03`. This falsely rejected contrasts with the
opposite orientation. Version 1.1 corrected the classifier to use absolute
magnitude, added signed and magnitude fields, and added a negative-gap
regression test. No prompts, rows, measurements, or threshold were changed.

All eight families passed every behavioral and source-intervention gate:
clean/counterfactual answer accuracy was eligible for both operations and both
source interventions were sufficient. Layer-24 route-gap magnitudes were:

- private belief `0.02934`;
- latest update `0.03058` (formal pass);
- key-value lookup `0.02230`;
- two-hop pointer `0.02905`;
- conditional selection `0.02721`;
- maximum score `0.02967`;
- constraint elimination `0.02113`;
- temporal slot `0.03676` (formal pass).

Formal verdict: `INSUFFICIENT_HETEROGENEOUS_FAMILIES`, with two of eight
passing the frozen `0.03` cutoff. Substantively, six of eight lie in a narrow
`0.02721`--`0.03676` band and four formal misses are within `0.00279` of the
cutoff. With only 15 rows per family, the binary verdict is therefore dominated
by an arbitrary threshold rather than behavioral failure or loss of the
mechanism. The sign also reverses across families, indicating that the
operation-induced route contrast is task-conditioned rather than a universal
ordered scalar.

This result must not be relabelled as a formal pass or rescued by lowering the
frozen threshold. It does motivate a new preregistered, independent-world test
that treats signed route response as a continuous outcome and asks whether
controller geometry prospectively predicts its magnitude and orientation
across heterogeneous computations.

Corrected frozen protocol SHA-256:
`1A7E511A42D104D91A1414E95E50F4DFCCA9F41BB4CA3BA5C3F4CF5215816A62`.

Result SHA-256:
`5A4847912FF2798DCC8B6DADD400A4E6E5E81EB5BC067D5F1497C6326E146D8E`.

## 2026-07-26: cross-family causal-subspace test — frozen

The next experiment does not rescue the heterogeneous screen by changing its
threshold. It asks a different, stronger question with 35 source-target pairs
that were absent from the screen: whether controllers learned from seven
computational families span a compact causal subspace for the eighth.

Each leave-one-family-out fold estimates donor controllers on 10 rows, infers
the held-out family's coordinates and route orientation on 10 separate
non-interventional calibration rows, then freezes a prediction artifact before
any intervention on 15 untouched test rows. The causal rank curve (1, 2, 3,
7), exact interchange, within-family upper bound, donor mean, rank-7
orthogonal residual, instruction-position control and three matched random
directions distinguish a shared compact manifold, a shared higher-rank
infrastructure, task-specific causal residuals and lexical/nonspecific
artifacts.

Unlike the earlier assay, route sign is a prospective family-specific
prediction. Test response is continuous and no test route-gap threshold can
exclude a family. A cheap geometry gate stops the run before the causal phase
if cross-family reconstruction is implausible.

Expected T4 runtime is 4--7 minutes after model load for a stopped geometry
run, or roughly 55--75 minutes if the full causal phase is earned.

Frozen protocol SHA-256:
`FDB053176FC5FCF4D059FAEEB562BAD7822FBB5FFA08DB55E3C3204091F98C2B`.

## 2026-07-26: cross-family causal-subspace test — completed

Formal verdict:
`CAUSAL_ASSAY_OR_ORIENTATION_UNRESOLVED`.

The non-interventional geometry result was strong: median held-out rank-3
reconstruction energy was `0.856`, versus `0.861` at full donor rank 7.
However, this geometry did not predict causal response. The pooled correlation
between reconstruction energy and causal effect was negative (Spearman
`-0.181`, Pearson `-0.199`).

Calibration orientation itself was not the problem. All eight families had a
calibration route-gap magnitude above `0.010`, and all eight retained the same
gap sign on the untouched test rows. The causal reference failed instead:
exact natural answer-prefix interchange and the full within-family controller
were jointly resolved only for `maximum_score`. Most other families showed
one-directional, negligible or sign-reversed route movement even under the
exact interchange. Consequently, neither rank 3 nor rank 7 passed in any
family.

Key adjudication:

- assay resolved: `maximum_score` only;
- rank-3 passes: none;
- rank-7 passes: none;
- task-specific residual dominance: `latest_update` only;
- rank-3 geometry-to-effect correlation: unsupported;
- all tested controller arms remained functionally evaluable, so this is not
  a behavioral-ineligibility result.

The substantive conclusion is negative for the preregistered shared linear
causal-subspace hypothesis. The highly compressible cross-family geometry is
not a reusable bidirectional route controller under this intervention. Given
the identical command tokens and output contract, the dominant shared
variance is likely lexical/template structure, while causal route response is
task-conditioned, asymmetric, nonlinear, or not captured by the layer-24
mediation scalar.

Do not lower thresholds, reinterpret the one-directional effects as a pass, or
launch the reserved cross-wording/random confirmation. The core did not earn
it. Any next Paper-2 hypothesis must explain the dissociation between strong
shared representational geometry and absent shared causal transport.

Geometry artifact SHA-256:
`E9A940A3CDEC8D988FD8956CA0B34D4B370D5D7B1ABFCF39969559E4221B436B`.

Prediction artifact SHA-256:
`5DB21FE63355B18E03CED2EF0320007D6A7E9835810A9A2354101ABE078FF1A9`.

Result SHA-256:
`0C76BC1BE9454C70C24CF93D0EDDC1A23CAA66CEFDA032265A73D1B4C8E8F94D`.

## 2026-07-26: final exact-transplant locus diagnostic — frozen

One final diagnostic will separate three explanations for the failed shared
linear controller: wrong fixed locus, incomplete layer-24 measurement, and
direction-asymmetric/nonlocal control. It compares the previously successful
`maximum_score` family with the one-way `two_hop_pointer` family.

Twelve histories use only the six directed color pairs absent from all prior
50-pair experiments. Exact opposite-operation states are transplanted in both
directions across eight frozen layers and five position groups: answer prefix,
instruction occurrence, all differing positions, and matched three- and
six-position identical-token controls. Strictly downstream layer-24 and
layer-27 state convergence is measured alongside value-answer preservation.
Same-layer comparisons are disallowed.

This result closes the current experiment loop. It will be interpreted under
the frozen diagnostic taxonomy and will not trigger prompt, threshold, layer
or position rescue runs.

Expected T4 runtime is 10--20 minutes after model loading.

Frozen protocol SHA-256:
`C1D9FF28058F80EA51461794BB35BC974E2842CC010708FFC46AFE0A10E0DBEA`.

## 2026-07-26: final exact-transplant locus diagnostic — completed; loop closed

Formal verdict:
`COMPUTATION_DEPENDENT_CAUSAL_LOCUS`.

Both families exhibited broad, bidirectional exact causal state transport on
12 histories built exclusively from previously unused directed pairs.
`maximum_score` passed 25 experimental cells and `two_hop_pointer` passed 23;
neither family passed any matched identical-token control.

At the exact layer-21 answer-prefix intervention used by the earlier causal
subspace experiment:

- `maximum_score`, layer 21 to checkpoint 24: progress `0.896/0.925`,
  median target-distance ratios `0.354/0.243`, value accuracy `1.0/1.0`;
- `two_hop_pointer`, layer 21 to checkpoint 24: progress `0.932/0.958`,
  median target-distance ratios `0.335/0.346`, value accuracy `1.0/1.0`;
- both remained bidirectional at terminal checkpoint 27.

Transplanting all six differing positions strengthened transport further.
For two-hop at layer 21, progress reached `0.970/0.986` at checkpoint 24 and
`0.983/0.992` at checkpoint 27 with value accuracy preserved. By contrast,
the six-position identical-token control at layer 26 produced only about
`0.001` progress in either direction and a target-distance ratio near `1.0`.

The formal best cells differed:

- `maximum_score`: layer 23, all six differing positions, checkpoint 24,
  bidirectional score `0.9971`;
- `two_hop_pointer`: layer 26, all six differing positions, checkpoint 27,
  bidirectional score `0.9967`.

Because many common cells already passed with near-saturated transport, the
scientifically stronger interpretation is not merely that the best absolute
locus differs. Exact per-example operation-state transplantation works
bidirectionally and specifically in both computations, including at the
original layer-21 answer prefix. The previous apparent one-way failure arose
from the layer-24 mediation scalar and from replacing exact row-conditioned
states with an averaged additive controller.

This closes the loop with a sharper boundary:

> Shared low-dimensional representational geometry is not itself a reusable
> additive causal direction. Operation control is carried by distributed,
> instance-conditioned states that can be transported exactly, while averaging
> them into a global vector destroys causal equivalence.

This is a mechanistic result, not yet a general breakthrough claim. It is
post-hoc diagnostic evidence from Qwen-7B, two selected computations, shared
command tokens and one prompt contract. A future Paper-2 program may test the
instance-conditioned control-field hypothesis prospectively across wording and
architectures, but no further run is authorized in this closed loop.

Result SHA-256:
`EA18E755B04DE9CAC1499B5A6A8D8AB553D0C1E19EAC17205D0A45E475026AAC`.

## 2026-07-27: predictive conditional transport — frozen, not launched

The exact-transplant result establishes causal state transport but uses the
opposite-operation activation as an oracle. The next and only licensed test
asks whether that displacement can instead be predicted from the origin state
alone.

For each of eight heterogeneous computation families, a reduced-rank linear
predictor is trained on the other seven families and selected only on
donor-validation rows. The held-out test uses twelve histories whose six
directed source-target pairs are disjoint from both donor splits. Target-family
predictions are hashed before counterpart test states are recaptured and
before any causal intervention.

The exact transplant is retained only as an upper bound. Global mean
displacement, target-state centroid, nearest donor, row-shuffled prediction,
instruction-position and matched identical-token controls distinguish a
predictable conditional transformation from a generic template or
nonspecific perturbation. Direct checkpoint convergence at layers 24 and 27
replaces the earlier mediation scalar.

The complete fixed design, gates and stopping rule are in
`PAPER2_PREDICTIVE_CONDITIONAL_TRANSPORT_PROTOCOL.md`. No prompt, split,
hyperparameter, threshold or locus rescue follows the run.

## 2026-07-27: predictive conditional transport — completed; hypothesis not supported

The full T4 run completed with prediction-freeze SHA-256
`9EFF4E595C08FEF87AEB5056E5AF38B13A8210D1A964C8A267C131006AC094C2`.
The protocol SHA-256 was
`B0B08F1DDD55CE381856F9C04D75F39B037E45ACB857451F33EDADD69DC74A58`.

Exact matched answer-prefix transplantation generalized strongly: all eight
held-out computation families passed bidirectionally, with terminal
bidirectional progress from `0.553` to `0.838` and no position-control
failure. This substantially broadens the exact causal-transport observation
beyond the two families used to discover it.

The source-only reduced-rank predictor did not establish state-conditioned
generalization. Across families its mean bidirectional progress was `0.424`,
versus `0.696` for exact transplantation, `0.447` for the target-state
centroid, and `0.422` after deterministically shuffling predicted
displacements across rows. Conditional prediction exceeded the best global
baseline in only three of eight families (mean gain `-0.027`) and exceeded
the shuffled control by only `0.002` on average. No family passed the complete
prospective conditionality gate.

### Adjudication correction

The original result file printed `GLOBAL_TEMPLATE_TRANSPORT`. Its raw
measurements are valid, but that label exposed a code error: `global_pass`
used route progress without requiring the already-recorded value-accuracy and
target-distance gates. In `latest_update` and `constraint_elimination`, the
target-centroid arm had 0% answer accuracy yet contributed to that verdict.
Reapplying the frozen per-direction gates without changing any threshold
leaves only two globally passing families (`private_belief` and
`temporal_slot`), below the required six. The corrected verdict is therefore:

`ORACLE_ONLY_STATE_TRANSPORT`.

The correction changes no activation, intervention, metric or threshold and
does not license a rerun. It is preserved in a separate derived adjudication
artifact keyed to the immutable raw-result hash.

Supported boundary:

> Exact row-matched operation states are broadly causally transportable, but
> neither a global template nor a donor-trained linear function of the origin
> state recovers that transport across unseen computations. The causally
> required variation is computation-specific, nonlinear, or inseparable from
> the matched state under this representation.

## 2026-07-27: within-family conditional transport — frozen, not launched

The final control-law test changes only the training scope. For each of the
eight computations independently, fit source-only BELIEF-to-SEARCH and
SEARCH-to-BELIEF displacement predictors on 24 directed pairs, select rank and
ridge on eight disjoint validation pairs, and test on twelve histories from
the six remaining directed pairs.

All interventions, direct L24/L27 convergence metrics, exact upper bound,
global templates, nearest-neighbour, row-shuffle and position controls are
unchanged. The corrected functional adjudicator requires answer accuracy for
every arm, and a regression test explicitly prevents the verdict-only bug
found in the cross-family result.

At least six of eight families must pass the complete conditionality gate to
establish `FAMILY_SPECIFIC_CONTROL_LAWS` and continue the research branch.
Fewer than three with valid exact references closes the learnable control-law
branch. The complete frozen design is in
`PAPER2_WITHIN_FAMILY_CONDITIONAL_TRANSPORT_PROTOCOL.md`.

## 2026-07-27: within-family conditional transport — completed

The frozen T4 run completed under protocol
`57BA943FA1C24D68358C214A4C4861B33C1BA36703934927060C13A240BBC0FA`.
The raw-result SHA-256 is
`E3A8E9D11F7A15291B20888007C05131ABF2F5B021401D1A8BC663F26F7AA99A`.

Exact matched-state transport passed all eight families. Conditional
source-only predictors also caused substantial bidirectional movement while
preserving the value answer (minimum directional accuracy was 0.83), and all
position controls passed. However, **zero of eight** families passed the
complete conditionality gate. Conditional gains over the best same-family
global template ranged from `-0.017` to `+0.029`, and gains over row-shuffled
predictions ranged from `+0.003` to `+0.050`; both are far below the frozen
`+0.10` requirement. Simple same-family global templates themselves passed
in six of eight families.

Frozen verdict: `WITHIN_FAMILY_GLOBAL_TEMPLATE_ONLY`.

Decision: close the proposed learnable state-conditioned control-law branch.
The supported result is narrower and cleaner:

> Exact operation-state transport generalizes across computations. Its
> portable intervention is computation-dependent, but within a computation
> most tested variation behaves like a family-specific template rather than
> a source-state-conditioned linear law.

This does not negate the exact causal locus. It rejects the stronger proposed
theory that a hidden-state-conditioned linear operator explains the required
variation, and it does not license further rescue runs under the frozen
stopping rule.

## 2026-07-27: theory pivot — context-conditioned causal geometry

This is a research-direction note, not an empirical result.

The universal additive-controller branch is closed. A generic
`instruction representation -> circuit selector` claim is also not a
sufficient Paper 2 target: recent work already reports instruction vectors
acting as circuit selectors, and prompt-specific circuit variation makes a
single compact selector an unnecessarily strong assumption.

The retained observations instead motivate the following tentative
factorization:

`h(context, operation) = carrier(context) + T_context z_operation`.

- Paper 1 localized the causal interface and its write, anchor, checkpoint
  and readout sequence.
- Early Paper 2 experiments showed that a relatively low-energy, distributed
  component can change the operation while much of the carrier remains
  stable.
- Exact matched-state transport passed all eight computation families.
- Fixed within-family templates often worked, while universal and
  source-conditioned cross-family linear predictors did not.

The new hypothesis is therefore **covariant causal control**: an abstract
operation effect may be shared, but its activation-space coordinates are
determined by the current computation's local downstream geometry. The
context map `T_context` may be implemented by distributed attention/MLP
gating, a circuit selector, or another nonlinear mechanism; no implementation
is assumed in advance.

The decisive missing result is a prospective transformation law. Given
training computations and only a held-out computation's unaltered baseline,
can a constrained estimate of its local causal response map predict the
intervention that produces the same downstream operation change, without
observing the held-out opposite-operation state?

Success would explain why exact states and family templates work while global
vectors fail. Failure would favor unrelated computation-specific templates
over a shared abstract operation. Any mapping must be low-complexity, frozen
before held-out causal outcomes, behavior-preserving, and compared with
global, family-template, shuffled, norm-matched and exact-oracle controls.
Unconstrained nonlinear alignment is explicitly out of scope.

Research strategy now changes to **width before depth**: screen several
distinct mechanistic explanations with small matched experiments, then fund
only the branch that makes a new held-out causal prediction. Do not begin
another exhaustive confirmation battery until a candidate beats its strongest
simple alternative.

## 2026-07-27: context-geometry width screen — frozen, not launched

The first width-first run combines two orthogonal diagnostics in one model
load across private belief, two-hop pointer, maximum-score comparison and
constraint elimination.

First, exact same-row operation residuals are applied at seven signed doses.
Smooth monotonic responses support local geometry; threshold-like responses
support gating; irregular responses support independent templates or a wrong
representation level.

Second, four calibration-only family templates define a frozen rank-four
intervention basis. Matched positive and negative probes estimate the entire
L24/L27 final-token response vector at unchanged test states, after removing
the exact direct residual identity carry-through. This prevents the skip path
from manufacturing shared geometry or linear dose curves. The diagnostic is
deliberately different from the earlier scalar route-readout Jacobian. It asks
whether the downstream-computed multi-output maps replicate within
computation families and differ across them.

Only smooth dose response plus stable context-specific maps licenses a
held-out covariant-transport pilot. Gated response plus stable maps licenses a
selector-mediation factorial. Mixed or unstable results license neither.
The frozen design and all thresholds are in
`PAPER2_CONTEXT_GEOMETRY_WIDTH_SCREEN_PROTOCOL.md`.

## 2026-07-27: context-geometry width screen — completed

The run completed cleanly under frozen protocol hash
`265E236CC043C6F7F3DA2E504632A83C48FC7F7611128FC3409EC2BB30781D99`.
This was not a behavioral-ineligibility failure: all probe arms preserved
`100%` value-answer accuracy and the post-identity response was nontrivial.

The dose screen passed decisively. All four computation families were smooth
in both BELIEF-to-SEARCH and SEARCH-to-BELIEF directions. At L27, exact-dose
processed progress ranged from `0.598` to `0.910`; every curve had Spearman
`1.000`, linear-fit R-squared from `0.987` to `0.999`, and no largest adjacent
step above `0.336`. The processed target retained `0.956` to `0.976` of the
raw target norm. This rejects the discrete-gate explanation at the tested
causal interface and shows a graded downstream response after removing the
tautological residual identity path.

The response-map screen was stable but did not pass context specificity. At
L27, median within-family map cosine was `0.806`, median cross-family cosine
was `0.677`, and their separation was `0.128`, below the frozen `0.150`
threshold. L24 separation was only `0.084`. The L27 map verdict was therefore
`MIXED_RESPONSE_MAPS`, and the frozen overall verdict was
`NO_DEEPER_BRANCH_LICENSED`.

Interpretation: the evidence supports smooth local causal dynamics, but not
the clean computation-specific transformation law required for the proposed
covariant-control pilot. The maps appear to contain both substantial shared
structure and weaker computation dependence. The `0.128` result is not to be
rescued as a near-pass by changing the threshold after observation.

Decision: close this predeclared branch without another GPU rescue run. The
smooth dose result is a useful mechanistic observation, but it does not by
itself establish a predictive cross-context law or an ICML/ICLR-level Paper 2
claim.

Artifacts:

- result JSON SHA-256:
  `7BC9AA6460ABE18E94200B1FFF68A094FE8F574F5D2F4D2A41185C474911F5C0`
- response-map NPZ SHA-256:
  `49B64B7A97B87E1EC90595B8B29E5A75D7F66B15C06EAAA5B01F138D742EFF72`
