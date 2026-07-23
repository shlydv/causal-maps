# Research roadmap

Living checklist for the two-paper program. Keep entries brief, strike through
completed work, and add new ideas only when they imply a concrete test or
decision. Detailed protocols and results remain in their existing files.

## Paper 1 — robust arXiv preprint / TMLR-level submission

**Core claim:** LLMs expose a token-anchored write interface for in-context
values; later computation reads those anchors, while a decodable intermediate
checkpoint need not be causally substitutable by the same token-local patch.

### Evidence already closed

- [x] ~~Write-interface and consequence-law batteries on Qwen and Mistral.~~
- [x] ~~Qwen2.5-14B headline replication with exhaustive anchor, full-depth
  checkpoint/readout trajectory, grouped probes, and row-level statistics.~~
- [x] ~~Reverse-base control falsified the verbalization/quorum explanation.~~
- [x] ~~DeepSeek-R1-Distill-Llama-8B replicated the broad write/readout result
  and exposed a real boundary: value content can transfer without strict
  address specificity.~~
- [x] ~~Checkpoint probe result calibrated: strong within-surface decoding,
  poor cross-surface transfer, and near-zero token-local causal substitution.~~
- [x] ~~Gemma-3-12B independently confirmed all workspace cells, the
  address-specific anchor write, inert checkpoint, and late causal readout.~~
- [x] ~~Qwen-14B multi-token locus curve localized sufficient causal support
  to source anchors through L32; summary/random loci fail and support leaves
  pre-query positions after L32.~~

### Close before submission

- [x] ~~Finish and archive the frozen Gemma-3-12B confirmation.~~
- [x] ~~Run the prespecified Qwen-14B multi-token/locus curve with cumulative,
  leave-one-out, full-prefix, and size-matched random controls.~~
- [x] ~~Run one **naturalized robustness surface** with longer, less templated
  prose under frozen tokenizer/behavior gates. The belief anchor versus
  checkpoint/readout result replicated; report and unrelated-address arms
  exposed a behavioral-eligibility boundary, which is frozen without rescue.~~
- [x] ~~Rebuild all evidence tables/macros from immutable row-level artifacts;
  report
  effect sizes, bootstrap CIs, exact/randomization tests, exclusions, and
  behavioral eligibility—not seed counts presented as independent samples.~~
- [x] ~~Freeze the claim ledger narrowly: *decodable but not causally
  substitutable under the tested matched-state intervention*; do not claim
  absence of information, a universal lack of world state, or quorum behavior.~~
- [x] ~~Freeze explicit alternative explanations and controls for the
  manuscript: off-manifold patch,
  distributed/multi-token code, surface dependence, value/address separation,
  quantization, synthetic-task scope, model-size ceiling, and output-token
  steering.~~
- [x] ~~Finish the pre-writing reproducibility package: exact configs, prompts,
  token positions,
  hardware, package versions, checksums, analysis scripts, per-row outputs,
  and one-command evidence regeneration.~~
- [x] ~~Complete the evidence-grounded first draft: abstract through conclusion,
  limitations/ethics, reproducibility appendix, three generated figures,
  citation-key audit, clean evidence rebuild, TeX compilation, and page-by-page
  PDF inspection.~~
- [ ] Pre-release editorial pass: author review, genuinely independent red-team
  read, venue formatting, public repository snapshot, and arXiv upload.

### Useful but not blocking

- [x] ~~Close official Llama-3.1-8B as redundant family coverage after the
  DeepSeek-Llama and Gemma confirmations; it is not a Paper 1 gate.~~
- [x] ~~Close an unmotivated larger-scale point; revisit only with a stated
  scaling prediction in Paper 2 or a post-preprint revision.~~
- [ ] Convert to TMLR submission after arXiv feedback and robustness revisions.

## Paper 2 — discovery program for ICML/ICLR

**Target thesis:** in-context state is factorized into value/content and
address/routing mechanisms, with query-dependent transport from distributed
source support to a late, causally usable readout. The goal is to predict and
control when binding succeeds, spills over, or fails—not merely to add models.

### Priority A: establish the new mechanism

- [ ] **Value × address factorization:** independently manipulate value and
  address; measure correct-address transfer, same-valued spillover, swaps,
  collisions, and wrong-address effects layer by layer.
- [ ] **Minimal causal support:** use cumulative span patches, leave-one-out,
  sparse subset search, and necessity ablations to locate the smallest set of
  source positions that preserves each downstream consequence.
- [ ] **Query-conditioned transport:** test whether one pre-query state supports
  multiple unseen queries; learn transformations only on donor worlds and
  evaluate held-out worlds, values, queries, and surfaces.
- [ ] Compare identity, mean direction, Procrustes, ridge, random, and nonlinear
  mappings. A learned map counts only if it causes correct, query-specific
  consequences and beats matched complexity controls—not if it is merely an
  external answer predictor.
- [ ] **Path necessity and rescue:** identify attention heads/edges carrying
  address and value signals; ablate them, predict the resulting error pattern,
  and rescue it with the corresponding targeted intervention.

### Priority B: demonstrate generality and predictive power

- [ ] Derive preregistered predictions from the factorized account: when
  address specificity emerges, when content-only transfer occurs, and how
  cross-talk scales with addresses, depth, and competing values.
- [ ] Require one intervention to change multiple coherent consequences of the
  same state (separate queries or a full continuation), with a natural-rewrite
  upper bound and necessity test.
- [ ] Extend from synthetic bindings to relational inference, state transitions,
  and a natural task where the same mechanism predicts successes and failures.
- [ ] Replicate the strongest frozen result across at least three architecture
  families and meaningful scales; use model additions to test predictions,
  not to accumulate a model zoo.
- [ ] Test capacity/composition: more addresses, simultaneous writes, conflicts,
  multi-hop depth, delayed queries, distractors, and paraphrases.
- [ ] Build an automatic discovery pipeline that proposes loci/subspaces on a
  discovery split and evaluates them once on a locked confirmation split.

### Conference bar / decision rule

- [ ] Advance the central claim only if the mechanism makes held-out predictions
  and supports both necessity and rescue. Otherwise report the strongest
  boundary as the result and redesign the theory before scaling experiments.
- [ ] Lock the Paper 2 thesis, benchmarks, baselines, and confirmation protocol
  after the first decisive factorization + minimal-support study; then execute
  frozen cross-model confirmation and write toward ICML/ICLR.
