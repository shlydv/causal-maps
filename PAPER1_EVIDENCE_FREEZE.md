# Paper 1 evidence freeze

Status: **FROZEN WITH A PRESPECIFIED NATURALIZED-SURFACE BOUNDARY**. This document
is the pre-writing claim and experiment gate. Detailed protocols/results stay
in their existing files; `paper_token_anchored/generated/evidence_manifest.json`
is the machine-checked artifact ledger.

## Claim ledger

| Claim | Evidence | Allowed wording | Boundary |
|---|---|---|---|
| A neutral-carrier value write is consumed like a textual substitution | Widened Qwen-7B, Mistral-7B, DeepSeek-Llama-8B, Gemma-3-12B batteries | A portable residual write reproduces native downstream value consequences in behaviorally eligible cells | Not every task/model is behaviorally eligible; do not count ineligible cells as causal failures |
| Addressed writes can propagate through relational inference | Qwen city families, Qwen-14B entity pilot, Gemma city families | The write changes direct and two-hop consequences while preserving an unrelated entity in eligible city worlds | Keys two-hop is often ineligible; DeepSeek shows content transfer without universal address specificity |
| Textual source anchors and a designated pre-query checkpoint dissociate | Qwen/Mistral widened batteries; exhaustive Qwen-14B; DeepSeek-Llama; Gemma | Source-anchor interventions are causally effective while matched single-token `STATECHECK` substitution is nearly inert over tested depths | This is intervention-relative non-substitutability, not absence of encoded information or every possible distributed state |
| The checkpoint is decodable but not causally substitutable in the tested way | Qwen-14B grouped probe plus full-depth causal sweep | Within-surface state is decodable at the checkpoint despite near-zero token-local causal substitution | Cross-surface checkpoint decoding is poor; decodability is not causality |
| Causal support is localized to source anchors before a late readout handoff | Frozen Qwen-14B multi-token locus curve | Source-anchor states are bidirectionally sufficient through L32; marker, local summary, edited-anchor-removed, and size-matched random loci fail; support later appears at the query readout | Six-anchor sufficiency plus leave-one-out establishes necessity of the edited anchor; the exact edited-anchor-only sufficiency arm is pending the closeout run |
| Verbalization is not a quorum/tamper-evidence mechanism | Frozen Qwen-14B reverse-base discriminator | Results are consistent with an asymmetric Paris/default prior, not witness quorum | Do not present verbalization as a positive mechanism claim |
| The belief-anchor/checkpoint/readout headline survives longer, less templated prose and a held-out transition | Frozen Qwen-14B Paper 1 closeout | On 30 held-out naturalized worlds, the belief write transfers at 1.012× with 100% target accuracy and p=.0476; checkpoint substitution remains near zero (max .00214×) and late readout is sufficient | The report counterfactual and unrelated-belief specificity arms fail behavioral eligibility, so naturalized report transfer and address specificity are not established |

## Experiment decisions closed

- **Official Meta Llama-3.1-8B:** not required. License delay was bypassed by a
  DeepSeek-Llama architecture replication, and Gemma supplies a clean
  independent-family confirmation. Another 8B model adds coverage but does not
  resolve a live Paper 1 identification question.
- **Jamba:** tokenizer preflight passed; execution is incompatible with Kaggle's
  missing fast Mamba CUDA kernels. This is infrastructure, not negative data.
- **Qwen3 or another Qwen scale:** not required; Qwen family already has 7B and
  14B coverage. A larger point without a frozen scaling prediction is a model
  zoo addition.
- **More random seeds on fixed structured worlds:** not data replication and
  therefore not required. Uncertainty is computed over distinct rows.
- **More task families, planning, learned alignment, attention-path rescue,
  value/address factorization:** Paper 2 discovery program, explicitly outside
  the Paper 1 evidence freeze.

## Mandatory limitations, not open experiments

- Predominantly synthetic controlled worlds; the naturalized closeout changes
  surface realism but is not a natural corpus benchmark.
- Open-weight instruction models up to 14B and the tested quantization/runtime
  recipes; no claim of universal scaling.
- Causal conclusions apply to specified residual interventions, token loci,
  depths, prompts, and behaviorally eligible rows.
- Qwen checkpoint probes establish decodability on one headline model; causal
  dissociation has broader model coverage than probe coverage.
- DeepSeek's address-spillover boundary argues for separable value/content and
  routing mechanisms; it is a Paper 2 hypothesis, not silently averaged away.

## Freeze gate

Paper 1 experiments are closed only when all conditions hold:

- [x] All existing confirmatory artifacts archived with SHA-256 hashes.
- [x] Row counts, uniqueness, behavioral exclusions, and depth grids pass the
  deterministic evidence audit.
- [x] Qwen-14B exact full-prefix intervention sanity passes at every depth.
- [x] ~~Frozen exact edited-anchor-only and naturalized-surface closeout completes.~~
- [x] ~~Closeout artifact is archived, bootstrapped, macro-generated, and passes
  `audit_evidence.py --require-closeout`.
  The machine status is `PAPER1_EVIDENCE_FROZEN_WITH_BOUNDARY`.~~
- [x] ~~Roadmap contains no unchecked Paper 1 experiment or data item.~~

After this gate passes, new empirical ideas go to Paper 2 or a documented
post-preprint revision; they do not delay Paper 1 writing.
