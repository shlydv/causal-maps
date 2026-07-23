# Paper 1 preprint plan — "Token-Anchored Memory"

Status: plan of record, revised 2026-07-21 (Sahil: optimize everything for the
arXiv preprint, then ICLR). ICLR 2027 deadlines are not yet official; we use a
late-September planning assumption based on the prior cycle. arXiv target:
**Aug 10–15**, leaving approximately one month of conference-polish runway if
that cadence holds.

Execution update: Qwen2.5-7B, Mistral-7B, Qwen2.5-14B, DeepSeek-R1-Distill-
Llama-8B, and Gemma-3-12B confirmatory batteries are complete. The exhaustive
Qwen2.5-14B multi-token locus study is also complete. One frozen Qwen2.5-14B
closeout battery (exact `ac`-only swaps plus held-out naturalized surfaces) is
complete. The Paper 1 evidence is frozen with its prespecified naturalized-
surface boundary.

## 1. Thesis and contributions

**Title (working):** Token-Anchored Memory: Where In-Context State Lives —
and Where It Doesn't.

**Thesis:** across the tested decoder LLMs and tasks, stated fact tokens expose
a writable, causally sufficient value interface. A designated token-local
summary checkpoint can make the relevant state perfectly decodable without
being causally substitutable. Multi-token tracing further localizes a handoff:
source anchors are sufficient through mid-depth, followed by late
query-specific readout. This does **not** establish the universal absence of
distributed or consolidated representations elsewhere in the residual stream.

Contributions (each with its evidence base already in `runs/`):
1. **Write interface.** Neutral-carrier prototype writes are consumed by
   downstream computation as if stated across Qwen, Mistral, DeepSeek-Llama,
   and Gemma families, with tokenizer-ineligible cells excluded by contract.
2. **Laws.** Full-domain consequence law (ratio ≈ 1.0, 100%, 2 families);
   argument composition incl. mixed injected+textual; capacity ≥ 8.
3. **Register structure.** Single-occupancy conflict rule; orthogonal
   prototype geometry; no affine codebook (confirming negative).
4. **World-state editing with inference propagation.** Entity/city two-hop
   at ratio 0.95–1.0 across 7B-int8, 7B-NF4, 14B-AWQ; disjoint-vocabulary
   control; role specificity (other-entity shift 0.00).
5. **The dissociation and causal locus (headline).** Derived world state is 100% decodable
   within surface at a summary token yet causally inert (0.0003×), while
   anchors carry 1.001× and the late readout carries 0.999× — a matched
   positive/negative/process-control sequence in one world,
   pre-registered both times. Exhaustive multi-token swaps show that source
   anchors remain sufficient through mid-depth, then lose sufficiency as the
   late query-specific readout becomes sufficient. This provides evidence for
   an "anchors + recomputation/readout" account on these tasks, not a universal
   inventory of every representation used by the models.
6. **Boundaries, honestly.** Program-writes null; two-scale capability
   anatomy (belief scales, action-under-CF doesn't); verbalization ledger and
   its falsifying reverse-base control (Paris/default prior, not quorum).

Figures: F1 theory schematic; F2 matrix heatmap (3 families × 5 cells);
F3 consequence-law transfer curve; F4 the dissociation bar (probe acc vs
causal load, checkpoint vs anchor); F5 entity two-hop across scale/quant;
F6 coverage table + boundary map.

## 2. The current field bar (what "industry standard" means now)

Internal robustness targets for this paper (not formal venue requirements):
- **Causal-identification statement** in the abstract/intro: claim type,
  identification strategy, assumptions, one stress test (per 2605.08012 —
  our pre-registration + frozen gates + bug-first audits map directly).
- Aim for **≥3 model families and ≥2 scales**. Use genuine independent row
  draws where they exist, $n\geq30$ distinct rows for headline cells when the
  tokenizer contract permits, row-level confidence intervals, and exact
  randomization p-values. Repeated random-control seeds are not data
  replications. Llama is useful coverage, not a formal requirement.
- **Code + configs + per-row artifacts released**; every kernel's frozen
  protocol doc doubles as the reproducibility appendix.
- Explicit **limitations**: synthetic surfaces, single-token readouts,
  ≤14B, layer coverage of the negative.

## 3. Gap matrix (final empirical closeout)

Original MUST list before arXiv (M1, M3, M4, and M5 are now complete):
- **M1 n/seeds upgrade:** workspace and entity cells at 30 unique rows across
  three genuine generator seeds; anchor/checkpoint on the exhaustive
  tokenizer-aligned structured-world set. The Mistral anchor set is the
  mechanically selected largest aligned bucket (18/30), not a hand-selected
  sample. Original runs remain discovery pilots. ~3 batched model kernels.
- **M2 COMPLETE via DeepSeek-R1-Distill-Llama-8B and Gemma-3-12B:** all 12
  eligible workspace cells pass and the checkpoint/readout dissociation
  replicates; the anchor is content-effective but fails address specificity.
  Gemma supplies an independent architecture/family replication: all 15
  workspace cells pass, city retrieval/two-hop passes, and the exhaustive
  anchor write passes all content, invariant, and address controls. Running
  official Meta Llama is now redundant coverage rather than an open claim gate.
- **M3 COMPLETE:** 14B full-depth layer set incl. late layers and
  question/readout controls; checkpoint max 0.0003×, readout max 0.999×.
- **M4 COMPLETE:** grouped probes reach 100% within-surface checkpoint
  accuracy; checkpoint cross-surface transfer is only 16.7–18.8% (chance
  12.5%).
- **M5 COMPLETE:** reverse-base cell rejects quorum and identifies a strong
  Paris/default-prior confound.

Final C1 COMPLETE: exact matched-state `ac`-only swaps are sufficient through
L32. On held-out naturalized case-note, witness-transcript, and curator-
narrative surfaces, belief transfer plus checkpoint/readout dissociation
replicates, while report and unrelated-address arms fail their frozen
behavioral gates. This boundary is reported without prompt/model rescue.
Further scale or architecture expansion requires a new prediction and belongs
after Paper 1.

Explicitly OUT of Paper 1: Floor 2 (architecture dissociation → Paper 2);
Floor 3 beyond M5 (→ Paper 3 candidate); injection/steering applications.

## 4. Timeline

- **Completed:** M1 Qwen/Mistral widening; Qwen-14B M3 + M4 + M5 in one
  dual-T4 load; immutable evidence and generated macros updated.
- **Completed:** independent-family coverage via DeepSeek-Llama and Gemma;
  exhaustive Qwen multi-token causal-locus study; deterministic evidence audit,
  provenance capture, generated statistics/macros, and claim ledger.
- **Completed:** frozen C1 closeout adjudicated; its row-level artifact,
  bootstrap intervals, generated macros, and deterministic audit are archived
  under `PAPER1_EVIDENCE_FROZEN_WITH_BOUNDARY`.
- **Completed:** evidence-grounded first draft, three immutable-data figures,
  limitations/ethics, reproducibility appendix, official model citations,
  Tectonic compilation, and page-by-page PDF inspection.
- **Now:** author/editorial red-team, venue formatting, public code snapshot,
  and arXiv submission preparation.
- **Aug 4–10:** full draft, red-team pass against §2 bar, code release
  prep.
- **Aug 10–15: arXiv v1.** Then 5 weeks of strengthening (NICE items,
  feedback) → ICLR abstract Sep 19, paper Sep 24.

## 5. Risks

- Naturalized surface shift may expose a boundary; the frozen protocol reports
  that boundary and stops rather than tuning a rescue condition.
- Exact `ac`-only swaps may narrow the depth range of anchor sufficiency; report
  the full trajectory rather than selecting layers.
- J-lens follow-up scoops the dissociation → arXiv early; the laws +
  interface remain ours regardless.
