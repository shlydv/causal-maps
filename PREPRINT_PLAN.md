# Paper 1 preprint plan — "Token-Anchored Memory"

Status: plan of record, revised 2026-07-15 (Sahil: optimize everything for the
arXiv preprint, then ICLR). ICLR 2027 deadlines are not yet official; we use a
late-September planning assumption based on the prior cycle. arXiv target:
**Aug 10–15**, leaving approximately one month of conference-polish runway if
that cadence holds.

Execution update: Qwen2.5-7B and Mistral-7B confirmatory batteries completed on
2026-07-15. The 14B anchor/checkpoint/probe kernel is frozen but was not started
because Kaggle reported the 30-hour weekly quota exhausted.

## 1. Thesis and contributions

**Title (working):** Token-Anchored Memory: Where In-Context State Lives —
and Where It Doesn't.

**Thesis:** in-context memory in decoder LLMs is a bank of single-occupancy,
near-orthogonal value prototypes anchored at the tokens where facts were
stated; downstream conclusions are recomputed from anchors at query time;
no consolidated world-state buffer exists to edit — but the anchors are a
full read/write interface obeying quantitative laws.

Contributions (each with its evidence base already in `runs/`):
1. **Write interface.** Neutral-carrier prototype writes are consumed by
   downstream computation as if stated (3 families; 14/15 matrix cells).
2. **Laws.** Full-domain consequence law (ratio ≈ 1.0, 100%, 2 families);
   argument composition incl. mixed injected+textual; capacity ≥ 8.
3. **Register structure.** Single-occupancy conflict rule; orthogonal
   prototype geometry; no affine codebook (confirming negative).
4. **World-state editing with inference propagation.** Entity/city two-hop
   at ratio 0.95–1.0 across 7B-int8, 7B-NF4, 14B-AWQ; disjoint-vocabulary
   control; role specificity (other-entity shift 0.00).
5. **The dissociation (headline).** Derived world state: readable at a
   summary token (probe arm, to run) yet causally inert (0.001×), while
   anchors carry 1.001× — a matched positive/negative in one world,
   pre-registered both times. Answers J-lens's stated open problem
   (binding/structure) with "anchors + recomputation," corroborated at the
   mechanism level by the rebinding-circuit results (2606.08644).
6. **Boundaries, honestly.** Program-writes null; two-scale capability
   anatomy (belief scales, action-under-CF doesn't); verbalization ledger
   (migration real + content-specific; single-site edits vetoed; quorum-vs-
   prior cell decides the paragraph).

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

## 3. Gap matrix (experiments still required)

MUST before arXiv (quota-gated; ~10–12 kernels ≈ 2 weekly resets):
- **M1 n/seeds upgrade:** workspace and entity cells at 30 unique rows across
  three genuine generator seeds; anchor/checkpoint on the exhaustive
  tokenizer-aligned structured-world set. The Mistral anchor set is the
  mechanically selected largest aligned bucket (18/30), not a hand-selected
  sample. Original runs remain discovery pilots. ~3 batched model kernels.
- **M2 Llama-3.1-8B-Instruct:** matrix + entity + anchor/checkpoint pair.
  Kaggle-hosted official model (license acceptance may need Sahil's click).
  ~3 kernels. If Llama G0-blocks cells, that is reported (anatomy table).
- **M3 checkpoint-null completion:** 14B full-depth layer set incl. late
  layers + question-token positions. 1 kernel. Kills "didn't look
  everywhere."
- **M4 probe arm:** activation dump at checkpoint + anchors (1 kernel),
  linear probes trained locally (CPU) → F4's "readable" bars.
- **M5 (tiny, optional-but-cheap):** reverse-base quorum cell → converts
  the verbalization section from "mixed" to one decisive paragraph.

NICE (post-preprint, pre-ICLR): Gemma-2-9B replication; 32B-AWQ scale
point; one natural-text surface variant; emergence sandbox (local/free,
runs in background; feeds Discussion or Paper 3).

Explicitly OUT of Paper 1: Floor 2 (architecture dissociation → Paper 2);
Floor 3 beyond M5 (→ Paper 3 candidate); injection/steering applications.

## 4. Timeline

- **Now → quota reset (free):** LaTeX skeleton; evidence tables
  auto-extracted from `runs/*.json`; methods section from the protocol
  docs; identification-assumptions statement; reviewer red-team checklist;
  M1 row-pool widening code + offline preflights; M4 probe code
  (CPU side); sandbox trainer (background).
- **Week of reset 1:** batched Qwen/Mistral M1; Qwen-14B M3 + M4 in one load.
  Freeze v1 numbers.
- **Week of reset 2:** M2 (Llama) + M5 + any M1 stragglers. FREEZE all
  results.
- **Aug 4–10:** full draft, red-team pass against §2 bar, code release
  prep.
- **Aug 10–15: arXiv v1.** Then 5 weeks of strengthening (NICE items,
  feedback) → ICLR abstract Sep 19, paper Sep 24.

## 5. Risks

- Llama license/mount friction (fallback: Gemma-2-9B as third family).
- M1 widening changes a headline number → report both (pre-registered
  widening; original runs stand as pilot).
- J-lens follow-up scoops the dissociation → arXiv early; the laws +
  interface remain ours regardless.
