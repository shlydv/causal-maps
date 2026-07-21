# Causal Maps of Agentic Micro-Skills at 7B

**Status:** FROZEN — 2026-07-12 (full-send)  
**Repo:** `/Users/sahilyadav/causal-maps` (local only; never pushed)  
**Models:** Qwen2.5-{1.5B,7B}-Instruct · greedy · answer logit-diff · activation patching / direction ADD  
**Living log:** `CAUSAL_MAPS_LOG.md` (append-only source of truth)

---

## 1. Question

Observational interpretability at 1.5B–3B (prior month) failed calibrated nulls: structure tracked generation dynamics, not task. Foundational small-model results (ROME, IOI) were *interventional*. We ran intervention.

**This project:** causal maps of load-bearing micro-skills at 7B (with 1.5B scale contrast), controlled pairs, pre-registered gates. No unearned grand-cognition labels.

The sharp result was not the heatmaps. **Site-maps failed template-replication (P1); direction-transfer passed** — so the reusable unit, under our gates, is a low-dimensional residual direction at a prompt-conditioned site, not a template-stable layer×position map.

---

## 2. Method (brief)

- **IE patching:** IE(L, p) = logit_diff(clean; patch h←h_cf at L,p) − logit_diff(clean); logit_diff = logit(a_cf) − logit(a_clean) at the final token. Patch = post-layer residual.
- **Direction transfer:** Δ = mean(h_cf − h_clean) at (L*, slot) on donor templates; ADD αΔ onto held-out clean runs at their slot. Cell PASS: ΔIE>0, p<0.01 vs 100 random same-norm dirs, cross/within ≥ 0.5.
- **Elicitation:** chat template + primed single-token answers (raw Instruct completions silently destroy elicitation — hit and fixed at P0).
- **Compute:** Kaggle T4; 7B bitsandbytes 8-bit (residual patched in fp16); 1.5B full precision.
- **Protocol:** living log; gates written before results; every metric beside a null; failed gate = result (no fishing).

| Skill | Bit | Role |
|-------|-----|------|
| Binding (entity–attribute) | which attribute binds | **P0 positive control** |
| Completion state | boolean task-complete flag | novel |
| Variable substitution | stored symbol → value | novel |
| Instruction vs mention | execute vs describe | dropped at elicitation |

---

## 3. Instrument — Gate P0 PASS

**P0 (7B binding, N=30, chat-templated):** patching at the queried entity's attribute token moves answer log-odds vs matched random-position null (p=0.001); split halves both p=0.001. Top site = `a1_slot`. Behavioral: 100% greedy both arms.

False starts (logged, not hidden): unverified Kaggle → silent CPU; P100 unsupported; raw Instruct → 0% greedy with deceptive relative-logit "pass." Fixed elicitation; then P0 clean. Downstream negatives are not a broken instrument.

---

## 4. Elicitation

| Skill | Hand-10 | ×50 | Outcome |
|-------|---------|-----|---------|
| Variable | 100% greedy | 50/50, ~100%, 5/5 templates | elicited |
| Completion (explicit if-then) | 100% | 40/40, 100%, 4/4 | elicited |
| Completion (implicit, no rules) | — | **0%** | not elicited |
| Instruction vs mention | 0% then relative-only / **0% greedy** | — | **dropped** (two failures) |

Completion without an explicit rules table does not elicit — standing caveat that maps may partly reflect in-context rule lookup.

---

## 5. Gate P1 @7B — site-maps FAIL (methodological foil)

Pre-registered: template-disjoint half maps, Spearman ρ > 0.5 and > pair-partition null.

| Skill | Expected-site IE | Top site | P1 ρ | P1 |
|-------|------------------|----------|------|-----|
| Variable | **+53.3 @ val_slot, p=0.001** | L2, pos 27 = val_slot | 0.37 | FAIL |
| Completion | +0.07 @ bit_slot, p=0.58 | L2, **pos 62** (not bit@61) | 0.14 | FAIL |

Heatmaps look sharp (top 5% sites ≈97% of |IE| mass). P1 asks whether that *shape* replicates across templates.

**Variable fragility:** mean pairwise full-grid Spearman ρ = **0.264**; expected-column alone correlates ≈ **1.0** across all template pairs. Stable site effect; fragile off-site geometry. P1 scores the whole grid — and fails.

**Completion:** mass one token after the bit; does not track the pre-registered site across surfaces.

Artifacts: `runs/p1_7b/`.

---

## 6. Direction transfer — the reusable unit

**Hypothesis (pre-registered):** P1 failed because it scored site-maps (wrong coordinate system), not because the mechanism is non-reusable.

### 6.1 Discovery — Variable @7B

Δ at (L=2, val_slot) from donors {X,Y}; ADD onto {Z,W,K}.

**PASS — DIRECTION_REUSABLE.** Cross/within ≈ 1.04 / 1.14 / 0.83, all p≤0.01. LOO mean ratio ≈ 1.07. Flip-rate 0% (baselines peaked; metric is log-odds). Artifacts: `runs/delta_transfer/`.

### 6.2 Full-send controls & generalization (pre-registered; all run)

| Kernel | Test | Verdict |
|--------|------|---------|
| **1a** Robustness | α∈{0.5,1,2} @L2; L∈{1..4}@α=1; flip hunt α∈{3,4,5} | α=1,2 and L1–4 **PASS**; flips only at α≥3 (steering, not demo) |
| **1b** Embed control | Δ_embed ADD at layer-0 | **NONTRIVIAL** (0/3 pass; ratio_e/L2≈0.005) |
| **1c** Wrong-value + anti-Δ | re-forward unrelated value; ADD −Δ | **GENERIC_BOOST** (wrong PASSes 3/3, ratio≈0.59, cos≈0.77); anti-Δ hurts ✓ |
| **2** Completion | donors A,B→C,D at **pos62**, L=2 | **COMPLETION_DIRECTION_REUSABLE** (ratio≈1.0, **100% flips**) |
| **3** Scale | Variable @1.5B, L=2 exactly | **SCALE_TRANSFER_OK** (3/3, ratio≈1.02) |
| **4** Cross-position | inert prefix; short↔long both dirs | **POSITION_FREE** (27↔40, both 3/3) |

Cached-activation "shuffle" in K1 was a math no-op (mean-pooled derangement ≡ matched Δ); withdrawn and replaced by wrong-value re-forwards (1c).

Artifacts: `runs/delta_var_robust/`, `runs/delta_var_shufflefix/`, `runs/delta_completion/`, `runs/delta_var_1p5b/`, `runs/delta_var_crosspos/`.

---

## 7. Finding

1. **Site-maps are the wrong object for "is the mechanism reusable?"** Under P1, Variable and Completion fail. The value-token / peak-site *effects* are real; the full-grid map is template-fragile.
2. **Directions transfer.** Across variable-name templates, Completion surfaces, 1.5B, and ±13 token positions, ADD of a donor Δ moves held-out log-odds at within-template strength.
3. **Not an embedding artifact** (1b NONTRIVIAL). **Signed** (anti-Δ hurts). **Not value-binding-specific** under the pre-registered wrong-value gate (1c GENERIC_BOOST): an unrelated value word still yields a transferring Δ (weaker, ~59% of matched). Honest reading: a reusable *value-slot update* direction, not a proven binder of a specific lexical value.
4. Completion's direction lives at **pos62** (P1 peak), not bit@61 (‖Δ‖≡0 at the bit) — consistent with site-map localization and with the explicit-rules elicitation caveat.

---

## 8. What we are not claiming

- Not a full circuit (sites/directions ≠ circuits).
- Not that we "discovered directions" as a field contribution — the contribution is the **coordinate-system result**: site-replication gates can miss a reusable direction that ADD recovers.
- Not a clean value-specific binder (GENERIC_BOOST blocks that claim).
- Not that Instruction is impossible — only not elicitable here after two documented attempts.
- Not that Completion is a latent agent "done" flag — explicit rules were required.

---

## 9. Limitations

Greedy-only; one model family; prompt-level skills; 8-bit 7B weights; Completion = explicit if-then only; Instruction dropped; wrong-value control PASSed (GENERIC_BOOST); no public release / no GitHub remote; correlation of causal sites ≠ circuit.

---

## 10. Deliverables

- Patching + direction-ADD harness, Kaggle orchestrator (`src/causal_maps/`, `kernel/run_kaggle.py`).
- `CAUSAL_MAPS_LOG.md` — full append-only record including pre-registrations.
- Artifacts under `runs/` for P0, ×50 behav, P1, delta_transfer, and full-send K1–K4.
- This note (**frozen**).

**No public drafts / X / arXiv until a separate decide-to-publish step.** arXiv-clean path was closed by K1c; any writeup must carry GENERIC_BOOST in the headline claim.

---

## 11. One-line summary

Activation-patching site-maps of elicited micro-skills at 7B fail template-replication, but the residual *direction* at the causal site transfers across templates, skills, scale, and position — reusable under ADD, layer-computed, not value-specific under a wrong-value control.
