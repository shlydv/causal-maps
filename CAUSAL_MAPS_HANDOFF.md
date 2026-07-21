# PROJECT HANDOFF: Causal Maps of Agentic Micro-Skills at 7B

**To:** Fresh Fable instance (project lead)
**From:** Prior Fable instance, closing a month-long research arc with the same researcher (Sahil)
**Date:** July 2026
**Action:** Spawn a NEW repository (suggested name: `causal-maps`) and run this project. Do not build inside the old repo.

---

## 1. Who you are working with, and how

Sahil is a senior data scientist running independent ML research solo, evenings and weekends. Compute: an Apple Silicon laptop (MPS, MLX available) and free Kaggle GPU sessions (T4/P100, ~9h caps, sessions die — everything must checkpoint and resume). He works through agent instances (you, plus "Composer" and GPT instances for engineering) and steers by voice notes. He is technically strong (BITS Pilani dual degree, published research), learns fast, and wants a genuine shot at novel, high-impact work — not toy demos, not another observational fishing trip.

Communication: be direct, be honest about novelty ceilings and failure branches, never inflate. He responds well to being pushed back on. He has explicitly asked for calibrated claims after a month of hard lessons. He also runs a public X account posting research updates with charts — posts are drafted from *finished, verified* results only, in a casual human voice (lowercase, no em-dashes, no fragment-stack "AI cadence").

**The single most important cultural artifact from the prior arc is the working protocol. It is non-negotiable and it is the reason the last month produced finished science instead of wreckage:**

1. One living log file per project (here: `CAUSAL_MAPS_LOG.md`), sole source of truth. Every phase appends: what ran, exact commands, numbers, PASS/FAIL verdict.
2. Pre-registered gates: success/failure criteria written into the log BEFORE results exist. No phase begins until the prior gate has a written verdict.
3. Every metric is reported beside a null/control. No best-layer or best-config cherry-picking without a null.
4. A failed gate is a result, not a problem. No rescue variants, no configuration fishing. Branch instructions must be exhaustive in advance.
5. Contradictions with predictions get logged explicitly in a status header, not buried in prose.
6. Any model accuracy wildly below published numbers is a HARNESS BUG until proven otherwise, and blocks all downstream gates. (This exact bug appeared three times in the prior arc: 0%, 0%, 3.3% — all harness, never the model.)
7. Agents restate the plan in their own words + raise objections in the log before writing any code.

---

## 2. Context: what the prior month established (do not relitigate)

Sahil ran a disciplined arc of interpretability and continual-learning experiments on small models (mostly Qwen2.5-Coder-1.5B / Qwen2.5-1.5B-Instruct, some VibeThinker-3B). Full details live in the OLD REPO, which Sahil will point you at. Key files there: `RULEWORLD_JSPACE_READOUT_AUDIT_FOR_FABLE.md`, `CL_RULEWORLD_MERGE_LOG.md`, `TRENCHCOAT_LOG.md`, `experiments/HANDOFF_FABLE.md` (microscopy brief), `experiments/exp05_state_phenomenology/`.

Findings that are CLOSED — treat as established, cite, do not re-test:

- **Behavioral:** 1.5B rule-world answers are largely surface heuristics. A logistic regression on problem-surface features matches the model's answers (54.7% vs 22.3% majority) and BEATS it on ground truth (78–80% vs 58%). When the model deviates from the heuristic, accuracy collapses to ~24% (chance). "Lookup table in a trenchcoat."
- **Observational internals, 1.5B–3B:** logit lens, Anthropic J-lens replication, template probes, HMM state models, graph-homomorphism similarity, episode segmentation — ALL failed calibrated nulls. Convergent finding, replicated ~8 ways: internal observables at this scale organize around generation dynamics (position, phase, surface form), not task structure. Coarse "regimes" are blind-annotator-visible (S0 PASS) but carry no information beyond position (S1 FAIL: position baseline beat HMM on held-out likelihood).
- **Continual learning:** designed conceptual overlap did not causally order forgetting under strict gates (11 seeds, magnitude-matched controls); update magnitude dominates.
- **Methodological:** the whole observational quadrant ("can I READ structure out of activations?") is retired at this scale. The lens-vs-raw predictor race was recognized as structurally rigged (a projection cannot contain more information than its source).

The strategic conclusion that launches THIS project: correlation/observation failed; **causal intervention was never run**. The field's foundational small-model interpretability results (ROME causal tracing, IOI circuit) were interventional and were done on GPT-2-class models SMALLER than what Sahil can run. This is the unplayed move.

---

## 3. The project in one line

**Build causal heatmaps — via activation patching on minimal pairs — of the micro-skills that agentic work depends on, at 7B, with a 1.5B scale contrast, in domains where we control ground truth.**

Not grand cognition ("planning," "verification" — banned as unearned semantic labels, per hard lessons). Small, load-bearing, testable competencies:

- **Skill A — State tracking:** does the model maintain a persistent "this step is already done" flag in its activations? Minimal pair: identical agent-style contexts, except one line records that an action already happened. Patch activations from done-context into not-done-context across (layer × position); find where patching flips the behavior from re-doing to skipping. Mechanistic core of why small agents loop/repeat — a known, painful, UNLOCALIZED failure mode. To our knowledge genuinely open at this scale.
- **Skill B — Instruction/data separation:** the same string as a command vs as quoted data under discussion. Where in the network is "this is to be obeyed" decided? Prompt-injection at mechanism level. Thin scattered literature; high relevance.
- **Skill C — Binding (positive control):** entity–attribute binding has prior art (binding-ID literature). It exists to CALIBRATE the harness: if patching cannot recover known binding behavior, the instrument is broken. This is the instrument-validation-gate pattern (Gate T1 pattern from the prior arc) applied to interpretability.

Core measurement per skill: indirect effect **IE(layer, position) = f(x_clean; h←h_counterfactual) − f(x_clean)** on answer log-odds, swept over all layers and positions, averaged over N pairs. The heatmap is a MEASUREMENT, not a hypothesis — concentrated, migrating, or diffuse are all reportable shapes. The pre-registered CONTRASTS carry the science:

1. Does the state-tracking "done" flag localize (few sites) or smear?
2. Is the instruction/data decision early (embedding-adjacent) or late (pre-emission)?
3. **Scale contrast:** same maps at 1.5B vs 7B. If 7B maps are crisp where 1.5B smears, that single figure retroactively explains the entire prior month (observational failure was about scale, not just method) and is the flagship result.
4. Optional if time: base vs R1-distill at 1.5B — does reasoning distillation reorganize causal pathways or just outputs?

Honest novelty ceiling (state this in the log and never inflate past it): the METHOD is textbook; binding is studied; instruction/data has related work; agentic state-tracking maps at this scale appear open. Contribution = the questions + controlled-pair rigor + the scale contrast. Realistic outcome: strong citable community/workshop result. Main-track upside only if something sharp appears (e.g., a clean, movable "done" flag).

---

## 4. What to reuse from the old repo (ask Sahil for the path)

- **Rule-world generator** (`rule_world.py`, `rft/make_problems.py`, single-token label machinery, strict answer extraction): a synthetic entailment domain with known ground-truth derivations, controllable depth/vocabulary, minimal-pair construction almost free. Use it for harness bring-up and as a controlled fourth domain if useful. Its tokenizer-validation pattern (every target label verified single-token under EVERY model's tokenizer) must be copied.
- **Null-calibration and gate scripts** as templates (`TRENCHCOAT_LOG.md` shows the fixed permutation-null construction — fit once, permute labels, NO refitting in the loop).
- **Known traps, all previously hit — do not re-hit:** VibeThinker/bf16 tensors need float32 cast before numpy export; chat-template mismatch between train/eval silently destroys accuracy; permissive answer extraction corrupts labels (always strict); leading-space tokenization can land anchors on the wrong token; Kaggle sessions die (checkpoint + resume everything, 20-min smoke test before any full run).
- The prior notes (trenchcoat, exp05, CL) as citable context for the writeup.

Do NOT port: any lens/probe/HMM analysis code paths (retired quadrant), the CL oracle machinery, the debugger platform (frozen; this project needs plain hooks, not that stack).

---

## 5. Execution plan with pre-registered gates

**Phase 0 — Repo + restatement.** New repo `causal-maps`. Write `CAUSAL_MAPS_LOG.md` with: this plan restated in your own words, objections raised NOW, and all gate criteria below copied in verbatim before any code. Stack: HuggingFace transformers + hooks (PyTorch), bf16; 7B = `Qwen2.5-7B-Instruct`, small = `Qwen2.5-1.5B-Instruct` (same family, honest scale contrast). Local MPS for pair design + 1.5B; Kaggle for 7B sweeps. Greedy decoding everywhere. Metric = answer log-odds delta, strict extraction.

**Phase 1 — Harness + Gate P0 (positive control).** Implement patch-at-(layer,position) with clean/corrupt caching, batched pairs. Bring up on rule-world minimal pairs locally at 1.5B (cheap, generator exists) purely as plumbing check. Then Gate P0 at 7B on Kaggle: N=30 binding pairs (design from binding-ID literature patterns: two entities, two attributes, question probes which attribute binds to which entity; counterfactual swaps the binding). **P0 PASS = patching at the expected token positions moves answer log-odds in the predicted direction with effect > a random-position patch null (matched layer distribution, p<0.01) and the effect replicates across two disjoint 15-pair halves.** FAIL → fix harness; if it fails after two documented fix rounds, the project stops and the writeup is "patching harness could not recover known effects at 7B under our constraints" (do not proceed on a broken instrument).

**Phase 2 — Pair libraries for Skills A and B.** This is where scientific taste lives — involve Sahil directly in pair design, do not fully delegate to yourself. Requirements per skill: ≥50 pairs; exactly ONE token-span differs between pair members; multiple surface instantiations per underlying template (≥5 templates × ≥10 instantiations) so Gate P1 can test template-robustness (the prior arc's Exp02 lesson: single-template effects are usually template artifacts); behavioral pre-check: the model must actually BEHAVE differently on the two pair members (e.g., for Skill A: re-does the action in not-done context, skips in done context) on ≥70% of pairs at 7B, measured before any patching — pairs where behavior doesn't differ are excluded and the exclusion rate logged. If behavioral pre-check fails wholesale (<40% of pairs), the skill is not elicitable in this setup at 7B; log it, drop the skill, and say so in the note — that itself is reportable.
**Gate P1 (per skill):** heatmaps computed independently on template-disjoint halves correlate (Spearman over the layer×position grid) > 0.5, and > a pair-shuffled null. FAIL → the map is template noise; report as such.

**Phase 3 — The maps + pre-registered contrasts.** Full (layer × position) IE sweeps, N≥50 pairs per skill, 7B and 1.5B. Before viewing ANY heatmap, log predictions for contrasts 1–3 (localize vs smear; early vs late; crisp-at-7B vs smear-at-1.5B), each with a quantitative operationalization chosen in advance (e.g., localization = top-5% of sites carrying >50% of total |IE|; crispness compared via normalized map entropy, 7B vs 1.5B, with a matched-null band). Report every contrast whichever way it lands.

**Phase 4 — Freeze + note.** `reproduce_all.py` regenerating every heatmap and table from cached activations/JSONs, pinned versions. `CAUSAL_MAPS_NOTE.md`: question, method, P0 calibration, pair-design discipline, maps, contrasts, limitations (greedy-only, one model family, prompt-level skills, correlation-of-sites ≠ full circuit), and explicitly what is NOT claimed. X-post drafts only after the note is frozen, from verified numbers only, in Sahil's voice rules.

**Standing stop condition:** if P0 passes but BOTH novel skills fail P1 (all template noise), stop after documenting — do not invent Skill D, E, F. The note then reports the calibrated instrument + the negative, which is still a complete deliverable.

---

## 6. Open threads in the old repo (context, not your job)

- Trenchcoat Gate T2 GSM8K harness bug (3.3% accuracy = broken rollouts, diagnosed not fixed) — a separate agent session owns this; it may finish before you start. Its models/rollout harness overlap with your Phase 2 model set; coordinate through Sahil if useful.
- Three writeup notes from the prior arc are being drafted in parallel (CoT-not-a-clock; microscopy negatives; trenchcoat). Cite them; don't rewrite them.

## 7. First message you should send Sahil

Ask for: (1) the old repo path/access, (2) confirmation of the 7B model choice and Kaggle account readiness, (3) a 30-minute session to co-design Skill A and B pair templates before you write any pair-generation code. Then post your Phase 0 restatement + objections in the log.

Run it like the last month was run at its best: gates first, nulls always, finished notes as the only currency.
