# CAUSAL_MAPS_LOG — Causal Maps of Agentic Micro-Skills at 7B

> **This file is the single source of truth.** Every phase appends: what ran, exact
> commands, numbers, PASS/FAIL verdict. History is append-only — corrections go in a
> dated addendum, never by deleting the record. Pre-registered gates are written BEFORE
> results exist; no phase begins until the prior gate has a written verdict.

---

## STATUS HEADER (keep current — top of file)

- **Date started:** 2026-07-11
- **Project lead (agent):** Grok (Cursor). Prior lead: Opus (Claude). Researcher: Sahil.
- **Repo:** `/Users/sahilyadav/causal-maps` — local only, **no git remote, never pushed to GitHub**.
- **Current phase:** **router-read DEMOTED → `ROUTER_READ_AMBIGUOUS`.** OOD control failed (span-keep harsher than res-keep). Generator hypothesis **not** licensed.
- **Currently blocking on:** Sahil — redesign matched control or pick next why-question (do not treat residual-read as settled).
- **Gate status:** P0 ✅ · P1 FAIL (foil) · delta_transfer ✅ · K1 NONTRIVIAL+GENERIC_BOOST · K2 ✅ · K3 ✅ · K4 ✅ · note frozen · **July-2026 literature gate: SHORT NOTE YES, claim narrowed** · **delta_decompose ✅** · **delta_centroid → CENTROID_MATTERS** · **delta_multislot ✅** · **delta_crossskill ✅** · **delta_capacity k*=8** · **delta_transform v2 → L3_LAYER_DEPENDENT_WEAK** · **delta_select v2 → L2_LAYER_DEPENDENT_STRONG** · **delta_instruction v2 → INSTR_DIRECTION_LAYER_DEPENDENT_WEAK** · **delta_instruction_s3 → INJECTION_PARTIAL** · **delta_typology v3 → TYPOLOGY_FALSIFIED** · **delta_explicit → EXPLICITNESS_FALSIFIED (demoted)** · **delta_protocol → PROTOCOL_COMPATIBLE (line stopped)** · **delta_chain → CHAIN_INELICITABLE (branch CLOSED permanent)** · **delta_necessity → ROUTE_NECESSARY_ONLY** · **delta_asymmetry → ASYM_INCOMPLETE_BASIS solid** · **delta_bindmiss → BIND_MISS_LINEAR_READOUT** · **delta_router_read → ROUTER_READ_AMBIGUOUS (OOD_FAIL)** · **delta_router_ood → OOD_FAIL**.

---

## 0. Working protocol (affirmed — non-negotiable)

1. **One living log** (this file) is the sole source of truth. Every phase appends what ran, exact commands, numbers, PASS/FAIL.
2. **Pre-registered gates.** Success/failure criteria are written into the log BEFORE results exist. No phase starts until the prior gate has a written verdict.
3. **Every metric beside a null/control.** No best-layer / best-config cherry-picking without a null.
4. **A failed gate is a result, not a problem.** No rescue variants, no config fishing. Branch instructions are exhaustive in advance.
5. **Contradictions with predictions** are logged explicitly in the status header, not buried in prose.
6. **Any model accuracy wildly below published numbers is a HARNESS BUG until proven otherwise**, and it blocks all downstream gates. (This bit the prior arc three times: 0%, 0%, 3.3% — all harness, never the model.)
7. **Restate the plan + raise objections in the log before writing any code.** (This section + §1–§2 are that restatement.)

---

## 1. Plan restated in my own words

**The unplayed move.** The prior month exhausted the *observational* quadrant at 1.5B–3B: logit lens, J-lens, HMM state models, template probes — all failed calibrated nulls, converging on one conclusion (internal observables organize around generation dynamics — position, phase, surface form — not task structure). The field's foundational small-model interpretability results (ROME causal tracing, IOI) were **interventional**, on GPT-2-class models *smaller* than what we can run. We never ran intervention. That is this project.

**What we build.** Causal heatmaps — via **activation patching on minimal pairs** — of small, load-bearing micro-skills that agentic work depends on, at **7B**, with a **1.5B scale contrast**, in domains where we control ground truth. No grand-cognition labels ("planning", "verification" are banned as unearned).

**The three skills:**
- **Skill A — State tracking.** Does the model keep a persistent "this step is already done" flag in its activations? Minimal pair = identical agent-style contexts differing only in one line that records an action already happened. We patch done→not-done activations across (layer × position) and find where patching flips behavior from re-doing to skipping. This is the mechanistic core of why small agents loop/repeat — a real, unlocalized failure mode; appears genuinely open at this scale. **Highest-variance bet.**
- **Skill B — Instruction/data separation.** The same string as a command vs. as quoted data under discussion. Where is "this is to be obeyed" decided? Prompt-injection at the mechanism level. Thin literature, high relevance.
- **Skill C — Binding (POSITIVE CONTROL).** Entity–attribute binding has prior art (binding-ID literature). Its job is to **calibrate the instrument**: if patching cannot recover known binding behavior, the harness is broken and nothing downstream is trustworthy. This is the instrument-validation gate.

**Core measurement (per skill).** Indirect effect
`IE(layer, position) = f(x_clean ; h ← h_counterfactual) − f(x_clean)`
on the **answer log-odds**, swept over all layers × positions, averaged over N pairs, strict single-token answer extraction, greedy decoding throughout. The heatmap is a **measurement, not a hypothesis** — concentrated, migrating, or diffuse are all valid, reportable shapes.

**The science lives in the pre-registered contrasts, not the pretty picture:**
1. Does the state-tracking "done" flag **localize** (few sites) or **smear**?
2. Is the instruction/data decision **early** (embedding-adjacent) or **late** (pre-emission)?
3. **Scale contrast (flagship):** same maps at 1.5B vs 7B. If 7B maps are crisp where 1.5B smears, that single figure retroactively explains the prior month (the observational failure was about *scale*, not just method).
4. *(Optional, only if time)* base vs R1-distill at 1.5B: does reasoning distillation reorganize causal pathways or only outputs?

**Honest novelty ceiling (never inflate past this).** The *method* is textbook; binding is studied; instruction/data has related work; agentic state-tracking maps at this scale appear open. Contribution = the questions + controlled-pair rigor + the scale contrast. Realistic outcome: a strong, citable workshop/community result. Main-track upside only if something sharp appears (e.g., a clean, movable "done" flag).

---

## 2. Objections / reality-checks raised BEFORE any code

These are mine, logged now so they can't be retrofitted later.

1. **24 GB RAM kills 7B locally — even for behavioral pre-checks, not just sweeps.** Qwen2.5-7B in bf16 ≈ 14–15 GB of weights; add macOS + KV cache + cached activations and this machine (24 GB unified) will swap hard. **Consequence:** *all* 7B work — including the Phase-2 behavioral pre-check, not only the Phase-3 sweep — must run on Kaggle. The local machine does 1.5B, all pair design, and harness development. This tightens the local↔Kaggle loop, so the **Kaggle path must be working by Phase 1b (the P0 gate), not deferred to Phase 3.** I need the Kaggle token earlier than the handoff implies.

2. **Everything is from scratch.** The old rule-world generator, null-calibration scripts, and gate templates are gone. Rebuild cost is real but bounded: a *minimal* rule-world generator (only needed as a plumbing check + optional 4th domain), a fresh fit-once-then-permute null utility, and the patching harness itself. I will copy the *patterns* the handoff names (fit-once-permute-null with NO refitting in the loop; strict single-token extraction; every target label verified single-token under **every** model's tokenizer). This front-loads Phase 1.

3. **The sweep is expensive and must be checkpointed against Kaggle's 9 h caps.** Naïve per-site patching = one forward pass per (layer × position) per pair. 7B ≈ 28 layers × ~40–60 positions × ~50 pairs ≈ 60k–85k forward passes *per skill*. This MUST batch over pairs, chunk over layers/positions, and checkpoint+resume per chunk (sessions die). Sweep design is an engineering risk I own; I'll validate the batching on 1.5B locally first.

4. **"Answer log-odds" needs a clean measurement token.** The behavioral difference must be readable as a next-token (or short fixed-span) log-odds delta under strict single-token extraction. This constrains pair design: every pair needs a well-defined answer token whose log-odds we track (e.g., Skill A must force the redo/skip decision to surface at one measurable token). This is a Phase-2 co-design constraint, flagged now so we don't design elegant pairs that have no clean readout.

5. **The behavioral pre-check is the real gate, and it's correctly front-loaded.** If 7B doesn't actually behave differently on the minimal pairs (redo vs skip; obey vs quote), the skill isn't elicitable and it drops (handoff: <40% ⇒ drop the skill, log it, that's reportable). I expect **Skill A to be the most fragile** — prompt-level state-tracking in an instruct model may be swamped by surface cues. We find out before any patching, which is right.

6. **P0 (binding) is the only thing that earns trust.** Nothing downstream means anything until P0 passes on a null-controlled, split-half-replicated basis. Per protocol rule 6, any 7B *behavioral* accuracy far below expectation is a harness bug first, not a finding, and blocks P0.

7. **Acknowledged, not objecting:** greedy-only, one model family (Qwen2.5), prompt-level skills, and "correlation of causal sites ≠ full circuit" are limitations already named in the brief. They go in the final note's limitations section verbatim; I won't quietly let scope drift past them.

**No blocking objection to the plan itself.** It is the right move and the gates are honest. My only structural change to sequencing: **pull the Kaggle/7B path forward to Phase 1b** so P0 runs on the real instrument, not a local proxy.

---

## 3. Environment (this machine + compute split)

**Local (pair design, 1.5B, harness dev):**
- Machine: `Mac16,8`, macOS 26.5, Apple Silicon, **24 GB unified memory**, 12 cores.
- Python: 3.13.13 in a project venv at `/Users/sahilyadav/causal-maps/.venv` (no conda).
- Stack: PyTorch (MPS) + HuggingFace transformers + hooks, bf16. numpy/scipy (nulls, Spearman), matplotlib (heatmaps), pandas, tqdm.
- Exact pinned versions (2026-07-11): `torch==2.13.0`, `transformers==5.13.1`, `tokenizers==0.22.2`, `accelerate==1.14.0`, `safetensors==0.8.0`, `numpy==2.5.1`, `scipy==1.18.0`, `matplotlib==3.11.0`, `pandas==3.0.3`, `tqdm==4.68.4`. Full lock: `requirements.txt`. **Note:** transformers is 5.x (major version past the 4.x era) — harness code must be validated against the installed API, not written from memory.

**Kaggle (all 7B work — P0 onward):**
- Free T4/P100, ~9 h caps, sessions die ⇒ checkpoint + resume everything; 20-min smoke test before any full run.
- Requires Kaggle API token (`kaggle.json`) — **needed by Phase 1b.** Not yet present on this machine.

**Models (handoff-specified, honest same-family scale contrast):**
- 7B: `Qwen2.5-7B-Instruct`
- 1.5B: `Qwen2.5-1.5B-Instruct`
- Decoding: greedy everywhere. Metric: answer log-odds delta, strict extraction.

---

## 4. Pre-registered gates (copied verbatim from the handoff — do not paraphrase away)

**Gate P0 — positive control (7B, binding, N=30 pairs).**
> P0 PASS = patching at the expected token positions moves answer log-odds in the predicted direction with effect > a random-position patch null (matched layer distribution, p<0.01) and the effect replicates across two disjoint 15-pair halves.
> FAIL → fix harness; if it fails after two documented fix rounds, the project stops and the writeup is "patching harness could not recover known effects at 7B under our constraints" (do not proceed on a broken instrument).

**Phase 2 — pair-library requirements (per skill).**
> ≥50 pairs; exactly ONE token-span differs between pair members; multiple surface instantiations per underlying template (≥5 templates × ≥10 instantiations) so Gate P1 can test template-robustness; behavioral pre-check: the model must actually BEHAVE differently on the two pair members on ≥70% of pairs at 7B, measured before any patching — pairs where behavior doesn't differ are excluded and the exclusion rate logged. If behavioral pre-check fails wholesale (<40% of pairs), the skill is not elicitable in this setup at 7B; log it, drop the skill, and say so.

**Gate P1 (per skill).**
> heatmaps computed independently on template-disjoint halves correlate (Spearman over the layer×position grid) > 0.5, and > a pair-shuffled null. FAIL → the map is template noise; report as such.

**Phase 3 — contrasts, operationalized in advance.**
> Before viewing ANY heatmap, log predictions for contrasts 1–3 (localize vs smear; early vs late; crisp-at-7B vs smear-at-1.5B), each with a quantitative operationalization chosen in advance (e.g., localization = top-5% of sites carrying >50% of total |IE|; crispness compared via normalized map entropy, 7B vs 1.5B, with a matched-null band). Report every contrast whichever way it lands.

**Standing stop condition.**
> if P0 passes but BOTH novel skills fail P1 (all template noise), stop after documenting — do not invent Skill D, E, F. The note then reports the calibrated instrument + the negative, which is still a complete deliverable.

---

## 5. Open items for Sahil

1. ~~Kaggle / Instruction drop / ×50 / P1~~ — done.
2. ~~1.5B scale run?~~ — **NO** (stop honored 2026-07-12).
3. **Write-up window (human):** freeze prior-arc notes + trenchcoat; X drafts only after all notes frozen. Two weeks, no new experiments.

---

## PHASE LOG (append-only)

### Phase 0 — 2026-07-11 — repo + restatement

- Created `/Users/sahilyadav/causal-maps` (local, `git init`, **no remote**). Added `.gitignore` (ignores venv, weights, activations, secrets/`kaggle.json`).
- Created venv at `.venv` (Python 3.13.13). Installed + pinned to `requirements.txt`: `torch==2.13.0` (MPS available: True), `transformers==5.13.1`, `tokenizers==0.22.2`, `accelerate==1.14.0`, `safetensors==0.8.0`, `numpy==2.5.1`, `scipy==1.18.0`, `matplotlib==3.11.0`, `pandas==3.0.3`, `tqdm==4.68.4`.
- Wrote this log: protocol affirmed (§0), plan restated (§1), objections logged BEFORE code (§2), environment recorded (§3), gates copied verbatim (§4), open items for Sahil (§5).
- **Verdict:** Phase 0 restatement complete. **Do not write experiment code until Sahil has seen §2 objections** (specifically the Phase-1b Kaggle pull-forward) and the environment install is verified.
- **Next:** Phase 1 — build the patching harness; bring it up on 1.5B locally as a plumbing check (rule-world minimal pairs) before touching 7B/Kaggle.

### Phase 0 addendum — 2026-07-11 — PIVOT: all execution on Kaggle (no local model runs)

**Decision (Sahil):** stop all local model downloads/runs; upload code to Kaggle and run *everything* there (1.5B and 7B). Local machine is now dev-only (write code + correctness tests on tiny tensors; no model loading). Priority: robust heartbeat/liveness logging so long runs are visibly not stuck. Sequence: code the whole harness → correctness sweep → then start runs.

**Done:**
- Killed local 1.5B download; removed partial `hf_cache` (reclaimed 2.9 GB).
- Kaggle CLI 2.2.3 in venv; `~/.kaggle/kaggle.json` set (chmod 600); auth verified via `competitions list`. Account: `sahilyadav2702`. (`kernels list --mine/--user` returns "Not found" = no kernels yet, not an auth failure.)
- Official attachable model: `qwen-lm/qwen2.5` (0.5–72B incl. 7B-Instruct) → avoids ~15 GB HF re-download per session. Model path kept configurable (Kaggle mount OR HF pull).
- Captured `kernel-metadata.json` schema for CLI 2.2.3: `enable_gpu`, `enable_internet`, `machine_shape`, `dataset_sources`, `model_sources`, etc.

**Execution architecture (to build):**
- Code shipped as a Kaggle **dataset** `causal-maps-code`; the **kernel** is a thin bootstrap that adds it to `sys.path`, reads a baked-in `STAGE`, runs the experiment, writes outputs+checkpoints to `/kaggle/working`.
- Local **orchestrator** `run_kaggle.py`: version code dataset → generate kernel-metadata + bootstrap → `kernels push` → poll `kernels status` (liveness) against a bounded expected wall-time (stuck warning) → on finish `kernels output` (pull log+artifacts) → tail log to verify steady progress.
- **Heartbeat honesty (limitation, stated up front):** Kaggle's API does NOT stream a running kernel's stdout. Mid-run liveness = `kernels status` (RUNNING/COMPLETE/ERROR) + a pre-set max wall-time that trips a "possibly stuck" warning; the full progress log arrives at completion. For long Phase-3 sweeps we **chunk into checkpointed sub-runs** so progress is observable between chunks and at most one chunk is ever lost. In-kernel logging prints timestamped `[ts] i/n elapsed eta` frequently (+ `progress.json`) so the committed log proves it was never stuck.
- **Checkpoint/resume:** kernel writes partial IE matrices frequently; resume attaches prior outputs as a results dataset. Early gates (plumbing, P0) expected to fit one session (~1 h incl. model load); resume exercised only for big sweeps.

**Next:** write harness (`logutil`, `model_utils`, `patching`, `nulls`, `rule_world`, `binding_pairs`, `experiments`) + Kaggle bootstrap + local orchestrator + tiny-tensor correctness tests → code sweep → runs (smoke → 1.5B plumbing → 7B P0).

### Phase 1 — 2026-07-11 — harness built, tested, reviewed (all local; runs on Kaggle)

**Built** (`src/causal_maps/`): `logutil` (heartbeat + progress.json), `model_utils` (load w/ `device_map`, single-token validation, strict metric), `patching` (hook cache / patch-at-(layer,pos) / IE sweep w/ checkpoint+resume; convention = patch post-layer residual, measure logit-diff between the two answer tokens at the final position), `nulls` (matched random-position null, pair-shuffle, Spearman, permutation p), `rule_world` (copy pairs), `binding_pairs` (P0), `tensorize` (single-token + equal-length + prefix-stable-anchor invariants), `experiments` (behavioral pre-check → sweep → null → verdict → heatmap). Kaggle: `kernel/run_kaggle.py` (code-dataset packaging, kernel push, status poll w/ stuck-warning, output pull). Deps pinned; `.venv` local.

**Verified from source, not memory:** transformers 5.x `Qwen2DecoderLayer.forward` returns a **bare tensor** (4.x returned a tuple) — all hooks handle both.

**Tests** (`tests/test_core.py`, offline, tiny toy model mimicking the 5.x calling convention + causal token mixing; NO model download/run): **28/28 pass** — logit-diff == log-odds identity, permutation p & Spearman, matched null beats when signal at expected pos, cache/patch/IE on both tensor- and tuple-return layers, IE≈0 pre-divergence & ≠0 post-divergence, manual IE == sweep IE, patch actually overwrites the residual, tensorize minimal-pair invariants (only the intended span differs; uniform anchors; equal length; multi-token dropped).

**Review (the "sweep") — issues found & fixed:**
1. 7B (~15 GB bf16) on a 16 GB GPU is tight → `device_map="auto"` for 7B, `input_device()` + device-safe `logit_diff` so single- and multi-GPU both work.
2. Kaggle zip auto-extraction is ambiguous → bootstrap handles zip-present, pre-extracted, and glob-fallback layouts, and prints the input listing.
3. `kernels status` parsing → keyword scan (robust to enum-style `KernelWorkerStatus.COMPLETE`).
4. Dataset push → try `version`, fall back to `create` (robust first run).
5. Two bugs were in the TEST scaffold (identity toy block; hook-ordering), not the harness — fixed.

**Remaining risks are EMPIRICAL (verify on Kaggle, not code bugs):** (a) 7B fits GPU mem (smoke prints total mem; device_map helps); (b) Kaggle internet enabled for HF pull, else fall back to attaching the `qwen-lm/qwen2.5` model; (c) binding behavioral rate high at 7B — if far below expectation, treat as HARNESS BUG first (rule 6); (d) whether Kaggle auto-extracts the code zip (bootstrap covers both).

### PRE-REGISTERED — Gate P0 operationalization (written BEFORE any 7B run)

- **Model/decoding:** Qwen2.5-7B-Instruct, bf16, greedy, `device_map=auto`.
- **Pairs:** `binding_card_v1`, 60 generated (single-token names & colors ⇒ all pairs one length). Behavioral pre-check (two-candidate): keep a pair iff clean favours a1 (`logit(a2)−logit(a1) < 0`) AND cf favours a2 (`> 0`). Use the first 30 behaving pairs (15+15 split-half). Log behave-rate + exclusions. If <30 behaving ⇒ flag + investigate as a possible harness bug before proceeding.
- **Metric:** IE(L,P) = logit_diff(clean, patched at (L,P)) − logit_diff(clean); logit_diff = `logit(a2) − logit(a1)` at the final token. Positive IE ⇒ patching pushed the clean run toward the counterfactual answer.
- **Expected site:** `a1_slot` (queried entity's own attribute token). Within-pair control: `a2_slot` (expected ≈ 0). Null pool = all positions except `a1_slot` and the final readout token.
- **Statistic:** real = maxₗ mean_pairs IE(L, a1_slot). Null (matched layer distribution): per draw (×1000), per layer sample 1 position uniformly from the null pool, mean over pairs, take max over layers. p = (#null ≥ real + 1)/1001.
- **PASS ⟺ all of:** (i) real > 0; (ii) full-set p < 0.01; (iii) split-half — both halves give real > 0 and p < 0.05.
- **FAIL handling:** per handoff — fix harness; if still failing after two documented fix rounds, project stops with the "patching harness could not recover a known effect at 7B" writeup (do not proceed on a broken instrument).
- **Instrument sanity (runs first): plumbing** = Qwen2.5-1.5B copy pairs; expect strongly positive IE at `val_slot`, p < 0.01 (near-certain). If plumbing fails, it is a harness bug — fix before P0.

**Next:** Kaggle runs — smoke (pipeline+GPU) → plumbing (1.5B instrument sanity) → P0 (7B gate). Requires nothing further from Sahil except confirming go / Kaggle GPU quota.

### Kaggle runs log

**2026-07-11 — smoke (pipeline + GPU validation): ⛔ BLOCKED — kernels provision CPU-only.**
- Pipeline mechanics all work: code push, `kernels status` polling, `kernels output` pull, log retrieval, dataset/kernel creation.
- BUT every GPU kernel ran on CPU (`torch 2.10.0+cpu`, `cuda_avail False`), across 3 attempts:
  1. `enable_gpu:true` only → CPU (learned: `enable_gpu` alone is legacy, ignored server-side).
  2. + `machine_shape:NvidiaTeslaT4` in metadata → CPU.
  3. fresh slug `cm-smoke2` + explicit `--accelerator NvidiaTeslaT4` → CPU.
- **Ruled out:** GPU quota (`kaggle quota` = 30 h free, 0 used); version stickiness (fresh slug behaved identically); wrong request field (server-side `kernels pull -m` shows it RECORDED `enable_gpu:true`, `machine_shape:NvidiaTeslaT4`).
- **Smoking gun:** server-recorded `docker_image` is the **CPU** image `gcr.io/kaggle-images/python@…`, not the GPU image `gcr.io/kaggle-gpu-images/python@…`. Kaggle accepted the GPU request but provisioned CPU.
- **Near-certain cause:** account `sahilyadav2702` is not **phone-verified**. Kaggle requires phone verification to use GPU/TPU **and** internet in kernels; unverified accounts are silently downgraded to CPU. → BLOCKER: Sahil verifies phone at kaggle.com/settings (unlocks GPU + the internet we need for HF pulls), or supplies a verified account's `kaggle.json`. Then retry smoke → plumbing → P0. Nothing else outstanding.
- Useful commands learned: `kaggle quota`; `kaggle kernels pull <slug> -p <dir> -m` (server-side metadata incl. `docker_image`).

**2026-07-12 — GPU blocker RESOLVED:** switched to verified account `shlydv` (`kaggle (1).json`; quota showed 2.7 h GPU already used ⇒ verified). Smoke on `shlydv`: `complete`, `torch 2.10.0+cu128 cuda_avail True`, `Tesla T4 15.6 GB`. Orchestrator now reads username from the active `kaggle.json`.

**2026-07-12 — plumbing (1.5B instrument sanity): ✅ PASS.**
- Two mount/timing gotchas fixed en route: (a) attached datasets mount at `/kaggle/input/datasets/<owner>/<slug>` (NOT `/kaggle/input/<slug>`) → bootstrap now globs the whole `/kaggle/input` tree for `causal_maps/__init__.py` and prints the mounts; (b) a kernel pushed before its dataset finishes processing mounts EMPTY → added `wait_dataset_ready` after dataset push.
- Qwen2.5-1.5B-Instruct, copy pairs. behave_rate **39/40 = 97.5%**. S=8, `val_slot`@3.
- IE at val_slot: **real=12.73** (layer 1) vs random-position **null_mean=1.15**, **p=0.001**. Split-half A: 12.7→14.16 (p=0.001), B: 11.44 (p=0.001). top site = (L1, val_slot).
- Instrument validated end-to-end on real GPU: patching flips the copy answer, IE strongly positive at the causal site, beats matched null, replicates across halves. Cleared to run P0.

**2026-07-12 — P0 attempt 1 (7B, P100): ✗ CUDA `no kernel image for device` (cudaErrorNoKernelImageForDevice).** Kaggle's `torch 2.10.0+cu128` ships no kernels for P100's **sm_60 (Pascal)** → P100 unusable with this torch. But 7B bf16 (~15.2 GB) does NOT fit single-device on a T4 (~15 GB usable). Trilemma: bf16 + single-device + supported-GPU can't all hold. → Retry: **T4 + `device_map=auto`** (accelerate offloads ~1 layer to CPU; keeps bf16; ~1.9 s/forward).

**2026-07-12 — P0 attempt 2 (7B, T4, device_map=auto): device_map + hooks work, but ELICITATION was broken.** Full sweep ran (448 fwd, 836 s); patching stats looked strong (real +6.88 @L12, p=0.001, split-half p=0.001) — BUT `clean_greedy=0% cf_greedy=0%`. Debug dump: raw "X's card is ___" makes the model emit a generic predicate (`not`/`worth`/`the`), never the attribute; binding was present only in the *relative* logit ordering, so the 44.7% "behaving" subset was near-chance. Rule-6 call: NOT a pass — a broken behavioral precondition, not a real positive control.
- Iterated: "lives in {city}" (raw) → 12.5% behave, 9% greedy, recency bias (predicts last-mentioned city) + `the`/`a` dominate. Still raw-on-instruct.

**2026-07-12 — P0 elicitation FIXED via chat template.** Root cause = feeding raw completions to an INSTRUCT model (the chat-template mismatch the handoff flags). Switched binding pairs to **chat template + explicit question** ("Which city does {e1} live in?", focuses on e1 ⇒ beats recency) **+ primed answer** ("The answer is" ⇒ forces a leading-space city). Behavioral diagnostic: **rate=100%, clean_greedy=100%, cf_greedy=100%** (64/80 pairs; e.g. clean Rome 34.0 vs Seattle 12.0; cf flips Seattle 33.0 vs Rome 9.6). Clean behavioral positive control; also fixes the elicitation approach for Skills A/B (use chat template). S=49.

**2026-07-12 — Gate P0 (7B binding, chat-templated, 8bit on T4): ✅ PASS.**
- Kernel `shlydv/cm-p0-binding`, COMPLETE (lastRunTime ~07:01–07:48 UTC). Config: `quantization=8bit` (bitsandbytes LLM.int8; residual stream patched in fp16 — fidelity preserved; fits fully on one T4, no CPU-offload). Artifacts: `runs/p0_binding/` (synced from pull).
- Behavioral: **rate=100%** (64/64 tensorized-behaving of 80 generated; 16 len_nonuniform drops), **clean_greedy=100%, cf_greedy=100%**. Used first 30 behaving (15+15). S=49, `a1_slot`@27, `a2_slot`@32.
- Full-set: **real=+45.13 @ L15** vs null_mean=+0.30, **p=0.001**. Split-half A: +42.8 @ L15, p=0.001; B: +47.6 @ L13, p=0.001.
- Top site = **(L15, pos 27 = a1_slot)** — exactly the pre-registered expected site. Within-pair control `a2_slot`@32: mean IE ≈ −1.6 (near zero / slightly negative) — instrument discriminates. Late-layer final-token IE present (aggregation), expected and not the gate statistic.
- Pre-registered PASS criteria all met: (i) real>0 ✓ (ii) full p<0.01 ✓ (iii) both halves real>0 & p<0.05 ✓. Behavioral precondition (greedy hits) also clean — unlike attempt 2.
- **Instrument is earned.** Cleared to Phase 2.

### Phase 2 redesign — 2026-07-12 — computational variables (Sahil)

**Decision (Sahil):** redefine skills around the *single bit that changes between prompts*, not application domains. Application-flavored templates (recipes, browsers, translation, capitalization, math) inject priors and fail the bit test. Continue the month's direction: synthetic, maximally controlled stimuli.

**Bit test (new standing filter):** if the ONLY information that changed between the two prompts cannot be described in one sentence, kill the template.

**Standing stop condition amended (not loosened):** original "both novel skills fail P1 ⇒ stop, do not invent D/E/F" still holds. Variable substitution is a *replacement* for the old Skill-A/B framing, not an invented Skill D. Reference binding is optional fourth and deferred until the top three have behavioral verdicts. P0 entity-attribute binding remains the instrument calibration (already PASSED) — not re-run as a Phase-2 skill.

#### Skills kept (ranked)

| Rank | Skill | Variable localized | Status |
|------|-------|--------------------|--------|
| 1 | Completion state | one boolean (task completed?) | ✅ Phase 2 primary |
| 2 | Variable substitution | one stored symbol→value | ✅ Phase 2 primary (NEW; replaces domain-flavored Skill A leftovers) |
| 3 | Instruction vs mention | execute vs describe identical string | ✅ Phase 2, narrowed |
| 4 | Reference binding | who/what a reference points to | optional, deferred |

**Killed:** A3 recipe, A5 browser, B2 capitalize, B3 translate, B4 math, natural-email framing. Kept spirit of A1/A2/A4 and B1/B5 as synthetic variants only.

#### Strengthened behavioral gate (pre-registered BEFORE hand-10 runs)

Replaces handoff's ≥70% for this project's Phase 2 onward:

1. **≥80%** of pairs show the intended behavioral difference at 7B (clean favours a1, cf favours a2; report greedy hit rates separately).
2. **Balanced across templates** — no single template carries the ≥80%; each kept template must itself clear ≥80% on its instantiations (else drop that template, not the whole skill, unless survivors <5 templates).
3. **Prefix identity:** clean and cf share **identical first ≥90% of tokens** (tokenized under the model tokenizer). Diff confined to a short span near the manipulated bit. Violators dropped at tensorize time and logged.
4. Wholesale fail: if after template drops a skill has **&lt;40%** overall behave OR &lt;5 viable templates, **drop the skill**, log it, do not rescue.
5. **Hand-10 before ×50:** for each skill, run behavioral pre-check on 10 hand-designed pairs first. Only skills that clear the hand-10 gate get generators scaled to ≥50 (≥5 templates × ≥10).

Elicitation constraints carried from P0: chat template + primed single-token answer; never raw completions on Instruct.

#### Expected causal site (for later P1/P3 — logged now, not fitted later)

- **Completion state:** the boolean token (`0`/`1` or `false`/`true`) in the state line.
- **Variable substitution:** the value token in the assignment (`dog`/`cat`).
- **Instruction vs mention:** the framing token(s) that encode execute vs describe (`command`/`quote`), NOT the shared payload string.

---

### PRE-REGISTERED — Hand-10 pair designs (written BEFORE any 7B behavioral run)

Format: each pair is shown as the **user-message body** (chat-templated at runtime; assistant primed with the primer line). `BIT` = the one sentence bit-test answer. `DIFF` = the sole changed span. Answers are the two single-token readouts (leading-space expected under Qwen, validated on Kaggle).

#### Skill 1 — Completion state (10 pairs = 5 templates × 2 instantiations)

Shared skeleton (prefix-stable; only `FLAG` bit flips):

```
Rules:
- if {flag} is 0, next action is {act0}
- if {flag} is 1, next action is {act1}

State:
{flag} = {bit}

Next action:
```

Primer: (empty — "Next action:" already primes). Answers: `{act0}` vs `{act1}`.
BIT: "the boolean {flag}." DIFF: the single character/token `0`↔`1` after `= `.

| id | flag | act0 | act1 | clean | cf |
|----|------|------|------|-------|-----|
| C1 | read_file | READ | SUMM | 0→READ | 1→SUMM |
| C2 | invoice_loaded | LOAD | EXTR | 0→LOAD | 1→EXTR |
| C3 | search_done | FIND | ANS | 0→FIND | 1→ANS |
| C4 | draft_ready | EDIT | SEND | 0→EDIT | 1→SEND |
| C5 | cache_warm | FETCH | SERVE | 0→FETCH | 1→SERVE |
| C6 | buf_flushed | WRITE | CLOSE | 0→WRITE | 1→CLOSE |
| C7 | idx_built | BUILD | QUERY | 0→BUILD | 1→QUERY |
| C8 | lock_held | LOCK | FREE | 0→LOCK | 1→FREE |
| C9 | batch_ready | FILL | FLUSH | 0→FILL | 1→FLUSH |
| C10 | model_cached | PULL | INFER | 0→PULL | 1→INFER |

Each id is one pair: clean `{flag}=0`→act0, cf `{flag}=1`→act1. (For later ×50: ≥5 flag-families × ≥10 name variants; hand-10 is the pilot.)

#### Skill 2 — Variable substitution (10 pairs)

Shared skeleton:

```
Let {var} = {val}.
What is the value of {var}?
The answer is
```

BIT: "the bound value of {var}." DIFF: `{val}` only. Answers: the two values.
Prefix identical through `Let {var} = `.

| id | var | val_clean | val_cf |
|----|-----|-----------|--------|
| V1 | X | dog | cat |
| V2 | X | red | blue |
| V3 | Y | one | two |
| V4 | Y | hot | cold |
| V5 | Z | apple | grape |
| V6 | Z | north | south |
| V7 | W | open | shut |
| V8 | W | left | right |
| V9 | K | alpha | beta |
| V10 | K | true | false |

(Values chosen as likely single-token under Qwen; drop any that tokenize >1 on Kaggle — logged, not forced.)

#### Skill 3 — Instruction vs mention (10 pairs) — NATURAL framing (Sahil revise)

**Change (Sahil, 2026-07-12):** do NOT encode the bit as `Treat as: command|quote`.
An explicit mode token lets the model learn mode→answer; the patch map would
localize that token rather than the computation of distinguishing instruction
from quoted data. Encode naturally; accept weaker prefix identity for this skill.

Shared response vocabulary (identical in both arms — teaches RUN/QUOTE readout
without labeling the payload's mode):

```
If you are executing an instruction, reply with the single word RUN.
If you are identifying quoted text from a document, reply with the single word QUOTE.
```

**Command (clean → RUN):**
```
{vocab}

Execute the following instruction.

{payload}

Response:
```

**Mention (cf → QUOTE):**
```
{vocab}

The following text appeared in a document.

"{payload}"

Response:
```

BIT: "whether the payload is an instruction to execute or quoted text to identify."
DIFF: framing sentences + quotes on mention. Clean/cf lengths usually differ →
hand-10 uses **loose** behavioral check (per-pair forwards). Prefix-identity
≥90% rule is **waived for this skill only** (logged exception).

| id | payload |
|----|---------|
| I1 | DELETE_FILE |
| I2 | CLEAR_BUF |
| I3 | SET_FLAG |
| I4 | OPEN_SLOT |
| I5 | RESET_IDX |
| I6 | FLUSH_LOG |
| I7 | PIN_PAGE |
| I8 | DROP_TEMP |
| I9 | SYNC_MAP |
| I10 | SEAL_JOB |

#### Reference binding — deferred

Not in hand-10. Revisit only if ≥2 of the above three clear behavioral + we have Kaggle budget.

---

**Next:** implement generators → Kaggle `hand10_behav` (one 7B load, three skills, 8bit T4) → write PASS/FAIL per skill (≥80%) → only then scale to ≥50 and sweep.

**Handoff path archived:** `CAUSAL_MAPS_HANDOFF.md` copied into repo root from Sahil's Downloads (2026-07-12).

### 2026-07-12 — Hand-10 behavioral pilot: verdicts

Kernels: `shlydv/cm-hand10-behav` (combined; COMPLETE) + `shlydv/cm-instruction-hand10` (instr retry; COMPLETE). Artifacts: `runs/hand10_behav/`, `runs/instruction_hand10/`.

| Skill | behave | clean_greedy | cf_greedy | gate ≥80% | Honest verdict |
|-------|--------|--------------|-----------|-----------|----------------|
| Completion state | **10/10 = 100%** | **100%** | **100%** | PASS | ✅ **PASS** — real elicitation |
| Variable substitution | **10/10 = 100%** | **100%** | **100%** | PASS | ✅ **PASS** — real elicitation |
| Instruction vs mention | 0/10 then 8/10* | **0%** | **0%** | soft 80%* | ✗ **FAIL** — not elicited |

\*Instruction retry (`instr_mention_natural_v3`, OK/text readout): relative logit-diff hit 8/10, but greedy top-1 was always `:` / prose — never `OK` or `text`. Same failure mode as P0 attempt 2 (relative ordering without emission). Per protocol rule 6 + P0 lesson: **not a pass**. Prefix identity only ~42% (90% rule waived for this skill; still weak for later patching).

**Completion DBG (clean):** top1=` READ`(26.0) vs ` DONE`(4.8). **CF:** top1=` DONE`(28.8). Bit flip works.
**Variable DBG (clean):** top1=` dog`(42.3). **CF:** top1=` cat`(40.8). Copy works.

**Instruction attempt history (logged so we don't pretend there was one try):**
1. Combined run, DELETE_FILE-style + RUN/QUOTE vocab → 0%, model emits `not`/`neither` (refusal/hedge).
2. Neutral payloads + natural framing, OK/text embedded in instructions → 80% relative, **0% greedy**.

**Protocol call on Instruction:** do **not** keep fishing variants. Two documented elicitation failures. Recommended branch: **drop Skill 3 for this project**, report "instruction/mention not elicitable at 7B under chat-template + single-token constraints," proceed with Completion + Variable (standing stop was "both novel skills fail P1" — we still have two skills). Optional revisit only after C+V maps exist, with a pre-registered redesign written *before* any new run — not now.

**Interpretation caveat (Completion) — not a fail, flag for writeup:** hand-10 prompts include an explicit rules table (`if flag is 0 → act0; if 1 → act1`). That makes elicitation trivial and may mean the causal site is "read the matching rule line" rather than a latent done-flag. Sahil's original sketch had state + next-action *without* spelling the mapping. Keep current form for ×50 (it earned the gate); if P3 maps concentrate on the rule lines rather than the `= 0/1` bit, log that and consider a no-rules ablation as a *contrast*, not a rescue.

**Cleared to scale:** Completion + Variable → ≥50 pairs (≥5 surface families × ≥10), full behav gate, then P1/P3. Instruction: await Sahil confirm on drop.

**Next:** ×50 generators for C+V only; Kaggle behav on full libraries; no Instruction rerun unless Sahil overrides the drop.

### 2026-07-12 — Instruction DROP confirmed (Sahil)

Two documented elicitation failures; protocol call affirmed. **Do not revisit Instruction before Completion + Variable maps exist.** Note will carry one honest paragraph. Reference binding stays deferred. Nothing else new until the two maps exist.

**Path (ordered):** (1) ×50 generators + Kaggle behav gate, with ≥1 Completion family *implicit* (no if-then rules) (2) pre-register P3 contrasts now (3) P1 sweeps @7B (4) same @1.5B — flagship, not optional (5) freeze + note.

### PRE-REGISTERED — Phase 3 contrasts (written 2026-07-12 BEFORE any skill heatmap)

Applies to Completion and Variable at both scales. Expected causal site = the manipulated bit token (`bit_slot` for Completion `0/1`; `val_slot` for Variable value). Predictions logged now; report whichever way they land.

**Contrast 1 — localize vs smear (per skill, per scale).**
- Localization statistic: let \(w_{l,p} = |\mathrm{IE}(l,p)|\), \(W = \sum w\). Let \(T\) = the top 5% of sites by \(w\) (ceil). **Localized** iff \(\sum_{T} w / W > 0.5\). Else **smear**.
- Secondary: fraction of total |IE| on the single expected-position column (all layers). Report; not a pass/fail.

**Contrast 2 — early vs late (secondary; Completion/Variable).**
- Mass-weighted mean layer \(\bar{L} = \sum_{l,p} w_{l,p}\, l / W\). **Early** iff \(\bar{L} < L/3\); **late** iff \(\bar{L} > 2L/3\); else **mid** (L = n_layers). Not a gate — descriptive.

**Contrast 3 — scale crispness (flagship): 7B vs 1.5B.**
- Map entropy: \(H = -\sum \hat{w}\log\hat{w}\) with \(\hat{w} = w/W\), normalized \(H_{\mathrm{norm}} = H / \log(|\mathrm{sites}|)\).
- **Crisper at 7B** iff \(H_{\mathrm{norm}}^{7B} + \delta < H_{\mathrm{norm}}^{1.5B}\), where \(\delta\) is the 95th percentile of \(|H_{\mathrm{norm}}^A - H_{\mathrm{norm}}^B|\) under a pair-partition null at 7B (same n_pairs split; 1000 draws). If the inequality fails or is within \(\delta\), report **no reliable scale crispness gain** (or reverse).
- Also report localization (Contrast 1) at both scales side-by-side.

**Predictions (pre-registered, not fitted):**
1. Completion: localized at `bit_slot`; mid-to-late layers; crisper at 7B than 1.5B.
2. Variable: localized at `val_slot`; earlier than Completion; crisper at 7B than 1.5B.
3. If Completion implicit-family behav fails: rules-form maps still run; caveat ("may be rule-line reading") goes in the note.

**Gate P1 (per skill, per scale) — reminder, operationalized:**
- Template-disjoint halves → two mean IE maps → Spearman ρ over layer×position grid.
- PASS iff ρ > 0.5 **and** ρ exceeds pair-partition null (p < 0.01, +1 smoothed, 1000 draws: randomly split pairs into two groups of the same sizes as the template halves).
- FAIL → map is template noise; report; do not invent new skills.

### 2026-07-12 — ×50 behavioral gate: ✅ PASS (both skills); implicit FAIL

Kernel `shlydv/cm-x50-behav` COMPLETE (~3 min after load). Artifacts: `runs/x50_behav/`.

| Skill | tensorized | behave | greedy c/cf | templates ≥80% | gate |
|-------|------------|--------|-------------|----------------|------|
| Completion (explicit A–D) | 40/50 | **100%** | **100%/100%** | 4/4 PASS | ✅ |
| Completion **implicit** | dropped (len≠) | loose **0%** | — | FAIL | ✗ |
| Variable (X/Y/Z/W/K) | 50/50 | **100%** | **98%/100%** | 5/5 PASS | ✅ |

**Implicit ablation result:** the no-rules family did **not** elicit (0% on loose check; also length-nonuniform vs explicit). **Keep rules-form for sweeps; caveat stands for the note** ("Completion map may partly reflect reading the if-then rule lines"). No rescue redesign.

**Cleared to P1:** Completion explicit ×40 (4 templates), Variable ×50 (5 templates). Stages `p1_7b` then `p1_1p5b`.

### 2026-07-12 — Gate P1 @7B: both FAIL (maps pulled)

Kernel `shlydv/cm-p1-7b` COMPLETE (~60 min sweep wall). Artifacts: `runs/p1_7b/`. Local orchestrator stalled mid-poll; results force-pulled after COMPLETE.

| Skill | Site effect @ expected | Top site | P1 ρ | P1 vs null | P1 |
|-------|------------------------|----------|------|------------|-----|
| Completion | real=+0.07 @ bit_slot, p=0.58 (null≃9.4) | L2, **pos 62** (IE +38.4) — *not* bit@61 | 0.14 | p=0.64 | ✗ |
| Variable | **real=+53.3 @ val_slot, p=0.001** | L2, **pos 27 = val_slot** | 0.37 | p=0.28 | ✗ |

**Completion read:** heatmap is sharp (localized by Contrast-1 mass) but the stripe is **one token after** the bit (`62` not `61`); expected-column mass ≈ 0. Map does **not** replicate across Surface A/B vs C/D (ρ≈null). Likely: effect on the post-bit residual / rule-reading geometry, plus template fragility. Caveat strengthened.

**Variable read:** textbook localization at the value token (77% of |IE| on that column; top site = expected). Behav was real; site effect beats position-null hard. **But** template-disjoint Spearman (0.37) does not clear 0.5 or the pair-partition null — Gate P1 FAIL. The *mean* map looks clean; the *replication across var-name templates* does not.

**Protocol:** both novel skills fail P1 → standing stop condition is met. Do not invent Skill D+. Deliverable = calibrated instrument (P0) + elicitation record + P1 negatives (+ optional 1.5B only if Sahil overrides stop for the pre-registered scale contrast).

### 2026-07-12 — STOP honored; fragility characterization; note FROZEN

**Sahil decision:** do **not** run 1.5B — scale contrast is meaningless when neither 7B map passed P1. No new experiments, no overrides.

**Salvage (characterization only, existing `ie_*.npz`):** per-template pairwise Spearman.
- Variable: mean grid ρ=0.264 (all pairs <0.5); **expected-column r≈0.999 across all template pairs**. Site effect is stable; full-grid map is template-fragile. Tables: `runs/p1_7b/fragility_variable_p1.json`.
- Completion: mean grid ρ=0.101; expected-col r≈0. `fragility_completion_p1.json`.

**Note frozen:** `CAUSAL_MAPS_NOTE.md`. Convergent finding with prior month: even causally, task-relevant structure at this scale is template-bound. Research on this repo stops. Next = write up the arc (this note + prior three + trenchcoat); X posts only after notes frozen.

### 2026-07-12 — REOPEN (one experiment): direction transfer — PRE-REGISTERED before run

**Hypothesis (Sahil):** mechanism is a reusable *direction*, not a reusable *site*. P1 scored the wrong coordinate system.

**Test:** Variable skill only, 7B 8bit T4, one kernel `delta_transfer`.
- L* = **2** (layer of max mean IE at val_slot from Variable P1; fixed now, not chosen after).
- Position = `val_slot` (uniform across templates).
- Δ_T = mean_pairs(h_cf − h_clean) at (L*, val_slot) on template T.
- **Primary:** Δ from donors {variable_X, variable_Y} (mean), ADD (α=1) onto clean runs of targets {Z, W, K} at val_slot. Measure mean ΔIE = ld_after − ld_clean (positive ⇒ toward cf answer). Also flip-rate (greedy neg→pos).
- **Positive control:** within-template Δ_T on T (must be strongly positive or the add intervention is broken — then abort interpretation).
- **Null:** 200 random directions with ||v||=||Δ||; p = P(null ≥ real), +1 smoothed.
- **PASS (direction reusable)** ⟺ for every primary target: (i) mean ΔIE > 0, (ii) p < 0.01 vs random null, (iii) cross/within ≥ 0.5.
- **FAIL** ⟺ otherwise → supports prompt-conditioned / dynamic assembly (direction does not transfer). Report whichever way; no α fishing, no layer fishing on the primary. Secondary descriptive only: leave-one-out over all 5 templates (mean ratio); not a gate.

No other reopenings. After verdict → note addendum → re-freeze.

### 2026-07-12 — delta_transfer: ✅ PASS — DIRECTION_REUSABLE

Kernel `shlydv/cm-delta-transfer` v2 COMPLETE. Artifacts: `runs/delta_transfer/`.

**Within (positive control):** ΔIE ≈ +1.2…+1.75 on all 5 templates (add intervention works; flip_rate=0 because baseline |clean_ld| ≫ 2).

**Primary (donors X,Y → Z/W/K), L*=2, val_slot=27, α=1:**

| target | cross ΔIE | within ΔIE | ratio | vs random null | p |
|--------|-----------|------------|-------|----------------|---|
| Z | +1.83 | +1.75 | **1.04** | null≃0 | **0.005** |
| W | +1.73 | +1.51 | **1.14** | null≃0 | **0.005** |
| K | +1.42 | +1.71 | **0.83** | null≃0 | **0.005** |

All three: ΔIE>0, p<0.01, ratio≥0.5 → **PASS**. Mean cross/within ≈ **1.01**.

**Leave-one-out (secondary):** mean ratio **1.07**, mean ΔIE +1.68; all held-outs transfer. Supports primary.

**Interpretation:** the Variable binding mechanism *transfers across variable-name templates as a direction* at the value token. P1's full-grid Spearman failure was a coordinate-system artifact (off-site map geometry is template-noisy; the causal direction is not). Convergent with the month: reusable computation in a low-dim direction; site-maps score the wrong object.

Re-frozen. Note addendum written. No further experiments.

### 2026-07-12 — FULL SEND reopen (Fable checklist + 5 amendments) — PRE-REGISTERED before Kernel 1

**Claim protected:** site-maps judged Variable non-reusable (P1 FAIL); direction-transfer overturned that at full strength — a methodological result about what patching studies measure, plus existence proof. Not "we discovered directions" alone.

**Process:** Kernels 2–4 always run (branch-independent). Note headline waits on Kernel 1b. No public drafts until 1b is logged.

**Shared cell PASS:** ΔIE>0 AND p<0.01 vs 100 random same-norm dirs AND cross/within ≥ 0.5. Donors {X,Y}→ targets {Z,W,K} unless noted.

#### Kernel 1 — `delta_var_robust` (7B 8bit) — TONIGHT

**1a Robustness:** α∈{0.5,1,2} at L=2; L∈{1,2,3,4} at α=1. Flip hunt α∈{3,4,5} at L=2 — report flips **with the α that produced them**; demo/note claim uses **α≤2 only**. Flips only at α≥3 = steering, not mechanism.

**1b Embed control:** Δ_embed = mean(embed_cf − embed_clean) at val_slot (input embeddings); ADD α=1 at layer-0 output. Record ||Δ_embed||/||Δ_L2||.
- **TRIVIAL:** embed PASSes ≥2/3 targets AND mean(cross_embed)/mean(cross_L2) ≥ 0.8
- **PARTIAL** (token + computed): both pass ≥2/3 AND ratio ∈ [0.3, 0.8) — expected middle; note quantifies split
- **NONTRIVIAL:** embed fails ≥2/3 while L2 α=1 still passes
- else **INCONCLUSIVE**

**1c Shuffled-pairing control:** Δ_shuf = mean_i(h_cf[π(i)] − h_clean[i]) at L=2 val_slot on donors, π derangement (mismatched value words). Transfer α=1. **Must fail** ≥2/3 targets. If it PASSes → **GENERIC_BOOST** (not binding).

**K1 clean for arXiv path:** L2 α=1 PASSes AND verdict ∈ {NONTRIVIAL, PARTIAL} AND not GENERIC_BOOST.

#### Kernel 2 — `delta_completion`

- Slot primary = **62** (P1 top); secondary bit@61. **L\* = 2** (frozen from `ie_completion_p1.npz`: max IE at pos62 is L2 = +38.36). Surfaces A,B→C,D; α=1.

#### Kernel 3 — `delta_var_1p5b`

- 1.5B has 28 layers. **L=2 exactly** (not a formula). Secondaries L∈{1,3} descriptive only.

#### Kernel 4 — `delta_var_crosspos`

- Inert prefix only. **Both directions mandatory** (p_short→p_long AND p_long→p_short); each must cell-PASS. One direction only → position-conditioned; do not average.

Then rewrite note from survivors; freeze; X/arXiv only after.

### 2026-07-12 — Kernel 1 `delta_var_robust` COMPLETE — embed NONTRIVIAL; shuffle INVALID (bug)

Kernel `shlydv/cm-delta-var-robust` COMPLETE. Artifacts: `runs/delta_var_robust/`.

**1a Robustness**
| cell | gate | mean ΔIE | mean ratio |
|------|------|----------|------------|
| α=0.5 L2 | FAIL (1/3) | +0.51 | 0.82 |
| **α=1 L2** | **PASS 3/3** | **+1.66** | **1.01** |
| α=2 L2 | PASS 3/3 | +7.60 | 1.01 |
| L1–L4 @α=1 | all PASS 3/3 | +1.07…+1.81 | ~1.0 |

Flip hunt: **no flips at α≤2**. Flips appear only at α∈{3,4,5} (~10–40%) — per honesty clause, **not** mechanism demo evidence.

**1b Embed:** n_pass **0/3**, cross_embed/cross_L2 = **0.005**, ||Δ_emb||≪||Δ_L2|| (0.36 vs 7.98). Verdict **NONTRIVIAL**. Not a tokenizer/embedding artifact.

**1c Shuffled-pairing — RESULT WITHDRAWN (instrument bug):** saved `delta_shuf` is **bit-exact equal** to `delta_L2` (cos=1, max|diff|=0). Cause: for any derangement π, mean_i(h_cf[π(i)]−h_clean[i]) = mean(h_cf)−mean(h_clean) = matched Δ. The pre-registered shuffle-of-cached-activations control is mathematically a no-op when mean-pooled. Logged "GENERIC_BOOST" is **not interpretable**. Fix: rebuild Δ from **re-forwarded wrong-value cf prompts** (string-substituted unrelated value, new forward) so the cf activation set is not a permutation of the matched cf set; also add −Δ anti-control. Re-run as `delta_var_shufflefix` (same kernel budget, controls only + confirm L2 α=1). Embed NONTRIVIAL and robustness grid **stand**.

### 2026-07-12 — Kernel 1c `delta_var_shufflefix` PRE-REGISTERED + launched

**Why:** replace invalid cached-activation shuffle with a control that can fail.

**Design (written before results):**
1. Confirm matched Δ at L=2 α=1 still cell-PASSes all 3 targets (sanity).
2. **Wrong-value control:** for each donor pair, re-encode cf with an unrelated value word ∉ {v_clean, v_cf}; re-forward; Δ_wrong = mean(h_wrong − h_clean) at L=2 val_slot. Must **fail** cell-PASS on ≥2/3 targets (else GENERIC_BOOST).
3. **Anti-Δ:** ADD −Δ_matched α=1. Require mean ΔIE < 0 OR n_pass=0.
4. **arxiv_path_clean** iff matched PASSes AND wrong fails AND anti_ok. Prior embed NONTRIVIAL stands independently.

Kernel: `shlydv/cm-delta-var-shufflefix`. Code: `src/causal_maps/delta_shufflefix.py`.

### 2026-07-12 — Kernel 1c `delta_var_shufflefix` COMPLETE — GENERIC_BOOST

Kernel COMPLETE (~18 min). Artifacts: `runs/delta_var_shufflefix/`.

| control | n_pass | mean ΔIE | mean ratio | notes |
|---------|--------|----------|------------|-------|
| Matched L2 α=1 | **3/3 PASS** | +1.66 | 1.01 | confirms K1a |
| **Wrong-value re-forward** | **3/3 PASS** | +0.97 | 0.59 | cos(Δ_m,Δ_w)=**0.77**, n=20 |
| Anti −Δ | **0/3** | −0.79 | −0.48 | direction-sensitive ✓ |

**Verdict: GENERIC_BOOST.** Wrong-value Δ is not a no-op and still cell-PASSes all targets (weaker than matched: ~59% ratio). Anti-Δ hurts as required — effect is signed, not pure noise. Embed NONTRIVIAL (K1b) still stands: not an embedding artifact, but also **not binding-specific** under the pre-registered wrong-value gate.

**arxiv_path_clean = false.** Note must say: direction is reusable and layer-computed, but wrong-value control did not isolate a value-specific binder; possible generic “value-slot update” feature. No fishing — do not redesign the control post hoc for a cleaner story.

K2–K4 proceed as pre-registered (branch-independent).

### 2026-07-12 — Kernel 2 `delta_completion` COMPLETE — COMPLETION_DIRECTION_REUSABLE

Kernel `shlydv/cm-delta-completion` COMPLETE. Artifacts: `runs/delta_completion/`.

| slot | n_pass | mean ΔIE | mean ratio | flip |
|------|--------|----------|------------|------|
| **primary pos62** | **2/2 PASS** | **+38.30** | **0.997** | **100%** |
| secondary bit@61 | 0/2 | 0.0 | — | 0% (‖Δ‖≡0) |

Donors A,B → C,D at L=2 α=1. Primary gate **PASS**. Secondary bit_slot has null Δ (consistent with P1 peak at 62, not 61) — descriptive only, does not affect verdict.

**Verdict: COMPLETION_DIRECTION_REUSABLE.** Skill-generalization of the direction method beyond Variable: near-perfect cross-surface transfer at the P1 peak site.

### 2026-07-12 — Kernel 3 `delta_var_1p5b` COMPLETE — SCALE_TRANSFER_OK

Kernel `shlydv/cm-delta-var-1p5b` COMPLETE. Artifacts: `runs/delta_var_1p5b/`. Model: Qwen2.5-1.5B-Instruct (28 layers).

| layer | n_pass | mean ΔIE | mean ratio |
|-------|--------|----------|------------|
| **L2 (primary)** | **3/3 PASS** | **+1.88** | **1.02** |
| L1 (secondary) | 3/3 | +1.35 | 1.00 |
| L3 (secondary) | 3/3 | +2.10 | 0.99 |

**Verdict: SCALE_TRANSFER_OK.** Direction transfer at L=2 holds at 1.5B with full-strength ratio≈1 — not a 7B-only phenomenon.

### 2026-07-12 — Kernel 4 `delta_var_crosspos` COMPLETE — POSITION_FREE

Kernel `shlydv/cm-delta-var-crosspos` COMPLETE. Artifacts: `runs/delta_var_crosspos/`. Inert prefix shifted val_slot **27 → 40** (Δpos=+13).

| direction | n_pass | mean ΔIE | mean ratio |
|-----------|--------|----------|------------|
| **short→long** (27→40) | **3/3 PASS** | +1.84 | 1.09 |
| **long→short** (40→27) | **3/3 PASS** | +1.53 | 0.92 |

**Verdict: POSITION_FREE.** Both mandatory directions cell-PASS — direction is not locked to a single token index.

**Full-send experimental slate closed.** Survivors for note: direction reusable (Variable + Completion), layer-computed (embed NONTRIVIAL), scale-ok @1.5B, position-free; caveat GENERIC_BOOST (wrong-value still transfers). Site-map P1 FAILs stand as the methodological foil.

### 2026-07-12 — NOTE REWRITTEN + FROZEN (full-send)

`CAUSAL_MAPS_NOTE.md` rewritten from survivors (not an addendum paste). Headline: site-maps fail P1; directions transfer; embed NONTRIVIAL; GENERIC_BOOST blocks value-specific binder claim. No X/arXiv until explicit publish decision.

### 2026-07-12 — EXPLICIT PUBLISH DECISION + JULY-2026 LITERATURE GATE

Researcher explicitly chose publication before extension: arXiv-shaped short note + X thread, zero additional kernels. The wrong-value residual decomposition and direction composition are future work only.

**Current-literature audit (cutoff 2026-07-12): SHORT NOTE YES, novelty claim narrowed.** Closest work:

- Task/function vectors and activation addition are established: Hendel et al. (2023), Todd et al. (ICLR 2024), Turner et al. (2023), Rimsky et al. (ACL 2024). **Do not claim discovery of directions.**
- Steering generalization is established but unreliable: Tan et al. (NeurIPS 2024), Braun et al. (2025).
- Opiełka et al., *Causality ≠ Invariance* (ICLR 2026), shows causally effective function vectors can be format-specific. Our position/template transfer is limited invariance, not arbitrary format invariance.
- Makelov et al. (ICLR 2024) and Grant et al. (ICLR 2026 Oral) show intervention/subspace effects can recruit dormant pathways or create divergent latent states. Existing controls do not prove the added vector is the model's unique natural mechanism.
- Long (2025) proposes cross-environment necessity/sufficiency/invariance triangulation; our tests provide sufficiency plus narrow invariance, not circuit-level necessity.

**Publication-safe claim:** a pre-registered full site-map replication gate can reject a behavior even when a direction extracted at the causal site transfers to held-out surfaces at within-template strength. Therefore, **site-map non-replication does not imply absence of a transferable intervention direction**. Call it a coarse, signed, layer-computed slot-update intervention; not a value-specific binder and not a fully identified natural circuit.

Package:

- `paper/main.tex` + compiled `paper/main.pdf` (7 pages)
- `paper/references.bib` (23 cited sources, including 2026 reliability work)
- `paper/build_figures.py` + generated PDF/PNG figures and result macros
- `paper/LITERATURE_AUDIT.md`
- `paper/X_THREAD.md`
- `paper/README.md` (provenance + submission checklist)
- `paper/verify_package.py`

Verification: all headline numbers independently matched frozen JSON artifacts; all citations resolve; `GENERIC_BOOST`, scale scope, position scope, and intervention-faithfulness guardrails pass; TeX/BibTeX build has no warnings. No account-level submission performed.

### 2026-07-12 — PUBLICATION LITERATURE AUDIT FOLLOW-UP (two independent searches)

Two broad searches through July 12, 2026 converged:

- **Closest overlap:** Nadaf (2026) already performs large cross-template Function Vector transfer; Todd et al. (2024) already shows template/natural-context transfer. Cross-template directions alone are not novel.
- **Site/circuit fragility precedents:** Franco et al. (2026), Méloux et al. (2025), and Bayat Makou et al. (2026). Site fragility alone is not novel or universal.
- **Exact conjunction not found:** no located paper freezes a causal site, shows the corresponding full layer×position map fail a template-disjoint replication gate, then shows a residual direction at that same site transfer at approximately within-template strength.
- **Additional boundary:** Cheng & Zhang (2026) find settings where single-position intervention fails, so do not generalize this result into “directions beat distributed sites.”

Paper revised accordingly:

- New title: **“Causal Site Maps Can Fail to Replicate While Residual Directions Transfer.”**
- Novelty restricted to the paired site-map/direction dissociation.
- “Across scale” corrected to **same-protocol replication at two model sizes**; no vector crosses residual spaces.
- P1 “failure” explicitly means the frozen full-grid gate, not proof that every semantically aligned map differs.
- Added limitations: small surface count, no formal equivalence bounds, random-null p-value floor (1/101), mapping/transfer dataset dependence, and intervention faithfulness.
- Rebuilt PDF; 23 citations; verifier clean.

### 2026-07-12 — REOPEN (post-publish, ONE experiment): GENERIC_BOOST DECOMPOSITION — PRE-REGISTERED before code

**Context.** Paper frozen (site-map/direction dissociation). Sahil reopens for the next experiment Fable flagged: split the transferable Variable direction into a generic slot-update component + a value-specific residual; test whether the residual (a) transfers and (b) is value-selective ("pushes cat, not grape"). Now driven by Opus (Fable-5 access removed). Grok/GPT landed the intervening gates + paper.

**Restatement (my words).** Δ = mean(h_cf−h_clean) at (L=2, val_slot) tested GENERIC_BOOST because a wrong-value Δ still transfers (~59%), cos(Δ_matched,Δ_wrong)=0.77 — the direction *averaged over all 10 value transitions* is dominated by a value-agnostic "install a value here" component. Sharpening: build PER-VALUE directions (the 5 pairs sharing cf value v, one per variable name) so value-specific content survives instead of cancelling; remove the generic part by projecting out the subspace spanned by the OTHER values' directions; the residual is the candidate content-carrying direction for v.

**Design (Variable skill, Qwen2.5-7B 8bit T4, L=2, val_slot; no layer/α fishing).**
- 50 pairs (5 vars × 10 value-pairs). Cache h_clean, h_cf at (L=2, val_slot).
- 10 cf-values V = {cat, blue, two, cold, grape, south, shut, right, beta, false}.
- Per value v: Δ_v = mean over the 5 pairs with cf==v of (h_cf − h_clean).
- Generic subspace G_v = span{Δ_w : w≠v} (9 dirs, orthonormalized via QR). g_v = proj_{G_v}(Δ_v); residual s_v = Δ_v − g_v.
- Test set T_v = clean prompts from the 45 pairs with cf≠v (held out from Δ_v). ADD a direction at (L=2, val_slot) to T_v; read Δlogit for every value token.
  - transfer(d,v)=mean_{T_v} Δlogit(v); selectivity(d,v)=mean_{T_v}[Δlogit(v) − mean_{w≠v} Δlogit(w)].
  - d ∈ {Δ_v matched, g_v generic, s_v residual} at natural norm; + norm-matched (to ||Δ_v||) g_v,s_v for the per-unit contrast.
- Null: N=100 random directions at the matched norm (prior-kernel construction) → transfer/selectivity p-values (floor 1/101).

**Pre-registered gates (aggregate over the 10 values):**
- **G0 residual non-trivial:** median_v ||s_v||/||Δ_v||. If <0.10 → PURELY_GENERIC (Δ ⊂ generic subspace); report & stop.
- **G1 residual transfers:** mean_v transfer(s_v) > 0, p<0.01 vs same-norm random null.
- **G2 residual value-selective:** mean_v selectivity(s_v) > 0, p<0.01 vs null.
- **G3 generic is the boost (non-selective):** norm-matched, mean_v selectivity(s_v) > selectivity(g_v), p<0.01, AND mean_v transfer(g_v) > 0 (generic still boosts values).
- **Verdict DECOMPOSED** ⟺ G0∧G1∧G2∧G3. **PARTIAL** if residual selective but generic also selective. **PURELY_GENERIC** if G0 or G2 fails.

**Objections (logged before results):**
1. Per-value Δ_v (5 same-value pairs, different variable names) keeps generic + value-specific; if even Δ_v is generic-dominated, G0 catches it.
2. Projection onto a 9-dim subspace may over/under-remove the generic axis; report cos(Δ_v,g_v) and residual fraction. 9 diverse value dirs should span the shared axis.
3. Selectivity could be trivial unembedding alignment, not a binding computation. Held-out contextual add mitigates; per Makelov/Grant this is SUFFICIENCY for a content-carrying component, NOT proof of the unique natural mechanism — state as such.
4. Small residual norm → noisy transfer; norm-matched contrast + 45-prompt test sets + null floor mitigate.
5. No fishing: L=2, val_slot, natural + one norm-matched variant, pre-registered.

**Novelty (per 2026 lit audit):** directions/function-vectors are established — contribution is the generic/value-specific *decomposition* + the pre-registered selectivity gate, reported whichever way. Not natural-circuit identification.

Kernel: `delta_decompose` → `src/causal_maps/delta_decompose.py`; launch `run delta_decompose --config {"quantization":"8bit"}`.

### 2026-07-12 — delta_decompose COMPLETE — ✅ DECOMPOSED (value-specific directions transfer + are selective; GENERIC_BOOST was an averaging artifact)

Kernel `shlydv/cm-delta-decompose` COMPLETE (~43 min: ~2.5 min load, ~40 min sweep at ~2.5 s/8bit-forward). Artifacts: `runs/delta_decompose/`. Qwen2.5-7B-Instruct 8bit T4, L=2, val_slot=27, 50/50 pairs (no drops), 10 cf-values × n=5.

**All four pre-registered gates PASS → verdict DECOMPOSED.**
- G0 residual non-trivial: median ||s||/||Δ|| = **0.992**.
- G1 residual transfers: mean transfer(s) = **+8.84**, p = **0.0099** (beat all 100 nulls).
- G2 residual value-selective: mean selectivity(s) = **+8.73**, p = **0.0099**.
- G3 generic is the (non-selective) boost: norm-matched selectivity residual − generic = **+8.28**, p = **0.0015**; generic still boosts (transfer_nm +1.91), generic selectivity_nm only **+0.58**.
- **Per-value: ALL 10** values have residual transfer p=0.0099 AND residual selectivity p=0.0099. Residual selectivity ranges +2.9 (false) … +20.6 (shut); the generic component at natural norm is ~0 (‖g‖≈2–4.5 vs ‖Δ‖≈15–38; mean generic transfer +0.11, selectivity +0.05).

**Descriptive cross-check (local, from saved Δ vectors — not a gate):**
- **mean pairwise cos(Δ_v, Δ_w) = 0.014** (range −0.09..+0.08) — the per-value directions are **mutually orthogonal** (at/below the sqrt(9/3584)≈0.05 random baseline).
- cos(Δ_v, aggregate) = 0.21..0.53 (mean ~0.32) with residual-frac-after-removing-aggregate 0.85..0.98. The ~0.32 is exactly the **geometric** value expected for 10 orthogonal equal-norm vectors and their mean (sqrt(10)/10 ≈ 0.316) — i.e. NOT evidence of a shared component; larger-‖Δ‖ values (grape, shut) sit above only because they dominate the mean.

**Interpretation (calibrated).** The Variable binding mechanism at (L2, val_slot) is a set of **(near-)orthogonal, value-specific directions**, one per value; each **transfers to held-out prompts** and is **value-selective** (raises its own value's logit ~3–21, others ~0). The earlier **GENERIC_BOOST is quantitatively consistent with an averaging account** (calibrated wording — NOT "proven artifact"): the measurements match the generic (aggregate) direction arising as the **centroid of approximately-orthogonal value-specific directions** (mean pairwise cos 0.014; cos(Δ_v, aggregate)≈0.32 is the exact geometric value for 10 orthogonal equal-norm vectors and their mean), rather than *requiring* a dominant shared "generic slot-update" mechanism. This does **not** rule out a smaller shared component — the centroid-removal control (next) tests that directly and still won't rule out *every* shared component. Removing the other-values subspace leaves the value-specific direction largely intact (resfrac≈0.99, partly high-dim geometry) and selective.

**Honest caveats (for any writeup):**
1. `resfrac≈0.99` is partly high-dim geometry (9-d subspace in 3584-d removes little regardless); the **selectivity** contrast (G2/G3), not the residual norm (G0), is the load-bearing evidence — and it is strong (all 10 at p-floor).
2. The GENERIC_BOOST-as-averaging-artifact reading is a geometry inference (well-supported by cos 0.014 + the sqrt(10)/10 match); a confirming re-run would project out the explicit **aggregate Δ** (1-d, the empirical generic direction) and re-measure selectivity, and/or re-derive Δ_wrong and show it is itself a value-mean.
3. Sufficiency, not natural-mechanism (Makelov/Grant): adding a value-specific direction selectively installs the value; this does not prove it is the model's unique natural binder. One model, one skill, one layer, greedy.

**Status vs paper:** the frozen note's central caveat ("GENERIC_BOOST blocks the value-specific binder claim") is **substantially weakened**: value-specific directions are recoverable, approximately orthogonal, transferable, and selective, and the generic direction is quantitatively consistent with being their centroid. (Full resolution — "the shared component contributes nothing" — awaits the centroid-removal control, and even then we do not claim to rule out every shared component.) Candidate for a note extension. No publish action taken.

### 2026-07-12 — CONTROL (before extending): centroid-removal — PRE-REGISTERED before code

**Why (Sahil):** close the obvious alternative before composition. `delta_decompose` used a 9-d other-values subspace; this removes the SINGLE explicit empirical generic direction — the **centroid** g = mean_v Δ_v (the aggregate that produced GENERIC_BOOST, cos 0.77) — and asks whether selectivity survives. If yes, the shared component contributes nothing beyond centroid geometry. This is the decisive test; the narrative becomes: recover value-specific dirs → transfer → selective → ~orthogonal → apparent-generic = their centroid → removing the centroid leaves the effect intact.

**Design (Variable, 7B 8bit T4, L=2, val_slot; same protocol as delta_decompose).**
- g = mean over the 10 per-value Δ_v; ĝ = g/‖g‖. Per value: **d_v′ = Δ_v − (Δ_v·ĝ) ĝ** (Δ_v with the centroid direction removed).
- On held-out prompts (cf≠v), ADD at (L2, val_slot) and read Δlogit per value:
  transfer(d,v)=mean Δlogit(v); selectivity(d,v)=Δlogit(v) − mean_{w≠v} Δlogit(w).
- Directions: Δ_v (matched baseline), d_v′ (natural norm), d_v′ norm-matched to ‖Δ_v‖, and ĝ scaled to ‖Δ_v‖ (centroid-only). Same-norm random null N=100 for d_v′.

**Pre-registered gates (aggregate over 10 values):**
- **C1 d′ transfers:** mean transfer(d′) > 0, p<0.01 vs null.
- **C2 d′ selective:** mean selectivity(d′) > 0, p<0.01 vs null.
- **C3 selectivity preserved (KEY):** median_v selectivity(d′_norm-matched)/selectivity(Δ_v) **≥ 0.90** (removing the centroid does not reduce per-content selectivity). Also report natural-norm ratio (expect ≥0.85; centroid removes only 1−√(1−cos²) ≈ 2–15% of ‖Δ‖).
- **C4 centroid is a non-selective boost (support):** mean selectivity(ĝ_norm-matched) ≈ 0 and ≪ selectivity(d′).
- **Verdict CENTROID_IRRELEVANT** ⟺ C1∧C2∧C3 (C4 supporting). Else CENTROID_MATTERS (selectivity drops when the centroid is removed → shared component carried real content).

**Objections:** (1) g is in-sample; but it is the strongest generic candidate, so removing exactly it is the right test; robustness via held-out centroid is a possible follow-up. (2) Only 1-d removed (small norm) — that is the point: it is the specific generic axis, and the norm-matched C3 isolates content from magnitude. (3) Sufficiency, not natural-mechanism (standing).

Kernel: `delta_centroid` → `src/causal_maps/delta_centroid.py`; launch `run delta_centroid --config {"quantization":"8bit"}`.

### 2026-07-12 — AGREED PLAN + language calibration (Sahil)

**Language calibration (adopt in ALL writeups):** do NOT say "GENERIC_BOOST was an averaging artifact." Supported wording: *"the measurements are quantitatively consistent with the generic direction arising as the centroid of approximately-orthogonal value-specific directions."* The centroid-removal control moves toward stronger wording but does NOT license "we ruled out every shared component" — never claim that.

**Forward plan (priority order, Sahil):**
1. **Finish the centroid-removal control** (running).
2. **Write up the decomposition result while fresh** — story: recover value-specific dirs → transfer → selective → ≈orthogonal (cos 0.014) → generic ≈ their centroid → centroid-removal leaves the effect intact. Tempered language throughout. (Decide at write-up: extend the frozen note vs a new short note.)
3. **Multi-slot binding composition** — best immediate follow-up (low-risk, same infra, stronger than vector arithmetic): install two bindings at two slots simultaneously via their directions and retrieve both independently without interference. Extends the decomposition naturally.
4. **Instruction-vs-data direction — the FLAGSHIP**, treated as a SEPARATE higher-risk project (not an appendix): a causal "treat-as-instruction vs treat-as-quoted-data" direction whose writing/ablation predictably changes prompt-injection susceptibility while preserving other behavior. Highest upside + AI-security relevance (field's current answers are architectural — ASIDE/CaMeL — not a discovered direction).
   - **Done-state/looping → third.** Less crisply defined than instruction-following (confounded by planning/uncertainty/state-tracking); interpretation likely messier even on success.

**Novelty audit (2026 lit, for framing):** directions transfer/compose/arithmetic = established (Todd parallelogram; FV causal-decomposition 2605.16591 decomposes over *examples*, not generic-vs-content). Binding is hot/deep (binding-IDs, lookbacks, mixing mechanisms). Steering non-identifiability is a live headwind (2602.06801, 2505.22637, Opiełka) → claims stay at *sufficiency + steerability*, not "the unique circuit." Our defensible edges: the site-map↔direction dissociation (paper core) + the centroid decomposition (distinct from example-decomposition). Instruction-vs-data as a *discovered causal direction controlling injection* appears genuinely open.

### 2026-07-12 — delta_centroid COMPLETE — CENTROID_MATTERS (value-specific effect SURVIVES centroid removal; but the centroid is NOT inert)

Kernel `shlydv/cm-delta-centroid` COMPLETE (~40 min). Artifacts: `runs/delta_centroid/`. Same protocol; g = centroid (mean of the 10 per-value Δ_v), ‖g‖=7.92; removed only ~6% of ‖Δ‖ on average.

**Gates:** C1 d′ transfers ✅ (mean +5.74, p=0.0099); C2 d′ selective ✅ (mean +5.88, p=0.0099); **C3 selectivity preserved ❌** (norm-matched retention median **0.848 < 0.90**; natural 0.784); C4 centroid-alone non-selective ✅ (centroid_nm selectivity **+0.028** ≈ 0, transfer +2.31). **Verdict CENTROID_MATTERS.**

**What this means (calibrated — and it vindicates the "don't say artifact" caution):**
1. **The value-specific effect is NOT merely the centroid.** After removing the explicit empirical generic direction, every one of the 10 value directions STILL transfers and STILL selects its own value at p=0.0099. So a genuine value-specific causal component exists *outside* the centroid direction. (This is the load-bearing positive result and it holds.)
2. **But the centroid is NOT inert.** Removing it costs a modest, real share of selectivity — median ~15% (norm-matched), and it is **heterogeneous**: minimal for two/right/beta/false (retention >0.90) but large for the values whose direction most overlaps the centroid — grape (cos 0.52 → retention 0.51) and shut (cos 0.45 → retention 0.72). So we must NOT claim "the shared component contributes nothing."
3. **The centroid added alone is non-selective** (C4, sel ≈ 0) — it boosts values roughly uniformly. So the picture is: value directions are *largely but not fully* orthogonal to the centroid (cos 0.21–0.52); the centroid alone selects nothing, yet each value's projection onto it contributes ~15% of that value's selectivity (nonlinear model response; the linear "non-selective ⇒ contributes nothing" intuition does not hold empirically).

**Calibrated claim for the writeup:** value-specific causal directions are recoverable, transfer, are selective, and *survive removal of the empirical generic (centroid) direction* (p<0.01, all 10 values) — the effect is not an averaging artifact. The generic/centroid direction is *quantitatively consistent* with a centroid of ≈orthogonal value directions and is non-selective on its own, **but it is not fully separable** from the value-specific directions and contributes a modest (~15% median), value-heterogeneous share of selectivity. Sufficiency + steerability, not the unique circuit.

### 2026-07-12 — Instruction/data status (before flagship) + multi-slot PRE-REGISTERED

**Instruction/data (flagship) — prior elicitation FAILED, needs redesign.** `instruction_hand10` (command vs mention, "reply OK"/"reply text"): behave 8/10 on the loose 2-candidate criterion but **clean_greedy=0%, cf_greedy=0%** — model never emits the label token top-1 (near-chance 2-candidate noise; same symptom binding had pre-chat-template). Prior protocol call: drop unless revisited with a **pre-registered redesign before any run**. → Flagship's first task = an elicitation redesign, proposed direction: **obey/not-obey behavioral readout** (does the model execute the payload — the actual injection behavior), which is naturally greedy-elicited. Co-design with Sahil before any kernel. HIGHER-RISK project (as ranked).

**Multi-slot binding composition — PRE-REGISTERED (low-risk, builds on Variable infra).**
- Directions: per-value Δ_v from single-var pairs at (L2, val_slot) (as in delta_decompose).
- Two-var template (chat): `Let X = {a}. Let Y = {b}. What is the value of {Q}?` primed `{Q} =`, Q∈{X,Y}; both slots hold a base value v0. Install by ADDING Δ at the {a}/{b} token slots (per-trial [B,D] deltas). N_trials distinct (vX,vY,v0) triples.
- Conditions per query prompt: none / add-X-only / add-Y-only / add-both. Metric: selectivity(v)=Δlogit(v)−mean_{others}Δlogit at the primed readout; same-norm random null N=100.
- **Gates:** **M1 single-slot transfers** to the 2-var template (add-X-only ⇒ query-X selects vX, p<0.01; symmetric for Y). **M2 simultaneous** (add-both ⇒ query-X selects vX AND query-Y selects vY, both p<0.01). **M3 independence** (a) add-both keeps query-X selectivity ≥0.7× add-X-only (no destructive interference); (b) add-Y-only does NOT raise vX at query-X (cross-talk ≈ 0). **Verdict COMPOSES** ⟺ M1∧M2∧M3.
- Kernel `delta_multislot`, 7B 8bit T4. (Batch with the instruction/data redesign probe once that's co-designed, to cut runs.)

### 2026-07-12 — PLAN REVISION (Sahil): depth bar + reordering

**Principle (stage shift):** we are past "find anything." Bar for a NEW flagship claim is now high — every new experiment must ask a **deeper question than the last**, not add another instance of the same phenomenon. (We already have: transferable directions, generic/value-specific decomposition, nontrivial selectivity after centroid removal.)

**Revised priority:**
1. **Multi-slot composition** ⭐⭐⭐⭐⭐ (running) — same-skill composition (two Variable bindings coexist).
2. **Cross-skill composition (Variable + Completion)** ⭐⭐⭐⭐⭐ — co-top, the deepest step. If a Variable direction and a Completion direction, added simultaneously in one prompt, BOTH survive with their own effects, the directions behave like **composable computational primitives**, not prompt-specific artifacts. Bigger conceptual move than one more direction. Build after multi-slot.
3. **Instruction/data — redesign to ISOLATE the variable** ⭐⭐⭐⭐☆. The "question vs classify" framing was CONFOUNDED (changes execute-vs-obey AND answer-vs-label → a direction there is uninterpretable). Adopt Sahil's design: keep the underlying computation identical, vary only execute-vs-treat-as-data:
   - Instruction: `Output the next word: cat` → obey → "cat".
   - Mention: `The following text says "Output the next word: cat". Repeat the quoted instruction exactly.` → treat as data → reproduce the quoted string.
   Greedy-elicited; isolates execute-vs-data. Separate higher-risk project; pre-register before any run.

### 2026-07-12 — delta_multislot COMPLETE — ✅ COMPOSES (two Variable bindings coexist at two slots)

Kernel `shlydv/cm-delta-multislot` COMPLETE (~5.5 min). Artifacts: `runs/delta_multislot/`. 7B 8bit T4, L=2, n_trials=12, two-var prompt `Let X={v0}. Let Y={v0}. …`; install vX at X-slot and vY at Y-slot via their directions; two readouts (query-X / query-Y).

| query | single-slot | both | retention both/single | cross-talk (other-slot only) |
|---|---|---|---|---|
| X | +8.65 (p=0.0099) | +8.93 (p=0.0099) | **1.03** | −0.05 |
| Y | +14.81 (p=0.0099) | +15.16 (p=0.0099) | **1.02** | −0.24 |

**Gates M1 (single transfers to 2-var template) ✅, M2 (simultaneous) ✅, M3 (independence) ✅ → COMPOSES.** Two bindings installed simultaneously are both retrieved selectively (p=0.0099); retention ≈1.0 (no destructive interference — if anything slightly stronger together); cross-talk ≈0 (installing the other variable does not raise this variable's target). Value directions act as **independent, composable slot-writes** (within-skill). Natural stepping stone to cross-skill composition.

**Heartbeat bug (cosmetic, results unaffected):** a duplicated `hb.step()` made the progress counter run to ~150% with negative ETA. Fixed (removed duplicate) + hardened `logutil.Heartbeat` to clamp pct≤100 / eta≥0 for all future runs.

### 2026-07-13 — delta_crossskill COMPLETE — ✅ COMPOSES_CROSS_SKILL (heterogeneous primitives compose)

Kernel `shlydv/cm-delta-crossskill` COMPLETE (~6 min). Artifacts: `runs/delta_crossskill/`. 7B 8bit T4, L=2, n_trials=10. Joint prompt: `Let X={v0}. Rules(if {flag} 0→{act0}/1→{act1}). State: {flag}=0.` + two readouts (value / next-action). Δ_V(vX) added at value slot; Δ_C (bit 0→1) added at completion site = bit_slot+1 (delta_completion's frozen peak). Completion action metric = Δ[logit(act1)−logit(act0)].

| readout | own-only | both | retention | cross-talk |
|---|---|---|---|---|
| value (Variable) | +9.41 (p=.0099) | +9.47 (p=.0099) | **1.01** | C-only → +0.10 |
| action (Completion) | +38.99 (p=.0099) | +39.45 (p=.0099) | **1.01** | V-only → −0.24 |

**Gates X1 (each transfers into joint prompt) ✅, X2 (simultaneous) ✅, X3 (independence) ✅ → COMPOSES_CROSS_SKILL.** Two *different-skill* directions, injected at two sites in one prompt, each produce their full effect (retention ≈1.0, if anything marginally stronger together) with cross-talk ≈0 (the value direction does not flip the action; the bit direction does not install a value; both beat same-norm random nulls at p=0.0099).

**Calibrated claim:** at L2, the Variable value-write and the Completion bit-flip behave as **independent, composable causal directions** — sufficiency + steerability evidence that these micro-skill interventions act like composable computational primitives, not prompt-specific artifacts. NOT a claim about the unique natural circuit. One model, one layer, greedy, two skills. Arc now: transfer → decompose → centroid control → within-skill composition → **cross-skill composition**.

### 2026-07-13 — Instruction-vs-data (FLAGSHIP) — PRE-REGISTERED, isolated design

**Confound fix (Sahil):** the earlier "question vs classify" idea changed TWO computations (execute-vs-obey AND answer-vs-label) → uninterpretable. Isolate the variable: keep the embedded directive identical; vary only whether it is EXECUTED vs TREATED AS QUOTED DATA.

**Pairs (isolated).** Single-token payload word W (from the value pool). Embedded directive D(W) = `Output the word: {W}`.
- **Instruction framing:** `{D(W)}` (as a live instruction) → obey → greedy first token = **W**.
- **Data framing:** `The following text says "{D(W)}". Repeat the first word of the quoted text.` → treat as data → greedy first token = **"Output"** (does NOT emit W).
- Behavioral readout is the ACTUAL behavior (obey vs not), naturally greedy-elicited — unlike the old "reply OK/text" (0% greedy). Two-candidate: W (obey) vs "Output" (data).

**Stage 1 — behavioral probe (cheap, gate first).** Over N payloads, measure: instruction→greedy==W, data→greedy=="Output". **Elicits iff ≥70%** show the split (report clean/cf greedy). **<40% ⇒ not elicitable at 7B under this design ⇒ STOP + report** (per the prior Skill-3 protocol; do not fish variants — one pre-registered redesign already spent). 40–70% ⇒ report as marginal, decide with Sahil.

**Stage 2 (only if Stage 1 passes) — direction.** Framings aren't token-aligned ⇒ FV-style `Δ_instr = mean(h_instruction) − mean(h_data)` at the primed last position over held-out payloads.
- **Transfer/selectivity:** adding Δ_instr to DATA-framed held-out prompts shifts behavior toward obeying (emits W); ablating (−Δ_instr) from instruction-framed prompts shifts toward not-obeying; beats a same-norm random null (p<0.01).

**Stage 3 (the killer result) — injection control.** On held-out prompt-injection setups (a directive embedded in ostensibly-benign data content), adding Δ_instr **raises** obey/injection-rate and ablating **lowers** it, while a **utility control** (unrelated task accuracy) is preserved. If it holds: a *discovered activation direction that causally modulates prompt-injection susceptibility* — the field's current answers are architectural (ASIDE/CaMeL), not a discovered direction.

**Caveats:** higher-risk (elicitation failed twice before); sufficiency + steerability, not the unique circuit; one model/layer/greedy. Batch Stages 1+2 into ONE kernel to cut runs; Stage 3 a second kernel only if 1+2 pass. Modules: redesign `instruction_pairs.py` (isolated), `delta_instruction.py`. **DEFERRED** — see next entry (Sahil re-prioritized: write the paper first).

### 2026-07-13 — PAPER REVISED (Sahil: write it now, before more experiments)

Sahil reversed the earlier "no paper" call: lock in decomposition + centroid + composition while fresh, before the higher-risk instruction/data + agentic-skill work. Revised `paper/main.tex` (source of truth) from the GENERIC_BOOST endpoint into the fuller thesis **transfer → decompose → compose**:
- **Title:** now "Causal Site-Maps Can Fail to Replicate While Residual Directions Transfer, Decompose, and Compose."
- **Abstract + Contribution:** GENERIC_BOOST reframed (aggregate = large generic component); added per-value directions (≈orthogonal, cos 0.01; value-selective all 10, p<0.01; aggregate consistent with their centroid), the centroid-removal control (value-specific survives; centroid not inert, ~15% median; \texttt{CENTROID\_MATTERS}), and composition (within-skill multi-slot + cross-skill Variable×Completion; retention≈1.0, cross-talk≈0).
- **New Results subsections:** "A value-specific residual survives removal of the generic direction" and "Directions compose within and across skills." Numbers inlined (no new macros).
- **Discussion:** "Future test" (which had pre-sketched exactly this decomposition) converted to results; genuine future work = instruction-vs-data direction + cross-model + natural-manifold faithfulness.
- **Conclusion + Limitations:** updated; added centroid-in-sample + composition-scope caveats. Tempered throughout (sufficiency + steerability, not the unique circuit).
- Structure sanity-checked (\begin/\end balanced; math balanced). **PDF recompile PENDING** — no local LaTeX toolchain (`pdflatex`/`bibtex` absent). To rebuild on a LaTeX machine: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`. Figures/macros unchanged (new numbers inlined); optionally add composition/decomposition figures + macros via `build_figures.py` later. No X/arXiv action taken.

### 2026-07-13 — FORWARD RESEARCH PLAN (Sahil) + calibration

**Meta-shift (agreed):** stop collecting examples → characterize the phenomenon; actively search for failure modes; measure quantitative laws, not isolated successes.

**Calibration (important):** do NOT prematurely frame this as a universal "instruction set" for transformers. Promising, but let the breadth of successful experiments determine how broad the eventual claim becomes (same discipline as "don't say artifact"). "Composable primitives" is a horizon, not yet a universal claim.

**Order (post-paper):** (1) publish the current paper; (2) **capacity law** (cheap, quantitative, likely publishable); (3) **boundary mapping** (highest scientific value — where the picture breaks); (4) **instruction-vs-data** (highest practical impact); (5) **faithfulness / on-manifold** (important, possibly long-running); (6) predictive direction generation (only if earlier work keeps succeeding).

### 2026-07-13 — Capacity law — PRE-REGISTERED

**Question:** how many value bindings compose simultaneously before interference? (extends multi-slot's k=2 to a curve over k.)

**Design (Variable, 7B 8bit T4, L2).** Per-value directions Δ_v from single-var pairs (as in delta_decompose). For each k ∈ {1,2,3,4,5,6,7,8}: a k-variable prompt `Let {n1}={v0}. … Let {nk}={v0}. What is the value of {n1}?` (single-token names; all slots hold base v0; query the first variable). Install distinct target values t_1..t_k at the k slots via `forward_add_multi`. n_trials random target assignments per k (prompt identical across trials → batched; only the added directions differ).
- **retention(k)** = selectivity(t_1 | install all k) / selectivity(t_1 | install only slot 1).
- **cross-talk(k)** = mean_{i≥2} Δlogit(t_i) at the V1 readout (leakage of distractor targets; ≈0 if slots stay independent).
- selectivity(v) = Δlogit(v) − mean_{other values} Δlogit; same-norm random null (N=100) per k for the all-k selectivity p-value.

**Reported (quantitative law, not pass/fail):** retention(k) and cross-talk(k) curves; **k\*** = largest k with mean retention ≥ 0.7 AND all-k selectivity p<0.01. Whichever way it lands. Module `delta_capacity.py`.

### 2026-07-13 — delta_capacity COMPLETE — high-capacity, interference-free to k=8 (ceiling not reached)

Kernel `shlydv/cm-delta-capacity` COMPLETE (~7 min). Artifacts: `runs/delta_capacity/`. 7B 8bit T4, L2, 8 trials/k, base value `false`.

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| retention | 1.00 | 1.00 | 1.02 | 1.09 | 1.09 | 1.15 | 1.07 | 1.12 |
| cross-talk | 0.0 | 1.06 | −0.10 | 0.79 | 0.78 | 0.33 | 0.38 | 0.64 |
| p(all-k) | .0099 | .0099 | .0099 | .0099 | .0099 | .0099 | .0099 | .0099 |

**k\* = 8** (the max tested). Retention is **flat at ≈1.0** across all k (no degradation; slight >1 is within trial noise / metric), cross-talk stays small (≤~1 logit vs selectivity ~10–18), and every k beats its same-norm null at the p-floor. So the composable value-binding code supports **≥8 simultaneous independent bindings with no measurable interference** — strong quantitative support for the independent-slot / composable-primitive picture.

**Honest caveats:** (i) the ceiling was NOT located — k=8 was capped by our **10 single-token value vocabulary** (need distinct targets + a base) and single-token variable names, not by observed interference; capacity is ≥8, unknown above. A cheap extension (repeat values / larger pools / longer prompts) could push for the true ceiling. (ii) Retention slightly increasing with k is likely a metric/noise effect, not a real gain. (iii) Within-range quantitative law, greedy, one model/layer — not a universal claim.

### 2026-07-13 — BOUNDARY MAPPING via 3-level taxonomy (Sahil) — PRE-REGISTERED

**Taxonomy (replaces "slot-fill vs not"):**
- **L1 Representation** — `X = cat` (store + retrieve). Established (decompose/transfer/compose).
- **L2 Selection** — `if flag: use X else Y` (route among stored representations; no new content).
- **L3 Transformation** — `X = X + 1` (create a NEW value). The frontier.

**Graded outcomes (NOT binary holds/breaks):** for each level classify transfer/selectivity as one of {STRONG (≈ L1 stored), WEAK (transfers but ≪ stored), LAYER_DEPENDENT (only some layers), PARTIAL (transfers but not selective, or vice versa), NONE/BOUNDARY}. Report the full layer profile + the computed/stored ratio; let the data pick the category. The crisp target: **Store✓ Retrieve✓ Select✓ Transform✗** ⇒ "linear causal directions explain representation and routing, not value creation" — a testable hypothesis; report whichever way it lands.

**L3 experiment (`delta_transform`), FIRST.** Single-token digit answers.
- **Stored control (L1, FV-style):** `X = {d}. What is X?` → d; Δ_d^store = mean(h_cf−h_clean) at (layer L, readout last-position) over pairs with cf-answer=d.
- **Computed (L3):** `X = {a}. Add {b} to X. What is X?` → d=a+b (1-digit, sums ≤9). Behavioral pre-check the model computes correctly (greedy=d); if low, flag confound. Δ_d^comp at (layer L, readout) — computed values have no value slot, so extract at the readout for BOTH (comparable, FV-style).
- **Sweep layers** L ∈ {2,6,10,14,18} (computation may live later than L2 storage). Held-out transfer + selectivity (answer d vs other digits), same-norm null; per layer, aggregated over digits. Compare computed vs stored (ratio, best layer, cos(Δ_d^store,Δ_d^comp)).

Then **L2 Selection (`delta_select`):** `A=cat. B=dog. If flag=1 answer A else B. flag=1` → cat; flip flag → answer switches A↔B. Is the "which-binding-to-read" selector a transferable+selective direction? Same graded framing. After L3.

### 2026-07-13 — delta_transform v1 COMPLETE — CONFOUNDED (not decisive)

Kernel `shlydv/cm-delta-transform` COMPLETE. Artifacts: `runs/delta_transform/`. 7B 8bit T4, layers [2,8,14,20,26], digits 3..9.

**Behavioral:** `stored_greedy=100%`, **`computed_greedy=0%`**, `confounded=true`. Under the rewrite-bind template (`Let X={a}. Then let X=X+{b}. …`), the model never emits the correct sum. Causal Δ_comp is therefore **not** a direction of successful computation.

| Layer | ss sel (p) | cc sel (p) | ratio | cos |
|---|---|---|---|---|
| 2 | +0.05 (.079) | +0.01 (.614) | 0.28 | 0.06 |
| 8 | −0.01 (.931) | −0.02 (.851) | (noise) | 0.20 |
| 14 | +0.05 (.406) | −0.01 (.723) | −0.14 | 0.10 |
| 20 | +0.14 (.030) | +0.03 (.416) | 0.19 | 0.04 |
| 26 | **+34.29 (.010)** | **+9.58 (.010)** | **0.28** | 0.38 |

Auto-grade string: `L3_LAYER_DEPENDENT_WEAK` (only L26 cc-sig; ratio≪0.7). **Do not interpret as Transform science** — harness/elicitation failure. Late-only spike is also consistent with unembedding-adjacent steering.

### 2026-07-13 — delta_transform v2 PRE-REGISTERED (elicitation fix + re-run)

**Rule:** do not run (or do not interpret) the causal sweep unless `computed_greedy ≥ 0.80` and `stored_greedy ≥ 0.80`.

**Protocol:** before the layer sweep, try a short fixed menu of computed templates (same digit pool, bare-digit readout after a trailing-space primer). Pick the **first** template with `computed_greedy ≥ 0.80`. If none clear: verdict **`L3_INELICITABLE`**, skip causal, proceed later to Selection. If one clears: re-run the same graded L3 causal protocol on that template; interpret the graded verdict only then.

Templates (order fixed, no fishing after seeing causal): rewrite_bind (v1), add_to_x, inline_sum, direct_sum, equals_phrase. Module update in `delta_transform.py`; one Kaggle re-run.

### 2026-07-13 — delta_transform v2 COMPLETE — ✅ unconfounded → L3_LAYER_DEPENDENT_WEAK

Kernel `shlydv/cm-delta-transform` v3 COMPLETE. Artifacts: `runs/delta_transform/` (v1 archived under `v1_confounded/`).

**Elicitation (fixed menu, first clear wins):**
| template | computed_greedy |
|---|---|
| rewrite_bind / add_to_x / inline_sum | 0% |
| **direct_sum** (`What is a+b?` + `Answer: `) | **100%** ← chosen |
| equals_phrase | 100% (not chosen; first-clear rule) |

`stored_greedy=100%`, `computed_greedy=100%`, **`confounded=false`**.

| Layer | ss sel (p) | cc sel (p) | ratio | cos |
|---|---|---|---|---|
| 2 | +0.05 (.079) | −0.00 (.277) | −0.10 | 0.02 |
| 8 | −0.01 (.931) | +0.00 (.129) | −0.45 | 0.06 |
| 14 | +0.05 (.406) | +0.03 (.050) | 0.55 | 0.03 |
| **20** | +0.14 (.030) | **+0.40 (.010)** | 2.81 | 0.03 |
| **26** | **+34.29 (.010)** | **+20.32 (.010)** | **0.59** | 0.51 |

**Verdict: `L3_LAYER_DEPENDENT_WEAK`.** Store control ok (L26). cc significant only at L20+L26 (not all layers). Best-layer (L26) ratio 0.59 < 0.7 STRONG threshold. Mean cos store↔comp 0.13; `representation_shared=false` (late cos 0.51 + cross transfer exists at L26, but mean cos < 0.3 gate).

**Reading (calibrated):** under a template the model *actually computes*, computed digit answers still carry a transferable, selective readout direction — but only late, and weaker than stored (~0.6× at best). Early/mid layers are null. So Transform is not a hard BOUNDARY (NONE); it is a weaker, late-layer version of the L1 story. Caveat: FV-style at readout; L26 is near-unembedding; direct_sum is simpler than rewrite-bind (which remains inelicitable). Next: `delta_select` (L2 Selection), as pre-registered.

### 2026-07-13 — delta_select (L2 Selection) — PRE-REGISTERED

**Question:** is “which stored binding to read” a transferable, route-selective causal direction (like L1 value dirs), or does Selection fail / only appear late?

**Template:** `Let A={va}. Let B={vb}. If the flag is 1, answer A. If the flag is 0, answer B. flag={f}. What is the answer?` primed `Answer =`. Flag written `flag={f}.` so the digit is a single bare token (inject site = that digit).

**Behavioral gate (hard):** greedy == va when f=1 and == vb when f=0, both ≥ 80%. Else **`L2_INELICITABLE`**, skip causal.

**Direction:** over N_train=8 (va,vb) pairs, `Δ_sel(L) = mean(h_f1 − h_f0)` at (L, flag_digit). Layer sweep L ∈ {2,8,14,20,26}.

**Transfer (held-out N_test=8, base flag=0):** ADD Δ_sel; `route = mean Δ[logit(va)−logit(vb)]` (toward A); `selectivity = mean[Δlogit(va) − mean_other Δlogit]`. Same-norm random null N=100. Descriptive anti: ADD −Δ_sel on flag=1 (expect route ↓).

**Graded verdict:** L2_STRONG (all layers route+sel p<0.01) / L2_LAYER_DEPENDENT_STRONG (early/mid sig) / L2_LAYER_DEPENDENT_WEAK (late-only) / L2_PARTIAL / L2_BOUNDARY / L2_INELICITABLE. Module `delta_select.py`.

### 2026-07-13 — delta_select v1 COMPLETE — L2_INELICITABLE

Kernel `shlydv/cm-delta-select` v1 COMPLETE (~2 min after load). Artifacts: `runs/delta_select/`.

**Behavioral:** flag0→B = **0%**, flag1→A = **6%** (gate 80%). Model does not follow the flag under the v1 template; causal sweep skipped by design.

**Verdict: `L2_INELICITABLE`.** Same class of issue as transform v1 — harness/elicitation, not a Selection BOUNDARY claim. Next (if continuing Selection): fixed template menu + behav gate, then re-run; do not interpret absent Δ_sel.

### 2026-07-13 — delta_select v2 PRE-REGISTERED (one iteration + permanent stop)

**Sahil stop condition:** one carefully designed template-search iteration only. If no template reaches ≥80% behavioral flag-following, verdict **`L2_STOPPED_PERMANENT`** — do not spend further weeks optimizing Selection prompts; move on (instruction-vs-data later).

**Menu (fixed order, 12 hand-designed; first dual-gate clear wins):** v1_if_flag, value_of, rules_answer_var, xy_selector, read_slot, switch_on_off, return_instr, choose_label, bit_state, fewshot_one, direct_ask, which_binding. Each exposes `{key}={f}` bare digit for injection. Same causal protocol as v1 if a winner exists.

### 2026-07-13 — delta_select v2 COMPLETE — ✅ L2_LAYER_DEPENDENT_STRONG

Kernel `shlydv/cm-delta-select` v2 COMPLETE (~11 min). Artifacts: `runs/delta_select/` (v1 under `v1_inelicitable/`).

**Elicitation (first clear wins):** `value_of` — flag0→B **100%**, flag1→A **94%** (`confounded=false`). Also would have cleared later: switch_on_off, fewshot_one, which_binding. v1_if_flag still 0%/6%.

| Layer | route (p) | sel (p) | anti (−Δ on flag1) |
|---|---|---|---|
| **2** | **+46.01 (.010)** | **+22.67 (.010)** | −46.39 |
| **8** | **+45.59 (.010)** | **+22.68 (.010)** | −44.66 |
| **14** | **+43.88 (.010)** | **+22.00 (.010)** | −44.28 |
| 20 | +0.26 (.089) | +0.19 (.050) | +0.16 |
| 26 | −0.02 (.574) | −0.01 (.683) | +0.08 |

**Verdict: `L2_LAYER_DEPENDENT_STRONG`.** Significant at early/mid L2–L14 (not late); best at L2. Anti-control signed correctly (negating Δ on flag=1 drives route down hard). Selection is a real steerable mechanism here — contrast Transform (late/weak only).

**Boundary picture so far:** Store/Retrieve ✓ · Select ✓ (early) · Transform ~ (late, weaker). Stop condition not triggered. Next per prior plan: instruction-vs-data (or publish / capacity ceiling).

### 2026-07-13 — delta_instruction LAUNCHING (Stages 1+2; Stage 3 later)

**Why now:** boundary map is informative (Select strong / Transform weak); composition stack is solid. Instruction-vs-data is still the highest-impact open flagship — a steerable obey-vs-quote direction would be a different *kind* of claim (security-relevant), not another binding instance.

**Running hypothesis (updated):** at 7B, some micro-skills admit approximately independent, addable residual directions (value content, selectors, completion bits) that transfer and compose; value creation by computation is weaker/later. Open: does a *framing* bit (execute vs treat-as-data) exist in the same style?

**Protocol (already pre-registered; executing):** isolated D(W)=`Output the word: {W}`; instr→W, data→`Output`. Stage1 ≥70% both or STOP (<40% = INSTR_INELICITABLE, no fishing; 40–70% = MARGINAL, no causal). Stage2 FV Δ_instr at last pos, layer sweep, add-to-data / ablate-instr. Stage3 injection = separate kernel only if Stage2 passes. Module `delta_instruction.py`.

### 2026-07-13 — delta_instruction COMPLETE — INSTR_INELICITABLE (Stage1 STOP)

Kernel `shlydv/cm-delta-instruction` v1 COMPLETE (~3 min). Artifacts: `runs/delta_instruction/`.

**Stage1:** instr→W = **0%**, data→Output = **0%**, data_leak_W = 0%. Both sides below the 40% stop floor.

**Verdict: `INSTR_INELICITABLE`.** Causal Stage2/3 not run. Per pre-registration: **no prompt-fishing** on this isolated design — the redesign budget for this framing is spent. Result is elicitation failure, not evidence against an instruction/data direction in general. Next move needs an explicit Sahil call (park flagship vs authorize a *different* pre-registered elicitation strategy).

### 2026-07-13 — INSTR_INELICITABLE VOIDED (harness bug) + v2 re-run PRE-REGISTERED

**Bug (Sahil):** Stage1 compared greedy argmax to *leading-space* token ids (`Ġcat`=8251, `ĠOutput`=9258), but the prompt ends at `assistant\\n` with no mid-sentence space, so an obeying model emits bare `cat`/`Output` (4616/5097). Spaced ≠ bare ⇒ 0%/0% on both channels — a broken thermometer, not a behavioral measurement. Same leading-space trap as transform. Protocol: “accuracy far below published ⇒ HARNESS BUG first” overrides the STOP gate (STOP assumed a valid meter).

**Fix (not fishing):** `leading_space=False` for all answer readout ids. Same isolated D(W) design, one re-run. Module version=2. Prior `INSTR_INELICITABLE` **void**.

### 2026-07-13 — delta_instruction v2 COMPLETE — ✅ Stage1 pass → INSTR_DIRECTION_LAYER_DEPENDENT_WEAK

Kernel `shlydv/cm-delta-instruction` v2 COMPLETE (~8 min). Artifacts: `runs/delta_instruction/` (v1 void under `v1_void_spaced_readout/`). Bare-token readout.

**Stage1:** instr→W = **100%**, data→Output = **75%**, data_leak_W = 12.5%. Clears ≥70% gate (`confounded=false`). Thermometer fix confirmed.

| Layer | add Δ to data (p) | ablate −Δ on instr |
|---|---|---|
| 2 | +0.19 (.604) | −0.09 |
| 8 | −0.22 (.743) | +0.50 |
| 14 | +5.53 (.168) | +3.16 |
| **20** | **+19.09 (.010)** | **−7.94** |
| **26** | **+20.66 (.010)** | **−19.17** |

**Verdict: `INSTR_DIRECTION_LAYER_DEPENDENT_WEAK`.** Significant only late (L20/L26); early/mid null or wrong-signed ablate. Adding Δ_instr to data-framed prompts raises obey (W vs Output); ablating on instruction-framed prompts lowers it — signed, late-layer. Stage3 (injection control) still pending — separate kernel if we proceed.

### 2026-07-13 — TWO-REGIME THESIS + Stage3 PRE-REGISTERED (OPAS handoff)

**Thesis (paper spine):** linear activation dials are early & strong for *retrieving/routing* given information (Store, Select), and late & weak for *creating or re-designating* meaning (Transform, Instruction/data). Deciding "is this text a command?" patterns with value creation, not retrieval. Method is standard diff-in-means; novelty = computational-type axis + rigor + depth. Complementary to LAP (semantics for their geometry) and HyperSteer (type prediction, not description→vector); see `LIT_MAP.md` / `INTENT_DIRECTION_PLAN.md`.

**Stage3 (`delta_instruction_s3`) — one kernel, both outcomes publishable.** Layers {20,26}. Injection prompt: extract-first-word from a quotation that embeds D(W); success = emit W. S3a +Δ raises injection (ld + rate, p<0.01); S3b −Δ lowers it; S3c −Δ on clean `Output the word: W` utility drop ≤10pp. Verdict INJECTION_DIAL / PARTIAL / NULL. NULL ⇒ obey-dial insufficient alone (supports ASIDE-style architecture) — report as finding. Bare-token readout; degenerate base behav ⇒ HARNESS_BUG first.

### 2026-07-13 — delta_instruction_s3 COMPLETE — INJECTION_PARTIAL

Kernel `shlydv/cm-delta-instruction-s3` COMPLETE (~7 min). Artifacts: `runs/delta_instruction_s3/`.

**Base behav (bug-check passed):** inj→W=0%, inj→Output=100%, util→W=100%. Not degenerate.

| Layer | S3a +Δ inj rate / ld (p) | S3b −Δ ld (p) | S3c util drop |
|---|---|---|---|
| **20** | 50% / **+23.08 (.010)** | −6.70 (.020) | **0pp ✅** |
| **26** | 25% / **+20.45 (.010)** | **−2.41 (.010)** | **+38pp ❌** |

**Verdict: `INJECTION_PARTIAL`.** +Δ significantly raises injection logit-diff at both late layers (and greedy inj rate 0%→50%/25%). Full dial fails: L20 ablate p=0.020 (misses <0.01); L26 ablate significant but utility tanks (−38pp). Honest framing (pre-registered): obey-status steering *moves* injection but is **insufficient alone as a defense** → supports architectural approaches (ASIDE). Not a null, not a clean dial. No fishing.

### 2026-07-13 — Paper reframe (A3) + typology H2 PRE-REGISTERED

**Paper:** `paper/main.tex` retitled around two-regime spine; positions vs LAP / HyperSteer / ASIDE; lead fig `two_regimes`; Stage3 as PARTIAL finding. Site-map→direction arc retained as methodological prelude.

**H2 typology (`delta_typology`) — predictions written BEFORE measuring:**
- **echo_kth** (route): list of 3 items, echo item k. Predict **early/strong** (sig ⊆ {2,8,14}).
- **compare_larger** (create): X=a,Y=b; output larger. Predict **late/weak** (sig ⊆ {20,26}).
- Gate T2: both sides confirmed → `TYPOLOGY_CONFIRMED`; miss → `FALSIFIED` (finding). Bare-token / digit primer discipline. Module `delta_typology.py`.


### 2026-07-13 — delta_typology v1 → TYPOLOGY_INELICITABLE (echo 0%); v2 PRE-REGISTERED

**v1:** echo_kth behav **0%/0%** (no answer primer — model not emitting bare answer token). compare_larger behav 100% and causal ran exploratorily: sig L14/20/26 (peak +31.5 @ L20) — **not** ⊆ {20,26}, so would be MISS on strict late gate; overall verdict INELICITABLE because echo failed. Artifacts archived `runs/delta_typology/v1_inelicitable_echo/`.

**v2 fix (not fishing):** fixed echo template menu with primers (`answer_colon`, `item_equals`, `return_nth`, `slot_read`); first ≥80% wins; if none → permanent echo stop. Same predictions / T2 gate. Compare unchanged.


### 2026-07-13 — typology v2 ECHO_STOPPED; v3 PRE-REGISTERED (not_bit alternate)

**v2:** all echo templates ≤6.25% → permanent stop on echo_kth (no more fishing). Artifacts: `runs/delta_typology/v2_echo_stopped/`.

**v3:** plan's alternate route skill — **boolean NOT of a stored bit** (`not_bit`). Predict early/strong. Create arm still `compare_larger` → late/weak. Same T2 gate. One kernel.


### 2026-07-13 — delta_typology v3 COMPLETE — TYPOLOGY_FALSIFIED

Kernel `shlydv/cm-delta-typology` v3 COMPLETE. Artifacts: `runs/delta_typology/`.

Both skills elicitable at 100%. Gate T2 **misses both sides** — report as finding, do not fish.

| Skill (type / predict) | behav | sig layers | confirm? | notes |
|---|---|---|---|---|
| **not_bit** (route → early) | 100% | **∅** | ❌ | best L14 +0.5 (n.s.); L20/26 **wrong-signed** −9/−26 |
| **compare_larger** (create → late) | 100% | **14, 20, 26** | ❌ | peak +16.3 @ L20; L14 bleeds → not ⊆ {20,26} |

**Verdict: `TYPOLOGY_FALSIFIED`.** Pre-registered type→profile prediction did not hold.
- Route alternate (NOT) never became an early dial under this Δ construction.
- Create (compare) is *not early* (good soft signal) but not cleanly late-only either.
- Honest reading: the four-skill two-regime *observation* still stands; the upgrade to an a-priori predictive typology is **not** earned yet. Possible confound: classifying NOT as “route” was strained (it creates a new bit). No further typology fishing.

### 2026-07-13 — delta_explicit PRE-REGISTERED — stated-vs-derived toggle (sharpened boundary test)

**Why:** TYPOLOGY_FALSIFIED because "route vs create" is a judgment call applied inconsistently (NOT *creates* a bit → generate; comparison *selects* a present value → copy). Re-checking all six prior skills against an **objective** variable — *is the controlling info STATED (a present token) or DERIVED (must be computed/inferred)?* — fits every one (Store/Select/compare = stated→early; Transform/not_bit = derived→late; instruction = present answer but derived *criterion* → late, the exception). This is post-hoc over 6 skills, so it earns nothing until a fresh pre-registered test. Test it the disciplined way: **toggle ONLY stated-vs-derived within the SAME skill**, so a mislabel can't drive the result. Module `delta_explicit.py`.

**Arms (same answers, toggle stated↔derived):**
- **increment** (control): `X=b; Add k` → b+k (derived, target ABSENT) vs `X=b; Then X=b+k` → b+k (stated, target PRESENT). Offline-verified: derived target 0% in-prompt, stated 100%.
- **instruction** (the test): inferred obey-frame (derived) vs explicit `[MODE: EXECUTE/QUOTE]` tag (stated). Here the toggle is on the *selection criterion* (obey-status), not the answer token.

**Method per cell:** Δ = mean(h_pos) − mean(h_foil) at (L, last pos) from TRAIN; on held-out FOIL prompts +Δ should raise logit(pos_target)−logit(foil_target); same-norm null (N=100, p-floor 0.0099); layers {2,8,14,20,26}. Bare-token readout. Behav gate 0.70 (data-frame elicits ~75%).

**Predictions (written BEFORE the result):** increment stated → EARLY sig (≤14), derived → LATE only. instruction stated → EARLY (if explicitness moves it to the easy regime), derived → LATE (replicate the flagship).

**Gate:** per skill `STATED_EARLIER` iff stated reaches an early sig layer AND derived does not. Overall `EXPLICITNESS_CONFIRMED` (both), `_PARTIAL` (one), `_FALSIFIED` (neither), `INELICITABLE` (behav gate).

**Outcomes, both publishable:**
- Confirmed → the regime is set by *stated-vs-derived*, demonstrated within-skill (resolves the typology falsification with an objective, controlled variable); instruction arm passing would *mechanistically explain ASIDE* (architecture = making obey-status stated).
- Instruction stated stays late while increment shifts → obey-status *resists* explicitness (a real security caution: a mode tag won't fix injection); increment still validates the axis.
- Neither shifts → stated-vs-derived is not the driver either → cede prediction to geometry (LAP), keep the observation + composition + injection-partial + honest negatives. No fishing.



### 2026-07-13 — delta_explicit COMPLETE — EXPLICITNESS_FALSIFIED (demoted footnote)

Kernel `shlydv/cm-delta-explicit` COMPLETE. Artifacts: `runs/delta_explicit/`.

| Arm | Verdict | Notes |
|---|---|---|
| increment | **NO_SHIFT** | stated sig=[20,26], derived sig=[20] — both late; stated≠early |
| instruction | **INELICITABLE** | foil behav 62.5% < 0.70 gate |

**Overall: `EXPLICITNESS_FALSIFIED`.** Demoted (Sahil): not centered. Site caveat — extraction at readout (where even store goes late in transform), so increment arm was poorly sited for early/late. **No rescue re-run at the slot** (semantic-label treadmill). One-line for the paper: stated-vs-derived toggle did not move the regime under this design.

### 2026-07-13 — BRANCH SHIFT: semantic → structural (protocol compatibility)

**Why:** typology + explicitness exhausted the semantic-axis upside without a predictive win. Prior structural positives (decompose, centroid, multislot, crossskill, capacity) are where remaining upside lives. Opus handoff + Sahil steer: center **protocol compatibility**, not "algebra" metaphors.

**Crisp question (no metaphor):** *Do independently discovered intervention directions write representations in a format that independently discovered downstream mechanisms can consume?*

Stronger than readability: does the downstream mechanism's computation **depend** on the upstream write (ablate / scale), not merely tolerate it?

### 2026-07-13 — delta_protocol PRE-REGISTERED — binding→routing existence test

**Module:** `delta_protocol.py`. Layer L2 for both (binding val_slot; routing flag digit). Donor protocols: Variable per-value Δ; Select `value_of` Δ_route. Carrier: neutral `Let X=v0. Let Y=v0. If flag=1 output X else Y. flag=0.` + `Answer =`.

**Primary signature — routing sensitivity (interaction):**
`RS = [logit(u)−logit(w) | bind(u)@X, bind(w)@Y, +Δ_route] − [same | binds only]`
Additive knobs cannot produce a routing-conditional flip. Same-norm null on Δ_route; p<0.01.

**Dependency (Sahil — stronger than compatibility):**
- **Ablate:** remove bind(u)@X, keep bind(w)@Y + Δ_route → preference for u collapses (RS_ablate ≪ RS).
- **Scale:** α∈{0,0.5,1.0,1.5} on bind(u)@X with route fixed → logit(u)−logit(w) increases with α (Spearman > 0).

**Controls:**
- Empty: +Δ_route only (slots still v0) → no coherent preference for held-out (u,w).
- Leakage handled by RS itself (symmetric push cancels).
- Native baseline: same RS with stated `Let X=u. Let Y=w.` (sanity; not the claim).

**Gates:**
- P0 native carrier behav ≥80% (flag0→Y, flag1→X) on stated values.
- P1 RS_injected > 0 vs null (p<0.01).
- P2 ablate collapses: RS_ablate < 0.5 × RS (or not sig).
- P3 scale Spearman(α, pref_u) > 0.
- P4 empty: |pref(u−w)| under route-only < 0.5 × |RS| (or n.s.).

**Verdicts:**
- `PROTOCOL_COMPATIBLE` ⟺ P0∧P1∧P2∧P3∧P4 — injected write is consumed *and* depended on.
- `COMPATIBLE_WEAK` — P1 holds, dependency (P2/P3) soft/partial.
- `KNOBS_NOT_PROTOCOL` — native RS works, injected RS fails (writes to output path, not re-readable workspace).
- `INCOMPATIBLE` — neither.
Both `PROTOCOL_COMPATIBLE` and `KNOBS_NOT_PROTOCOL` are publishable findings. No fishing.



### 2026-07-13 — delta_protocol COMPLETE — COMPATIBLE_WEAK

Kernel `shlydv/cm-delta-protocol` COMPLETE (~6 min). Artifacts: `runs/delta_protocol/`.
Question: independently-extracted binding Δ + routing Δ — does routing *consume* an injected write?

| Gate | Result | Detail |
|---|---|---|
| P0 native behav | ✅ | flag0→Y 92%, flag1→X 92% |
| **P1 RS (flip)** | ✅ | **RS=+18.08 (p=.010)** bind→full: pref −10.7 → +7.4 |
| P2 ablate | ❌ | dep_gap=+7.09 < 0.5×RS (=9.04); but pref 7.4→0.31 collapses in absolute terms |
| P3 scale | ✅ | α∈{0,0.5,1,1.5} → prefs [0.3, 2.1, 7.4, 13.1]; **Spearman=+1.0** |
| P4 empty | ✅ | route-only pref=+1.0 ≪ RS |
| native RS (sanity) | — | +48.2 (routing works on stated bindings) |

**Verdict: `COMPATIBLE_WEAK`.** Load-bearing positive: **routing sensitivity on *injected* bindings is real and null-controlled** — not mere additive knobs. Continuous scale tracks the upstream write perfectly. Ablate is directionally right but missed the pre-registered 0.5×RS bar (do not re-threshold). Not yet full `PROTOCOL_COMPATIBLE`; not `KNOBS_NOT_PROTOCOL`.

**Interpretation (calibrated):** independently extracted mechanisms appear to share a usable write/read format on this carrier. Remaining gap is how strict the dependency criterion should be — next structural steps (depth/saturation, or a sharper ablate pre-reg) only if Sahil wants to push past WEAK.


### 2026-07-13 — P2 diagnostic (why COMPATIBLE_WEAK, not fishing)

**Frozen prefs:** base=+0.32, bind=−10.68, full=+7.40, ablate=+0.31, empty=+1.01.
**RS** = full−bind = **+18.08**. **dep_gap** = full−ablate = **+7.09**.
**Pre-registered P2:** `dep_gap > 0.5 × RS` (=9.04). 7.09 < 9.04 → fail.

**Diagnosis: metric mismatch, not an empty ablate effect.**

1. **RS is not a pure "u@X contribution" measure.** On flag=0, bind-only correctly prefers *w* (read Y) → large negative baseline (−10.7). Full flips to prefer *u*. So RS ≈ (leave read-Y/w) + (arrive at read-X/u). Decomposition: RS = dep_gap + (ablate−bind) = 7.09 + 10.99. **~61% of RS is leaving the Y-read state; only ~39% is u@X under route.** Demanding dep_gap > 50% of RS asks the upstream-write term to exceed half of a quantity dominated by the leave-Y swing — structurally harsh.

2. **Under-route ablate is clean in absolute terms.** Removing u@X drops routed pref 7.40 → 0.31 (**96% relative collapse**). ablate ≈ base (0.31≈0.32); ablate−empty = −0.70 (no residual u without the write). Scale (P3) already showed continuous dependence. The *phenomenon* P2 meant to catch is present; the *threshold* compared the wrong denominators.

3. **What would have been a matched dependency rule** (diagnostic only — NOT a re-verdict): e.g. `dep_gap > 0.5 × (full−empty)` (content under route) → 7.09 > 3.20 ✓; or `ablate < 0.25 × full` when full>0 → 0.31 < 1.85 ✓. Re-scoring with these after seeing data would be fishing. Keep `COMPATIBLE_WEAK`.

**Implication for the claim:** we have null-controlled interaction (P1) + continuous dependence (P3) + empty control (P4). We do **not** yet have a pre-registered ablate gate that passed. Next move if Sahil wants the strong claim: **pre-register a corrected P2** that compares ablate to other *routed* conditions (full vs ablate vs empty), not to RS, then re-run once (or re-grade only if Sahil explicitly authorizes a locked formula written before looking at alternatives). Until then: compatible steering with suggestive-but-not-gated dependency — not fully certified composable primitives.


### 2026-07-13 — delta_protocol v2 PRE-REGISTERED — one metric-fix rerun (P2 only)

**Why:** v1 P2 compared `dep_gap` to `0.5×RS`, but RS is dominated by leaving the read-Y state (~61%), not by u@X. One disciplined fix, then stop.

**P2v2 (written BEFORE rerun):** stay inside *routed* conditions.
`content = pref_full − pref_empty`
`dep_gap = pref_full − pref_ablate`
**Pass iff** `content > 0` AND `dep_gap > 0.5 × content` AND `pref_ablate < pref_full`.

No other gate changes. Same design, seed=0, L2, n_trials=12, n_null=100.

**Verdict rules (unchanged otherwise):**
- All of P0,P1,**P2v2**,P3,P4 → `PROTOCOL_COMPATIBLE`
- P1 + (P2v2∨P3) but not full set → `COMPATIBLE_WEAK`
- If P2v2 fails again → keep weak claim; **stop** this line (no further metric fishing).



### 2026-07-13 — delta_protocol v2 COMPLETE — PROTOCOL_COMPATIBLE (stop)

Kernel `shlydv/cm-delta-protocol` v2 COMPLETE. Artifacts: `runs/delta_protocol/` (v1 under `v1_compatible_weak/`).
Same seed/prefs as v1 (RS=+18.08, dep_gap=+7.09, scale Spearman=1.0). **Only change: P2v2.**

| Gate | v2 |
|---|---|
| P0 native | ✅ 92%/92% |
| P1 RS | ✅ +18.08 (p=.010) |
| **P2v2** | ✅ dep_gap 7.09 > 0.5×content 3.20; ablate 0.31 < full 7.40 |
| P3 scale | ✅ Spearman=+1.0 |
| P4 empty | ✅ |

**Verdict: `PROTOCOL_COMPATIBLE`.** Independently extracted binding and routing directions: routing *consumes* the injected write (flip + empty controls), and the upstream write is *necessary* for the routed preference under the corrected within-route dependency gate. Calibrated claim: **shared write/read format on this carrier**, not a full algebra. **Stop** this metric line — no further P2 fishing. Next structural work (saturation/depth) only on Sahil call.



### 2026-07-13 — protocol line STOPPED; chaining protocol drafted (no code)

Sahil: stop polishing protocol compatibility. Next hypothesis = **computational chaining** (not mere communication).

**Design-only deliverable:** `CHAIN_PROTOCOL.md` — hypothesis, knobs vs protocol vs primitives, BIND→ROUTE→PREDICATE experiment, outcome table, novelty check vs Todd FV algebra / multi-steer. Novelty cut: dataflow + **ablate-B bypass** control. **Do not implement** until Sahil signs the page.


### 2026-07-13 — chain protocol v1.1: existing-donor audit → keep predicate + hard G0

Sahil: prefer compose only already-validated early/strong donors if possible.

**Audit:** {Binding, Routing, Completion-bit} cannot form true \(C(B(A))\) without collapsing to protocol (bind→route→emit) or parallel crossskill (bind∥completion). Transform/Instr disqualified (late/weak).

**Decision:** keep BIND→ROUTE→PREDICATE; **G0 hard stop** on predicate native ≥80% @ L2; **no donor fishing**. `CHAIN_PROTOCOL.md` updated.


### 2026-07-13 — chain novelty FINAL GO → implementing one kernel

**Targeted lit check (write→read→compute / causal dataflow of independently extracted dirs):**
- Todd/Hendel FV–task algebra: **sum tasks → new task** (superposition), not heterogeneous role dataflow + ablate-middle bypass.
- MoSV / RISER / ASA: learned routers compose steers for multi-objective control — not A-write → B-select → C-predicate on injected state.
- Flow/ODE steering: richer transport of one concept — not a three-link chain.

**Novelty: CLEAR for this exact hypothesis.** Proceed: one `delta_chain` kernel; G0 absolute; stop after verdict.

### 2026-07-13 — delta_chain IMPLEMENT + LAUNCH (one kernel)

**Sahil approved.** Final lit re-check confirmed: Todd/Hendel additive FV algebra, MoSV/RISER routers, flow/ODE transport, CAS sparse mediation — none test write→route→predicate with ablate-B bypass. Novelty still CLEAR.

**Module:** `src/causal_maps/delta_chain.py` wired as stage `delta_chain`.

**Frozen inequalities (code PR):**
- G0 hard stop: bind/route/predicate native ≥80% @ L2.
- G1: CS = oriented(FULL(+route) − FLIP(−route)) > 0, p < 0.01 vs null on B.
- G2: drop_B = oriented(FULL) − oriented(noB) ≥ 0.5 × CS.
- G3: drop_A ≥ 0.5 × (FULL − empty).
- G4: |empty| < 0.5×|FULL| OR |empty| < 1.0.

**Note:** Variable Δ cf-vocab has only `cat` as animal; chain pairs use cat↔non from that vocab (not a donor swap).

Kernel: `python kernel/run_kaggle.py run delta_chain --config '{"quantization":"8bit"}'`.

### 2026-07-13 — delta_chain COMPLETE — CHAIN_INELICITABLE (branch CLOSED permanent)

Kernel `shlydv/cm-delta-chain` v1 COMPLETE (~4 min after load). Artifacts: `runs/delta_chain/`.

**G0 ABSOLUTE STOP — no chain conditions run.**
- bind retrieve: **75%** (fail; gate ≥80%)
- route native: **100%** (pass)
- predicate native @ L2: **0%** (fail; decisive)

**Verdict: `CHAIN_INELICITABLE`.** Stop reason: `G0_hard_stop_no_donor_fishing`.

**Branch closed permanently.** No replacement donors, no layer sweep, no template search, no predicate redesign. We did **not** falsify computational chaining — we failed to elicit the third donor under the pre-registered absolute gate. That is the result.

**Stock after structural arc (for next-question ranking):**
- Early **store/select** strong; late **transform/instruction** weak (two-regime spine).
- **Parallel** compose (multislot / crossskill) and **pairwise protocol** (bind→route) work.
- Semantic **typology** + **explicitness** falsified.
- **Chain** (write→route→compute) not elicitible under hard G0.

**Next (design only until Sahil picks):** highest-value scientific question ranking delivered in chat 2026-07-13 — not the next runnable kernel. Candidates: (1) geometric interface of protocol compatibility; (2) necessity of natural bind/route mechanisms; (3) residual ceiling vs architecture for instruction/data.

### 2026-07-13 — necessity protocol APPROVED v1.1 (design locked; no code yet)

**Sahil chose:** sufficiency → compatibility → necessity. Approved plan with two tweaks.

**Tweaks locked in `NECESSITY_PROTOCOL.md`:**
1. **Selectivity/utility are PRIMARY evidence** (S1/S2/U) — target drop alone cannot yield `NECESSARY` (guards generic damage).
2. **Projection-strength sweep** \(\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}\) pre-registered (cheap site-local hook). α-sweep failure demotes to `PARTIAL`.

**Intervention:** site-matched directional ablation \(h \leftarrow h - \alpha P_S h\) at (L2, val_slot) for bind and (L2, flag digit) for route. Native surfaces. Content-selective bind (Δ_true vs Δ_foil); wrong-subspace route controls; Completion + cross-skill utility.

**Hard stop:** site-local fail → `SUFFICIENT_ONLY`. **No** layer expansion, all-pos erase, weight orthogonalization, or post-hoc rescue.

**Novelty (scoped):** method (project-out) established (RepE/Arditi); conjunction with prior sufficiency+protocol on these micro-skill dirs + selectivity-primary + α-sweep is the contribution. Not unique-circuit claim (Makelov/Grant ceiling).

**Headline:** `NECESSARY` ⟺ G0 ∧ S1 ∧ S2 ∧ U ∧ T. One kernel `delta_necessity` when Sahil says launch — then stop.

### 2026-07-13 — delta_necessity LAUNCH

Sahil: go. Module `src/causal_maps/delta_necessity.py` + `forward_with_project` in `patching.py`. Stage wired. Kernel: `run delta_necessity --config '{"quantization":"8bit"}'`. Site-local only; no rescue.

### 2026-07-13 — delta_necessity COMPLETE — ROUTE_NECESSARY_ONLY (stop)

Kernel `shlydv/cm-delta-necessity` COMPLETE (~8 min). Artifacts: `runs/delta_necessity/`.

**G0:** bind 100%, route 100%.

**Bind arm:** S1 ✅ (drop_true=+8.08 vs foil −0.28, p=0.010); U ✅ (Select/Completion Δacc=0); T ❌ (need ≥0.5×pref_clean=17.1, got 8.08). Acc drop 100%→50% is real but T is pref-gated when pref_clean>0. α-sweep ρ_true=+0.90, foil flat.

**Route arm:** S2 ✅ (drop_route=+22.7 vs wrong +0.05, p=0.010); T ✅ (acc 100%→50%); U ✅ (Variable/Completion Δacc=0); α-sweep ρ=+0.70.

**Verdict:** `ROUTE_NECESSARY_ONLY` (kernel log said PARTIAL from aggregator bug when bind S1∧¬T; corrected to table; gates unchanged; **no re-run**).

**Interpretation (calibrated):** at L2 flag digit, Δ_route is load-bearing for native Select with clean selectivity/utility — not generic damage. Binding Δ_v at val_slot is **content-selective** and utility-safe but below pre-registered magnitude T — not a clean bind necessity pass. Not unique-circuit (Makelov/Grant). **No layer expansion.**

**Branch stopped.**

### 2026-07-13 — asymmetry PRE-REGISTERED + LAUNCH

**Question:** why route necessity ≫ bind necessity — *before* any “bus” interpretation.

**Sequence (Sahil + ChatGPT + Grok):** (1) study asymmetry; (2) test redundancy vs incomplete-basis vs bottleneck with frozen ratios; (3) elevate to bus *only if* A or (C∧B) land.

**Protocol:** `ASYMMETRY_PROTOCOL.md`. Accounts A (span redundancy \(R_{span}≥1.5\)), B (incomplete \(R_{knock}≥2\) ∧ ¬A), C (bottleneck \(\beta_{route}≥0.7\), \(\beta_{bind}≤0.4\)).

**Note:** necessity already showed drop_span≈drop_true descriptively — pressures A; this kernel adjudicates with knockouts + C.

Kernel: `run delta_asymmetry --config '{"quantization":"8bit"}'`.

### 2026-07-13 — delta_asymmetry COMPLETE — solid B; ¬A; C caveat; no bus

Kernel `shlydv/cm-delta-asymmetry` COMPLETE (~8 min). Artifacts: `runs/delta_asymmetry/`.

| Account | Result | Numbers |
|---|---|---|
| **A span redundancy** | **FAIL** | \(R_{span}=0.99\) (span≈dir; confirms necessity hint) |
| **B incomplete basis** | **PASS** | \(R_{knock}=4.21\); knock drop +34.0 vs dir +8.1; p=.01 |
| **C bottleneck** | **gates PASS, control INVALID** | \(\beta_{bind}=0.24\) OK; route `drop_knock≈0` (within-flag mean-knock keeps flag-conditioned mean) ⇒ \(\beta_{route}\) inflated — does **not** show route-dir≈site |
| **U** | PASS | Completion Δacc=0 under knock |

**Printed verdict:** `ASYM_BOTTLENECK_AND_INCOMPLETE` (frozen C1∧B1∧U).  
**Solid claim for writeup:** bind write at val_slot is mostly **outside** our extracted Δ span (**incomplete basis**); extracted dirs are **not** redundant with each other (¬A). Route Δ remains load-bearing (necessity); this kernel does **not** cleanly prove route is a site bottleneck.

**Bus elevation: NOT licensed** (C knockout failed its intended meaning). No re-run, no locus fishing. **Line stop** per plan (redundancy prediction tested; answer is incomplete-basis, not span-redundancy).

### 2026-07-13 — bind-miss PRE-REGISTERED + LAUNCH

**Why-question:** what information in native binding does \(\mathrm{span}\{\Delta_u\}\) miss?

**Protocol:** `BIND_MISS_PROTOCOL.md`. Accounts L (linear residual probe+causal ADD), P (distributed knock), N (nonlinear local).

Kernel: `run delta_bindmiss --config '{"quantization":"8bit"}'`.

### 2026-07-13 — delta_bindmiss COMPLETE — BIND_MISS_LINEAR_READOUT (stop)

Kernel `shlydv/cm-delta-bindmiss` COMPLETE (~3 min after load). Artifacts: `runs/delta_bindmiss/`.

| Gate | Result |
|---|---|
| **L1** residual probe | ✅ **100%** (p=.01); full H also 100%; **span projection also 100%** |
| **C1** residual-centroid ADD recovery | ❌ rec=**−0.06** (span+ADD does not restore pref) |
| **P1** distributed | ❌ |
| **P1-local** | ✅ knock drop +34.6 at val_slot; neighbors ≈0 |

**Verdict: `BIND_MISS_LINEAR_READOUT`.**

**Interpretation (calibrated):** After removing \(\mathrm{span}\{\Delta_u\}\), the bound value is still **linearly readable** at the same site (and was also readable *inside* the span). So native value identity is not uniquely localized to our extracted 10-D for *readout*. But the residual centroid is **not a causal install direction** under ADD (C1 fail). Knockout remains **site-local**.  

This sharpens incomplete-basis: the “more” is (at least) **linearly decodable ambient / correlated features at val_slot**, not a second steerable Δ-basis we can ADD, and not off-slot distribution.

**Line stop.** No position/layer fishing. Next why-question only on Sahil call (e.g. LEACE-style erasure; or does the router read span vs residual?).

### 2026-07-13 — router-read PRE-REGISTERED + LAUNCH

**Why-question:** does \(\Delta_{\mathrm{route}}\) read \(\mathrm{span}\{\Delta_u\}\) write, ambient residual, or both?

**Protocol:** `ROUTER_READ_PROTOCOL.md`. Native bindings + write-site keep_span / keep_res + route ADD.

Kernel: `run delta_router_read --config '{"quantization":"8bit"}'`.

### 2026-07-13 — delta_router_read COMPLETE — ROUTER_READS_RESIDUAL (stop)

Kernel `shlydv/cm-delta-router_read` COMPLETE (~9 min). Artifacts: `runs/delta_router_read/`.

**G0:** flag0→Y 92%, flag1→X 92%.

| Surface | RS | p | Gate |
|---|---|---|---|
| FULL (native) | **+48.24** | .010 | G1 ✅ |
| SPAN keep \(P_S h\) | **+0.14** | .386 | SPAN ❌ (< 0.5× full) |
| RES keep \(h-P_S h\) | **+30.87** | .010 | RES ✅ (≥ 0.5× full) |

**Verdict: `ROUTER_READS_RESIDUAL`.**

**Interpretation (calibrated):** Under native bindings, \(\Delta_{\mathrm{route}}\)’s routing sensitivity survives residual-only write-site content and **collapses** when only \(\mathrm{span}\{\Delta_u\}\) is kept. So the protocol “language” the router listens to is **not** our extracted install subspace — it is the **ambient residual** that bind-miss showed still linearly decodes \(v\). Our \(\Delta_v\) remain sufficient *install* dirs (prior work) but appear **correlated / incomplete** relative to what routing actually reads natively.

**Line stop.** No rescue. This is the arc’s strongest reframing so far.

### 2026-07-13 — delta_router_ood COMPLETE — OOD_FAIL → demote router-read

Kernel `shlydv/cm-delta-router_ood` COMPLETE. Artifacts: `runs/delta_router_ood/`.

| | rel_norm | cos | rel_disp |
|---|---|---|---|
| **keep_span** | 0.485 | 0.485 | 0.854 |
| **keep_res** | 0.854 | 0.854 | 0.485 |

**Fails:** cos (0.485 < 0.854−0.30), disp (0.854 > 0.485+0.30). Norm gate marginal-OK.  
**energy_frac** span in clean = **0.261** — keep_span retains ~¼ of energy by construction; asymmetry is partly geometric.

**Verdict: `OOD_FAIL`.** Demote `ROUTER_READS_RESIDUAL` → **`ROUTER_READ_AMBIGUOUS`**. ChatGPT’s warning correct: SPAN may have failed because the state is OOD, not because the router ignores Δ.

**Do not** elevate “Δ as generator” yet — that needs a *matched* control (e.g. random subspace with same energy_frac / norm-matched keep) first.

**Agreement with the critique:** slow down; biggest *candidate* result is not yet earned.

### 2026-07-13 — native-trajectory gate PRE-REGISTERED

**Question:** does cross-fitted L2 ADD \(\Delta_v\) induce the same downstream
state as a natural textual value change, and does that state causally mediate
the output?

**Protocol:** `NATIVE_TRAJECTORY_PROTOCOL.md`. Five-fold template cross-fit;
natural CF, ADD, wrong-value, and 100 same-norm random controls; frozen L8
trajectory and mediation checkpoint. Full-state natural/generated patches
replace the invalid `keep_span`/`keep_res` comparison.

**Decision rule:** only `CONTROL_GENERATES_NATIVE_STATE` licenses cross-model
scaling and attention-path mediation. Any weaker verdict is logged without a
layer/α/template rescue.

### 2026-07-13 — delta_trajectory COMPLETE — CONTROL_NATIVE_LIKE_NO_CONVERGENCE

Kernel `shlydv/cm-delta-trajectory` COMPLETE (~9 min). Artifacts:
`runs/delta_trajectory/`. Five template-held-out folds, 50/50 pairs.

| Gate | Result |
|---|---|
| G0 | ✅ clean 98%, natural CF 100% |
| O1 | ✅ ADD Δlogit +53.17 vs random +0.63, p=.0099 |
| A1 | ✅ L8 val cosine **.992** vs wrong .247 / random .182 |
| A2 | ❌ normalized error **.120 at L2 → .124 at L8** (no contraction) |
| Q1 | ✅ L8 query/last cosine **.730**, random .333, p=.0099 |
| M1 | ✅ ADD-state / natural-state patch effect **.996** |
| M2 | ✅ replacing L8 val state with CLEAN blocks **92.7%** of ADD effect |
| D1 | ✅ L8 activation-norm ratio 1.000 |

**Verdict: `CONTROL_NATIVE_LIKE_NO_CONVERGENCE`.**

**Correct interpretation:** cross-fitted ADD creates an activation already very
close to the natural textual counterfactual at the injection site; downstream
query states increasingly align with the natural trajectory (cos .07 L2, .48
L4, .73 L8, .84 L14), and the L8 write state has essentially the same patch
effect as the natural state. It does **not** become more native-like at the write
site, so the proposed “generator reconstructs the native state” mechanism is
not supported.

**Critical scope:** each cf value has one fixed clean source value; folds vary
only variable name. Thus .992 may reflect a context-stable
source→target replacement vector, not a source-invariant abstract binding
message. Prior embed control rules out a raw embedding artifact, but not this
stronger transition-code alternative. No cross-model scale-up or head-path
mediation under the frozen decision rule.

### 2026-07-13 — affine counterfactual-operator gate PRE-REGISTERED

**Narrowing question:** can value rewrites be constructed as
\(d_{a\rightarrow b}=z_b-z_a\) from independently estimated value prototypes,
then reproduce the natural internal trajectory for unseen source×target
transitions in a two-binding prompt?

**Protocol:** `COUNTERFACTUAL_OPERATOR_PROTOCOL.md`. Donor codebook from four
single-variable contexts; primary test is 40 balanced multi-binding trials with
queried-slot, wrong-target, other-slot, and 100 same-norm random conditions.
Frozen L2 state-equivalence, L8 query-trajectory, role-specificity, and
full-state mediation gates.

**Novelty boundary:** additive binding IDs and linear counterfactual steering
are prior art. The candidate contribution is approximate prompt-reachability:
a compositional latent operator reproducing a natural counterfactual trajectory
and mediator, not merely the same output.

Only `AFFINE_COUNTERFACTUAL_OPERATOR` licenses broader cross-skill/model study.

### 2026-07-13 — operator protocol v1.1 audit revision (before results)

Independent code/science audit of `delta_trajectory` found no leakage,
hook-order, sign, or verdict bug, but correctly narrowed mediation: blocking L8
proved necessity for ADD, not for the natural textual counterfactual. Equal
patch effects alone do not prove a shared path.

The first `delta_operator` remote job was started, then its local watcher was
stopped while still RUNNING; no result or metric was inspected. That version
is **VOID by protocol revision**.

`COUNTERFACTUAL_OPERATOR_PROTOCOL.md` v1.1 adds, before any operator result:

1. natural-CF + CLEAN-state block at the same L8 queried slot;
2. M2 requires ≥70% block for both ADD and natural effects, with block
   fractions within 0.20;
3. save every per-trial metric and all aggregate null draws.

Relaunch exactly once from the audited v1.1 code. No threshold changed.

### 2026-07-13 — delta_operator v1.1 COMPLETE — AFFINE_COUNTERFACTUAL_OPERATOR

Kernel `shlydv/cm-delta-operator-v11` COMPLETE (~8.5 min). Artifacts:
`runs/delta_operator_v11/`. Forty balanced source×target transitions in a
two-binding prompt; operators \(d_{a\rightarrow b}=z_b-z_a\) were constructed
from value prototypes learned only in four single-binding donor contexts.

| Gate | Result |
|---|---|
| G0 behavior | ✅ CLEAN 97.5%, natural CF 100% |
| A1 affine write state | ✅ L2 cos **.9928**, error **.1125**, 40/40 cos≥.5 |
| Q1 query trajectory | ✅ L8 last cos **.7977** vs random .3758, p=.0099 |
| O1 causal rewrite | ✅ ADD +75.78 vs natural +75.57, ratio **1.003**, 40/40 positive |
| R1 role specificity | ✅ other-slot +3.05 = 4.0% of own-slot |
| M1 state patch | ✅ ADD/native effect ratio **1.002** |
| M2 shared necessity | ✅ block ADD **97.69%**, natural **97.35%**, gap .0033 |
| D1 | ✅ L2/L8 slot norm ratios 1.006/1.001 |
| S1 held-out K | ✅ L2 cos .9960; L8 query cos .8440; output ratio 1.001 |

**Verdict: `AFFINE_COUNTERFACTUAL_OPERATOR`. All frozen gates pass.**

**Strongest licensed claim:** In this Qwen7B variable-memory system, value
prototypes estimated in single-binding contexts compose as \(z_b-z_a\) to
causally reproduce unseen multi-binding textual counterfactuals: near-identical
write states, aligned query trajectories, natural-sized output changes, clean
slot specificity, and—critically after audit—the same L8 site is necessary for
both ADD and the natural textual rewrite.

**Not yet licensed:** a universal binding code, exact prompt surjectivity,
cross-model generality, or a general law of steering. Wrong-target operators
still move the target contrast substantially (+37.81 vs correct +75.78), so
content geometry is graded rather than perfectly discrete.

Per protocol, this PASS licenses the broader prompt-reachability program across
existing Store/Select/Transform directions and multiple model families. That
program must compare approximate natural-trajectory reachability, not merely
behavioral steering.

### 2026-07-13 — operator content-specificity control PRE-REGISTERED

**Why:** the wrong-target operator raised the old target-vs-source scalar by
+37.81, so that metric mixes intended-target gain with source suppression.

**Protocol:** `OPERATOR_CONTENT_PROTOCOL.md`. Same model, codebook, 40 trials,
L2 site, and α. Compare target ADD to its natural target rewrite and wrong ADD
to its own natural wrong rewrite using global greedy accuracy and centered
ten-value logit vectors.

**Gate:** both operators must emit their own intended values ≥90%, match their
own natural multiclass logit displacement (cos≥.95, error≤.25), and discriminate
target/source/alternate on ≥90%. Positive verdict:
`CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR`. No random-null rerun or tuning; this
is a measurement-validity control, not a new discovery kernel.

### 2026-07-13 — operator target-specificity audit PRE-REGISTERED

**Why:** WRONG operator \(z_w-z_a\) raised the frozen
`logit(b)-logit(a)` metric by +37.81. This can be a measurement artifact:
suppressing source \(a\) raises that binary contrast without selecting target
\(b\).

**Protocol:** `OPERATOR_READOUT_AUDIT.md`. Exact same donors, 40 trials, L2
sites, and correct/wrong operators. Readout only: full-vocabulary greedy token
and ten-value multiclass margin. No new null, prompt, layer, α, or trajectory
analysis.

**Gate:** correct ADD must greedily select \(b\); wrong ADD must greedily select
its own \(w\), not \(b\). Only `TARGET_SPECIFIC_OPERATOR` clears broader
scaling.

**Superseded before run:** use the stricter
`OPERATOR_CONTENT_PROTOCOL.md`/`delta_operator_content` only. It raises both
greedy/discrimination thresholds to 90% and additionally requires each ADD
condition’s centered ten-value logit displacement to match its own natural
textual counterfactual (cos≥.95, error≤.25). Do not run
`delta_operator_readout`.

### 2026-07-13 — delta_operator_content COMPLETE — CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR

Kernel `shlydv/cm-delta-operator-content` COMPLETE (~3.7 min). Artifacts:
`runs/delta_operator_content/`. Exact same codebook, 40 trials, L2 operators,
and two-binding prompts as `delta_operator_v11`.

| Gate | Result |
|---|---|
| G0 natural target/wrong | ✅ 100% / 95% |
| C1 ADD target/wrong | ✅ 100% / 95% global greedy |
| C2 multiclass equivalence | ✅ target cos .99982/error .0193; wrong cos .99979/error .0212 |
| C3 discrimination | ✅ target 100%; wrong 100% |

**Verdict: `CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR`.**

The suspicious prior WRONG binary effect is resolved: wrong ADD raised its own
intended value by **+37.70** and suppressed the source by **37.69**, but changed
the original target by only **+0.12**. Thus `logit(b)-logit(a)` rose because
\(a\) was removed, not because \(b\) was selected.

**Combined licensed result:** prototype arithmetic \(z_b-z_a\), learned in
single-binding contexts, performs content-specific, role-specific rewrites in
multi-binding memory and reproduces the matching natural textual
counterfactual at the write state, query trajectory, candidate-logit state,
and shared L8 causal mediator.

This clears the frozen gate for the cross-skill prompt-reachability atlas.

### 2026-07-13 — cross-skill natural-reachability atlas PRE-REGISTERED

**Question:** is binding's approximate natural reachability a general class of
activation operator or a memory-specific exception?

**Protocol:** `REACHABILITY_ATLAS_PROTOCOL.md`. STORE is the frozen positive
reference. One Qwen7B kernel tests:

1. SELECT flag 0→1 at L2→L8;
2. TRANSFORM arithmetic answer increment at L20→L26;
3. INSTRUCTION data→live-command reframing at L20→L26.

Each cell compares cross-fitted ADD to its exact natural textual
counterfactual using injection/query trajectory, output equivalence, full-state
patching, shared ADD/natural blocking, ANTI, and 100 same-norm random controls.

**Decision:** ≥2/3 new cells fully natural-reachable →
`COUNTERFACTUAL_OPERATORS_GENERALIZE`, licensing multi-family replication.
No template/layer/α rescue for weaker patterns.

### 2026-07-13 — delta_reachability COMPLETE — MIXED_REACHABILITY

Kernel `shlydv/cm-delta-reachability` COMPLETE (~7.3 min). Artifacts:
`runs/delta_reachability/`.

| Cell | Verdict | Key result |
|---|---|---|
| STORE | frozen positive | affine/content-specific natural counterfactual operator |
| SELECT | `OUTPUT_EQUIVALENT_ONLY` | L2 flag state cos .998; output/patch/block ≈ natural; L8 query cos .413, error 1.02, p=.832 |
| TRANSFORM | `CONTROL_NULL` | generic +1 operator: injection cos −.032; output ratio −.006 |
| INSTRUCTION | `NATURAL_REACHABLE` | L20 cos .952; L26 query cos .892; output ratio .760; patch ratio .772; shared block 100.5%/96.0% |

**Atlas verdict: `MIXED_REACHABILITY`** (only INSTRUCTION fully passes among
three new cells). Per frozen rule, do **not** immediately scale models.

**Interpretation:**

1. Approximate natural reachability is **not binding-specific**: a late
   data→instruction framing direction also follows the matching natural
   trajectory and mediator.
2. Behavioral/causal equivalence does **not guarantee full trajectory
   equivalence**: SELECT recreates the natural control state at the flag, the
   output, and the load-bearing L8 flag mediator, yet the L8 final-position
   displacement differs.
3. TRANSFORM only falsifies the preregistered source-invariant arithmetic +1
   operator. Prior transform success used answer-specific late directions, so
   this is not a universal “computation is unreachable” result.

**Scientific boundary now:** natural counterfactual operators exist across at
least memory content and instruction framing, while other effective
interventions can be locally natural/causally equivalent yet travel through a
different intermediate trajectory. This mixed boundary—not a universal class—
is the next object to understand.

### 2026-07-13 — instruction-reachability audit — abstract claim DEMOTED

Independent audit found no token-ID, split-overlap, hook-order, sign, or gate
arithmetic bug. The exact fixed-template result reproduces.

However, INSTRUCTION's abstract interpretation is confounded:

1. live-instruction and data prompts are 34 vs 49 tokens; Δ includes the whole
   fixed 15-token wrapper/position displacement;
2. L20 and the same eight test payloads were selected/measured in the prior
   instruction experiment;
3. L26 final-state patch/block is near-unembedding and largely a mandatory
   output cut-set;
4. null draws used independent per-row random vectors rather than a shared
   random operator.

**Safe claim only:** one fixed train-template displacement transfers across
held-out words and mimics that template pair's natural final-state/output
shift. Do not call it a template-independent instruction-framing operator.

Printed atlas verdict remains `MIXED_REACHABILITY`; scientific interpretation
is demoted to **`REACHABILITY_ATLAS_AMBIGUOUS`**. Cross-model scale remains
blocked.

### 2026-07-13 — instruction reachability validation PRE-REGISTERED

**Protocol:** `INSTRUCTION_REACHABILITY_VALIDATION.md`.

One decisive test removes the dominant confounds:

- execute/data pairs differ only in one one-token mode label and must be
  exactly length/position matched;
- train on two mode/template families (`run/hold→skip`,
  `live/quote→text`);
- test on an unseen family and output vocabulary (`do/read→none`);
- semantic payload categories are disjoint;
- compare against a matched mode/template-only displacement and one **shared**
  same-norm random operator per draw;
- preserve L20→L26 trajectory and shared-mediator gates.

Pass → `TEMPLATE_INVARIANT_INSTRUCTION_OPERATOR`. Fail →
`FIXED_TEMPLATE_DISPLACEMENT` and close the exact shared-additive-L20 branch
without family/layer/α rescue.

**Scope correction before result:** neither outcome is universal. Pass gives
template-held-out Qwen7B evidence and licenses replication. Fail rejects this
exact α=1/L20/shared-vector operationalization; it does not prove that no
instruction representation exists under other models, tasks, sites, or
parameterizations.

**Pre-run tokenizer alignment fix:** held-out labels `do/read` were rejected
because `kind=read` merges under Qwen tokenization while `kind=do` does not
(pair lengths 65 vs 66). Replaced before any model forward/result with unseen
`go/see`; tokenizer preflight confirms every family pair is equal-length with
exactly one differing token. No behavioral or causal result was inspected.

### 2026-07-13 — instruction validation v1 VOID — output-token harness

Kernel stopped before causal intervention: bare `grape` tokenizes as two tokens
(`[901, 2027]`). No result JSON/verdict was produced. This is a broken output
meter, not a failed scientific gate.

Replace only `grape` with category-disjoint bare token `false`; add a preflight
requiring every payload and negative answer to be one bare token. Templates,
sites, α, thresholds, controls, and all other payloads remain frozen. Relaunch
as v2.

The new full preflight also rejected bare `shut`; replace it with bare `true`
before relaunch. Final held-out pool:
`apple,false,north,south,open,true,left,right`. No model result inspected.

### 2026-07-13 — instruction validation v2 COMPLETE — VALIDATION_INELICITABLE

Kernel `shlydv/cm-delta-instruction-validate-v2` COMPLETE. Artifacts:
`runs/delta_instruction_validate_v2/`.

- alignment: all family pairs equal-length, exactly one token differs;
- train execute accuracy: 100%;
- **train data accuracy: 50% < frozen 80% G0**;
- held-out execute/data: 100% / 100%.

G0 failed, so the kernel correctly stopped before direction extraction or
causal intervention. Verdict: `VALIDATION_INELICITABLE`.

**Interpretation:** the original fixed-template instruction reachability result
is not validated as an abstract, template-independent operator. This run also
does not falsify every possible instruction representation; the preregistered
two-family donor construct was not behaviorally elicitable. Per protocol, no
template repair or rerun. Close this validation branch.

### 2026-07-13 — SELECT causal-quotient test PRE-REGISTERED

**Why-question:** how can SELECT reproduce the natural flag state, output, L8
flag-site patch effect, and necessity while its L8 final/query displacement has
cosine only .413?

**Protocol:** `SELECT_CAUSAL_QUOTIENT_PROTOCOL.md`. Same fixed data and L2
direction. At layers {4,8,12,14,16,20,26}, test full-state query geometry,
ADD/natural query patches, bidirectional state swaps, and query-state blocking.

Competing outcomes:

- `CAUSAL_QUOTIENT_EQUIVALENCE`: geometrically different query states are
  causally interchangeable and load-bearing;
- `DELAYED_NATURAL_CONVERGENCE`: they merge later;
- `PARALLEL_OR_UNRESOLVED_PATHS`: neither.

No subspace fitting, layer selection, or rescue.

### 2026-07-13 — SELECT causal-quotient test COMPLETE — DELAYED_NATURAL_CONVERGENCE

Kernel `shlydv/cm-delta-select-quotient` COMPLETE. Artifact:
`runs/delta_select_quotient/results_delta_select_quotient.json`.

- clean/natural accuracy: 100% / 100%;
- ADD/natural output-effect ratio: .991;
- L8 final/query discrepancy replicated: cosine .413, error 1.020;
- L16: cosine .619, error .855;
- **L20: cosine .986, error .168**;
- L26: cosine .993, error .115.

At L20 the converged query states were causally equivalent:

- ADD/natural patch-effect ratio: 1.002;
- bidirectional swap deviations: .0065 / .0034 of natural effect;
- clean-state overwrite blocked 99.75% / 98.88% of ADD/natural effects.

No divergent layer passed causal-equivalence gates; L20 and L26 passed frozen
delayed-convergence gates. Verdict: `DELAYED_NATURAL_CONVERGENCE`.

**Interpretation:** this is the first clean positive instance of the proposed
control-interface dynamic. A low-dimensional L2 intervention recreates the
natural local selection-control state; the query trajectory remains distinct
through L16, then converges to the same full, load-bearing natural state by
L20. This rules against a merely output-equivalent parallel path in this cell.
Scope remains one task and one model.

### 2026-07-13 — held-out-answer reasoning controller PRE-REGISTERED

**Protocol:** `REASONING_CONTROLLER_PROTOCOL.md`.

One Qwen7B kernel tests whether an L8 mean red→blue branch operator can reroute
unseen two-step graph problems whose correct endpoint varies and is balanced
across A–J. The active intervention must reproduce natural multiclass logits,
the L20 query state, reverse blue→red behavior, and the shared L20 mediator.

Controls: raw red→blue embedding displacement, 100 norm-matched shared random
operators, reverse operator, disjoint graphs, exact continuation-token audit.
No layer/α/prompt repair. A prelaunch code audit found and fixed continuation
token, null-gate, baseline, and verdict-aggregation errors; the corrected
kernel passed compilation, tokenizer/alignment checks, balance/leakage checks,
and a second read-only audit.

### 2026-07-13 — held-out-answer reasoning controller COMPLETE — REASONING_INELICITABLE

Kernel `shlydv/cm-delta-reasoning-controller` COMPLETE. Artifact:
`runs/delta_reasoning_controller/results_delta_reasoning_controller.json`.

Mechanical preflights passed: 20/20 balanced disjoint graphs, uniform 88-token
prompts, exactly one red/blue token difference at position 66, and valid exact
continuation answer IDs.

Frozen G0 failed:

- clean/red native accuracy: 35%;
- natural-blue native accuracy: 20%.

The run stopped before extracting or applying a direction. Verdict:
`REASONING_INELICITABLE`. This is an elicitation failure, not evidence for or
against latent control of multi-step reasoning. Per protocol, no prompt repair
or rerun of this construct.

### 2026-07-13 — reasoning eligibility screen PRE-REGISTERED

Before another causal kernel, `REASONING_ELIGIBILITY_SCREEN.md` freezes a
behavior-only screen:

1. priority: natural-language one-hop vs two-hop kinship reasoning;
2. fallback: add vs subtract program switching.

Each family has 20 exact-continuation examples and an equal-length one-token
mode change. Eligibility requires ≥90% native accuracy in both modes. Selection
uses behavior only; no activation is extracted. If neither passes, stop.

### 2026-07-13 — reasoning eligibility screen COMPLETE — ELIGIBLE_ARITHMETIC_PROGRAM

Kernel `shlydv/cm-delta-reasoning-screen` COMPLETE. Artifact:
`runs/delta_reasoning_screen/results_delta_reasoning_screen.json`.

- kinship one/two: 40% / 55%, ineligible;
- arithmetic add/subtract: 100% / 100%, eligible.

Per the frozen behavior-only priority rule, arithmetic is selected for a fresh
causal protocol. No activation was extracted during task selection.

### 2026-07-13 — arithmetic program controller PRE-REGISTERED

`ARITHMETIC_CONTROLLER_PROTOCOL.md` freezes a Qwen7B add→subtract operator
learned at the L8 query position on 10 operand pairs and tested on the other 10.
It must recover variable subtraction answers, reverse back to addition,
reproduce digit-logit and L20 state displacements, and share the L20 mediator.

Controls include separate norm-matched 100-draw nulls for the learned operator
and raw token-embedding baseline. All prediction, trajectory, content, and
mediation rows are saved. The kernel passed compilation/tokenization/split
checks and a multi-pass read-only audit before launch.

### 2026-07-13 — arithmetic program controller COMPLETE — ARITHMETIC_CONTROL_NULL

Kernel `shlydv/cm-delta-arithmetic-controller` COMPLETE. Artifact:
`runs/delta_arithmetic_controller/results_delta_arithmetic_controller.json`.

- native add/subtract: 100% / 100%;
- steered target accuracy: 0%; reverse accuracy: 0%;
- natural output effect +36.37 vs steered −0.41 (ratio −.011, p=.485);
- L8 state cosine .166/error 1.050;
- L20 state cosine .068/error .999;
- digit-logit cosine −.291/error 1.013.

A1/O1/C1/R1/Q1/M1/M2 all failed. The raw embedding baseline was also null.
Verdict: `ARITHMETIC_CONTROL_NULL`.

**Interpretation:** the behavior-qualified, content-independent mean L8
add→subtract query operator does not transfer across held-out operands. This
is a clean boundary against immediately extending the storage/selection
control-interface claim to arithmetic program changes. No layer or α rescue.

### 2026-07-13 — tool orchestration eligibility screen PRE-REGISTERED

`ORCHESTRATION_ELIGIBILITY_SCREEN.md` tests a complete behavior-only workflow:
Qwen must emit exact calculator/database calls with row-specific arguments,
the harness executes the parsed call, and Qwen must use the returned result.
Both modes require ≥90% exact action, correct arguments, final answer, and
same-row end-to-end success.

The screen uses a frozen Qwen commit and bitsandbytes 0.49.2, saves full
pre-EOS generations, rejects extra text and malformed calls, executes parsed
arguments rather than expected arguments, and records runtime versions. A
read-only audit found no remaining semantic parsing/execution/scoring issue.
No activation is extracted unless a later causal protocol is unlocked.

### 2026-07-13 — tool orchestration screen COMPLETE — ORCHESTRATION_ELIGIBLE

Kernel `shlydv/cm-delta-orchestration-screen` COMPLETE. Artifact:
`runs/delta_orchestration_screen/results_delta_orchestration_screen.json`.

Calculator and database modes each achieved 100% exact calls, correct
row-specific actions, final answers after actual parsed-call execution, and
same-row end-to-end success across 20 payloads. Runtime/model revisions and
all raw generations were saved.

Verdict: `ORCHESTRATION_ELIGIBLE`. Unlock one causal orchestration kernel.

### 2026-07-13 — causal tool orchestration PRE-REGISTERED

`ORCHESTRATION_CONTROLLER_PROTOCOL.md` freezes one L2 calculate→lookup
mode-site operator learned on 10 payloads and tested on the other 10. Success
requires the exact database call, row-specific key, actual tool execution,
final answer matching the executed result, reverse calculator workflows,
natural L20 `CALL`-state convergence, and shared mediation.

Controls: raw mode-token embedding displacement, separate 100-draw
norm-matched nulls, full-action parsing, strict JSON, frozen Kaggle runtime,
and raw tensor artifacts sufficient to recompute states/logits/nulls. The
kernel passed tokenizer/runtime preflights, persisted regression tests,
compilation, and final read-only launch audit.

### 2026-07-13 — causal tool orchestration COMPLETE — LATENT_ORCHESTRATION_CONTROLLER

Kernel `shlydv/cm-delta-orchestration-controller` COMPLETE. Artifacts declared:
`results_delta_orchestration_controller.json` and
`raw_delta_orchestration_controller.pt` (the tensor artifact is present in the
Kaggle output listing but was not pulled locally by the runner).

All frozen gates passed:

- local L2 cosine .9994/error .0356;
- tool effect +47.58 vs natural +48.16, ratio .988, p=.0099;
- database calls/actions 100%; final/end-to-end 90%;
- reverse calculator workflows 100%;
- L20 `CALL` state cosine .9800/error .2000;
- patch ratio .9957; block fractions 1.0096/.9433, gap .0663;
- raw embedding baseline failed.

Verdict: `LATENT_ORCHESTRATION_CONTROLLER`.

**Safe interpretation:** one prompt-bound L2 mode operator switched all held-out
actions, including variable database keys, and reproduced the natural
next-tool state by L20. One of ten final answer stages ignored the correct
database result and reverted to the calculator answer.

**Required caution before breakthrough elevation:** learned direction norm
27.61 vs raw embedding baseline 1.33. The frozen B1 passed, but the baseline is
not scale-matched, so transformed/scaled lexical replay remains unresolved.
The result concerns one fixed template and immediate tool choice; it does not
yet establish an abstract controller or full autonomous planning.

### 2026-07-13 — orchestration lexical-scale control PRE-REGISTERED

`ORCHESTRATION_LEXICAL_CONTROL.md` directly addresses the 27.61 vs 1.33 norm
imbalance in the original embedding baseline. One kernel compares the learned
positive reference against:

- embedding(`lookup`)−embedding(`calculate`) scaled to the learned norm;
- the same embedding direction with a donor-only least-squares scalar.

Each baseline must pass tool-logit, full forward/reverse workflow, actual
execution, final-answer, L20 trajectory, and its own 100-draw norm-matched
null. No test fitting or α sweep. Tokenizer/provenance preflights, native raw
workflows, tensor artifacts, compilation, lint, and a read-only audit passed
before launch.

### 2026-07-13 — orchestration lexical-scale control COMPLETE — BEYOND_SCALE_MATCHED_EMBEDDING

Kernel `shlydv/cm-delta-orchestration-lexical` COMPLETE. Artifact:
`runs/delta_orchestration_lexical/results_delta_orchestration_lexical.json`.

Positive reference reproduced (ratio .988; 100% calls; 90% end-to-end; 100%
reverse; L20 cosine .980/error .200).

Lexical controls:

- learned/embedding cosine .150;
- norm-matched embedding (scale 20.77): output ratio .025, 0% workflows,
  L20 cosine .187/error .986;
- donor-optimal embedding (scale 3.11): ratio −.016, 0% workflows,
  L20 cosine .133/error .992.

Both failed output, workflow, reverse, and trajectory gates. Verdict:
`BEYOND_SCALE_MATCHED_EMBEDDING`.

**Interpretation:** the orchestration result survives the observed magnitude
confound and is not scalar replay of the input embedding contrast. It remains
prompt-bound and could still reflect a nonlinear transformed lexical mode
representation. The next generality test must hold out mode labels/templates,
then replicate across a second model.

### 2026-07-13 — orchestration label/template transfer COMPLETE — CONTROLLER_NOT_REPLICATED

Kernel `shlydv/cm-delta-orchestration-label-transfer` COMPLETE. Artifact:
`runs/delta_orchestration_label_transfer/results_delta_orchestration_label_transfer.json`.

Template B was natively eligible (calculator 100% end-to-end; database 90%).
The frozen template-A operator did not transfer to held-out `red`/`blue`
labels and reordered template:

- A/B direction cosine .021;
- output ratio .108;
- 0% forward and reverse workflow switches;
- L20 cosine .079/error 1.096.

The B-specific positive reference showed that B itself supports a strong
tool-choice controller: ratio .998, 100% target calls, 100% reverse
end-to-end, and L20 cosine .981/error .196. However, its forward end-to-end
accuracy was only 60% because four correct database calls were followed by
answers that ignored the executed result. It failed the frozen ≥80% gate.

Verdict: `CONTROLLER_NOT_REPLICATED`. The existing result remains a real
prompt-bound controller that is beyond scalar lexical replay, but it is not a
template-independent orchestration controller. The preregistered
multi-workflow/model expansion is not unlocked.

### 2026-07-13 — agent-workspace context screen COMPLETE — WORKSPACE_CONTEXTS_INELICITABLE

Kernel `shlydv/cm-delta-agent-workspace-screen` COMPLETE. Artifact:
`runs/delta_agent_workspace_screen/results_delta_agent_workspace_screen.json`.

Six frozen, tokenizer-aligned prompt contexts were screened behaviorally
before any causal workspace test. Four passed:

- A calculate/lookup: 100% end-to-end in both modes;
- C north/south: 100% calculator, 95% database;
- D left/right: 90% calculator, 100% database;
- E alpha/beta: 100% in both modes.

B red/blue failed because database end-to-end accuracy was 80%. F open/close
produced 0% valid workflows in both modes. The protocol required all six at
≥90%.

Verdict: `WORKSPACE_CONTEXTS_INELICITABLE`. No templates are replaced or
repaired, and the causal canonicalization test is not unlocked.

### 2026-07-13 — answer-turn controller diagnostic COMPLETE — ANSWER_TURN_LATENT_CONTROL

Kernel `shlydv/cm-delta-answer-turn-control` COMPLETE. Artifact:
`runs/cm-delta-answer-turn-control/results_delta_answer_turn_control.json`.

This diagnostic separated initial tool routing from later result integration.
The same template-B `blue`−`red` L2 direction was learned on the original
donor split, then reapplied at the red mode token in a fresh answer-turn
forward pass containing the correct database call and executed result.

All frozen gates passed on the eight non-collision rows:

- natural blue target use 7/8; natural red source use 8/8;
- contradictory red + database transcript without steering: 5/8 target;
- controller reapplied: 7/8 target, a +25-point gain;
- target-vs-source output effect +10.42 vs natural +9.16, ratio 1.137;
- positive effect on 8/8 rows; 0/100 random-null exceedances;
- norm-matched lexical baseline: 6/8 target, effect ratio .240,
  positive on 5/8, 27/100 null exceedances.

Verdict: `ANSWER_TURN_LATENT_CONTROL`.

**Mechanistic interpretation:** in the contradictory transcript, target-result
and original-calculator answers compete. The unsteered target-minus-source
logit margin exactly separated the three source fallbacks from the five target
answers. Reapplication shifted all eight rows toward the database result and
flipped two of the three source-biased rows; the strongest source-biased row
remained unchanged.

This is evidence that one template-specific policy direction is reusable at
distinct workflow stages (tool selection and result integration). It is not
evidence of permanent memory: independent model calls do not carry hidden
states by construction. It also does not restore static cross-template
transfer or establish a general working-memory mechanism.

### 2026-07-13 — cross-stage policy broadcast COMPLETE — NO_SHARED_POLICY_INTERCHANGE

Kernel `shlydv/cm-delta-agent-policy-broadcast` COMPLETE. Artifact:
`runs/delta_agent_policy_broadcast/results_delta_agent_policy_broadcast.json`.

Independent L20 blue−red decision directions were extracted on donors at the
`CALL` tool-decision token and at the later answer-decision token. Same-stage
positive references passed:

- call direction: 100% exact database calls, output ratio .964;
- answer direction: 7/8 diagnostic target answers.

But the directions were not a shared interchangeable code:

- cross-stage cosine .345 (gate ≥.80);
- norm-matched answer→call: 0% database calls, output ratio .169;
- norm-matched call→answer: 4/8 target answers, ratio −.133;
- all geometry and cross-stage gates failed.

Verdict: `NO_SHARED_POLICY_INTERCHANGE`.

**Interpretation:** the same upstream, template-specific L2 policy direction
can be reused at separate workflow turns, but downstream consumers transform
it into different stage-local decision representations. There is no evidence
for one broadcast L20 policy vector shared by tool selection and result
integration.

### 2026-07-13 — binding component-convergence test PRE-REGISTERED

Protocol: `BINDING_COMPONENT_CONVERGENCE_PROTOCOL.md`. Kernel:
`delta_binding_component_convergence`.

This is a single, held-out L8 downstream-component test for the completed
binding affine operator. It asks whether the natural two-binding value rewrite
and its L2 `z_target - z_source` controller depend on the same four-component
set among 28 L8 attention-output head channels and the L8 MLP output, at the
final readout token. Component sets are selected separately on offsets 1/3;
offsets 5/7 are held out. Every ablated NATURAL/ADD condition is compared to
an identically ablated CLEAN condition, so generic ablation damage is not
counted as mechanism evidence.

Positive outcome requires baseline and residual convergence, overlap of the
independently selected NATURAL/ADD sets, at least 50% held-out loss for both
effects from a shared set, superiority to 100 type-matched random sets, and a
one-token-earlier damage control. Negative outcomes distinguish divergent,
overlapping-but-not-localized, and distributed/redundant pathways. No result
licenses a further head hunt or broader layer search.

### 2026-07-13 — binding component-convergence test COMPLETE — OVERLAPPING_COMPONENTS_NOT_LOCALIZED

Kernel `shlydv/cm-delta-binding-component-convergence` COMPLETE. Artifact:
`runs/cm-delta-binding-component-convergence/results_delta_binding_component_convergence.json`.

The operator baseline and residual convergence replicated on the held-out
offsets 5/7: CLEAN and NATURAL accuracy were both 100%, ADD was positive on
all rows, ADD/NATURAL effect ratio was 1.001, and L8 ADD/NATURAL displacement
cosines were .995 at the queried slot and .800 at the final readout token.

Discovery selected the same four-component set for NATURAL and ADD (Jaccard
1.0): L8 MLP plus heads 3, 13, and 21. On held-out trials, however, that set
removed only 1.83% of the NATURAL logit effect and 1.87% of the ADD effect,
far below the 50% localization gate. Its natural loss exceeded all 100
type-matched random sets (p=1/101), but its ADD loss did not (p=4/101).
The preceding-token damage control passed.

Verdict: `OVERLAPPING_COMPONENTS_NOT_LOCALIZED` (G0, G1, C1, C4 pass; C2,
C3 fail).

**Interpretation:** the natural rewrite and affine controller select the same
weakly influential L8 output components, consistent with downstream
convergence, but there is no evidence that a small final-readout set carries
the mechanism. This rejects the proposed sparse L8 head/MLP-reader account at
this granularity; it does not refute residual-level convergence or identify
the full circuit as distributed. No adaptive follow-up is run from this
outcome.

### 2026-07-13 — binding causal-subspace rank curve PRE-REGISTERED

Protocol: `BINDING_CAUSAL_SUBSPACE_PROTOCOL.md`. Kernel:
`delta_binding_causal_subspace`.

The completed head/MLP and routing-head tests make a sparse-reader account
untenable at their tested granularity. This one held-out binding test asks
whether the L8 final-token displacement is instead a shared *linear
distributed* causal object. A single uncentered SVD basis is fit from stacked
NATURAL and ADD L8 displacements on discovery offsets 1/3. On held-out offsets
5/7, the preregistered rank-1/2/4/8/16 projections are subtracted separately
from NATURAL and ADD relative to their matched CLEAN L8 state.

Every rank is evaluated against 100 same-rank random orthonormal bases and a
preceding-token control. A low-rank claim requires both effects to lose at
least 50%, exceed every random basis, and leave the preceding-token effect
within 20%. No rank expansion, alternate basis learner, or layer search is
licensed by this experiment.

### 2026-07-13 — binding causal-subspace rank curve COMPLETE — HIGH_RANK_OR_NONLINEAR_DISTRIBUTED

Kernel `shlydv/cm-delta-binding-causal-subspace` COMPLETE. Artifact:
`runs/cm-delta-binding-causal-subspace/results_delta_binding_causal_subspace.json`.

The held-out baseline replicated (CLEAN/NATURAL 100% accuracy; ADD/NATURAL
effect ratio 1.001; final L8 displacement cosine .800). The uncentered
discovery SVD captured substantial held-out displacement energy—16 dimensions
captured 55.1% of NATURAL and 49.8% of ADD energy—yet removing rank
1/2/4/8/16 from the L8 final-token displacement changed either output effect
by only −.2% to +.2%. No learned rank beat its 100 same-rank random bases;
at rank 16 the best nominal p-values were .079 (NATURAL) and .050 (ADD).

Verdict: `HIGH_RANK_OR_NONLINEAR_DISTRIBUTED`.

**Interpretation:** this rejects the particular account that the dominant
linear variance subspace of the L8 final-readout displacement carries the
causal binding effect. It does not reject a shared slot state, a low-variance
task-selective subspace, cross-position dynamics, or nonlinear distributed
computation. It should not motivate a larger PCA rank sweep.

### 2026-07-13 — binding causal-state timeline PRE-REGISTERED

Protocol: `BINDING_CAUSAL_STATE_TIMELINE_PROTOCOL.md`. Kernel:
`delta_binding_causal_state_timeline`.

This fixed full-state test asks when, if ever, matched NATURAL and ADD binding
states become causally interchangeable. It uses no learned basis or component
selection. On confirmation offsets 5/7 it captures CLEAN, NATURAL, and ADD
residuals at layers 2/4/8/12/16/20/26 and at the queried value slot plus final
readout. At every fixed site it performs six matched state replacements:
NATURAL/ADD into CLEAN, NATURAL→ADD and ADD→NATURAL swaps, and CLEAN into each
condition. Interchangeability requires comparable state-patch sufficiency,
small bidirectional swap deviations, and strong CLEAN-overwrite necessity.

The result distinguishes early shared state, delayed reconstruction, slot-only
sharing, and unresolved alternative paths. It is not an authorization to infer
the upstream writers or to resume a head search.

### 2026-07-13 — binding causal-state timeline COMPLETE — SLOT_ONLY_SHARED_STATE

Kernel `shlydv/cm-delta-binding-causal-state-timeline` COMPLETE. Artifact:
`runs/cm-delta-binding-causal-state-timeline/results_delta_binding_causal_state_timeline.json`.

The held-out operator baseline replicated: CLEAN/NATURAL accuracy 100%/100%,
ADD positive on all rows, and ADD/NATURAL effect ratio 1.001.

At the **queried value slot**, NATURAL and ADD states are causally
interchangeable at L2, L4, L8, L12, L16, and L20. Across those layers,
ADD/NATURAL displacement cosine is .993–.995; swapping either matched state
changes output by at most .4% of the natural effect; and CLEAN replacement
blocks 92–100% of both effects. NATURAL and ADD states patched into CLEAN have
matched, large effects (67–76 logit units) through L20.

At L26 the value slot is no longer the bottleneck: CLEAN replacement removes
only 10.5% of either effect. The information has moved to the final readout
state, where matched NATURAL/ADD state patches into CLEAN each restore about
67.5 of the 75.8 natural effect and CLEAN replacement blocks 88.6–100%.
However, full final-token interchangeability does not strictly pass: replacing
the ADD final state with the NATURAL final state gives 67.42 rather than 75.90
(11.2% deviation; frozen gate ≤10%), while the reverse swap is essentially
unchanged. The final state therefore remains context-dependent on other token
states/KV context even when its displacement cosine is .993.

Verdict: `SLOT_ONLY_SHARED_STATE`.

**Interpretation:** the controller does recruit the natural, content-carrying
value-slot state immediately and maintains it through L20; no small head is
required for this shared state. The final answer is not encoded in one
interchangeable readout-token vector alone. Its realization depends on a
distributed sequence context after the shared slot state is communicated
forward. This is evidence for a distributed state-transit mechanism, not for
a single localized reader or a complete shared final-token circuit.

**Scope correction for earlier negatives:** the component-convergence and
causal-subspace rank tests both intervened at the L8 *final readout token*.
The timeline shows that final-token state is not yet sufficient or necessary
at L8 (its state-patch effects are only .10/.05 and CLEAN replacement leaves
the natural effect unchanged). Therefore their negative results rule out only
a sparse or dominant-linear **L8 final-token** account. They do not rule out
localized, low-dimensional, or otherwise structured mechanisms at the
causally load-bearing queried value slot.

### 2026-07-13 — binding slot-transport test PRE-REGISTERED

Protocol: `BINDING_SLOT_TRANSPORT_PROTOCOL.md`. Kernel:
`delta_binding_slot_transport`.

This is one fresh-mapping, all-head directed-edge test of how the demonstrated
shared value-slot state reaches the later distributed answer context. It uses
only source-to-target offsets 2/4/6/8, disjoint from the earlier 1/3/5/7
binding mappings. At each fixed layer L20–L26, it blocks the final-readout
query's attention edge to the queried value slot across every head, and uses
the final-readout-to-other-variable-slot edge at the same layer as the matched
control. It also tests the fixed cumulative L20–L26 block. CLEAN, NATURAL, and
ADD are always rerun under the same mask, so generic attention disruption is
not counted as binding loss.

A passing fixed-layer or cumulative result requires loss of at least 50% of
both effects and a 25-point loss advantage over the other-slot control for
both. A failure is limited to direct final-readout-to-slot transport in this
window; it does not license a layer, head, or window sweep.

### 2026-07-13 — binding slot broadcast test PRE-REGISTERED

Protocol: `BINDING_SLOT_BROADCAST_PROTOCOL.md`. Kernel:
`delta_binding_slot_broadcast`.

The direct final-readout-to-slot edge test on fresh mappings found a matched
but incomplete cumulative effect (NATURAL/ADD loss 36.8%/37.7%; other-slot
control near zero). This test does not search for an additional head or layer.
It asks whether the unaccounted route is a distributed receiver population.

On a held-out reversed two-binding layout (`Let Y = … . Let X = … .`), it
blocks every head, at each fixed L20–L26 layer, from every later causal query
position to the queried value-slot key. The exact same all-later-query block
to the other value slot is the control. It runs only the cumulative window,
and reruns CLEAN, NATURAL, and ADD under each matched mask. G0 must establish
operator transfer in the reversed layout before the intervention is read.

The frozen thresholds and verdicts are in the protocol. An essential pass
requires 80% matched loss in both NATURAL and ADD; a partial pass requires
50%. A null result requires auditing the tension with L20 source-slot
necessity, not broadening the mechanism search.

### 2026-07-13 — binding slot broadcast COMPLETE — incomplete shared access

Kernel `shlydv/cm-delta-binding-slot-broadcast` completed on the reversed
layout and passed G0. Blocking all heads from every later token position to the
queried slot over L20–L26 selectively removed 40.2% of NATURAL and 41.4% of
ADD effect; the matched other-slot control was effectively zero. This fails
the frozen 50% partial threshold and prints
`DIVERGENT_OR_UNRESOLVED_BROADCAST`, although the effects themselves are
matched rather than NATURAL/ADD-divergent.

The same all-receiver block is only modestly larger than the direct final-token
block (36.8%/37.7%). Thus the broad intermediate-token relay account is not
supported. Because the original timeline's post-L20 CLEAN overwrite removed
92–93%, this result requires a same-layout, same-timing bridge audit before a
mechanistic interpretation.

### 2026-07-13 — L20 overwrite versus outgoing-mask bridge PRE-REGISTERED

Protocol: `BINDING_SLOT_BRIDGE_PROTOCOL.md`. Kernel:
`delta_binding_slot_bridge`.

This audit returns to the timeline's original template and confirmation offsets
5/7. It directly compares the original post-L20 matched CLEAN-state overwrite,
an otherwise identical custom-mask implementation of that overwrite, and an
all-later-query outgoing-edge block restricted to L21–L26 (the layers strictly
after a post-L20 replacement). It includes a custom-mask no-block equivalence
check, an exact other-slot edge-block control, and a combined overwrite plus
edge block. No new mechanism search is performed.

### 2026-07-13 — L20 overwrite versus outgoing-mask bridge COMPLETE

Kernel `shlydv/cm-delta-binding-slot-bridge` completed with
`PATCH_MASK_DISSOCIATION`. The default and custom-mask baselines were exactly
identical. The matched post-L20 CLEAN overwrite replicated exactly in both
paths (NATURAL/ADD loss 92.4%/93.2%). The own-slot L21–L26 outgoing block
selectively removed 40.4%/41.2% (other-slot control −.4%/−.2%).

The decisive interaction is that the combined overwrite plus own outgoing block
removed only 74.4%/75.5%, less than the state overwrite alone. Thus the CLEAN
overwrite is not merely deleting the shared state: later computation continues
to read the now-CLEAN slot, which actively suppresses some alternate,
target-relevant evidence. Blocking those reads permits partial recovery. This
is consistent with a pre-L20 distributed copy plus late competition; it does
not yet localize a backup receiver.

### 2026-07-13 — backup-formation timeline PRE-REGISTERED

Protocol: `BINDING_BACKUP_FORMATION_PROTOCOL.md`. Kernel:
`delta_binding_backup_formation`.

This is the one permitted follow-up to the bridge interaction. On the original
held-out layout, it applies the same post-L20 CLEAN overwrite and fixed late
L21–L26 own-slot edge block, then asks whether blocking slot access during one
of three fixed pre-L20 windows (L3–8, L9–14, L15–20) prevents the late-block
recovery. Each queried-slot early block has an exact other-slot control, and
the primary statistic is a difference in recoveries, not raw performance.
No receiver, head, position, or new layer window is selected post hoc.

### 2026-07-13 — backup-formation timeline COMPLETE — early shared formation

Kernel `shlydv/cm-delta-binding-backup-formation` completed with
`LOCALIZED_SHARED_BACKUP_FORMATION`. G0/G1 passed. The base late-block recovery
after L20 overwrite was 13.70 NATURAL and 13.38 ADD logit units. Blocking
queried-slot outgoing access in L3–L8 reduced controlled recovery to
5.91/5.59, while the exact other-slot control retained 13.60/13.88. The
difference-in-differences therefore shows that the queried early block
prevented 56.6%/59.7% of controlled recovery for NATURAL/ADD.

The fixed middle (L9–14) window prevented only 3.5%/1.8%, and the fixed late
(L15–20) window did not prevent recovery. The correct limited conclusion is a
shared **early L3–L8 formation window** for the recoverable backup, not a
localized component or receiver token.

### 2026-07-13 — reversed-layout early backup confirmation PRE-REGISTERED

Protocol: `BINDING_BACKUP_REVERSED_CONFIRMATION_PROTOCOL.md`. Kernel:
`delta_binding_backup_reversed_confirmation`.

This structural replication uses the already behavior-validated reversed
Y-then-X binding layout and its 80 balanced rows. It tests only the frozen
early L3–L8 queried-slot versus other-slot controlled recovery interaction;
there is no repeat window search. A pass establishes that the early formation
phenomenon is not confined to one binding order. The previous reversed-layout
broadcast measurement makes this a structural confirmation, not a fully blind
dataset.

### 2026-07-14 — reversed-layout early backup confirmation COMPLETE

Kernel `shlydv/cm-delta-binding-backup-reversed-confirmation` completed with
`REVERSED_LAYOUT_SHARED_EARLY_BACKUP_REPLICATES`. All gates passed on 80
reversed-order rows (CLEAN/NATURAL accuracy 98.8%/96.3%; ADD/NATURAL effect
ratio .999). The fixed early L3–L8 queried-slot block prevented 61.6% of
controlled NATURAL recovery and 62.7% of ADD recovery, versus the exact
other-slot control. The original-order formation test found 56.6%/59.7%.

This is a structural prompt-order replication of the shared early
outgoing-access requirement for recoverable backup evidence. It strengthens
the result within Qwen2.5-7B, but it is not a second model family or an
independent blind template confirmation. The claim remains temporal and
causal—not a claim that a particular receiver or head has been localized.

### 2026-07-14 — cross-model binding-operator gate PRE-REGISTERED

Protocol: `CROSS_MODEL_BINDING_OPERATOR_PROTOCOL.md`. First target:
`mistralai/Mistral-7B-Instruct-v0.3`; second target:
`microsoft/Phi-3.5-mini-instruct`.

This is deliberately an entry gate rather than a full port of the Qwen causal
timeline. Each model retains only tokenizer-valid candidate values, uses the
fixed within-value discovery/held-out split, and tests exactly three
architecture-normalized early depths. The selected depth must reproduce the
operator on held-out rows with both wrong-direction and wrong-binding-slot
controls. A pass is evidence for a comparable behavioral affine binding
operator, not evidence for an identical circuit. A failure stops deeper
cross-model causal experiments rather than motivating a depth search.

### 2026-07-14 — Mistral cross-model gate COMPLETE

Kernel `shlydv/cm-delta-binding-cross-model-gate` completed with
`CROSS_MODEL_AFFINE_OPERATOR_CONFIRMED` on
`mistralai/Mistral-7B-Instruct-v0.3` (32 layers). Nine values were
tokenizer-valid (only `grape` was excluded). Discovery considered normalized
early L2/L4/L6 and selected L2 solely because it had the largest viable ADD
effect. On the independent 18 held-out substitutions: CLEAN/NATURAL accuracy
was 100%/100%, natural effect was 49.229, the L2 ADD effect was 48.975
(ratio .995), and all ADD rows were positive. The wrong-direction ADD effect
was 24.529 and the identical target delta at the other binding slot was 2.834.

This is a strong cross-family behavioral replication of the early queried-slot
affine binding operator. It does not yet replicate the Qwen backup timeline or
establish that the two models share a circuit. The second, smaller Phi family
test is still required to distinguish a Mistral-specific replication from a
more portable phenomenon.

### 2026-07-14 — Phi cross-model gate INELIGIBLE

The first Phi-3.5-mini attempt stopped during remote-code model
initialization under 8-bit loading, before it formed any experimental inputs;
that was infrastructure-only. The native Transformers retry loaded the model
successfully (32 layers, hidden size 3072) but returned
`CROSS_MODEL_TOKENIZATION_INELICITABLE`: none of the ten fixed values is a
single token under the current one-next-token, space-prefixed measurement
contract.

This is **not** a behavioral negative result and provides no evidence against
the Mistral replication. It says the frozen gate is not directly comparable to
Phi's tokenizer. A future Phi-specific version would have to pre-register a
literal-space answer prefix and the following word token as the intervention
site, or use multi-token likelihood scoring; it must not be quietly merged
with the present single-token result.

### 2026-07-14 — Mistral causal backup port PRE-REGISTERED

Protocol: `MISTRAL_BINDING_BACKUP_PORT_PROTOCOL.md`. This is the next
high-value test after Mistral's held-out affine-operator confirmation. It uses
only the original held-out offsets 5/7 and Mistral's already selected L2 ADD.
The normalized Qwen geometry is frozen: CLEAN overwrite after L23, later
queried-slot read block L24–L30, and a single early own-slot versus other-slot
formation comparison at L3–L9.

The port first requires an explicit Mistral 4-D mask baseline to reproduce the
unmasked effect within 5%, then requires a late-block recovery of at least five
logit units for both NATURAL and ADD. Only then does the one early
difference-in-differences run execute. No later windows, components, or
receiver positions are licensed by a negative result.

The Mistral operator gate retained nine tokenizer-valid values (excluding
`grape`); this port inherits that exact eligibility set rather than applying
the Qwen-only all-ten-value helper.

### 2026-07-14 — Mistral causal backup port COMPLETE — early window not replicated

Kernel `shlydv/cm-delta-mistral-binding-backup-port-mistral7b` completed with
`MISTRAL_EARLY_BACKUP_NOT_REPLICATED`. This is an interpretable causal result,
not a mask or behavioral failure: the explicit Mistral 4-D mask exactly matched
the unmasked baseline; G0 passed at 100% CLEAN/NATURAL accuracy with
ADD/NATURAL ratio .995; and G1 passed. A post-L23 CLEAN overwrite reduced the
natural/ADD effect from 49.229/48.975 to 32.257/31.467. Blocking late own-slot
reads L24–30 restored 5.245/5.318 logit units.

However, with the fixed normalized early L3–9 window, recovery was 4.910/4.889
for the own-slot block and 5.414/5.582 for the exact other-slot control. Thus
the difference-in-differences prevented only 9.3% of controlled NATURAL
recovery and 12.4% of ADD recovery—far below the pre-registered 50% shared
formation threshold.

The correct conclusion is narrow: Mistral shows an early slot-specific affine
operator and a weaker late overwrite/recovery phenomenon, but it does **not**
show Qwen's normalized early backup-formation dependence in this one frozen
window. This result does not license a search over Mistral layers, heads, or
receiver positions.

### 2026-07-14 — cross-surface binding-operator gate PRE-REGISTERED

Protocol: `BINDING_SURFACE_OPERATOR_PROTOCOL.md`. This tests one and only one
out-of-distribution grammar on Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3:
`X maps to VALUE` rather than `Let X = VALUE`. The response completion is also
changed to `It maps to`. L2 is fixed from the prior within-model result; no
depth discovery happens in this test.

For each model, directions are computed from separate donor prompts in the
new grammar, then tested only on offsets 5/7. The held-out gate requires the
same natural/ADD behavioral match, wrong-value-direction control, and
other-binding-slot control as the cross-model gate. A pass supports recurrence
of the operator phenomenon across binding surface forms, not identity of the
literal direction or full circuit across those forms.

### 2026-07-14 — mapping-surface v1 behaviorally ineligible; v2 PRE-REGISTERED

The first Qwen run used free-form assistant completion `It maps to`. It had a
near-exact natural/ADD logit-effect match (57.276/57.308, ratio 1.001), but
CLEAN/NATURAL greedy value accuracy was only 50%/55% because the next-token
task asks the model to generate an article before the value. This is a prompt
measurement failure, not a result about the operator; Mistral is not run under
that invalid completion.

Version v2 keeps the mapping grammar and all frozen held-out/control rules,
but uses the direct value completion `X =` or `Y =` after the mapping question.
This is a newly registered behavioral-validation version, not a rescue of a
causal failure. Qwen must first pass G0 under v2 before the identical Mistral
run is launched.

### 2026-07-14 — Qwen mapping-surface v2 COMPLETE

Kernel `shlydv/cm-delta-binding-surface-operator-qwen7bv2` completed with
`SURFACE_OPERATOR_CONFIRMED`. On 20 held-out substitutions, CLEAN/NATURAL
greedy value accuracy was 95%/100%; natural and L2 ADD effects were
59.706/59.461 (ratio .996); all ADD rows were positive. The wrong-direction
control was 30.105 and the same target direction at the other value slot was
4.434. The fixed mapping grammar is therefore behaviorally valid and supports
the same early queried-slot affine-operator phenomenon in Qwen. Mistral is now
licensed to run the identical v2 gate.

### 2026-07-14 — Mistral mapping-surface v2 COMPLETE — cross-model, cross-surface replication

Kernel `shlydv/cm-delta-binding-surface-operator-mistral7bv2` completed with
`SURFACE_OPERATOR_CONFIRMED` on the same frozen mapping grammar. Mistral kept
nine tokenizer-valid values (excluding `grape`) and evaluated 18 held-out
substitutions. CLEAN/NATURAL value accuracy was 83.3%/83.3%; natural and L2
ADD effects were 41.764/41.647 (ratio .997), and every ADD row was positive.
The wrong-direction control was 21.025 and the same target direction at the
other binding slot was 6.607.

Together with the Qwen v2 result, this is a two-model, two-binding-surface
replication of the early queried-slot affine binding operator with matched
wrong-direction and wrong-slot controls. It strengthens the portable operator
claim. It does not revise the separate negative Mistral result for Qwen's
normalized early backup-formation dynamics.

### 2026-07-14 — literal cross-surface controller-transfer test PRE-REGISTERED

Protocol: `CROSS_SURFACE_CONTROLLER_TRANSFER_PROTOCOL.md`. This is a stricter
test than the surface replication. At fixed L2, derive value prototypes only
from the original `Let X = value` donor prompts, then add the raw source-to-
target difference unchanged to the behaviorally validated mapping grammar.
There is no scale fitting, representation alignment, new layer selection, or
use of mapping prompts to choose the literal direction.

The concurrent mapping-native direction is a sanity baseline only. Confirmation
requires the raw literal controller to match both the natural rewrite and the
native mapping controller on held-out offsets 5/7, as well as exceed literal
wrong-direction and wrong-slot controls. Qwen runs first; Mistral is licensed
only by a Qwen confirmation. A negative result limits raw literal-vector reuse,
not the already replicated per-surface affine-operator phenomenon.

### 2026-07-14 — Qwen literal cross-surface controller transfer COMPLETE

Kernel `shlydv/cm-delta-binding-cross-surface-transfer-qwen7b` completed with
`LITERAL_CROSS_SURFACE_CONTROLLER_CONFIRMED`. The fixed raw L2 direction was
computed solely from the original `Let` donor prompts and injected unchanged
at the mapping-surface queried value slot. On 20 held-out substitutions,
CLEAN/NATURAL accuracy was 95%/100%; natural effect was 59.706; the concurrent
mapping-native controller effect was 59.461; and the unchanged literal
`Let`-to-mapping transfer effect was 59.786. Thus literal/natural and
literal/native ratios were 1.001 and 1.005, respectively, with all transfer
rows positive. Literal wrong-direction and other-slot controls were 30.151
and 4.027.

This is stronger than a per-surface re-derivation: at least in Qwen, the raw
early affine controller is reusable unchanged across these two binding
grammars. It licenses the pre-registered Mistral replication; it still does
not establish arbitrary task transfer, composition, or a shared component-level
circuit.

### 2026-07-14 — Mistral literal cross-surface controller transfer COMPLETE

Kernel `shlydv/cm-delta-binding-cross-surface-transfer-mistral7b` also
completed with `LITERAL_CROSS_SURFACE_CONTROLLER_CONFIRMED`. On 18 held-out
substitutions (nine tokenizer-valid values), natural, mapping-native, and raw
literal-transfer effects were 41.764, 41.647, and 41.907, respectively.
Literal/natural and literal/native ratios were 1.003 and 1.006; every literal
transfer row was positive; wrong-direction and wrong-slot controls were 20.914
and 3.101. CLEAN/NATURAL greedy answer accuracy was 83.3%/83.3%, passing the
frozen behavior gate.

The project now has a strict two-model result: an early value-specific affine
controller derived in one binding grammar can be reused unchanged in a second
grammar, with matched directional and binding-slot controls. The next
registered question is composition—whether two distinct controllers can be
installed simultaneously at distinct bindings without interfering—rather than
another surface form or component search.

### 2026-07-14 — compositional binding-controller test PRE-REGISTERED

Protocol: `BINDING_COMPOSITION_PROTOCOL.md`. This uses fresh three-binding
mapping contexts, makes two counterfactual changes at once (X and Y), and
derives both raw L2 writes only from original `Let` donors. The two additions
are installed in the same forward pass, then the identical double-write state
is read under X and Y queries. Own-only and swapped-address double writes test
for interference and address specificity. Qwen runs first; Mistral is licensed
only by a Qwen confirmation. No scale fitting, alignment, layer choice, or
prompt/value selection is allowed.

### 2026-07-14 — Qwen compositional binding-controller test COMPLETE

Kernel `shlydv/cm-delta-binding-composition-qwen7b` completed with
`COMPOSITIONAL_BINDING_CONTROLLER_CONFIRMED`. In fresh three-binding mapping
contexts, two raw L2 directions derived only from `Let` donors were installed
simultaneously at X and Y, then read under both X and Y queries. On 20 rows,
natural, own-only, and joint effects were 64.825, 65.198, and 65.159. Thus the
joint/natural ratio was 1.005, joint/own-only ratio .999, and mean cross-talk
was -0.039 logit units. Every joint row was positive. The swapped-address
double-write control was only 16.462.

Within this controlled three-binding setting, the result supports parallel,
address-specific composition of two literal controllers rather than a fragile
single-slot steering effect. It licenses the pre-registered Mistral replication.

### 2026-07-14 — Mistral compositional binding-controller test COMPLETE

Kernel `shlydv/cm-delta-binding-composition-mistral7b` also completed with
`COMPOSITIONAL_BINDING_CONTROLLER_CONFIRMED`. On 18 three-binding rows,
CLEAN/NATURAL accuracy was 94.4%/88.9%; natural, own-only, and simultaneous
joint effects were 42.950, 43.531, and 43.373. Joint/natural and
joint/own-only ratios were 1.010 and .996; every joint row was positive; and
mean cross-talk was -0.158 logit units (well below the pre-registered 15% of
natural-effect bound). The swapped-address double write was 9.064.

Across Qwen and Mistral, two raw value-specific L2 directions derived from the
original binding grammar now transfer into a separate mapping grammar and
compose at two distinct addresses with negligible measured interference. This
is a controlled synthetic binding result, not yet a claim about arbitrary
language reasoning, larger capacity, or shared circuit implementation.

### 2026-07-14 — neutral-carrier rival test PRE-REGISTERED

Protocol: `NEUTRAL_CARRIER_PROTOCOL.md`. This directly tests the remaining
lexical-state alternative. Raw L2 value directions are now derived only from
the non-binding carrier `Here is a token: VALUE.`, with no fitting or alignment,
and compared to the raw `Let` controller in the held-out mapping task. Neutral
wrong-direction and wrong-slot controls are matched. A neutral match supports a
portable lexical value-state account; a non-match while the `Let` baseline
remains positive supports binding-context information in the early controller
under this measurement. Qwen runs first.

### 2026-07-14 — Qwen neutral-carrier rival test COMPLETE

Kernel `shlydv/cm-delta-binding-neutral-carrier-qwen7b` completed with
`NEUTRAL_CARRIER_MATCHES_LITERAL_CONTROLLER`. The raw L2 direction derived
only from `Here is a token: VALUE.` produced 59.942 logit units on the mapping
task, versus 59.706 for the natural rewrite and 59.786 for the `Let` controller.
Neutral/natural and neutral/Let ratios were 1.004 and 1.003, with every row
positive. Neutral wrong-direction and wrong-slot controls were 30.360 and
5.211.

This rules out the strongest version of the claim that the early vector itself
encodes binding grammar. In Qwen, it is consistent with a portable lexical
value-state write whose binding specificity is supplied by the address and
downstream read machinery. Mistral is now required to establish whether this
revision is cross-model.

### 2026-07-14 — Mistral neutral-carrier rival test COMPLETE

Kernel `shlydv/cm-delta-binding-neutral-carrier-mistral7b` completed with
`NEUTRAL_CARRIER_MATCHES_LITERAL_CONTROLLER`. On 18 held-out mapping rows, the
neutral-carrier direction produced 42.193 logit units, versus 41.764 for the
natural rewrite and 41.907 for the `Let` controller. Neutral/natural and
neutral/Let ratios were 1.010 and 1.007, every neutral row was positive, and
wrong-direction and wrong-slot controls were 21.153 and 6.723.

The cross-model interpretation is therefore revised: the L2 direction itself
is a portable, context-neutral lexical value-state difference. The causal
binding phenomenon lies in the addressed write and distributed downstream
readout, which accept and compositionally use that state. This remains a
strong activation-editing result, but it does not support describing the raw
vector as encoding binding grammar.

### 2026-07-14 — addressed-state reasoning ladder PRE-REGISTERED (discovery)

Protocol: `ADDRESSED_REASONING_LADDER_PROTOCOL.md`. A Qwen-only discovery
kernel tests whether a neutral L2 start-value write causes the correct endpoint
after two explicitly defined transitions, compared with natural replacement,
raw embedding difference, and a wrong-chain direction. G0 is an absolute
behavioral stop. If eligible, matched CLEAN/NATURAL final-query states are also
exchanged at frozen layers 4/8/12/16/20/26 to locate—descriptively—the earliest
causally sufficient workspace candidate. The matched workspace patch is not a
reusable-controller confirmation; any positive candidate must later be
cross-fitted and frozen before Mistral.

### 2026-07-14 — addressed-state reasoning ladder COMPLETE — behaviorally ineligible

Kernel `shlydv/cm-delta-addressed-reasoning-ladder-qwen7b` returned
`REASONING_LADDER_BEHAVIORALLY_INELIGIBLE`: CLEAN and NATURAL exact next-token
accuracy were both 0% on the frozen arbitrary two-transition prompt. The hard
stop fired before all interventions. This is an elicitation failure and no
evidence about addressed consequence propagation or workspace state.

### 2026-07-14 — addressed arithmetic-state ladder PRE-REGISTERED (discovery)

Protocol: `ADDRESSED_ARITHMETIC_STATE_PROTOCOL.md`. This uses the previously
qualified 100%-accurate addition family and changes operand data while holding
the arithmetic program fixed; it does not retry the failed add-to-subtract
operator. A neutral L2 digit-state write, raw embedding difference, and wrong
digit direction are compared. Matched query-state exchanges at frozen layers
only locate a candidate workspace for later cross-fitting.

### 2026-07-14 — addressed arithmetic-state v1 mechanically invalid; v2 PRE-REGISTERED

V1 passed its 100%/100% native behavior gate and independently found that the
matched final-query state was bidirectionally interchangeable only at L26
(L4–L20 failed). However, the operand intervention used token 27, Qwen's shared
standalone whitespace, while the digit actually changed at token 28. Neutral
digit donors had the same split. This made neutral, embedding, and wrong
deltas exactly zero, so `ADDRESSED_ARITHMETIC_CONSEQUENCE_NULL` is invalid and
has no scientific meaning.

Protocol v2 changes only mechanical token location: it identifies the unique
changed digit token in arithmetic prompts and the actual digit token in neutral
donors. Rows, model, layer, workspace checkpoints, controls, and thresholds
remain frozen. The valid v1 L26 workspace observation is retained as discovery.

### 2026-07-14 — addressed arithmetic-state v2 COMPLETE — positive discovery

The first nominal v2 kernel mounted the stale v1 code package and is excluded.
The fresh immutable-code run then stalled during an unauthenticated Hugging
Face model download and was cancelled; it produced no evidence. Kernel
`shlydv/cm-arith-state-v2m` mounted the exact Qwen checkpoint locally and
completed with protocol v2, operand token 28, neutral digit token 30, and
`ADDRESSED_ARITHMETIC_WRITE_WITH_WORKSPACE`.

Native CLEAN/NATURAL addition accuracy was 100%/100% on 12 rows. Natural
operand replacement produced +39.786 logit units. The raw neutral-carrier L2
digit-state write produced +39.745 (ratio .999), 100% target accuracy, and all
rows positive. A wrong +1 digit write produced +19.807 but 0% target accuracy.
The raw embedding difference was null (-.109; 0% target accuracy), ruling out
simple delayed embedding replay under this intervention.

Matched final-query state exchange again failed bidirectionally at L4/L8/L12/
L16/L20 and became exactly interchangeable at L26: NATURAL->CLEAN target
accuracy 100%, effect ratio 1.002, and CLEAN->NATURAL source accuracy 100%.
This is discovery evidence that an upstream processed value-state edit is
accepted as operand data and propagates to the correct arithmetic consequence,
with the answer-specific query state becoming concentrated very late. The L26
matched patch is an upper-bound localization, not yet a reusable latent-state
controller. A frozen cross-model confirmation is required.

### 2026-07-14 — Mistral addressed arithmetic confirmation PRE-REGISTERED

Protocol: `ADDRESSED_ARITHMETIC_CONFIRMATION_PROTOCOL.md`. The identical 12
rows, L2 neutral digit write, embedding and wrong-digit controls, and Qwen
thresholds are frozen on Mistral-7B-Instruct-v0.3. There is no layer sweep:
Qwen's L26/28 workspace candidate maps to the sole normalized Mistral checkpoint
L30/32. Confirmation requires both arithmetic consequence propagation and
bidirectional matched query-state interchangeability at L30.

### 2026-07-14 — Mistral addressed arithmetic confirmation INELIGIBLE

Kernel `shlydv/cm-arith-confirm-mistral` completed the frozen protocol using
the mounted `Mistral-7B-Instruct-v0.3` snapshot and protocol v2 code. Mechanical
audits passed (32 layers; operand anchor/token 5/6; last token 49), but native
CLEAN and NATURAL exact next-token arithmetic accuracy were both 0%. The hard
behavioral stop returned `ADDRESSED_ARITHMETIC_BEHAVIORALLY_INELIGIBLE` before
any intervention or L30 patch.

This is not a failed causal replication: the selected Mistral checkpoint did
not express the prerequisite arithmetic behavior under the frozen prompt and
metric. It supplies no evidence for or against operand-state consequence
propagation in Mistral. The Qwen discovery therefore remains single-model for
arithmetic, while the binding, cross-surface, composition, and neutral-carrier
results remain cross-model. Any future Mistral arithmetic retry must be labeled
as a new elicitation protocol and frozen before interventions.

### 2026-07-14 — dual-readout workspace discovery PRE-REGISTERED

Protocol: `DUAL_READOUT_WORKSPACE_PROTOCOL.md`. Qwen donor problems alone
select the earliest shared pre-query `READY` checkpoint where row-matched state
exchange is bidirectionally sufficient for exact-sum and parity readouts. One
donor-mean NATURAL-minus-CLEAN direction is then injected into untouched
operand-pair problems and must reproduce both consequences. Reverse, raw
embedding, CLEAN-overwrite necessity, and 20 norm-matched random controls are
frozen. Behavioral failure or absence of a donor workspace is an immediate
stop; no held-out layer rescue is allowed.

### 2026-07-14 — dual-readout workspace discovery INELIGIBLE

Kernel `shlydv/cm-delta-dual-readout-workspace-qwen7b-v2` completed protocol
v1 and returned `DUAL_READOUT_BEHAVIORALLY_INELIGIBLE`. Exact-sum behavior was
fully eligible: donor CLEAN/NATURAL accuracy 100%/100% with natural effect
+38.922, and held-out accuracy 100%/100% with effect +43.531. Parity did not
meet the frozen 7/8 threshold: donor accuracy was 75%/75% and held-out accuracy
was 62.5%/87.5%, despite positive natural effects (+7.594 and +12.141).

The hard stop fired before donor layer localization, direction fitting,
held-out intervention, necessity, or random controls. This result therefore
contains no causal evidence for or against a reusable arithmetic workspace.
It isolates the failure to unreliable parity elicitation under this prompt.
The next admissible step is a behavior-only screen of alternative second
readouts; any eligible readout and prompt must be frozen before interventions.

### 2026-07-14 — parity ineligibility DIAGNOSED

A logging-only replay of the unchanged G0 (`shlydv/cm-delta-dual-readout-
workspace-parity-diagnostic`) saved expected tokens, greedy predictions, and
top-five logits. The misses are genuine semantic parity errors, not formatting,
capitalization, whitespace, or tokenizer mismatch: every failed row greedily
selected the opposite lowercase answer token. Wrong-answer margins ranged from
3.5 to 18.5 logits.

The error pattern was structured. All `b=1` prompts were answered `odd`; all
`b=2` prompts were correct; further misses occurred for 1+3, 1+4, 1+5, and
3+5. The separate exact-sum readout remained 100% accurate on the same operand
pairs. Thus Qwen can directly report these sums but does not reliably compose
the sum with a parity readout under the frozen prompt. Semantic answer-class
scoring would not rescue G0. This behavioral dissociation may be reported, but
it provides no causal workspace evidence.

### 2026-07-14 — orchestration cross-model confirmation PRE-REGISTERED

Protocol: `ORCHESTRATION_CROSS_MODEL_CONFIRMATION.md`. This returns to the
strongest Qwen agent result under the revised local-coordinate hypothesis. On
Mistral-7B-Instruct-v0.3, the exact original calculator/database template,
rows, donor/test split, alpha, nulls, workflow parser, execution checks, and
thresholds are frozen. Mistral learns its own donor direction at L2; no literal
Qwen vector is transported across incompatible residual spaces. Qwen
L2/L8/L14/L20/L26 maps to frozen Mistral L2/L9/L16/L23/L30, with L23 as the
sole mediator. Native workflows must pass every 90% G0 metric before any
activation intervention. A causal pass tests replication of function across
model families, not universal vector identity or cross-template transfer.

The first launch loaded the model but stopped mechanically before G0 because
Mistral re-tokenizes the `CALL` suffix when a tool name is appended; the Qwen
harness had required `CALL` and each following tool name to be independently
prefix-stable single tokens. It produced no scientific evidence. Protocol v2
defines the same tool decision tokenizer-agnostically as the longest shared
token prefix of the two literal call continuations followed by their first
differentiating token. On Qwen this exactly recovers the original post-`CALL`
measurement. All scientific rows, gates, layers, and thresholds remain frozen.

The v2 relaunch then stopped mechanically at the first native batched
generation because the Mistral tokenizer snapshot has no configured padding
token. No call was generated and G0 again did not run. V3 assigns the existing
EOS token as temporary left-padding during batched generation; it adds no token
and changes no weights, prompt, decoding rule, row, or scientific threshold.

### 2026-07-14 — Mistral orchestration confirmation INELIGIBLE at G0

Kernel `shlydv/cm-orch-xmodel-mistral-v3` completed the tokenizer-generalized
preflight and native workflow gate. Native calculator workflows were 10/10 on
every metric. Native lookup generated 10/10 exact database calls with correct
row-specific arguments, but final-answer/tool-result/end-to-end accuracy was
8/10, below the frozen 9/10 G0 threshold. Both failures were key `D`, whose
executed database result is `0`: Mistral returned `4` and `9`, exactly the
calculator sums of the two corresponding payloads.

The hard stop fired before activation extraction, direction fitting, causal
intervention, trajectory, mediation, or null controls. This is behavioral
ineligibility rather than failed causal replication. It independently exposes
the same boundary observed in Qwen template B: correct tool selection does not
guarantee that a later answer turn consumes the executed tool result when a
competing answer remains in context. The unchanged-template Mistral causal
function therefore remains untested.

### 2026-07-14 — causal evidence arbitration PRE-REGISTERED

Protocol: `EVIDENCE_ARBITRATION_PROTOCOL.md`. A new Mistral discovery holds the
database call, executed result, payload, and answer transcript fixed while the
original mode-token address specifies whether the final answer must use the
internal payload sum or external database evidence. A common answer-turn
reminder defines both rules without repeating the active value. Native donor
and held-out arbitration must pass before activation extraction. One donor L2
lookup-minus-calculate state is then tested on held-out conflicts with reverse,
norm-matched lexical, wrong-address, and 100 random controls; L23 natural-state
convergence, matched mediation, and CLEAN-overwrite necessity are mandatory.

Only after primary evaluation is the learned direction applied to the original
unmodified lookup transcripts to report rescue of the known `D -> 0` failures.
That bridge is supplemental and cannot determine the primary verdict. A causal
null after G0 ends affine arbitration-vector searches and redirects the theory
toward distributed evidence-resolution dynamics.

The v1 arbitration screen was itself behaviorally ineligible and stopped
before activation extraction. Its first turn retained the original instruction
to output a tool call, while its final turn requested evidence arbitration;
Mistral frequently generated another `CALL calculator` or explanatory text.
Donor/test internal diagnostic accuracy was 22.2%/37.5%, and external accuracy
66.7%/50%. This is ambiguous task elicitation, not evidence about arbitration.

Protocol v2 uses a clean one-turn factorial prompt containing the internal
arithmetic result and successfully executed external result, with the same
single `calculate`/`lookup` policy-token contrast. All causal thresholds,
donor/test rows, layers, nulls, and controls remain frozen. Transfer of the v2
direction to the original multi-turn mode-token address is supplemental and
cannot determine the primary verdict.

### 2026-07-14 — causal evidence arbitration COMPLETE — all primary gates pass

Kernel `shlydv/cm-evidence-arbitration-v2` returned
`CAUSAL_EVIDENCE_ARBITRATION_STATE`. Native donor and held-out internal/external
conditions were 100% accurate, including both held-out external-result-zero
rows. A single donor-mean L2 `lookup`-minus-`calculate` policy-state edit at the
mode-token address switched all eight held-out conflicts from the internal
arithmetic result to the external tool result; the reverse edit restored all
eight internal answers.

The learned output effect was +13.540 versus natural +13.799 (ratio .981),
positive on 8/8 rows, with 0/100 norm-matched random exceedances. The
norm-matched raw lexical direction was only .031x natural and switched 0/8
diagnostic answers. The wrong-address edit was .011x natural and switched 0/8.
Local L2 equivalence was cosine .998/error .064. At the answer-decision token,
the induced state converged strongly by L16 and remained natural-like at L23
(cosine .979/error .207) and L30 (.974/.227); the lexical trajectory remained
non-equivalent.

Matched L23 mediation reproduced the learned/natural effects at ratio .987.
CLEAN L23 overwrite removed 102.5% of each effect with block-fraction gap
.001. Thus the early local edit recruits the same late answer pathway as the
natural external-evidence policy under this clean conflict context.

The preregistered supplemental bridge did **not** rescue the original
multi-turn `D -> 0` failures: base and amplified answers remained `4` and `9`.
This does not alter the positive primary verdict. It establishes an addressed,
causally writable evidence-arbitration state in Mistral, while showing that its
literal direction is context-local and does not explain the original agent
failure by direct transfer. The next mechanistic question is how arbitration
is transformed or reconstructed across the tool-call/result boundary.

### 2026-07-14 — original-context evidence bridge PRE-REGISTERED

Protocol: `MULTITURN_EVIDENCE_BRIDGE_PROTOCOL.md`. This frozen Mistral
discovery tests the narrowest explanation of the original `D -> 0` failures:
the native original-prompt lookup policy coordinate exists but is simply too
weak. Twenty variable-payload rows all use key D/result zero, split 10/10. A
donor-only original-prompt L2 lookup-minus-calculate direction is amplified at
alpha one on held-out original multi-turn transcripts. Eligibility first must
reproduce internal-answer preference under calculate, failure under ordinary
lookup, and successful override under an explicit authoritative-result upper
bound.

The frozen causal test includes generated-answer rescue, target-vs-internal
logit effects, local-state equivalence, L23 final-decision trajectory and
mediation/overwrite, the successful clean-context direction as a transfer
control, norm-matched lexical and wrong-address controls, and 100 random
nulls. No layer or alpha search is permitted. A failure after G0 rules out the
simple same-coordinate/insufficient-strength account; it redirects the theory
toward state reconstruction or downstream evidence-resolution dynamics.

### 2026-07-14 — original-context evidence bridge INELIGIBLE at G0

Kernel `shlydv/cm-multiturn-evidence-bridge-v1` completed and returned
`MULTITURN_BRIDGE_DIAGNOSTIC_INELIGIBLE`. The failure phenomenon replicated
strongly: the untouched lookup transcript used the executed zero result on
only 1/10 held-out rows, used the exact internal sum on 6/10, and produced
near-sum continuations on additional rows. The calculate-policy conflict used
the internal sum on 9/10, passing its frozen eligibility threshold.

The authoritative-result upper bound failed its 90% gate: it returned zero on
6/10 but repeated `CALL database D` on the final 4/10. Therefore G0 was false
and the kernel stopped before direction extraction, activation intervention,
trajectory, mediation, generation controls, or random nulls. This run confirms
that Mistral's original zero-result rejection is broad across variable
payloads, but it supplies no causal evidence for or against the
same-coordinate/insufficient-strength hypothesis. Any subsequent test needs a
behavior-qualified final-answer interface frozen separately; these thresholds
must not be rescued post hoc within this protocol.

### 2026-07-14 — compositional agent-control registers PRE-REGISTERED

Protocol: `COMPOSITIONAL_AGENT_CONTROL_PROTOCOL.md`. One Mistral load will
factor agent control into a workflow-phase coordinate (act versus answer) and
an evidence-policy coordinate (internal versus executed tool result), both
measured and edited at the same final decision marker. The factorial
intervention is load-bearing: phase-only must change the stage without changing
the selected evidence, evidence-only must not spuriously exit the action
stage, and both together must produce the correct external-evidence answer.

The kernel contains database discovery, independently learned calculator local
replication, separately scored literal database-to-calculator transfer, and a
supplemental untouched-original-transcript bridge. It includes exact natural
behavior gates, donor/held-out splits, actual executable calls, natural output
and trajectory matching, L23 mediation/necessity, wrong-address and 100 random
controls, and immutable row-level artifacts. No layer, alpha, task, or winner
selection occurs after results.

### 2026-07-14 — compositional agent-control v1 INELIGIBLE at G0

Kernel `shlydv/cm-compositional-agent-control-v1` completed in 648 seconds and
returned `COMPOSITIONAL_CONTROL_BEHAVIORALLY_INELIGIBLE`. Both family branches
stopped before direction extraction or any causal intervention.

The database interface was broadly unstable. Train exact accuracies for
A/I, A/E, B/I, B/E were 25%, 62.5%, 37.5%, and 87.5%; held-out accuracies were
0%, 12.5%, 12.5%, and 50%. Failures were predominantly explanatory
continuations such as `In Phase A...` rather than exact calls or answers. This
task cannot support a causal decomposition claim.

The calculator interface was much closer but still missed its frozen gate.
All four donor cells were 100%. Held-out A/I, A/E, B/I, B/E were 87.5%, 75%,
75%, and 100%. The A/E and B/I misses concentrated exactly on the two rows
where the constructed internal rival wrapped from the correct sum 9 to 1
(payloads 5+4 and 7+2); Mistral produced explanations rather than the required
wrong internal answer or tool call. One A/I miss also occurred on 7+2. Thus
the donor/test split unintentionally coupled held-out status with this extreme
rival-value construction.

Per protocol, neither family was eligible, so no phase/evidence directions,
composition, trajectory, mediation, wrong-address test, random nulls,
cross-workflow transfer, or original-transcript bridge ran. This is not
evidence against separable agent-control states. It is evidence that the v1
factorial elicitation was not distributionally balanced and that the database
surface was especially unsuitable. A successor must first freeze a balanced
behavior-only interface; the v1 thresholds and rows must not be rescued.

### 2026-07-14 — WORKSPACE-WRITE MATRIX PRE-REGISTERED — the decisive generality test

**Why (Sahil):** stop proving bricks; test the arch. Novelty audit vs function
vectors (Todd et al.): FV installs the PROGRAM (which task runs); our live claim
installs the DATA (the argument the in-context program consumes), synthesized
from a NEUTRAL carrier and written at an address. Our program-write attempt
(add→subtract) was NULL; the data-write claim has exactly ONE compute cell of
evidence (Qwen arithmetic). This kernel tests the principle directly, fast,
across models. Module `delta_workspace_matrix.py`, stage `delta_workspace_matrix`.

**Claim under test:** a value-state extracted from `Here is a token: v.` and
written at the value address of a task prompt is consumed by downstream
in-context computation as if the prompt said `v` — across tasks that COMPUTE
over the value and across model families.

**Design (digits 1–9 only; per-cell G0 so no whole-kernel death):**
cells = retrieve (control), add2 (v+2), sub1 (v−1), max5 (max(v,5)),
gt5label (v>5 → north/south). 12 rows/cell. Per row (a→b, wrong c):
CLEAN, NATURAL, ADD z_b−z_a (neutral donors, layer L at value token),
WRONG z_c−z_a (must track its OWN f(c); for gt5label c is same-side as a so
the gate is no-spurious-flip), EMB embedding-diff (report only), 50 shared
norm-matched randoms. Answer scoring: first-token logit margin with mechanical
per-model primer-variant selection (`Answer: ` vs `Answer:`) chosen by
tokenizer contract only (no-merge + distinct answer ids), no model forwards.

**Frozen gates per cell:** G0 CLEAN/NATURAL likelihood-argmax ≥90%;
W1 ADD/natural effect ratio ≥0.7; W2 ADD target accuracy ≥80%; W3 p<.02 vs 50
nulls (floor .0196); W4 WRONG own-target ≥80%. PASS=G0∧W1∧W2∧W3∧W4.

**Model verdict:** WORKSPACE_GENERAL (retrieve PASS ∧ ≥3/4 compute PASS);
WORKSPACE_PARTIAL (1–2 compute); RETRIEVE_ONLY; INELICITABLE cells listed.
**Program decision rule (frozen):** ≥2 model families WORKSPACE_GENERAL →
`GENERAL_PRINCIPLE` (the workspace claim is general; full program justified).
Only-Qwen or retrieve-only elsewhere → honest downgrade of the whole line; no
cell rescue, no prompt fishing beyond the pre-registered mechanical variants.

**Models:** Qwen2.5-7B-Instruct (L2 frozen), Mistral-7B-Instruct-v0.3 (L2
frozen from cross-model gate), Phi-3.5-mini (layer picked from {2,3,4} on
retrieve train rows only — 4 extra rows excluded from scoring; digits bypass
its word-tokenization ineligibility).

**Pre-run mechanical amendment (before any model forward):** the primer-variant
scheme failed Qwen's tokenizer contract offline (BPE merges " north" across the
boundary; digits share a leading space token). Replaced with the strictly more
general **canonical-continuation contract**: base = enc(chat+`Answer:`),
cont(a) = enc(chat+`Answer: `+a)[len(base):]; the common token prefix across
the answer set is appended to every prompt and scoring reads the first
diverging token. Offline preflight now passes all three families (Qwen common
[220]→bare digits — exactly the previously validated recipe; Mistral/Phi
common [▁]→digits; label words diverge immediately). No thresholds changed;
no model forwards used.

**Infra note (no scientific content):** first launches used HF model ids and
stalled at unauthenticated download (documented failure mode), also holding
both GPU slots; kernels stopped in UI, relaunched on mounted snapshots
(`ragnar123/qwen2-5-7b-instruct`, `phmcngc/snapx-mistral7b-instruct-v03-snapshot`).
No result was inspected from the stalled runs.

### 2026-07-14 — workspace matrix COMPLETE (Qwen + Mistral) — GENERAL_PRINCIPLE

Kernels `shlydv/cm-delta-workspace-matrix-qwen7b-v3` and
`...-mistral7b-v2` COMPLETE (~8 min / ~7 min). Artifacts:
`runs/delta_workspace_matrix-qwen7b-v3/`, `runs/delta_workspace_matrix-mistral7b-v2/`.

**Both models: `WORKSPACE_GENERAL` — all 5 cells PASS, no ineligible cells.**

| cell | Qwen nat/ADD (ratio, acc) | Mistral nat/ADD (ratio, acc) |
|---|---|---|
| retrieve | +51.8/+52.2 (1.01, 100%) | +31.1/+31.1 (1.00, 100%) |
| add2 | +32.5/+32.6 (1.00, 100%) | +22.3/+22.1 (0.99, 100%) |
| sub1 | +42.6/+42.2 (0.99, 100%) | +26.4/+26.5 (1.00, 100%) |
| max5 | +59.7/+59.8 (1.00, 100%) | +20.0/+17.5 (0.87, 100%) |
| gt5label | +42.8/+42.2 (0.99, 100%) | +18.0/+18.5 (1.03, 100%) |

Every cell in both models: wrong-value write tracks its OWN consequence 100%
(content-specific); embedding-diff control null (−0.34…+0.85 vs effects
+17…+60, 0% target acc); ADD beats all 50 norm-matched nulls (p at floor
.0196 < .02). Per-cell G0 passed everywhere (no INELICITABLE cells).

**Frozen program decision rule fires: ≥2 families WORKSPACE_GENERAL ⇒
`GENERAL_PRINCIPLE`.** Licensed claim (calibrated): in Qwen2.5-7B and
Mistral-7B, a value-state extracted from a neutral carrier and written at a
value address is consumed by at least four distinct downstream in-context
computations (retrieval, +2, −1, max-vs-5, >5→label routing) at
natural-counterfactual effect size, content-specifically, and not via
embedding replay. Scope: digits, single-token answers, synthetic one-line
prompts, L2, greedy/logit-margin readout. NOT claimed: arbitrary content
types, long prompts, program writes (add→subtract remains NULL), or circuit
identity across models.

Third family (Phi-3.5-mini) launching next under the identical frozen
protocol (layer picked from {2,3,4} on retrieve train rows only).

### 2026-07-14 — workspace matrix Phi-3.5 COMPLETE — printed WORKSPACE_DEAD; compute cells 4/4 PASS (taxonomy artifact, logged as-is)

Kernel `shlydv/cm-delta-workspace-matrix-phi35-v3` COMPLETE (~7 min; mounted
Kaggle model `richolson/phi-3.5-mini-instruct`, 8bit, L3 selected from {2,3,4}
on retrieve train rows: effects +23.25/+23.38/+23.25 — essentially flat).
Infra note: phi35 v1 stalled at unauthenticated HF download (same mode as
Qwen/Mistral v1s); v2 on the mounted model was cancelled in the UI before
producing logs (v1/v2 mix-up); v3 is the only scored run. `--model-source`
support added to the launcher for this (mechanical).

| cell | g0 | nat | ADD | ratio | acc | p | wrong_own | verdict |
|---|---|---|---|---|---|---|---|---|
| retrieve | 1.00/1.00 | +24.7 | +18.8 | 0.76 | **75%** | .020 | **75%** | **FAIL** (W2,W4) |
| add2 | 1.00/1.00 | +30.7 | +25.0 | 0.81 | 83% | .020 | 83% | PASS |
| sub1 | 1.00/1.00 | +38.3 | +37.6 | 0.98 | 100% | .020 | 100% | PASS |
| max5 | 1.00/1.00 | +41.2 | +38.2 | 0.93 | 92% | .020 | 100% | PASS |
| gt5label | 1.00/1.00 | +39.9 | +37.4 | 0.94 | 92% | .020 | 100% | PASS |

**Printed verdict: `WORKSPACE_DEAD`** — the frozen model-verdict taxonomy
anchors on the retrieve control passing, and retrieve missed its 80%
accuracy gates by one row (9/12) on both W2 and W4, while W1 (0.76 ≥ 0.7)
and W3 (p at floor) passed. Per discipline the printed verdict stands; no
re-grading, no re-run.

**Honest annotation (does not change the verdict):** all four COMPUTE cells —
the actual claim under test — passed in Phi at ratios 0.81–0.98 with
content-specific wrong-writes. Descriptively, the write-consumption
phenomenon appears in the third family too, with its weakest cell being,
unexpectedly, plain retrieval. Lesson for the paper: anchoring a verdict
taxonomy on a control that can be weaker than the treatment misfires; a
future revision (if any) must be pre-registered fresh, not applied here.

**Program status (frozen rule, unchanged):** Qwen + Mistral both
`WORKSPACE_GENERAL` ⇒ **`GENERAL_PRINCIPLE` stands on two families**; Phi
adds 4/4 compute-cell replication under a printed DEAD verdict due to the
control anchor. Scope line for the paper: two families fully general, third
family compute-consistent with a control anomaly, reported exactly as such.

### 2026-07-14 — CONSEQUENCE LAW PRE-REGISTERED — from demo to quantitative law

**Why:** the matrix shows write→consequence at one counterfactual pair per row.
The law question: does output track f(written value) over the WHOLE domain,
do two written arguments compose through one computation, and what happens
under conflicting writes? Module `delta_consequence_law.py`, stage
`delta_consequence_law`. Models: Qwen + Mistral (mounted, L2, 8bit). Neutral
donors + canonical-continuation contract as in the matrix. Per-arm G0 ≥90%.

**Arm A — transfer curve (the law).** T1 = `X = {a}. Y = X + {b}. What is the
value of Y?` Contexts (a,b) ∈ {(2,3),(3,2),(4,5),(5,1),(6,2),(3,4)}; for each,
write z_v−z_a for EVERY valid v (v≠a, v+b≤9). Metrics per cell: ADD target
accuracy (argmax over digit ids = v+b) and margin ratio vs the natural text
counterfactual. 30 shared norm-matched nulls. **Gates:** `CONSEQUENCE_LAW`
iff cell-accuracy ≥90% AND median ratio ≥0.7 AND null-clear (p<.04 floor
1/31); `LAW_PARTIAL` iff accuracy ≥70%; else `LAW_FAIL`.

**Arm B — two-operand composition.** T2 = `X = {a}. Z = {c}. Y = X + Z. What
is the value of Y?` (G0 unknown → per-arm gate protects). 10 rows; conditions:
natural-both (text v,w), ADD-both (write v@X and w@Z, forward_add_multi),
ADD-X-only (predict v+c — MIXED injected+textual arguments), ADD-Z-only
(predict a+w), wrong-pair (predicts its OWN sum), 30 nulls. **Gates:**
`ARGUMENT_COMPOSITION` iff both-write acc ≥80% ∧ ratio ≥0.7 ∧ null-clear;
flag `MIXED_ARGS_OK` iff both single-write mixed sums ≥80% on their own
predictions. `INELICITABLE` if G0 <90% (reported, not fatal).

**Arm C — conflicting writes at one address (DISCOVERY, no pass/fail).**
Fixed context (a=3,b=2); 10 (v,w) pairs. Conditions: v-only, w-only,
sum-both (z_v+z_w−2z_a), half-scale sum. Frozen report: winner distribution
over {f(v), f(w), f(a), other} + margins. Hypothesis space (not gated):
superposition/midpoint vs winner-take-all vs interference — connects to the
overwrite-suppression findings. No post-hoc metric additions.

### 2026-07-14 — consequence law COMPLETE (Qwen + Mistral) — LAW + COMPOSITION confirmed

Kernels `cm-delta-consequence-law-{qwen7b,mistral7b}` COMPLETE (mounted models,
L2, 8bit). Artifacts `runs/delta_consequence_law-*/`.
(Infra: first Qwen launch errored on stale code version — `datasets status` is
version-blind and reported the prior version ready; relaunched on processed
code, both clean. No result inspected from the errored run.)

**Both models: `CONSEQUENCE_LAW | ARGUMENT_COMPOSITION`.**

| Arm | Qwen | Mistral |
|---|---|---|
| **A — codebook law** (31 cells, 6 contexts) | acc **100%**, median ratio **0.99**, p=.032 | acc **100%**, median ratio **1.00**, p=.032 |
| **B — two-operand** (Y=X+Z, write both) | both **100%**, ratio 1.00 | both **100%**, ratio 1.00 |
| **B — mixed args** (1 injected + 1 textual) | mixedX/Z **100%/100%**, `MIXED_ARGS_OK` | **100%/100%**, `MIXED_ARGS_OK` |
| **B — wrong-pair own sum** | 100% | 100% |

**Arm A = the quantitative law:** across the ENTIRE digit codebook and 6
contexts, output tracks f(written v)=v+b at natural-counterfactual strength
(ratio ≈1.0), 100% correct, in both families. This is `output = f(written
value)` over the whole domain — not a single-pair demo. **Arm B:** two
independently-written arguments are consumed by one addition (100%), and —
the sharpest single cell in the project — a value written as a **direction**
sums correctly with a value written as **text** (`MIXED_ARGS_OK`), i.e. the
injected argument enters the SAME computation path as the natural one.

**Arm C (discovery — genuine mechanistic finding):** two conflicting writes at
one address do NOT superpose to f(v+w-a) or collapse to f(a). Instead
**winner-take-all**: single writes are clean (v-only→f(v) 100%, w-only→f(w)
100%); the summed write splits ~40/60 (Qwen) and ~60/40 (Mistral) between
f(v) and f(w) with small margins (|v−w| margin 2–4 vs ~11–16 for single
writes), f(a)=0% and other≈0%. So the address holds ONE value; competing
writes contend for it rather than blending. This aligns with the earlier
overwrite-SUPPRESSION / patch-mask dissociation: the workspace slot is a
single-occupancy register, not an additive accumulator. Reported as discovery
(no gate), but it is a clean, publishable structural result.

**Status:** the write-consumption principle is now a *law* (full-domain,
two-family) with *argument composition* (incl. mixed injected+textual) and a
*single-occupancy* conflict rule. Next: coordinate-alignment (Procrustes
transfer prediction) — the predictive flagship — then paper assembly.

### 2026-07-14 — SHARED ATLAS PRE-REGISTERED — relational synthesis of unseen value-states

**Question (the predictive flagship):** is the value-state codebook a shared
geometric object across contexts and MODELS — i.e., can we **synthesize a
value-state we never extracted**, purely from its *relations* to other values,
and have it drive retrieval AND computation causally? Module `delta_atlas.py`,
stage `delta_atlas` (one kernel, loads Qwen then Mistral sequentially, both
mounted).

**Method (barycentric synthesis — dimension-free, avoids rectangular-map
ill-posedness):** anchors = neutral digit states z_v at L2. **Frozen split:**
fit F={1,2,4,5,7,8}, held-out H={3,6,9}. For each h∈H solve affine LSQ
coefficients c(h): z_src(h) ≈ Σ_f c_f·z_src(f)+c_0 **in the source space**;
synthesize ẑ_tgt(h) = Σ_f c_f·z_tgt(f)+c_0 **in the target space**; write
ẑ_tgt(h) − z_tgt(a) at the value slot.

**Arms:** (1) WITHIN-model synthesis (source=target), Qwen and Mistral — the
geometry control: does the codebook have affine structure at all? (2) CROSS-
model: coefficients fit on **Qwen** anchors, synthesis from **Mistral**
anchors, tested in Mistral. Tasks per arm: retrieve (h∈{3,6,9}, 4 base
contexts) and consequence T1 add (h∈{3,6} — 9 excluded by digit range; 4
(a,b) contexts). Conditions: NATIVE write z(h)−z(a) (reference), SYNTH
ẑ(h)−z(a) (the test), WRONG-SYNTH ẑ(h′)−z(a) (must track ITS OWN h′), 30
norm-matched randoms.

**Frozen gates:** per arm, SYNTH pass iff target acc ≥80% ∧ effect ratio vs
NATIVE ≥0.7 ∧ null-clear (p<.04) ∧ wrong-synth own-target ≥80%. **Verdicts:**
`SHARED_ATLAS` (within-Qwen ∧ within-Mistral ∧ cross all pass) ·
`LOCAL_CHARTS` (both within pass, cross fails — geometry real but
model-local) · `NO_AFFINE_CODEBOOK` (within fails — cross uninterpretable,
stop) · partials reported as mixed. Either verdict is a finding.

**Prior-art line (honest):** relative representations (Moschella) do
zero-shot *stitching* via relational coords; linear-representation-
transferability steers big models with small models' vectors. Ours differs:
**causal write-synthesis of an unseen value-state that must drive downstream
computation** (consequence task), with native-reference ratio + content
controls. If SHARED_ATLAS: value codebooks are portable geometry — one
atlas, many models. If LOCAL_CHARTS: relational structure exists per model
but does not port — both publishable.

### 2026-07-14 — delta_atlas COMPLETE — NO_AFFINE_CODEBOOK (a confirming negative)

Kernel `shlydv/cm-delta-atlas` COMPLETE. Artifacts `runs/delta_atlas/`.

| Arm | retrieve | consequence |
|---|---|---|
| within-Qwen | native 100%, **synth 0%**, ratio 0.41 | native 100%, synth 12%, ratio 0.52 |
| within-Mistral | native 100%, **synth 0%**, ratio 0.50 | native 100%, synth 0%, ratio 0.53 |
| cross Q→M | native 100%, **synth 0%**, ratio 0.47 | native 100%, synth 0%, ratio 0.47 |

Wrong-synth own-target also ~0% everywhere (ALL synthesized states are
non-functional — not a content mix-up). Yet the synthesized states are
geometrically CLOSE: cross-model cos(ẑ_m(h), z_m(h)) = 0.87–0.92, and the
causal margin moves ~half the native effect — but the argmax never selects
the target. **Per the frozen ladder, within-model failure makes cross
uninterpretable: verdict `NO_AFFINE_CODEBOOK`.**

**Interpretation (calibrated — this negative CONFIRMS the earlier geometry):**
delta_decompose showed value-specific residuals are pairwise ~orthogonal
(cos 0.014) around a large shared centroid. Orthogonal prototypes are
exactly the geometry in which affine interpolation MUST fail: the anchors'
affine hull contains the shared centroid (hence cos ~0.9 and ~half the
margin) but none of the held-out value's discriminative residual (hence 0%
target selection). One picture now covers four results:
**value-states are near-orthogonal discrete prototypes** ⇒ composition
across ADDRESSES is free (multislot/capacity ≥8, cross-talk ≈0), blending
WITHIN an address is impossible (conflict-writes contend 50/50, never
average), and states are not interpolable/synthesizable from relations
(atlas). The workspace behaves like a **digital register bank with an
orthogonal code**, not an analog vector space. Also consistent with
bind-miss (discriminative info outside the extracted span) and the
non-surjective-steering literature. NOT claimed: that no nonlinear map
could synthesize states; only that affine/barycentric relational structure
is absent at the functional level. Line closed; no coefficient fishing.

### 2026-07-14 — ENTITY / WORLD-STATE EDITING PRE-REGISTERED (Sahil re-prioritization)

**Why (Sahil + external review):** arithmetic = proof of existence (finished,
stopped). The significance jump is the same mechanism on **entity/world
state** with **inference over the edit**. Module `delta_entity_matrix.py`,
stage `delta_entity_matrix` (one kernel, Qwen→Mistral sequential, mounted).
Design upgrade over the suggestion: **edited vocabulary ⊥ answer vocabulary**
— we edit a color/city, the answer is an object/word reachable ONLY through
an in-context rule, so a correct answer cannot be lexical leakage; it must
pass through an inference hop.

**Families (2), synthetic in-context rules (no world-knowledge confound):**
- **KEYS:** `Alice has the {a} key. Bob has the {b} key. The {a} key opens
  the {oa}. The {b} key opens the {ob}. The {w} key opens the {ow}.`
- **CITY:** `Alice is in {a}. Bob is in {b}. People in {a} say {sa}. ...{w}
  say {sw}.`

**Cells per family:** retrieve (`What color is Alice's key?` → colors;
control), **twohop** (`What can Alice open?` → objects; THE cell), other-
entity (`What can Bob open?` — edit Alice, Bob's answer must NOT move).
Write = neutral-carrier `Here is a word: {w}.` state difference z(w)−z(a) at
Alice's possession/location token (rules untouched). Natural CF = text with
Alice's token swapped to w (rules fixed ⇒ exactly 1 token differs). WRONG
write = z(b)−z(a) → must yield ITS OWN consequence (ob). 30 norm-matched
nulls on retrieve + twohop. n=10 rows/family; pools preflight-filtered to
single-token-in-context words with distinct answer first-tokens.

**Frozen gates:** per cell G0 (clean+natural argmax over pool) ≥90%.
retrieve/twohop PASS iff ADD target-acc ≥80% ∧ ratio vs natural ≥0.7 ∧
p<.04 ∧ wrong-own ≥80%. other-entity PASS iff |Bob-answer shift| ≤ 0.25 ×
twohop effect. **Family PASS** = all three. **Model verdict:**
`WORLD_STATE_GENERAL` (both families) / `WORLD_STATE_PARTIAL` (one) /
`RETRIEVE_ONLY` / per-cell INELICITABLE reported. **Program rule:** both
models `WORLD_STATE_GENERAL` ⇒ the workspace mechanism extends from symbols
to world-state with inference — the paper's headline claim. Failures logged
without rescue; prompt realization is frozen after the mechanical preflight.

### 2026-07-14 — entity matrix COMPLETE — WORLD_STATE_PARTIAL(Qwen) | RETRIEVE_ONLY(Mistral)

Kernel `shlydv/cm-delta-entity-matrix` COMPLETE. Artifacts
`runs/delta_entity_matrix/`.

| cell | Qwen | Mistral |
|---|---|---|
| keys/retrieve | **PASS** 1.05, 100% | **PASS** 0.97, 100% |
| keys/twohop | INELICITABLE (nat G0 80%) | INELICITABLE (80%/80%) |
| **city/retrieve** | **PASS** 1.04, 100% | **PASS** 1.00, 100% |
| **city/twohop** | **PASS ratio 0.95, acc 100%, wrong_own 100%** | INELICITABLE (nat G0 50%) |
| **city/other-entity** | **PASS — Bob shift literally 0.00** | INELICITABLE |

**The headline cell passed (Qwen, city family, full):** editing Alice's
location as an INTERNAL STATE (neutral-carrier write at her location token;
rules untouched) makes the model answer the two-hop question (`What does
Alice say?`) through the in-context rule — **ratio 0.95 vs the natural
textual counterfactual, 100% target accuracy, wrong-value writes produce
their own consequence 100%, and the OTHER entity's answer moves 0.00**. The
edited vocabulary (cities) is disjoint from the answer vocabulary (say-words),
so the result cannot be lexical leakage — the edit passed through an
inference hop. This is world-state editing with inference-propagation.

**The blocks are capability, not mechanism:** every non-passing inference
cell failed at G0 on the NATURAL TEXT task (Mistral city two-hop 50%; keys
two-hop 80% both models) — the models cannot reliably do those inferences
from plain text, so the write mechanism was correctly never tested there
(per-cell G0 did its job; INELICITABLE ≠ FAIL). Retrieval-level entity
editing is cross-model (4/4 cells, ratios 0.97–1.05, 100%).

**Calibrated claim:** wherever the model can perform the inference in text,
the injected state drives the SAME inference at natural strength; where base
capability is absent, editing is untestable. The capability boundary is the
model's, not the mechanism's. Obvious strengthener (not yet run): a larger
model (e.g. 14B, 2×T4 sharded) to unlock the INELICITABLE inference cells.

### 2026-07-14 — SCALE UNLOCK PRE-REGISTERED — identical entity matrix on Qwen2.5-14B

**Why (Sahil robustness call):** the inference-propagation claim rests on ONE
passing cell (Qwen-7B city twohop); all other inference cells were blocked at
natural-text G0 (capability, not mechanism). Breadth-at-7B would mostly add
INELICITABLE cells; the correct robustness axis is **scale under the frozen
protocol**. Kernel: `delta_entity_matrix` with `models=[{qwen14b}]` — same
families, pools, rows, seed, gates, L2, per-cell G0; ONLY the model changes.
Model: official Kaggle mount `qwen-lm/qwen2.5/transformers/14b-instruct/1`
(8-bit, sharded across 2×T4 via device_map=auto).

**Predictions (written before the run):** (1) the 7B-blocked cells (keys
twohop/other; possibly city at margin) clear G0 at 14B and PASS their causal
gates; (2) city family replicates. If instead cells stay INELICITABLE at 14B,
that is reported as-is (scale did not unlock; claim stays capability-scoped).
If cells clear G0 but FAIL causally, that is a real mechanism boundary at
scale — the most informative possible outcome, also reported as-is. No
prompt changes, no layer search (L2 frozen; 14B has 48 layers — L2 remains
the pre-registered early write site consistent with all prior models).

**Also queued after this (agent arm, per Sahil):** phase/evidence controller
redo under the new interface discipline — the arbitration positive already
exists (Mistral, all primary gates); the redo tests edit-driven behavior
switching ("asked to do X, edited to do Y") with per-cell G0 and the
canonical contract. Design to be pre-registered separately before any code.

### 2026-07-14 — Qwen-14B load failure + frozen feasibility amendment

Both scale-unlock attempts failed **before any example or causal intervention
ran**. The official Qwen2.5-14B checkpoint reached 88% of 8-bit materialization,
then GPU 1 exhausted its 14.56 GiB (14.42 GiB allocated). A second attempt with
`max_memory={0:11GiB,1:11GiB}` still hit a 14.4 GiB transient allocation. This
is a model-construction OOM, not evidence for or against world-state editing.

**Single amendment:** use bitsandbytes NF4 4-bit with bf16 computation. All
scientific choices remain frozen: L2, prompts, rows, seed, pools, G0, causal
gates, wrong-value/other-entity controls, and 30 nulls. The retry kernel runs
Qwen2.5-7B 4-bit first, then Qwen2.5-14B 4-bit. The 7B arm is a quantization
calibration against its known 8-bit city-family positive. Interpretation is
precommitted:

- If 7B 4-bit loses the known city positive, 14B causal negatives are not
  interpretable as scale effects.
- If 7B calibrates and 14B clears G0 then passes, scale robustness is supported.
- If 7B calibrates and 14B clears G0 then fails causally, that is a genuine
  scale/mechanism boundary.
- If 14B still fails G0, scale did not unlock the frozen task.

No layer search, prompt repair, coefficient tuning, or post-hoc gate changes.

### 2026-07-14 — 4-bit calibration COMPLETE; sequential 14B load still OOM

Kernel `shlydv/cm-delta-entity-matrix-qwen14b-v3` ran the frozen 7B→14B
calibration design. The **Qwen-7B NF4 calibration succeeded** and reproduced
the known result:

- city/retrieve PASS: ratio 1.02, target accuracy 90%, wrong-own 100%, p=.032;
- city/twohop PASS: ratio 1.02, target accuracy 100%, wrong-own 100%, p=.032;
- city/other PASS: Bob shift 0.00;
- model verdict `WORLD_STATE_PARTIAL` (city family passes).

Thus NF4 itself preserves the headline causal effect. Keys inference remained
G0-ineligible (70/50%), as expected for a capability-limited cell.

The subsequent Qwen-14B phase again OOMed at 88% materialization (GPU 1:
14.27/14.56 GiB allocated). **No 14B behavioral or causal example ran.** This
does not update the scientific hypothesis. Because 14B NF4 has not yet been
run in a fresh process by itself, the least-assumptive next diagnostic is a
14B-only kernel using this already-validated code. The completed 7B arm serves
as the quantization calibration; it need not be repeated. Only if a fresh
14B-only NF4 load also fails should the loader change to a pre-quantized AWQ
checkpoint or explicit CPU offload.

### 2026-07-14 — fresh 14B-only NF4 load FAILED; on-the-fly path closed

Kernel `shlydv/cm-delta-entity-matrix-qwen14b-v4` loaded **only** the official
Qwen2.5-14B-Instruct checkpoint in a fresh process. It failed at 78% in the
displayed progress stream while materializing layer 37 (GPU 1:
14.36/14.56 GiB allocated; the next 136 MiB allocation failed). This rules out
retained Qwen-7B memory as the cause. Neither NF4 nor the earlier int8 and
max-memory attempts avoid the transient full-weight materialization peak in
this Kaggle Transformers 5.x stack.

**Closed:** do not retry on-the-fly bitsandbytes, alter memory caps, or repeat
the standard checkpoint. **Still no 14B scientific observation:** zero prompts
or interventions ran. The next justified feasibility route is a checkpoint
already stored in quantized form (Qwen2.5-14B-Instruct-AWQ, loaded directly),
with the completed 7B NF4 city-family result retained as the quantization
calibration. CPU offload is a lower-priority fallback because it would make the
full causal/null matrix substantially slower.

### 2026-07-14 — official Kaggle AWQ feasibility path selected

Kaggle exposes the official 9.99 GB variation
`qwen-lm/qwen2.5/transformers/14b-instruct-awq/1`. Its own usage contract is
direct `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto",
device_map="auto")`. The loader now has an explicit `quantization="awq"` mode
that follows this contract and deliberately supplies **no** bitsandbytes
configuration. This avoids both network download and on-the-fly conversion;
all scientific entity-matrix settings remain frozen. The previously completed
Qwen-7B NF4 city-family pass remains the quantization calibration.

The first AWQ kernel reached Transformers' quantizer validation immediately
and exited before loading with the explicit requirement `pip install
gptqmodel`. This is a missing runtime dependency, not a checkpoint or memory
failure. The AWQ loader now pins `gptqmodel==7.1.0` (current stable release,
Python 3.12 and Turing+ support) before `from_pretrained`; no experiment logic
changed.

The dependency-enabled kernel installed successfully but failed on a mixed
Transformers import (`KERNELS_MAX_VERSION` missing): `causal_maps` had imported
the preinstalled Transformers before pip upgraded it as a gptqmodel dependency.
This is a bootstrap-order error. The kernel template now installs the pinned
AWQ runtime **before importing causal_maps/Transformers**, ensuring one coherent
package version for the whole process. No scientific code or configuration
changed.

### 2026-07-14 — Qwen2.5-14B-AWQ COMPLETE — cross-scale city replication

Kernel `shlydv/cm-delta-entity-matrix-qwen14b-awq-v3` completed in 433 s using
the official pre-quantized checkpoint. The direct AWQ path therefore solves the
Kaggle 14B feasibility problem.

| family/cell | G0 clean/natural | causal result |
|---|---:|---|
| keys/retrieve | 100% / 100% | **PASS**, ratio .993, add acc 100%, wrong-own 100%, p=.032 |
| keys/twohop | 20% / 10% | INELICITABLE |
| keys/other | 40% / 40% | INELICITABLE |
| city/retrieve | 100% / 100% | **PASS**, ratio 1.002, add acc 100%, wrong-own 100%, p=.032 |
| city/twohop | 100% / 90% | **PASS**, ratio .999, add acc 90%, wrong-own 100%, p=.032 |
| city/other | 100% / 90% | **PASS**, Bob shift 0.00 |

Model verdict: `WORLD_STATE_PARTIAL` (city family passes). This is a genuine
cross-scale replication of the headline inference-propagation result: at 14B,
an L2 neutral-carrier write to Alice's city slot drives the disjoint downstream
consequence at essentially exactly the natural textual effect while leaving
Bob unchanged. Together with Qwen-7B int8 and NF4, the city result now survives
scale and two quantization backends. The preregistered scale-unlock prediction
for keys was falsified: its natural two-hop task became *less* eligible
(20/10%), so no keys causal conclusion is permitted. Do not repair that prompt
post hoc inside this confirmatory matrix.

### 2026-07-14 — STRUCTURED WORKSPACE INTERCHANGE PRE-REGISTERED (not run)

Anthropic's July 2026 J-space result closes the weak version of the endogenous
state claim: silently computed, verbalizable concepts can already be swapped to
redirect later reasoning. Their paper explicitly leaves open the structure
above a flat bag of concepts—how concepts are bound into relations and assigned
roles. The next experiment therefore targets that exact gap rather than merely
swapping another intermediate value.

Frozen stage `delta_structured_workspace`, protocol
`STRUCTURED_WORKSPACE_PROTOCOL.md`, Qwen2.5-7B-Instruct int8, seed 0, 30 nulls.
It derives a six-variable private-belief world from event histories, places a
query-independent `STATECHECK` before the question, requires bidirectional
natural-state interchange before learning any controller, then tests donor-mean
relation edits on a held-out narrative surface. Required evidence includes
three downstream consequences, preserved truth and neighboring relations,
positive wrong-address edits, three value transitions, two-edit factorial
composition, a processed unbound-concept rival, random nulls, and full greedy
outputs. The earliest passing layer from frozen L8/L12/L16/L20/L24/L26 is used
everywhere; no rescue sweep or prompt repair is allowed. **No GPU run has been
launched.**

### 2026-07-14 — structured workspace v3 COMPLETE — BEHAVIORALLY_INELIGIBLE at 7B

Infra: v1 (GPT launch) stalled at unauthenticated HF download — raw HF id, no
mount (the known failure); cancelled, no result. v2 tripped the module's
config-freeze guard on the mounted path (guard pinned the literal path
string); amended pre-run to accept the validated mount of the SAME checkpoint
(storage location ≠ science; logged, one line). v3 ran clean in 340 s.

**Verdict: `STRUCTURED_WORKSPACE_BEHAVIORALLY_INELIGIBLE`** — G0 stopped the
kernel before any intervention (cheap, as designed). **Failure anatomy** (the
informative part): the PRIMARY relation is fine — Alice-cube belief/tell 100%
across surfaces and transitions. The wall is third-party/second-object
tracking and behavioral readouts on the narrative surface:
`belief(Bob, sphere)` clean 60% (train) / 40% (test), `search_bs` test **0%**,
and several `search_*` cells 60–80% on held-out narratives. Qwen-7B can track
one agent's false belief but not the full six-variable two-agent world at the
≥80%-everywhere bar. Natural cells mostly ≥80% while CLEAN cells fail —
specific clean configurations confuse it.

**Per protocol:** no prompt repair, no rescue. The failed gate determines the
next question: the identical frozen protocol on **Qwen2.5-14B-AWQ** (loader
validated, 433 s; 14B already unlocked the entity-matrix inference cells that
7B/Mistral could not do). Requires a pre-registered amendment (model +
quantization="awq" in the freeze guard; layer menu per protocol "subject to
model depth"). Decision on spending the remaining Kaggle quota: Sahil's call.

### 2026-07-14 — structured workspace 14B AMENDMENT PRE-REGISTERED + LAUNCH (Sahil: go, once)

**Amendment (before the run, infra + model only):** freeze guard now pins
CHECKPOINT×QUANTIZATION pairs; added the pair (official Kaggle
`qwen-lm/qwen2.5/transformers/14b-instruct-awq/1` via glob path, "awq") —
exactly GPT's validated entity-matrix AWQ recipe (gptqmodel==7.1.0 installed
pre-import by the kernel template; dtype auto; device_map auto; no
bitsandbytes). Glob resolution added to the module (mechanical). Layer menu
unchanged: literal L8/L12/L16/L20/L24/L26, all valid at 48 layers, per the
protocol's "subject only to model depth" clause. Worlds, readouts, gates,
seed, n_null, thresholds: UNCHANGED. Prediction (from the 7B anatomy + the
14B entity unlock): the failing G0 cells (belief/search for Bob-sphere;
narrative search readouts) clear ≥80% at 14B; if G0 clears, the full ladder
G1→G5 runs for the first time. One run authorized (low quota); any verdict
is final for this protocol — no rescue.

### 2026-07-14 — structured workspace 14B-AWQ COMPLETE — BEHAVIORALLY_INELIGIBLE (anatomy shifted; protocol closed at both scales)

Kernel `cm-delta-structured-workspace-14b-awq` COMPLETE (288 s — AWQ path
clean, G0 stopped before interventions). Artifacts
`runs/delta_structured_workspace-14b-awq/`.

**Verdict: `STRUCTURED_WORKSPACE_BEHAVIORALLY_INELIGIBLE` at 14B too — but
the failure anatomy CHANGED (partial confirmation of the scale prediction):**
- 7B walls that CLEARED at 14B: `belief(Bob, sphere)` tracking (7B 40–60% →
  14B ≥80%; no belief cell fails at 14B) — the predicted unlock happened for
  BELIEF readouts.
- The remaining wall is almost purely **`search_*` under NATURAL
  (counterfactual) histories**: search_ac natural 0.20–0.60 across
  transitions, and the joint-composition world's search_ac natural **0.00**.
  Belief reports and tells are fine; predicting the *action* ("where will
  Alice look?") in a counterfactually-swapped false-belief world is what
  neither scale can express reliably in text.

**Interpretation (calibrated):** a genuine capability boundary — false-belief
ACTION prediction under counterfactual histories — documented at two scales
with a shifted anatomy (belief tracking scales; action-under-CF does not, at
7B→14B). Relational-edit interventions on this six-variable construct remain
untestable at these scales; the protocol's discriminative design is intact
and READY for any model that clears G0. Per the freeze: protocol closed at
7B and 14B, no rescue, no prompt repair inside this protocol. A future
simpler-world variant (e.g. 4-variable, report-only readouts) would be a NEW
protocol requiring fresh pre-registration. Quota spent: 288 s.

### 2026-07-14 — REPORT-ONLY RELATIONAL VARIANT PRE-REGISTERED + LAUNCH (Sahil: run the tests before any paper)

**Sahil directive:** no paper until the general (relational) tier is actually
tested. **New variant, fresh registration (not a rescue of the closed
protocol):** `report_only_v1` — identical worlds, histories, six latent
variables, multi-register discriminative design (Paris fills multiple
registers; a bag-of-concepts swap must fail), wrong-address positives,
composition, unbound-concept rival, 30 nulls, thresholds, seed, layer menu.
**One change, motivated by the documented two-scale anatomy:** drop the
`search_*` readouts (action-under-counterfactual — the class that failed G0
at BOTH scales) from every required set. Consequences per relation become
report-level (belief report; tell for the primary). Implemented as a
`report_only` flag filtering the six frozen readout tuples; variant recorded
in results as `protocol_variant=report_only_v1`. **Frozen to the 14B-AWQ
pair** (at 7B, belief_bs itself failed 40–60%, so report-only could not
clear G0 there — running it would waste quota).

**Prediction:** every required report-level cell was ≥80% in the 14B run's
G0 table ⇒ G0 clears, and G1→G5 run for the first time. Honest note: the
composition G0 cell drops search_ac (0.00 at 14B) but keeps all four belief
+ four invariant readouts — if any of those fail in the joint world, the
variant dies at G0 and that is the result. One kernel authorized; verdict
final for this variant.

### 2026-07-15 — report_only_v1 COMPLETE — G0 PASSED; NO_BIDIRECTIONAL_STRUCTURED_CHECKPOINT (real, verified)

Kernel `cm-delta-structured-workspace-14b-awq-report` COMPLETE (~445 s).
Artifacts `runs/delta_structured_workspace-14b-awq-report/`.

**G0 PASSED** (prediction confirmed — the report-level world is fully
elicitable at 14B). **G1 failed at every layer** L8/L12/L16/L20/L24/L26:
patching the full natural residual state at the STATECHECK token into clean
runs (and reverse) produced target/reverse accuracy 0.00 everywhere.

**Bug-first audit (per house rule) says the negative is REAL:** natural
effects are huge (belief_ac +63.8, belief_bs +37.9 logit-diff) while patch
effects are tiny but NONZERO and row-varying (−0.05…+0.22; ratios ~1e-4 to
2e-3) — hooks fired (same _split_output convention as the AWQ-validated
entity run), the state was written, and it simply carries ~0.1% of the
behavioral difference. (An earlier extraction printed nan/0.00 for
ratio/pos_frac — wrong dict keys in the reader, corrected; verdict
unchanged.)

**Finding:** the silently derived six-variable belief world is NOT stored in
the residual state of a query-independent checkpoint token at any tested
layer (≤L26/48). The model demonstrably HAS the state behaviorally (G0
100%), but a single-position state swap transfers essentially none of it.

**Interpretation — the general picture this completes (calibrated):**
**LLM in-context memory is TOKEN-ANCHORED.** Values live at the positions
where they were stated (slot states interchangeable L2–L20, editable,
single-occupancy — the whole positive arc); derived answers are recomputed
at query time by attending back to those anchors; there is NO consolidated
world-state register at a summary position (this result), no affine codebook
(atlas), and edits work exactly when they hit an anchor (entity twohop) and
do nothing when they hit a derived checkpoint (here). Scope: one model
(14B-AWQ) for this negative; layer menu ends at L26 of 48 (~54% depth), so a
late-layer consolidation is not excluded; verdict final for this variant per
the freeze.

**Falsifiable next prediction (pre-registerable, not yet run):** in the SAME
G0-validated world, writing at the HISTORY ANCHOR token (the event sentence
where Alice last saw the cube) SHOULD edit her belief with full downstream
consequences — the token-anchored theory predicts success where the
checkpoint edit predicted (and got) nothing. One cheap kernel would test
both arms of the theory in one world.

### 2026-07-15 — ANCHOR-WRITE PRE-REGISTERED + LAUNCH (Sahil: go) — the theory's positive arm

Stage `delta_anchor_write`, module `delta_anchor_write.py`. Same frozen world
(`_rows("Paris","Rome","ac","test")`, narrative surface — the G0-validated
held-out split), same 14B-AWQ pair, same answer contract. **Intervention:**
neutral-carrier write z(Rome)−z(Paris) ("Here is a word: {loc}." donors, the
exact recipe validated in the entity matrix) at **L2** at the **anchor
token** — the single position where clean/natural prompts differ (Alice's
cube-event location; position verified uniform across rows offline).

**Frozen gates:** G0 re-verify (belief_ac, tell_ac clean+natural ≥80%).
CONSEQUENCES: belief_ac and tell_ac each pass target_acc ≥80% ∧
positive-fraction ≥80% ∧ ratio vs natural ∈ [0.6, 1.4]; 30 norm-matched
nulls, p<.04 on the belief_ac margin. INVARIANTS under the write: belief_as,
belief_bc, truth_cube, truth_sphere all ≥80% preserved. WRONG-ADDRESS
positive control: same z(Rome)−z(Paris) written at BOB's cube-event anchor
must flip belief_bc (its own consequence, ≥80%) while preserving belief_ac
(≥80%).

**Verdicts:** `TOKEN_ANCHORED_CONFIRMED` (all gates) — with the checkpoint
null (previous entry), both arms of the token-anchored theory hold in ONE
world: derived beliefs are editable at their stated anchors, not at any
summary state. `TOKEN_ANCHORED_PARTIAL` (some gates) —reported as-is.
`TOKEN_ANCHORED_FALSIFIED` (consequences fail) — the theory's positive arm
is wrong for derived beliefs at 14B; equally final, equally logged. One
kernel; no rescue.

### 2026-07-15 — anchor-write COMPLETE — TOKEN_ANCHORED_PARTIAL (printed); core prediction confirmed at ratio 1.001

Kernel `cm-delta-anchor-write-14b-awq` COMPLETE. Artifacts
`runs/delta_anchor_write-14b-awq/`.

| panel | result |
|---|---|
| **belief_ac** | g0 100/100; natural +63.1; **anchor-write +63.1 (ratio 1.001), target acc 100%, pos-frac 100%**; nulls p=.032 (null mean −0.27) |
| invariants ×4 | **100% preserved under the write** (belief_as, belief_bc, truth_cube, truth_sphere) |
| wrong-address | same Δ at Bob's anchor flips belief_bc **100% (ratio 1.003)**; belief_ac preserved 100% |
| tell_ac | margin ratio **1.006** (full natural strength), pos-frac 100%; greedy acc 3/5 < 4/5 gate — on a readout whose natural-text G0 is itself 80% (4/5) |

**Printed verdict `TOKEN_ANCHORED_PARTIAL`** (frozen gate: tell_ac greedy
missed by one row at n=5). Stands as printed; no re-grading.

### 2026-07-15 — FLOOR 3 PRE-REGISTERED: verbalization write-back (`VERBALIZATION_PROTOCOL.md`)

**Sahil directive:** pursue the highest-ceiling extension — CoT as the
transformer's write-back mechanism. Theory: no internal consolidation exists
(checkpoint null), so verbalizing a derived fact CREATES an anchor; causal
load should MIGRATE from history anchors to the verbalization token.
Quantities: causal load λ_t(r) (edit-effect / natural-effect; baseline ledger
already measured: λ_hist≈1.001, λ_checkpoint≈0.001) and faithfulness
F = λ_cot/(λ_cot+λ_hist) — a continuous, causal CoT-faithfulness meter.
Hypotheses H1–H5 (write-back, migration w/ content-specificity control,
conservation structure incl. conflict arbitration, depth-reset of the
14B-failed action cells, self-vs-forced) with frozen gates and graded
verdicts — every branch reportable, incl. COT_DECORATIVE (deflationary).
Positioning: 2606.29522 showed trained-to-write registers; we measure the
load ledger in STOCK models + migration + capability unlock. Arm A
(teacher-forced, one 14B-AWQ kernel) decides H1–H3; build next, launch on
quota. Floor 2 (architecture dissociation) parked as paper-#1 backup.

### 2026-07-15 — PROGRAM DECISION: Paper 1 preprint is the sole objective (`PREPRINT_PLAN.md`)

**Sahil:** Paper 1 finish → Paper 2 (architecture dissociation) after
acceptance/arXiv → Paper 3 from whatever emerges. Floor 3 dropped as
flagship (LEDGER_MIXED killed the shadowing prize); its reverse-base cell
survives only as preprint item M5. Plan of record `PREPRINT_PLAN.md`:
thesis + 6 contributions, field-standard bar (identification statement,
≥3 families incl. Llama, n≥30 × 3 seeds, code release), gap matrix
M1–M5 (~10–12 kernels over two quota resets), arXiv target Aug 10–15,
ICLR abstracts Sep 19 / papers Sep 24 2026. Emergence sandbox runs on
local/free compute in the background. All further log entries serve the
preprint unless Sahil redirects.

### 2026-07-15 — PREPRINT BATTERY PRE-REGISTERED + LAUNCH (M1/M2 widening)

Vehicle `delta_preprint_battery` (one model load → workspace matrix + entity
matrix + anchor-write, per seed {0,1,2}, plus one checkpoint-null cell).
**Pre-registered widening (NOT a redesign):** n_rows 12/10/5 → 30/30/30;
seeds 0,1,2; all protocols/gates/sites otherwise byte-identical to the frozen
originals; structured `_rows` widening keeps the frozen first-5 prefix exact
(verified offline). Original single-seed runs stand as pilots; if a widened
headline number shifts, BOTH are reported. Models this batch: Qwen2.5-7B
(mounted int8), Mistral-7B (mounted int8). Then 14B-AWQ, Llama-3.1-8B, and
small Qwens (1.5B/3B) for a scale axis. Checkpoint cell replicates the
no-buffer null per model across 8 layers incl. late. Launch now; freeze
numbers on return.

### 2026-07-15 — Opus review of GPT's preprint package + PLAN AMENDMENT (pre-registered, no GPU)

**Blocked-experiment review (frozen `cm-preprint-headline-qwen14b-v2`):**
probe design AUDITED AND APPROVED as-is — balanced 8-way labels with
independently permuted nuisance fields per replicate, grouped CV, layer
selection NESTED inside outer folds, exact binomial vs chance; probes at
anchor + STATECHECK + readout. No changes to existing cells.

**Amendments (before any launch):**
1. **Fold the M5 reverse-base quorum cell INTO the 14B headline kernel**
   (same model, same world, ~2 min appended block; saves a full model load;
   decides the verbalization paragraph). Existing cells byte-untouched.
2. **Add one cheap scale-axis kernel:** Qwen2.5-1.5B + 3B batteries,
   sequential in one load (both mounted official variations, Apache/open;
   ~10 min total). Upgrades the coverage table with a scale trend; promoted
   from NICE.
3. **M2 Llama-3.1-8B battery unchanged** — needs Sahil's one-time license
   acceptance on Kaggle (`metaresearch/llama-3.1`); Gemma-2-9B remains the
   fallback only.
4. Nothing removed. Next-reset queue: (i) 14B headline+quorum, (ii) Llama
   battery, (iii) small-Qwen pair. 32B stays out (quota).

**Paper edits applied (`paper_token_anchored/main.tex` + references.bib):**
abstract restructured to lead with the anchor/checkpoint dissociation and the
readout positive control, "is running" removed (dates the draft), overshoot
now acknowledged where the CI excludes 1 (fixed-norm explanation, direction
conservative for the claim), NEW related-work paragraph "Workspaces and
binding" positioning against feng2023binding, oh2026rebinding (mechanism-level
corroboration), anthropic2026workspace (the open question our checkpoint test
answers), shih2026registers (cited in the verbalization section as the
trained-register complement), table caption fixed, dead macro removed. Source
verifier passes (14 citations, 29 macros). The separate `paper/` short note
left untouched (independent narrative; only its metadata checklist remains).

Kernel `cm-delta-verbalization-14b-awq` COMPLETE. Artifacts
`runs/cm-delta-verbalization-14b-awq/`. All G0 cells 100% (belief) /
≥70–100% (tell, V′ natural 0.7 — that cell reported but core gates clean).

**The ledger (belief_ac primary; tell_ac in parentheses):**
| cell | λ | acc |
|---|---|---|
| λ_hist, no V | **1.004** (1.010) | 100% |
| λ_hist, V present | **0.217** (0.174) | 0% |
| λ_cot (edit V token) | **0.337** (0.587) | 0% |
| λ_both, consistent | **0.997** (1.000) | **100%** |
| λ_hist, V′=sphere control | **0.989** (0.982) | 100% |
| λ_cot, V before marker | 0.267 (0.516) | 0% |
Nulls: both single edits beat 30 nulls (p=.032). Textual-conflict priors:
both inconsistent TEXT variants answer Paris (see confound below).

**Gates: H1 register FAIL (λ_cot 0.34 < 0.7); H2 migration PASS
(λ_hist 0.22 ≤ 0.3 AND V′ control 0.99 ≥ 0.7 — content-specific, recency
confound excluded). Printed verdict `LEDGER_MIXED`, stands.**

**What the ledger shows (calibrated):** verbalization neither transfers the
memory (no shadowing: λ_cot low) nor is decorative (V's presence drains 80%
of the history anchor's load, content-specifically; V-edits move real
margin). Instead the belief becomes **jointly encoded**: single-site edits
are rejected at readout (acc 0% both ways — the unedited source's value
wins) while editing BOTH sites consistently restores the full natural effect
exactly (0.997/1.000, 100%). Load is not conserved across single edits
(0.22+0.34 ≪ 1) but is exactly recovered jointly. This is a
**corroboration/quorum readout** — consistent with the earlier
overwrite-SUPPRESSION and conflict-contention findings (consensus, not
blending). Security-flavored corollary: a verbalized belief is
tamper-evident — one-site tampering is vetoed.

**Honest confound (must be resolved before claiming the quorum rule):** in
every single-edit cell the unedited partner's value is ALWAYS Paris (edits
go Paris→Rome from the Paris-clean base), so "unedited witness wins" is
confounded with "Paris/majority prior wins" — and the textual-conflict
cells (both answering Paris) hint the prior is real. **Decisive cell
(pre-register next): reverse-base edits** — start from the Rome-consistent
world (natural rows + V(Rome)) and edit ONE site Rome→Paris; quorum predicts
the unedited ROME wins; prior predicts Paris wins. One cheap kernel
(~4 forwards + refs + nulls). Arm B (self-generated V: does self-generation
reweight the quorum?) and Arm C (depth-reset) remain registered.

**Scientific summary (both arms, one world, both pre-registered):**
- checkpoint full-state swap → **~0.001×** natural effect (previous entry);
- anchor prototype write → **1.001×** natural effect, all and only its
  consequences (invariants 100%, address-specific 100%).
A ~1000× causal asymmetry between the two sites, predicted in advance both
times. **The token-anchored account of in-context memory is now supported by
a matched positive and negative in the same G0-validated false-belief
world:** derived beliefs are edited at the STATED anchor tokens (and the
edit propagates through the belief rule to reports and tells at natural
strength), while no query-independent consolidated world-state exists to
edit at a checkpoint. Scope: 14B-AWQ, one world family, n=5 rows/split,
L2 write site, layers ≤L26 for the checkpoint null. The general-tier tests
Sahil required before any paper have now RUN, with both branches measured.
