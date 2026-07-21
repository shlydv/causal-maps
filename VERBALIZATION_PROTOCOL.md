# Verbalization write-back protocol (Floor 3)

Status: pre-registered 2026-07-15, design frozen before any GPU run.
Stage: `delta_verbalization` (Arm A first). Model: Qwen2.5-14B-AWQ (validated
loader), the G0-validated false-belief world from `delta_structured_workspace`
(report_only surface, narrative split). Positioning: "Do Models Read What They
Write?" (2606.29522) shows trained-to-write models use written states; this
protocol measures, in STOCK instruct models, the LOAD LEDGER — where causal
load lives, how verbalization migrates it, whether it is conserved, and what
capability it buys.

## Quantities

For readout r and token t: causal load
  λ_t(r) = effect(prototype edit z(Rome)−z(Paris) at t on r)
           / effect(full natural counterfactual on r).
Measured baseline (anchor-write run): λ_hist ≈ 1.001, λ_checkpoint ≈ 0.001.
Faithfulness of a verbalized statement V: F = λ_cot / (λ_cot + λ_hist).

## Arms and conditions (Arm A = teacher-forced V; one kernel)

V = "Alice believes the cube is in Paris." inserted between STATECHECK and the
question. n ≥ 10 rows (doubled world rows; new value pairs), 30 norm-matched
nulls per primary cell, wrong-address controls as in the anchor-write run.

Conditions (readouts: belief_ac, tell_ac; each with natural references):
1. no-V baseline (replicates anchor-write; λ_hist re-measured at n≥10);
2. V-consistent, edit a_h only  → λ_hist under V;
3. V-consistent, edit a_v only  → λ_cot;
4. edit both, consistent        → sanity (should ≈ natural);
5. edit both, CONFLICTING (a_h→Rome, V stays Paris; and the reverse)
   → arbitration rule (connects to single-occupancy/contention);
6. irrelevant-V control: V′ = "Alice believes the sphere is in {as}." —
   cube λ_hist must stay ≈ 1.0 (content-specific shadowing; kills recency);
7. V-position variation (early/late in the gap) — positional robustness;
8. textual-inconsistency baselines: natural text where history and V disagree
   (which source wins in TEXT is the behavioral prior the causal ledger is
   read against).

## Frozen gates

G0 per condition ≥ 80% (clean + natural). Primary passes at target_acc ≥ 80%,
positive-fraction ≥ 80%, ratio window [0.6, 1.4] vs the matched natural
reference, nulls p < .04. No prompt repair, no layer search (L2 write site,
frozen), no re-thresholding.

## Hypotheses → verdicts (graded; every branch reportable)

H1 write-back: λ_cot ≥ 0.7  →  VERBALIZED_REGISTER (stock model).
H2 migration: λ_hist (with V) ≤ 0.3 AND irrelevant-V leaves λ_hist ≥ 0.7
   →  WRITE_BACK_SHADOWING.
H3 structure: both ≥ 0.7 → WRITE_BACK_REDUNDANT (conflict cell reports the
   arbitration); λ_cot < 0.3 → COT_DECORATIVE. Mixed → ledger reported as-is.
H4 depth-reset (Arm C, second kernel): with V present, the 14B-failed
   search_ac natural G0 (20–60%) clears ≥ 80%; then edit at a_v propagates to
   the ACTION at ratio ≥ 0.7 → DEPTH_RESET_CONFIRMED. G0 still failing →
   DEPTH_RESET_INELIGIBLE (reported; not rescued).
H5 (Arm B, third kernel): self-generated V (greedy "First, state what Alice
   believes." → re-tokenize → edit the generated location token → teacher-
   force the question). λ_cot(self) vs λ_cot(forced) reported with the same
   gates.

Arm D (phase 2, separate registration): validation of F on a known-unfaithful
regime (bias-injected prompts, Turpin-style): predict F low where the
literature shows decorative CoT.

## Stopping

Arm A verdict is final for teacher-forced claims; no fishing. Arms B/C run
only if Arm A's G0 holds (the V-inserted prompts must elicit ≥80% — new
surface, real elicitation risk, priced as usual). Kernel budget: 3 × ~500 s
on 14B-AWQ; Arm A alone decides H1–H3.
