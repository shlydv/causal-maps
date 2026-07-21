# Literature Map — where our work sits in the activation-control field
*Prepared 2026-07 for the Causal Maps project. Honest positioning, not a pitch.*

## TL;DR (read this first)

The vision "read the model's internal dials and turn them to control behavior" is **not open territory** — it is the explicit program of a crowded, fast-moving subfield: **representation engineering / activation steering**. There is a 2025 survey and dozens of 2025–2026 papers, including ones that already do the two moves we thought were our novel swings:

- **Generating a steering vector from a natural-language description** → **HyperSteer** (2506.03292); weight-space analog **Text-to-LoRA** (2506.06105).
- **Predicting which concepts are linearly steerable** → **Linear Accessibility Profile / "Predicting Where Steering Vectors Succeed"** (2604.15557), **"The Geometric Canary"** (2604.17698), **"When is Your LLM Steerable?"** (2606.11599).

Our **extraction method is standard** (difference-in-means / function vectors — Todd et al. 2024; ActAdd — Turner 2308.10248; CAA — Rimsky). We did **not** invent a new mechanism.

**What is genuinely ours** (ranked, honest):
1. A **computational-type axis** for the steerability divide (*retrieve/route* vs *create/designate*), which gives a **semantic explanation** for what the geometric predictors (LAP) only measure.
2. **Rigor + depth on one primitive family**: pre-registered gates, null controls, "failed gate is a result," and a full arc (decompose → centroid control → multi-slot → cross-skill → capacity → boundary) — unusual in a literature the field itself calls "unreliable and often cherry-picked."
3. The **two-regime finding incl. instruction/data**: obey-vs-quote status patterns with *value creation* (late/weak), not *routing* (early/strong).

**Honest size:** a solid, well-organized mechanistic-interpretability paper that contributes the *semantic axis* the steerability literature is missing — respected and publishable. **Not** a paradigm-founding breakthrough; the paradigm is already founded.

---

## The landscape by branch

| Branch (what it enables) | Key work | Settled | Still open |
|---|---|---|---|
| Human turns concept "knobs" | SAE feature steering / Golden Gate; "mechanistic knobs" (2601.02978) | knobs exist, demoed at scale | reliability, side-effects |
| A behavior = one linear direction | **Refusal** (Arditi 2406.11717) | canonical result | most behaviors messier — see below |
| Not every behavior is cleanly steerable | "What Can We Actually Steer?" (2511.18284); "More to refusal than one direction" (2602.02132) | steering is **behavior-dependent & often unreliable** | *why* — the mechanism of the divide |
| **Predict** which concepts are steerable | **LAP / Predicting Where SVs Succeed** (2604.15557); Geometric Canary (2604.17698); When is Your LLM Steerable (2606.11599); geometric-predictor limits (2602.17881) | **geometric predictors exist** (A_lin predicts diff-of-means success ρ≈0.9) | **semantic** account: *which kinds of computation* are linear |
| Compose skills as vectors | Function vectors compose (Todd 2024); Steer2Adapt (2602.07276); task arithmetic | composition works; interference known | clean capacity/interference laws per skill type |
| **Generate direction from intent** | **HyperSteer** (2506.03292); Text-to-LoRA (2506.06105) | **done, zero-shot to unseen prompts** | type-/structure-conditioned generation |
| Read internal state (detector) | Task-drift "Are you still on track" (2406.00799); PIShield (2510.14005); hidden-state IPI probes | injection **detectable from activations** | robustness to adaptive attacks |
| Instruction/data separation | **ASIDE** (2503.10566, *architectural*); "Do LLMs know when they follow instructions" (2410.14516); instruction-following steering (2410.12877) | separability established; architectural defense works | **causal, retraining-free** obey-dial; is it strong enough to *defend*? |
| One model steers another | cross-model SV transfer (2507.12638); activation transport; latent multi-agent (2605.28214) | transfer possible but **limited**; latent attacks exist | reliable model→model control |
| Limits / risks | Non-surjective steering (2604.09839); steering→emergent misalignment (2606.08682) | steering **cannot reach every state** and can *cause* misalignment | safe operating envelope |

**Takeaway:** almost every branch has a 2025–2026 footprint. The field's own open wound is **reliability** — steering works for some things, fails for others, and the *predictors are geometric, not semantic*.

---

## The two results that most constrain us

### 1. HyperSteer ⇒ "intent→direction" is taken
HyperSteer trains a hypernetwork to map a NL steering prompt (+ optional activations) to a steering vector, beating per-task methods and generalizing zero-shot. **So "compile a description into a dial" is not our novelty.** What HyperSteer does *not* do: condition on the **computational type** of the skill, or explain *which* skills are reachable. That gap is where a type-conditioned or type-predictive angle can still live (see the experiment plan).

### 2. LAP / steerability prediction ⇒ "boundary map" is half-taken
LAP predicts diff-of-means steering success per layer from a linear-accessibility measure, across 24 concept families and 5 models, with a three-regime story (nonlinear-encoded ⇒ steering fails; linear-encoded ⇒ diff-of-means works). **This is the general form of "some computations are linear dials, some aren't."** What LAP does *not* provide: a **semantic taxonomy** — it measures A_lin but doesn't say *what kind of computation* is linear vs not. Our Store/Select (early/strong) vs Transform/Instruction (late/weak) result, framed as **retrieve/route vs create/designate**, supplies exactly that missing semantics. **Position us as complementary to LAP, not competing.**

---

## Method-novelty verdict (answer to "is our *way* different?")

- **Extraction mechanism:** standard. Difference-in-means at a token/layer = Arditi/Todd/CAA. Not novel.
- **What is different:**
  1. **Organizing axis** — computational role (store/route/create/designate), not behavior (refusal/sentiment) or pure geometry (A_lin). This is the one genuinely fresh conceptual contribution.
  2. **Experimental rigor** — pre-registration, matched nulls beside every metric, graded verdicts, "harness-bug-first," "failed gate is a result." The steering literature is criticized for the opposite; doing it cleanly is a real (if unglamorous) contribution.
  3. **Depth on one family** — the decompose→compose→capacity→boundary arc on binding directions, rather than a single steering demo.
- **So:** we did not find a *better mechanism*. We are doing the *same mechanism with more discipline, on a better axis, to greater depth*. That is a legitimate paper — just name it honestly.

---

## Calibrated size

- The "control AI minds" vision: **real and important, but the field's, not ours.**
- Our realistic prize: a **clean mechanistic paper** whose contribution is the **semantic axis behind the steerability divide** + a rigorous composition/capacity/boundary arc + the two-regime instruction/data result. A good conference/workshop paper. **Not** a breakthrough.
- The one way it gets genuinely bigger: turn the four data points into a **predictive theory** — *classify a new skill by computational type and predict its steerability before measuring* (see experiment plan). A confirmed a-priori prediction is what separates "we observed a pattern" from "we have a theory," and it's the part LAP/HyperSteer have *not* framed semantically.

---

## Sources
RepE survey 2502.17601 · Function vectors (Todd 2024) functions.baulab.info · ActAdd 2308.10248 · Refusal-direction 2406.11717 · More-than-one-direction 2602.02132 · What Can We Actually Steer 2511.18284 · Steering unreliability/limits 2602.17881 · Predicting Where SVs Succeed 2604.15557 · Geometric Canary 2604.17698 · When is Your LLM Steerable 2606.11599 · Steer2Adapt 2602.07276 · HyperSteer 2506.03292 · Text-to-LoRA 2506.06105 · ASIDE 2503.10566 · Do LLMs know they follow instructions 2410.14516 · Instruction-following steering 2410.12877 · Task drift 2406.00799 · PIShield 2510.14005 · Cross-model SV transfer 2507.12638 · Latent multi-agent 2605.28214 · SAE mechanistic knobs 2601.02978 · Non-surjective steering 2604.09839 · Steering→emergent misalignment 2606.08682
