# Validation: Is instruction reachability template-independent?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `VALIDATION_INELICITABLE`.***

## Question

Does a framing operator learned from equal-length execute/data prompt pairs
transfer to an unseen equal-length template family and unseen output
vocabulary, or was the atlas result a fixed-template/position displacement?

## Frozen model and sites

- Qwen2.5-7B-Instruct, 8-bit.
- Extract/inject: L20 final prompt position, α=1.
- Checkpoint: L26 final prompt position.
- No layer, α, or template search.

## Fixed template families

Each family contains both mode labels and changes only the final active-mode
token. The execute and data prompts must tokenize to exactly equal lengths and
final positions; otherwise that family is invalid.

### Train family A

- labels: `run` / `hold`
- data answer: `skip`
- rule: run executes `output the word W`; hold ignores it and emits `skip`.

### Train family B

- labels: `live` / `quote`
- data answer: `text`
- rule: live executes `output the word W`; quote treats it as text and emits
  `text`.

### Held-out family C

- labels: `go` / `see`
- data answer: `none`
- rule: go executes `output the word W`; see does not execute it and emits
  `none`.

The held-out family uses unseen labels, syntax, and negative-answer token.

## Payload split

- Train categories:
  `red, blue, one, two, dog, cat, hot, cold`.
- Held-out categories:
  `apple, false, north, south, open, true, left, right`.

No payload or semantic category crosses the split.

## Directions

### Active operator

For both train families and all train payloads:

\[
\Delta_{\mathrm{active}} =
\operatorname{mean}(h_{\mathrm{execute}}-h_{\mathrm{data}})
\]

at L20 final position.

### Matched mode/template control

Within each train family, set the command payload equal to that family's data
answer (`skip` or `text`), so execute and data modes request the same output.
Extract the corresponding mean mode-label displacement
\(\Delta_{\mathrm{control}}\).

This control captures label/template displacement without changing the desired
answer.

### Random null

Each of 100 draws is **one shared same-norm random vector**, repeated across all
held-out rows. Per-row independent random vectors are forbidden.

## Held-out conditions

On family C data-mode prompts:

1. CLEAN;
2. natural execute-mode counterfactual;
3. ADD \(\Delta_{\mathrm{active}}\);
4. ADD \(\Delta_{\mathrm{control}}\);
5. ADD shared random vector.

Save all per-row and null-draw metrics.

At L26:

- patch active-ADD state into CLEAN;
- patch natural execute state into CLEAN;
- overwrite active ADD with CLEAN state;
- overwrite natural execute with CLEAN state.

## Gates

### G0 — equal-length elicitation

- every family has exactly aligned execute/data token lengths;
- train execute/data greedy accuracy each ≥80%;
- held-out execute emits \(W\) and held-out data emits `none`, each ≥80%.

Fail → `VALIDATION_INELICITABLE`, stop.

### A1 — held-out natural-state equivalence

At L20:

- active cosine ≥0.80;
- active normalized error ≤0.60;
- p<0.01 vs shared-random null;
- active cosine exceeds control cosine by ≥0.30.

### Q1 — held-out downstream trajectory

At L26:

- cosine ≥0.50;
- normalized error ≤0.80;
- p<0.01 vs shared-random null.

### O1 — content-specific output equivalence

- active ADD greedily emits held-out \(W\) on ≥80%;
- natural execute emits \(W\) on ≥80%;
- ADD/natural \(W-\text{none}\) effect ratio in [0.70, 1.30];
- active effect p<0.01 vs shared-random;
- active effect ≥2× absolute control effect.

### M1/M2 — shared mediator

- active/natural L26 patch-effect ratio in [0.70, 1.30];
- CLEAN overwrite blocks ≥70% of both effects;
- block-fraction gap ≤0.20.

### D1 — coarse guard

Active/natural activation norm ratios at L20 and L26 are in [0.8, 1.2].

## Verdict

- `TEMPLATE_INVARIANT_INSTRUCTION_OPERATOR` iff all gates pass.
- `FIXED_TEMPLATE_DISPLACEMENT` if G0 passes but any substantive gate fails.
- `VALIDATION_INELICITABLE` if G0 fails.

Failure closes this exact branch: one shared additive L20 operator, α=1,
transferring across these equal-length mode-template families. It does **not**
prove that no abstract instruction representation exists under other
parameterizations, sites, models, or tasks; those are outside this project's
frozen claim.

Success establishes template-held-out evidence in Qwen7B—not a universal law.
It restores INSTRUCTION as a provisional natural-reachable atlas cell and
licenses fresh template/model-family replication before any broad claim.

## Result

All three families passed the strict alignment preflight: execute/data prompts
had equal lengths and exactly one differing token.

Behavior:

- train execute: 100%;
- train data: 50%;
- held-out execute: 100%;
- held-out data: 100%.

The frozen train-data ≥80% G0 failed, so no direction, intervention, or causal
gate was evaluated. Verdict: `VALIDATION_INELICITABLE`.

This does not falsify all abstract instruction representations. It fails to
elicit the preregistered two-family donor construct, so the original
fixed-template atlas result remains scientifically ambiguous and cannot support
an abstract instruction-operator claim.
