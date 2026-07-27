# Paper 2 discovery pilot: shared causal component of address spillover

Status: frozen before the first GPU run, 2026-07-23.

## Question

Paper 1 found a model-family divergence. A neutral-carrier content write at
Alice's belief anchor is address-specific in Qwen and Gemma, but in
DeepSeek-R1-Distill-Llama-8B it changes Alice's intended answer while also
changing an unrelated same-valued belief. The corresponding textual
counterfactual remains address-correct.

This pilot does **not** assume that the natural-minus-synthetic activation
difference is a routing representation. It asks the narrower falsifiable
question:

> Does that difference contain a donor-learned, low-dimensional causal
> component that predicts and controls spillover on held-out worlds?

Routing/address specificity is an interpretation licensed only after the
causal and alternative-explanation gates below pass.

## Frozen model, data, and split

- DeepSeek-R1-Distill-Llama-8B, the same 8-bit dual-T4 recipe as Paper 1.
- The same tokenizer-compatible 30-world structured census.
- First 15 mechanically selected worlds are donors; the remaining 15 are
  evaluation worlds. No evaluation row is used to estimate a component.
- CLEAN has `ac=bc=as=Paris` and `bs=Rome`.
- NATURAL changes only `ac` from Paris to Rome in text.
- SYNTHETIC keeps CLEAN text and adds the validated neutral-carrier
  `z(Rome)-z(Paris)` write at Alice's cube anchor at L2.
- Discovery layers: L2, L8, L16, L24, and L30.

## Component construction

For each frozen layer and pre-query locus, capture NATURAL and SYNTHETIC
residual states on donor worlds and compute

`R = h_natural - h_synthetic`.

The intervention component is the donor mean `mu = mean_rows(R)`. It is a
single fixed vector (or fixed multi-position vector) applied unchanged to all
evaluation worlds. Report its energy fraction, donor cosine consistency, and
the singular-value spectrum of `R`; these are descriptive dimensionality
measurements, not success gates.

Primary loci are the edited anchor, all four belief anchors, all six source
anchors, and the summary span. The marker token is a Paper-1 matched negative.
Seeded size-matched random pre-query loci are complexity controls. No locus,
layer, coefficient, sign, or norm is selected after results are observed.

## Behavioral eligibility

On the evaluation split:

1. CLEAN and NATURAL must answer `belief_ac`, `belief_bc`, and `belief_bs`
   correctly at >=80%.
2. SYNTHETIC must preserve the intended Alice edit (`belief_ac=Rome`) at
   >=80%.
3. SYNTHETIC must reproduce the Paper-1 spillover
   (`belief_bc=Rome`, rather than Paris) at >=80%.
4. SYNTHETIC must retain the legitimate same-valued control
   (`belief_bs=Rome`) at >=80%.

Failure yields `BEHAVIORALLY_INELIGIBLE`; it is not evidence against a causal
component.

## Causal gates

For each frozen layer/locus:

- **Rescue:** add `+mu` to SYNTHETIC. It must restore
  `belief_bc=Paris` at >=80%, while retaining both `belief_ac=Rome` and
  `belief_bs=Rome` at >=80%.
- **Necessity/reverse induction:** add `-mu` to NATURAL. It must induce
  `belief_bc=Rome` at >=80%, while retaining `belief_ac=Rome` and
  `belief_bs=Rome` at >=80%.
- **Specificity:** at least one primary locus must pass both directions and no
  size-matched random locus may pass both.

The two retained-Rome endpoints reject a generic Rome-suppression or confidence
calibration account. Reverse induction rejects a rescue-only compensator.
Held-out application rejects row copying. Random loci reject an arbitrary
distributed perturbation of matched size.

## Verdicts and next decision

- `SHARED_CAUSAL_COMPONENT`: all eligibility gates and at least one
  locus-specific bidirectional causal gate pass.
- `RESCUE_ONLY_COMPONENT`: a primary locus repairs spillover but no primary
  locus induces it in reverse.
- `NONLOCAL_OR_NONSPECIFIC_COMPONENT`: only random/summary-wide controls pass,
  or primary and random loci are indistinguishable.
- `NO_SHARED_CAUSAL_COMPONENT`: eligibility passes but no primary rescue.

A positive pilot licenses a locked confirmation across held-out values and
Qwen/Gemma matched controls, followed by attention-path necessity and targeted
rescue. A negative pilot stops this residual-component hypothesis before a
large model sweep.
