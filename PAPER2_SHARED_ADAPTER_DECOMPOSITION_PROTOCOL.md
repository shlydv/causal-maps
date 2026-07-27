# Paper 2: Shared-Backbone / Domain-Adapter Decomposition

Status: frozen after the leave-color-out result and before evaluation on the
second fresh color set.

## Hypothesis

The dominant cross-domain component is a shared but causally incomplete
backbone. A lower-energy, domain-specific component acts as an adapter that
makes that backbone causally usable in the target domain.

This hypothesis predicts a compositional interaction:

- the shared projection alone fails;
- the color residual alone fails;
- their sum restores bidirectional route control;
- matched random residuals do not restore it;
- graded addition of the true residual produces an orderly rescue.

## Frozen construction

- Model: Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Locus: all three answer-prefix positions at residual layer 21.
- Reconstruct the same location, ownership, and color controllers from the
  frozen 15-row donor sets.
- Construct the shared PC1 from location and ownership only, exactly as in the
  leave-color-out protocol.
- Orthogonally project the color donor controller onto the shared PC1:
  `P = <C,S>/<S,S> S`.
- Define the color adapter residual `R = C - P`; by construction `<P,R> = 0`.
- No evaluation activations enter P or R.

## Independent evaluation

Use 30 additional color histories, disjoint at the rendered-prompt level from
all earlier clean and natural color histories. Source-target shifts are 4--7
and distractor pairs are selected deterministically from unused ordered pairs.

The aggregate and per-world gates are unchanged from the preceding frozen
protocol: both movements >= 0.05 and >= half the original gap, at least 27/30
positive movements separately in both directions, exact one-sided sign-test
p <= 0.01, and all behavioral/source/depth gates passing.

## Full-curve arms

- shared PC1;
- color projection P;
- residual R alone;
- P + 0.25R;
- P + 0.50R;
- P + 0.75R;
- P + R (the reconstructed color oracle).

Report all cumulative L22--L27 curves. The dose sequence is evaluated without
selecting a coefficient after seeing outcomes.

## Causal controls

At layer-24 readout:

- add R at each single answer-prefix position and at each two-position subset
  on top of P;
- add R at the instruction-label positions on top of P;
- add R at 19 seeded sets of matched identical-token positions on top of P;
- replace R by 19 seeded random residuals, each orthogonal to the shared
  vector separately at every prefix position and matched to R's norm at that
  position, then add each to P.

The reconstructed controller must exceed both empirical nulls at p <= 0.05,
and the instruction-locus incremental rescue must remain below half the full
reconstruction score. All controls preserve the same total residual mass.

## Verdicts

- `COMPOSITIONAL_SHARED_PLUS_DOMAIN_ADAPTER`: target eligible; P and R alone
  fail; P+R passes; P+R exceeds each component by at least 0.02 in
  bidirectional score; both null tests pass; the instruction control is
  specific; and the five dose scores are nondecreasing within tolerance 0.005.
- `DOMAIN_ADAPTER_SUFFICIENT`: R alone passes.
- `SHARED_PROJECTION_SUFFICIENT`: P alone passes.
- `NONMONOTONIC_COMPOSITIONAL_RESCUE`: only P+R passes the main and specificity
  gates but the dose order is not monotonic.
- `COLOR_TARGET_UNRESOLVED`: original behavior or P+R oracle fails.
- `NO_SHARED_ADAPTER_COMPOSITION`: otherwise.

This is a discovery/confirmation bridge. If compositionality passes, its
architecture- and domain-general scope must subsequently be tested without
reusing this color evaluation set.
