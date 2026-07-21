# When Does the Binding Controller Enter a Native Causal State?

*Causal Maps · 2026-07-13*  
*Status: **PRE-REGISTERED — one fixed state-interchange timeline.***

## Question

The affine binding controller reproduces the natural rewrite's output and has
native-like L8 residual displacements, while no small L8 head/MLP set is a
causal bottleneck. The question is therefore temporal and state-level:

> Along the binding computation, at what layer and position—if any—are the
> controller and natural rewrite full residual states causally interchangeable?

This is a full-state, matched-run intervention. It does not identify an
upstream circuit, a head, a basis, or a component.

## Frozen setting

- Qwen2.5-7B-Instruct, 8-bit, seed 0, existing affine binding codebook.
- Existing 40 two-binding trials; offsets 1/3 remain discovery reference and
  offsets 5/7 are the 20-row confirmation set used for every verdict.
- NATURAL is the textual target-value rewrite; ADD is CLEAN plus
  `z_target - z_source` at L2 queried value slot.
- Fixed layers: \(L\in\{2,4,8,12,16,20,26\}\).
- Fixed positions at every layer: queried value **slot** and final query
  **readout** token.
- No layer, position, template, state operation, or threshold may be selected
  after observing results.

## Matched state interventions

For each layer/position and each matched row, cache full residual states
\(h_C,h_N,h_A\) from CLEAN, NATURAL, and ADD. Measure six interventions,
always against the original CLEAN output baseline:

1. patch \(h_N\) into CLEAN (natural-state sufficiency);
2. patch \(h_A\) into CLEAN (controller-state sufficiency);
3. replace ADD state with \(h_N\) (N→A swap);
4. replace NATURAL state with \(h_A\) (A→N swap);
5. replace ADD state with \(h_C\) (ADD necessity);
6. replace NATURAL state with \(h_C\) (NATURAL necessity).

All replacement states are actual model states from the exact matched prompt
and row. This avoids interpreting an arbitrary off-manifold vector as a
native state.

## Metrics

At each fixed layer/position report:

- ADD/NATURAL displacement cosine and normalized error;
- sufficiency effects from NATURAL- and ADD-state patches into CLEAN;
- ratio of the two sufficiency effects;
- bidirectional swap deviations from their original ADD/NATURAL effects,
  normalized by the natural effect;
- CLEAN-overwrite block fractions for ADD and NATURAL.

## Gates

### G0 — confirmation baseline

On confirmation trials: CLEAN and NATURAL greedy accuracy each ≥80%, mean
NATURAL and ADD effects are positive, ADD is positive on ≥80% of rows, and
ADD/NATURAL logit-effect ratio lies in [0.70, 1.30].

### Causal interchangeability at a site

A fixed layer/position is causally interchangeable iff all hold:

- both CLEAN patch effects are positive and their ratio is in [0.70, 1.30];
- both bidirectional swap deviations are ≤10% of the natural effect;
- CLEAN overwrite blocks ≥50% of both effects.

Geometric similarity is deliberately not a gate: geometrically distinct
states can be causally equivalent, and vice versa.

## Verdicts

| Verdict | Rule |
|---|---|
| `EARLY_SHARED_CAUSAL_STATE` | G0 and a final-readout site at L2, L4, or L8 is interchangeable |
| `DELAYED_SHARED_CAUSAL_STATE` | G0, no early final-readout site qualifies, and a final-readout site at L12, L16, L20, or L26 qualifies |
| `SLOT_ONLY_SHARED_STATE` | G0, no final-readout site qualifies, but at least one queried-slot site qualifies |
| `ALTERNATIVE_PATHS_OR_UNRESOLVED` | G0 and no fixed site qualifies |
| `BINDING_TIMELINE_INELICITABLE` | G0 fails |

## Interpretation limits

- A shared state means the two conditions are interchangeable and load-bearing
  at that site; it does not prove they used the same earlier writers.
- A delayed verdict supports reconstruction of a common downstream state, not
  a claim that the controller followed the native trajectory from its L2
  insertion onward.
- A negative verdict can reflect a cross-position, nonlinear, or distributed
  mechanism; it does not establish absence of a controller mechanism.
- This kernel does not license a component search. A later communication-versus-
  transformation experiment would require a new protocol and fresh trials.
