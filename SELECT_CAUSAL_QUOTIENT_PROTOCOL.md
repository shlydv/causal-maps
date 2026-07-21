# Protocol: Are divergent selection states causally equivalent?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `DELAYED_NATURAL_CONVERGENCE`.***

## Observation to explain

For SELECT flag 0→1:

- ADD recreated the natural L2 flag state (cos .998);
- output effect, L8 flag-state patch effect, and L8 flag-state necessity were
  essentially identical to the natural flag flip;
- but the L8 final/query displacement differed (cos .413, error 1.020).

The central question is whether this geometric divergence is causally silent.

## Competing accounts

1. **Causal quotient:** ADD and natural query states differ mainly in
   downstream-null or task-irrelevant components and are functionally
   interchangeable.
2. **Delayed convergence:** states differ at L8 but converge geometrically at
   later layers before output.
3. **Parallel paths:** distinct query states remain functionally
   non-interchangeable even though they produce the same answer.

## Frozen setup

- Qwen2.5-7B-Instruct.
- Existing `value_of` SELECT template.
- Existing seed-0 8 train / 8 test pairs.
- Existing cross-fitted mean flag 0→1 direction at L2 flag token, α=1.
- No new templates, pairs, layers chosen from results, or α tuning.

Capture CLEAN, natural flag-1, and ADD runs at final/query position for:

\[
L\in\{4,8,12,14,16,20,26\}.
\]

Output metric remains `logit(A value) − logit(B value)`.

## Per-layer interventions

At each fixed layer \(L\), final/query position:

1. patch ADD state into CLEAN;
2. patch natural state into CLEAN;
3. run ADD, overwrite query state with natural state;
4. run natural, overwrite query state with ADD state;
5. run ADD, overwrite query state with CLEAN state;
6. run natural, overwrite query state with CLEAN state.

All states are full residual vectors from real runs; no subspace projection.

## Metrics

- Euclidean displacement cosine/error for ADD vs natural.
- ADD-state / natural-state patch-effect ratio.
- Bidirectional swap deviation:
  output change caused by ADD→natural or natural→ADD state replacement,
  normalized by the natural effect.
- ADD and natural query-state block fractions.

## Gates

### G0 — replication

- clean flag-0 and natural flag-1 greedy accuracy each ≥80%;
- ADD/natural output-effect ratio in [0.90, 1.10].

### D0 — geometric discrepancy

At L8, cosine <0.50 or normalized error >0.80.

### Causal-equivalence layer

A fixed layer qualifies iff:

- geometric discrepancy remains: cosine <0.50 or error >0.80;
- ADD and natural query-state patches into CLEAN are positive, with ratio in
  [0.80, 1.20];
- replacing ADD query state with natural, or natural with ADD, changes output
  by ≤10% of the natural effect in each direction;
- overwriting the query state with CLEAN blocks ≥50% of both ADD and natural
  effects.

### Delayed-convergence layer

A later layer \(L>8\) qualifies iff cosine ≥0.80 and error ≤0.60.

## Verdict

- `CAUSAL_QUOTIENT_EQUIVALENCE` if G0∧D0 and any fixed layer qualifies as
  causally equivalent.
- `DELAYED_NATURAL_CONVERGENCE` if G0∧D0, no causal-equivalence layer
  qualifies, and a later convergence layer exists.
- `PARALLEL_OR_UNRESOLVED_PATHS` if G0∧D0 and neither condition holds.
- `DISCREPANCY_NOT_REPLICATED` if G0 passes but D0 fails.
- `SELECT_INELICITABLE` if G0 fails.

This is a mechanism test, not a license for broad claims from one task. A
positive quotient verdict would motivate a fresh cross-task prediction that
raw activation similarity can understate causal equivalence.

## Result

G0 and D0 passed:

- clean/natural accuracy: 100% / 100%;
- ADD/natural output-effect ratio: .991;
- L8 final/query cosine .413, error 1.020, exactly replicating the atlas
  discrepancy.

The trajectories then converged sharply:

- L16: cosine .619, error .855;
- L20: cosine .986, error .168;
- L26: cosine .993, error .115.

At L20 the ADD and natural query states were also causally interchangeable and
load-bearing: patch-effect ratio 1.002; bidirectional swap deviations .0065 and
.0034; clean overwrite blocked 99.75% and 98.88% of ADD and natural effects.
L26 gave the same pattern.

No geometrically divergent layer satisfied the causal-equivalence gate.
Therefore the preregistered verdict is:

**`DELAYED_NATURAL_CONVERGENCE`.**

Safe interpretation: the L2 selection intervention does not merely find a
parallel route to the same answer. It first creates a natural-like control
state at the flag site, while the query trajectory remains different through
L16, then reaches the same full, causally sufficient query state as the natural
flag change by L20. In this fixed Qwen7B selection task, this is direct evidence
for low-dimensional control followed by delayed reconstruction of native
downstream computation.
