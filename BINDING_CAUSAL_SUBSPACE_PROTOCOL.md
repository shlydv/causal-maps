# Is the Binding Controller a Shared Distributed Causal Subspace?

*Causal Maps · 2026-07-13*  
*Status: **PRE-REGISTERED — one held-out rank-curve test.***

## Motivation

The binding affine operator passed the residual-level natural-convergence and
causal-mediation tests, but a four-component L8 head/MLP ablation removed only
about 2% of either effect.  The corresponding policy experiments also found no
small routing-head bottleneck.  These tests reject a sparse final-readout
component account; they do not distinguish a low-dimensional distributed code
from a high-dimensional or nonlinear distributed process.

This protocol asks the next, falsifiable question:

> At the frozen L8 final readout state, does one low-rank residual subspace
> causally carry both the natural binding rewrite and the affine controller?

It is **not** a search over heads, layers, ranks, templates, or models.

## Frozen setting

- Qwen2.5-7B-Instruct, 8-bit, seed 0; same binding/operator harness.
- 40 deterministic two-binding source→target trials and the existing L2
  `z_target - z_source` controller.
- L8 output residual at the final query/readout token.
- Offsets 1/3 are discovery (20 trials); offsets 5/7 are held-out test (20).
- Fixed ranks: \(k\in\{1,2,4,8,16\}\).  One hundred seeded Haar-random,
  orthonormal bases are used at each rank.

## Basis fitting

For every discovery row, capture L8 final-token residuals for CLEAN, NATURAL,
and ADD.  Let

\[
\delta_N=h_N-h_C, \qquad \delta_A=h_A-h_C.
\]

Fit the *single shared basis* \(B_{16}\) from the top right singular vectors
of the uncentered stacked matrix \([\delta_N;\delta_A]\).  The rank-\(k\)
basis \(B_k\) is its first \(k\) columns.  Neither held-out states nor
outputs are used in this fit.

## Causal intervention

For a NATURAL or ADD held-out run, at L8 final token replace its residual
\(h\) by

\[
h' = h - B_kB_k^\top(h-h_C),
\]

where \(h_C\) is the matched CLEAN row's L8 residual.  This removes only the
condition-induced displacement within the learned subspace and preserves the
matched CLEAN state plus the orthogonal displacement.  On CLEAN itself this
operation is the identity, so effects remain measured against an identically
transformed CLEAN baseline.

For every rank, measure loss of NATURAL and ADD output logit effects.  Compare
each loss to the 100 same-rank random-basis losses.  Also apply each learned
basis at the preceding token as a position-matched damage control.

## Gates and verdicts

### G0 — held-out operator replication

- CLEAN and NATURAL greedy accuracy each ≥80%;
- ADD effect positive on ≥80% of rows;
- ADD/NATURAL logit-effect ratio in [0.70, 1.30];
- L8 final displacement cosine (ADD vs NATURAL) ≥0.50.

### Shared causal rank \(k\)

At a fixed preregistered rank, all must hold:

- NATURAL and ADD effect losses are each ≥50%;
- each exceeds all 100 same-rank random-basis losses (p=1/101);
- preceding-token losses are each ≤20%.

### Verdict

| Verdict | Rule |
|---|---|
| `SHARED_LOW_RANK_CAUSAL_SUBSPACE` | G0 and a shared causal rank \(k\le8\) |
| `SHARED_MIDRANK_CAUSAL_SUBSPACE` | G0, no rank ≤8 qualifies, rank 16 qualifies |
| `ONE_SIDED_OR_DIVERGENT_SUBSPACE` | G0 and some learned rank removes ≥50% of only one effect, beyond its random null |
| `HIGH_RANK_OR_NONLINEAR_DISTRIBUTED` | G0 and no fixed rank qualifies |
| `CAUSAL_SUBSPACE_INELICITABLE` | G0 fails |

## Interpretation limits

- A positive result shows a low-dimensional **L8 residual** object that is
  jointly necessary for both effects. It does not identify its upstream
  writers, prove it is the only representation, or generalize to other models.
- A negative result rejects only this linear, rank-≤16, final-token account. It
  remains compatible with a high-rank, nonlinear, earlier-token, or
  cross-position mechanism.
- No post-hoc rank expansion, alternate basis learner, layer search, or
  follow-up intervention is licensed by any outcome of this kernel.
