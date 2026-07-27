# Paper 2 discovery: conditional backup routing

Status: frozen before GPU output, 2026-07-24.

## Question

The L21 source state causally controls belief, tell, and search. The frozen
top-eight late heads mediate 59–81% of that effect, far above random clamps,
but leave a residual route. Is the residual completed by a small shared set of
heads, or by operation-specific backup routes?

## Frozen design

- Qwen2.5-7B-Instruct, 8-bit, Tesla T4.
- Same 30 compatible worlds, split before output:
  - first 15 for conditional head ranking;
  - final 15 for untouched bidirectional evaluation.
- Queries: `belief_ac`, `tell_ac`, `search_ac`.
- Upstream intervention: matched CLEAN/NATURAL Alice/cube source state after
  L21.
- Base blockade: the previously frozen top eight.
- Backup candidates: every remaining attention-output head at L22-L24. L21 is
  excluded because the source intervention occurs after L21.

For each query independently, on discovery rows only:

1. Apply the forward NATURAL-source-into-CLEAN intervention.
2. Clamp the frozen top eight to CLEAN.
3. Add one candidate head clamp to CLEAN.
4. Rank candidates by reduction in the absolute residual target-logit ratio.

Freeze nested complements K in {1,2,4,8}. On untouched evaluation rows, test
each complement together with the original top eight in both causal
directions. Use the existing mediation gate: at least 70% removal in both
directions and at least 80% restoration of both originating endpoints.

## Specificity and transfer

- For each query, compare its K=8 complement with nine seeded size-matched
  random complements drawn outside that query's frozen top eight candidates.
- Test every query-selected K=8 complement on every evaluation query,
  producing a 3×3 cross-query causal transfer matrix.
- No candidate is re-ranked from evaluation or cross-query output.

## Verdict

- `SHARED_SPARSE_COMPLEMENT`: at least one frozen K=8 complement passes all
  three evaluation queries and no random complement passes.
- `SPARSE_QUERY_COMPLEMENTS`: all three diagonal K=8 cells pass, no single
  complement passes all three queries, and no random complement passes.
- `PARTIAL_BACKUP_LOCALIZATION`: at least one but not all diagonal K=8 cells
  passes, with no random pass.
- `RESIDUAL_ROUTE_DISTRIBUTED_OR_OUTSIDE_HEAD_OUTPUTS`: no diagonal K=8 cell
  passes.
- `NONSPECIFIC_BACKUP`: any matched random complement passes.
- `SOURCE_SITE_INELIGIBLE`: an evaluation L21 source intervention fails.

This is a discovery screen. A sparse result must be followed by locked
confirmation and source-edge necessity/rescue before circuit-level claims.
