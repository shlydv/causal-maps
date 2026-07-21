# Bridge Audit: L20 State Overwrite versus L21–L26 Outgoing Attention Block

*Causal Maps · 2026-07-13*  
*Status: **PRE-REGISTERED — intervention-semantics audit.***

## Why this audit is needed

The causal-state timeline found that replacing the queried value-slot state
with its matched CLEAN state immediately **after L20** removes 92–93% of both
NATURAL and ADD effects. The later direct-edge and all-later-receiver masks
removed only 37–41%, despite selective other-slot controls. The broadcast
result used a reversed layout and its mask included L20, so it cannot directly
resolve the discrepancy.

This is an audit, not a new mechanism search. It runs all comparisons on the
timeline's original two-binding layout and held-out offsets 5/7.

## Fixed conditions

For CLEAN, NATURAL, and ADD, measure logit effects in each condition below.
ADD is the established L2 queried-slot `z_target-z_source` intervention.

1. **Default baseline:** ordinary model forward pass.
2. **Post-L20 CLEAN-state overwrite:** replace the queried slot's L20 output
   state in NATURAL and ADD with the matched CLEAN L20 state. This exactly
   reproduces the timeline operation.
3. **Custom-mask baseline:** the 4-D attention-mask forward path used by the
   edge experiments, but with no edges blocked. This checks mask equivalence.
4. **Outgoing attention block:** in L21–L26 only (the layers after the L20
   overwrite), block every head and every later causal query from attending to
   the queried value slot. Run the identical other-value-slot block as control.
5. **Combined operation:** post-L20 CLEAN-state overwrite plus the queried
   slot's L21–L26 outgoing block, under the custom mask.

No layout, layer, head, position, or alternate attention window is swept.

## Gates and verdicts

- G0 is the established operator/behavioral gate on these held-out rows.
- Custom-mask baseline effects must each lie within 5% of the default baseline.
- The post-L20 overwrite must remove at least 80% of both effects in both the
  default and custom-mask paths to count as reproduced.

| Verdict | Rule after G0 |
|---|---|
| `OUTGOING_CHANNEL_RECONCILED` | Mask-equivalence and overwrite-reproduction gates pass, and the own-slot outgoing block removes ≥80% of both effects with ≥50-point advantage over control. |
| `PATCH_MASK_DISSOCIATION` | Mask-equivalence and overwrite-reproduction gates pass, but the outgoing block fails the reconciled rule. |
| `MASK_SEMANTICS_UNVERIFIED` | Custom-mask baseline differs from default by >5%. |
| `TIMELINE_OVERWRITE_NOT_REPRODUCED` | The L20 overwrite replication fails. |
| `BRIDGE_INELICITABLE` | G0 fails. |

## Interpretation limit

`PATCH_MASK_DISSOCIATION` does not demonstrate a hidden component. It shows
that state replacement and edge deletion have different causal effects under
matched conditions, consistent with attention re-normalization or redundant
information elsewhere in the prompt. It licenses neither a head search nor a
claim of a complete circuit.
