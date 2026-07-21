# Does the Shared Binding Slot Broadcast to a Distributed Receiver Set?

*Causal Maps · 2026-07-13*  
*Status: **PRE-REGISTERED — one held-out-layout broadcast necessity test.***

## Rationale

The completed state timeline found that the queried value-slot state is
necessary through L20, while the final readout is not a sole bottleneck. The
completed direct-edge test found that blocking the final readout's direct edge
to that slot across L20–L26 selectively removes 36.8% of NATURAL and 37.7% of
ADD effect, but no individual edge is necessary. The remaining hypothesis is
that intermediate tokens read the same slot and relay its state to readout.

This test asks the minimal causal question:

> Is access from the queried value slot to the *distributed set of all later
> token positions* necessary for the common NATURAL/ADD effect?

## Held-out layout and operator confirmation

- Qwen2.5-7B-Instruct, 8-bit, seed 0; same single-binding donor codebook and
  L2 `z_target-z_source` intervention as the completed operator.
- The two bindings are written in **reversed textual order**: `Let Y = … . Let
  X = … .` This layout was not used by the completed binding timeline or the
  direct final-readout edge test.
- The 80 deterministic source→target rows balance offsets 1–8. These are not
  treated as an independent mapping-confirmation set; the held-out object here
  is the prompt layout.
- G0 requires CLEAN and NATURAL greedy accuracy at least 80%, positive ADD on
  at least 80% of rows, and ADD/NATURAL effect ratio in [0.70, 1.30].

## Intervention

At every layer L20–L26, block **all attention heads** on every causal query
position strictly after the queried value-slot position from attending to that
slot. This removes the slot's outgoing attention access for the final readout
and every possible intermediate relay token.

The matched control is the identical all-later-query block to the other
variable's value slot. CLEAN, NATURAL, and ADD are each rerun under the same
own-slot or other-slot mask. Effects are condition-minus-CLEAN under the
matched mask.

No per-layer, per-head, receiver-position, or alternate-window results are
computed in this kernel.

## Frozen interpretation

| Verdict | Rule after G0 |
|---|---|
| `SHARED_BROADCAST_ESSENTIAL` | Own block removes ≥80% of both effects and exceeds other-slot loss by ≥50 points for both. |
| `SHARED_BROADCAST_PARTIAL` | Essential criterion fails, but own block removes ≥50% and exceeds control by ≥25 points for both. |
| `DIVERGENT_OR_UNRESOLVED_BROADCAST` | Neither shared rule passes, or only one condition reaches the partial rule. |
| `BROADCAST_INELICITABLE` | G0 fails. |

## Limits

- A pass identifies a distributed **receiver set**, not individual readers.
- A partial result means direct slot access is shared but redundant with another
  route; it does not establish a complete path.
- A null result would conflict with the prior L20 slot-necessity result and is
  grounds for auditing intervention semantics, not expanding a search.
