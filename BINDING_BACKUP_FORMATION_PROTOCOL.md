# When Is the Compensatory Binding State Formed?

*Causal Maps · 2026-07-13*  
*Status: **PRE-REGISTERED — fixed causal backup-formation timeline.***

## Premise

On the original held-out binding layout, a matched post-L20 CLEAN-state
overwrite at the queried value slot removes 92–93% of NATURAL/ADD effect. A
queried-slot all-later outgoing attention block over L21–L26 removes only
40–41%, and combining it with the overwrite retains 24–26% of the effect. The
bridge audit verified that this is neither a custom-mask nor timing mismatch.

The most constrained explanation is that some target-relevant state was copied
from the queried value slot before L20. A post-L20 clean overwrite then emits a
misleading competing signal; blocking its late reads exposes part of the
earlier copy.

## Question

> During which fixed pre-L20 phase, if any, is the state required for this
> late-block recovery formed?

## Frozen setting

- Qwen2.5-7B-Instruct, 8-bit, seed 0; original two-binding template and
  timeline confirmation rows (offsets 5/7).
- NATURAL is the target-value rewrite. ADD is CLEAN plus the established L2
  queried-slot `z_target-z_source` controller.
- A matched CLEAN-state overwrite is always applied immediately after L20 at
  the queried value slot (`P`).
- The late queried-slot outgoing block is fixed to L21–L26 (`L`).
- Three and only three pre-L20 formation windows are tested:
  `early = L3–L8`, `middle = L9–L14`, and `late = L15–L20`.

## Difference-in-differences intervention

For each formation window `E_i`, run four conditions:

1. `P + E_i(own)`;
2. `P + L + E_i(own)`;
3. `P + E_i(other)`;
4. `P + L + E_i(other)`.

`E_i(own)` blocks all heads from every later causal query to the queried value
slot over the specified pre-L20 window. `E_i(other)` is the exact same mask to
the other value slot. In both `P+L` conditions, L always blocks the queried
slot. CLEAN, NATURAL, and ADD are rerun under every matched mask.

For each effect type, recovery under a control is

\[
R = E(P+L+E_i)-E(P+E_i),
\]

and selective formation dependence is the difference in recoveries

\[
D_i = R_{other}-R_{own}.
\]

This controls the generic impact of an early outgoing-edge block before asking
whether it prevents the late-block recovery.

## Gates and verdicts

- G0 is the established behavioral/operator gate.
- G1 requires the base late block to produce positive recovery of at least five
  logit units for both NATURAL and ADD: `E(P+L)-E(P) ≥ 5`.
- A window is a shared formation window iff, for both NATURAL and ADD,
  `R_other ≥ 5`, `R_own ≥ 0`, and `D_i / R_other ≥ 0.50`.
- A window is partial iff the same rules hold with `D_i / R_other ≥ 0.25`.

| Verdict | Rule |
|---|---|
| `LOCALIZED_SHARED_BACKUP_FORMATION` | G0/G1 and exactly one shared formation window. |
| `MULTIPHASE_SHARED_BACKUP_FORMATION` | G0/G1 and two or more shared formation windows. |
| `PARTIAL_OR_UNRESOLVED_BACKUP_FORMATION` | G0/G1 but no shared window; at least one partial window or no selective result. |
| `BACKUP_FORMATION_INELICITABLE` | G0 or G1 fails. |

## Limits

- Passing identifies a time window in which outgoing access from the slot is
  necessary for the recoverable copy. It does not identify a recipient token
  or head.
- It does not prove that all backup information is created in that window.
- No receiver-position, head, or additional-window sweep is licensed here.
