# Reversed-Layout Confirmation of Early Binding Backup Formation

*Causal Maps · 2026-07-13*  
*Status: **PRE-REGISTERED — fixed-prediction structural replication.***

## Purpose

The original-layout backup-formation timeline selected one shared formation
phase: queried-slot outgoing access in L3–L8. This kernel tests only that
fixed prediction in the existing reversed `Y-then-X` binding layout. That
layout independently passed the operator gate and the late outgoing-access
test, but it was not used to select the early window.

This is a structural replication, not an independent blind dataset: the
reversed layout was previously used for a late L20–L26 broadcast measurement.

## Fixed setting

- Qwen2.5-7B-Instruct, 8-bit, seed 0; same L2 controller and donor codebook.
- Reversed binding order: `Let Y = … . Let X = … . What is the value of …?`.
- The 80 deterministic rows balance source→target offsets 1–8.
- Matched CLEAN-state overwrite after L20 at the queried slot (`P`).
- Fixed queried-slot late outgoing block L21–L26 (`L`).
- The sole preregistered formation intervention is queried-slot or other-slot
  all-later-query outgoing block in L3–L8 (`E`).

## Measure

For NATURAL and ADD independently, compute the same controlled recovery
difference used in the discovery timeline:

\[
R_{own}=E(P+L+E_{own})-E(P+E_{own}),\quad
R_{other}=E(P+L+E_{other})-E(P+E_{other}),
\]

and the prevented-recovery fraction
\((R_{other}-R_{own})/R_{other}\).

G0 is the usual operator gate. G1 requires base recovery
`E(P+L)-E(P) ≥ 5` for NATURAL and ADD. Replication requires, for both, positive
other-slot recovery of at least five, non-negative own recovery, and prevented
recovery fraction at least 0.50.

## Verdicts

| Verdict | Rule |
|---|---|
| `REVERSED_LAYOUT_SHARED_EARLY_BACKUP_REPLICATES` | G0/G1 and both NATURAL/ADD meet the fixed early criterion. |
| `REVERSED_LAYOUT_EARLY_BACKUP_NOT_REPLICATED` | G0/G1 but the fixed early criterion fails. |
| `REVERSED_LAYOUT_BACKUP_INELICITABLE` | G0 or G1 fails. |

No other window, recipient, head, or prompt variant is evaluated in this
kernel.
