# Protocol: What does the router read — span write or ambient residual?

*Causal Maps · 2026-07-13 · post-`BIND_MISS_LINEAR_READOUT` + `PROTOCOL_COMPATIBLE`*
*Status: **DEMOTED — `ROUTER_READ_AMBIGUOUS`.** OOD control failed. Do not claim residual-read.*

---

## Scientific question

> **When routing works on bound values, does \(\Delta_{\mathrm{route}}\) consume the part of the write in \(\mathrm{span}\{\Delta_u\}\), the ambient residual that still linearly decodes \(v\), or both?**

Couples incomplete binding (readable residual outside span) to protocol interoperability.

---

## Design (frozen)

**Surface:** native protocol carrier (real text bindings), flag=0, inject \(\Delta_{\mathrm{route}}\) at flag digit (L2) — same geometry as `delta_protocol`.

```
Let X = u. Let Y = w. If flag=1 output X else Y. flag=0. What is the answer?
Answer =
```

**Write-site edits** at (L2, X val_slot) and (L2, Y val_slot), \(S=\mathrm{span}\{\Delta_u\}\):

| Cond | Write-site op | + route |
|---|---|---|
| **FULL** | none (native) | yes |
| **SPAN** | keep \(P_S h\) only (`h ← P_S h`) | yes |
| **RES** | keep residual only (`h ← h - P_S h`) | yes |
| **BASE** | none | no |
| **EMPTY** | none (optional: v0=v0 carrier) | yes |

**Primary metric:** routing sensitivity on that write state  
\(\mathrm{RS} = \mathrm{pref}(u)-\mathrm{pref}(w)\) under route minus same without route  
(for FULL/SPAN/RES: \(\mathrm{RS}_\bullet = \mathrm{pref}_{\bullet+route} - \mathrm{pref}_{\bullet}\) with \(\bullet\) write edit held fixed; BASE = no-route native).

Simpler freeze:  
\(\mathrm{pref} = \mathrm{logit}(u)-\mathrm{logit}(w)\).  
\(\mathrm{RS}_{full} = \mathrm{pref}_{FULL} - \mathrm{pref}_{BASE}\).  
\(\mathrm{RS}_{span} = \mathrm{pref}_{SPAN} - \mathrm{pref}_{BASE}^{span}\) where BASE\(^{span}\) = span-keep without route.  
Same for RES.

**Null:** random same-norm dir at flag instead of \(\Delta_{\mathrm{route}}\) on FULL write (and report on SPAN/RES).

---

## Gates

- **G0:** native flag0→Y and flag1→X ≥80% (same as protocol P0). Else `ROUTER_READ_INELICITABLE`.
- **G1:** \(\mathrm{RS}_{full} > 0\), p < 0.01 vs null — route works on intact native write.
- **SPAN:** \(\mathrm{RS}_{span} \ge 0.5 \times \mathrm{RS}_{full}\) and \(\mathrm{RS}_{span} > 0\) and p < 0.01 vs null on SPAN surface.
- **RES:** \(\mathrm{RS}_{res} \ge 0.5 \times \mathrm{RS}_{full}\) and \(\mathrm{RS}_{res} > 0\) and p < 0.01 vs null on RES surface.

**Verdicts**

| Verdict | Rule |
|---|---|
| `ROUTER_READS_SPAN` | G1 ∧ SPAN ∧ ¬RES |
| `ROUTER_READS_RESIDUAL` | G1 ∧ RES ∧ ¬SPAN |
| `ROUTER_READS_BOTH` | G1 ∧ SPAN ∧ RES |
| `ROUTER_READS_NEITHER` | G1 ∧ ¬SPAN ∧ ¬RES |
| `ROUTER_READ_WEAK` | not G1 (route barely moves native) |
| `ROUTER_READ_INELICITABLE` | G0 fail |

One kernel then stop. No layer expansion. No bus claim unless SPAN (or BOTH with SPAN-dominant) clearly lands — still prefer naming the readout account only.

---

## Interpretation cheat-sheet

- **SPAN:** protocol language = our extracted causal install subspace; ambient residual is epiphenomenal for routing.
- **RESIDUAL:** Δ’s install something *correlated* with what routing uses; biggest reframing.
- **BOTH:** redundant codes.
- **NEITHER:** protocol success was not carried by either slice under these ops — rethink.
