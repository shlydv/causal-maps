# Protocol: What is native binding missing from our extracted basis?

*Causal Maps · 2026-07-13 · post-`ASYM_INCOMPLETE_BASIS`*
*Status: **COMPLETE — `BIND_MISS_LINEAR_READOUT`.** Residual decodes \(v\); residual ADD does not recover. Site-local. Stopped.*

---

## Scientific question

> **What information exists in native binding that our extracted basis \(\mathrm{span}\{\Delta_u\}\) is missing?**

Not: make binding stronger. Not: bus. Not: does a direction exist.

Context: ADD \(\Delta_v\) installs binding; project-out of span only mildly hurts; site knockout destroys retrieve (\(R_{knock}\approx 4\)). Something at the site is used natively that our dirs do not span.

---

## Competing accounts (one kernel)

After forming residual \(r = h - P_S h\) at (L2, `val_slot`) with \(S=\mathrm{span}\{\Delta_u\}\):

| Account | Claim | Primary test |
|---|---|---|
| **L — more linear dirs** | Missing content is still linear in-slot | Centroid probe on \(r\) decodes \(v\); ADD of residual centroid direction is causal |
| **P — other positions** | Missing content is elsewhere in the prompt | Mean-knock at frozen offsets: off-slot drops ≥ 0.5× val_slot drop |
| **N — nonlinear / local** | Missing content is in-slot but not linear in \(r\) | Probe fails; knock local to val_slot |

---

## Frozen design

**Sites / model:** Qwen2.5-7B 8bit; L2; Variable native prompts; values = existing Δ vocabulary.

**S:** QR span of all per-value \(\Delta_u\) (same extractors as necessity/asymmetry).

**Position set (no sweep menu):**  
`{val_slot-1, val_slot, val_slot+1, last}`  
(skip any out-of-range; require val_slot and last always).

**Probe:** nearest-centroid classifier on \(r\) (train/test split by prompt id, stratified by value). Chance = \(1/|V|\).

**Causal residual ADD:** \(d_v = \bar r_v - \mathrm{mean}_w \bar r_w\), norm-matched to \(\|\Delta_v\|\); ADD at val_slot on held-out prompts after span-project (or on clean — freeze: **on clean held-out**, measure Δpref for gold \(v\)). Primary: span-ablate then ADD \(d_v\) recovers preference.

---

## Gates

- **G0:** native Variable retrieve ≥ 80%.
- **L1 (readout):** probe acc ≥ 0.50 **and** beats label-shuffle null (p < 0.01).
- **C1 (causal):** recovery  
  \(\mathrm{rec} = \dfrac{\mathrm{pref}_{span+ADD\,d_v} - \mathrm{pref}_{span}}{\mathrm{pref}_{clean} - \mathrm{pref}_{span}}\)  
  ≥ 0.50 when denom > 0; else fail C1.
- **P1 (distributed):** \(\max_{p\neq val}\mathrm{drop}(knock_p) \ge 0.5 \times \mathrm{drop}(knock_{val})\).
- **P1-local (support):** \(\mathrm{drop}(knock_{val}) \ge 2\times \mathrm{drop}(knock_p)\) for every other p in the set.

**Verdicts**

| Verdict | Rule |
|---|---|
| `BIND_MISS_LINEAR` | L1 ∧ C1 |
| `BIND_MISS_LINEAR_READOUT` | L1 ∧ ¬C1 |
| `BIND_MISS_DISTRIBUTED` | ¬L1 ∧ P1 |
| `BIND_MISS_NONLINEAR_LOCAL` | ¬L1 ∧ ¬P1 ∧ P1-local |
| `BIND_MISS_UNCLEAR` | else (G0 pass) |
| `BIND_MISS_INELICITABLE` | G0 fail |

One kernel then stop. No layer menu, no extra positions, no probe fishing.

---

## What this is for

If **L**: missing piece is more residual geometry → revisit extraction (why standard Δ missed it).  
If **P**: incomplete-basis asymmetry was partly a single-site illusion.  
If **N**: native bind uses nonlinear local features → hard ceiling for linear dir necessity.  

Later (not this kernel): does the router read the spanned part or the missing part? — couples to protocol.
