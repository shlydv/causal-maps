# Token-Anchored State preprint

`main.tex` is the new manuscript. It is intentionally separate from the older
`paper/` note so that neither research narrative overwrites the other.

## Rebuild evidence macros

```bash
python3 rebuild_evidence.py
python3 verify_preprint.py
```

`rebuild_evidence.py` regenerates paired-bootstrap summaries, TeX macros, and
all manuscript figures, then requires the deterministic evidence audit to return
an adjudicated frozen status (`PAPER1_EVIDENCE_FROZEN` or
`PAPER1_EVIDENCE_FROZEN_WITH_BOUNDARY`). `build_evidence.py` reads immutable discovery-pilot JSON under
`evidence/pilots/` and audited confirmatory artifacts under
`evidence/confirmatory/`. The generated manifest records hashes, row integrity,
behavioral exclusions, layer coverage, and final experiment verdicts.

## Build PDF

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# or: tectonic main.tex --keep-logs
```

The checked-in source has also been compiled with Tectonic 0.15.0 and visually
inspected page by page. `verify_preprint.py` checks citation keys, generated
macros, figure presence, frozen evidence status, required claim guardrails,
and common overclaims before release.

## Current manuscript status

- Drafted: calibrated abstract, introduction, identification strategy,
  intervention definitions, pilot results, related work, limitations, and
  artifact policy.
- Completed: Qwen-7B and Mistral-7B widened batteries, including full-depth
  checkpoint and late-readout controls. Immutable JSON copies live under
  `evidence/confirmatory/`.
- Completed on a dual-T4 Kaggle runtime: Qwen2.5-14B exhaustive anchor,
  full-depth checkpoint trajectory, grouped cross-surface probe, and frozen
  reverse-base M5 discriminator. The immutable artifact is
  `evidence/confirmatory/results_delta_preprint_battery_qwen14b_headline.json`.
- Resolved: M5 rejects the proposed verbalization-quorum interpretation; the
  manuscript now reports the Paris/default-prior result rather than rescuing
  the exploratory story.
- Completed: official DeepSeek-R1-Distill-Llama-8B frozen battery. It confirms
  the workspace write and checkpoint/readout dissociation in a Llama-family
  architecture while exposing a real address-specificity failure at the
  structured anchor. The immutable artifact is stored beside the other
  confirmatory JSON files.
- Completed: Gemma-3-12B independently confirms all workspace cells, the
  address-specific structured anchor, inert checkpoint, and late readout.
- Completed: the Qwen2.5-14B multi-token locus curve localizes sufficient
  causal support to source anchors through L32 and rules out marker, local
  summary, edited-anchor-removed, and size-matched random loci.
- Completed/final gate: the exact `ac`-only arm is sufficient through L32.
  The naturalized belief and checkpoint/readout arms replicate, while report
  and unrelated-address arms fail behavioral eligibility. The evidence is
  frozen with this prespecified boundary; no additional Paper 1 model or task
  battery remains.

Development/test dependencies are pinned separately in
`../requirements-dev.txt`; they do not alter the recorded Kaggle scientific
runtime.
