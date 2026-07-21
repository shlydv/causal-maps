# Token-Anchored State preprint

`main.tex` is the new manuscript. It is intentionally separate from the older
`paper/` note so that neither research narrative overwrites the other.

## Rebuild evidence macros

```bash
python3 build_evidence.py
python3 verify_preprint.py
```

`build_evidence.py` currently reads immutable copies of the discovery-pilot
JSON files under `evidence/pilots/`. Confirmatory battery outputs replace the
pilot macros and tables only after their frozen verdicts are audited.

## Build PDF

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The current laptop environment does not provide a TeX engine, so source-level
checks run locally and PDF compilation is deferred to an environment with
TeX Live or to arXiv's compiler.

## Current manuscript status

- Drafted: calibrated abstract, introduction, identification strategy,
  intervention definitions, pilot results, related work, limitations, and
  artifact policy.
- Completed: Qwen-7B and Mistral-7B widened batteries, including full-depth
  checkpoint and late-readout controls. Immutable JSON copies live under
  `evidence/confirmatory/`.
- Deferred by Kaggle's 30-hour weekly quota: Qwen-14B exhaustive anchor,
  full-depth checkpoint trajectory, and grouped cross-surface probe. The frozen
  launch is ready as `cm-preprint-headline-qwen14b-v2` after quota reset.
- Deliberately absent: claims that depend on unfinished confirmation, figures
  that would mix pilot and confirmatory numbers, and the unresolved
  verbalization-quorum interpretation.
