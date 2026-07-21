# Causal Site Maps Can Fail to Replicate While Residual Directions Transfer

ArXiv-ready short-note package built from the frozen causal-maps artifacts.

## Publication verdict

The result remains worth publishing as a short methodological note after two independent literature reviews through 12 July 2026. Difference-of-means activation directions, task vectors, function vectors, and large cross-template steering studies are established. The distinct result here is the paired failure/success at the same frozen causal site:

- full layer-by-position patching maps fail a pre-registered template-replication gate;
- directions extracted at the causal sites transfer to held-out surfaces at approximately within-template strength.

The paper claims a **transferable intervention direction**, not that directions are new or that the model's unique natural mechanism literally lives in the identified direction. See [`LITERATURE_AUDIT.md`](LITERATURE_AUDIT.md).

## Contents

- `main.tex`: seven-page manuscript.
- `main.pdf`: compiled review copy.
- `references.bib`: literature through July 2026, including intervention-faithfulness and vector-invariance work.
- `build_figures.py`: reads frozen JSON artifacts and regenerates all figures and TeX result macros.
- `figures/`: publication PDF figures and X-ready PNG versions.
- `generated/results_summary.json`: machine-readable numbers used by the manuscript.
- `generated/results_macros.tex`: generated TeX macros; do not edit manually.
- `X_THREAD.md`: ready-to-post thread and single-post version.
- `LITERATURE_AUDIT.md`: novelty and claim-strength assessment.

## Rebuild

From the repository root:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=paper/generated/.mpl \
  .venv/bin/python paper/build_figures.py

cd paper
tectonic main.tex --keep-logs --keep-intermediates
```

The figure builder fails if the frozen verdicts no longer match the manuscript assumptions.

## Artifact provenance

| Manuscript result | Frozen source |
|---|---|
| P0 instrument calibration | `runs/p0_binding/results_p0_binding.json` |
| P1 map failures and top sites | `runs/p1_7b/results_p1_both_7b.json` |
| Pairwise map/column correlations | `runs/p1_7b/fragility_{variable,completion}_p1.json` |
| Variable 7B direction transfer | `runs/delta_transfer/results_delta_transfer.json` |
| Layer, alpha, embedding controls | `runs/delta_var_robust/results_delta_var_robust.json` |
| Wrong-value and anti-direction controls | `runs/delta_var_shufflefix/results_delta_var_shufflefix.json` |
| Completion transfer | `runs/delta_completion/results_delta_completion.json` |
| Variable 1.5B transfer | `runs/delta_var_1p5b/results_delta_var_1p5b.json` |
| Cross-position transfer | `runs/delta_var_crosspos/results_delta_var_crosspos.json` |

The append-only experimental record is `../CAUSAL_MAPS_LOG.md`.

## Figure captions

1. **Coordinate system:** full layer-by-position map correlation versus expected-site-column correlation across templates.
2. **Transfer ratios:** every primary target's cross-template effect divided by its within-template effect.
3. **Controls:** matched, unrelated-value, negative, and embedding-direction mean held-out effects.

## Proposed abstract

Activation patching is commonly summarized as a layer-by-position map. We test whether replication of that map is necessary for a reusable intervention. In Qwen2.5-7B-Instruct, full patching maps for two elicited micro-skills fail a pre-registered template-disjoint replication gate. Directions extracted at their causal sites nevertheless transfer to held-out surfaces at approximately within-template strength. The same Variable protocol separately passes at 1.5B, and its 7B direction transfers in both directions across a 13-token position shift. An embedding control fails and the negative direction hurts, but an unrelated-value direction also transfers. The supported object is therefore a coarse, signed, layer-computed slot-update intervention, not a value-specific binder. This is a controlled counterexample to “non-replicating site-map implies no reusable intervention,” not evidence that the added direction is the model's unique natural mechanism.

## Preflight checklist

- [x] Every plotted and headline number is generated from frozen artifacts.
- [x] The P1 failures remain visible and are not reframed as passes.
- [x] `GENERIC_BOOST` is stated in the abstract, results, limitations, and X thread.
- [x] Completion is described as 7B-only; Variable carries the 1.5B scale result.
- [x] Position-free is limited to the tested 27-to-40 inert-prefix shift.
- [x] Direction novelty is disclaimed; task/function vector and CAA work is cited.
- [x] Intervention-faithfulness limits from Makelov et al. and Grant et al. are explicit.
- [x] The PDF builds without TeX or BibTeX warnings.
- [ ] Add stable paper and repository URLs to `X_THREAD.md`.
- [ ] Confirm author name, affiliation, ORCID, and contact metadata.
- [ ] Choose arXiv category (recommended primary: `cs.LG`; secondary: `cs.CL`).
- [ ] Make the code/artifact repository public or replace the repository promise.
- [ ] Human proofread the final PDF and figure accessibility.
- [ ] Upload to arXiv and inspect the rendered source build before submission.
- [ ] Post the X thread only after the stable paper URL resolves.
