# Frozen preprint launches

These commands package only `src/causal_maps/*.py`, version the private Kaggle
code dataset, load each model once, and write per-row JSON artifacts. Run from
the repository root with the repository virtual environment.

## 1. Qwen2.5-7B full confirmation

```bash
.venv/bin/python kernel/run_kaggle.py run delta_preprint_battery \
  --config '{"model_path":"Qwen/Qwen2.5-7B-Instruct","model_key":"qwen7b_confirm","quantization":"8bit","seeds":[0,1,2],"n_matrix":30,"n_entity":30,"n_world":30,"matrix_null":50,"entity_null":30,"anchor_null":99,"run_probe":false}' \
  --kernel-slug cm-preprint-battery-qwen7b-v1 \
  --slug-suffix qwen7b-confirm-v1 --max-wait 14400 --poll 45 \
  --accelerator NvidiaTeslaT4
```

## 2. Mistral-7B independent-family confirmation

```bash
.venv/bin/python kernel/run_kaggle.py run delta_preprint_battery \
  --config '{"model_path":"mistralai/Mistral-7B-Instruct-v0.3","model_key":"mistral7b_confirm","quantization":"8bit","seeds":[0,1,2],"n_matrix":30,"n_entity":30,"n_world":30,"matrix_null":50,"entity_null":30,"anchor_null":99,"run_probe":false}' \
  --kernel-slug cm-preprint-battery-mistral7b-v1 \
  --slug-suffix mistral7b-confirm-v1 --max-wait 14400 --poll 45 \
  --accelerator NvidiaTeslaT4
```

The structured arm automatically uses the largest tokenizer-aligned bucket
from the same frozen 30 candidates. The tokenizer preflight selects 18 Mistral
worlds and all 30 Qwen worlds. This selection is mechanical and logged.

## 3. Qwen2.5-14B headline mechanism run (v2, frozen 2026-07-21)

This run skips already-covered workspace/entity widening and spends the one
model load on the exhaustive anchor census, full-depth checkpoint trajectory,
and grouped cross-surface probe audit. It also runs M5's reverse-base
verbalization control in the same model load. The reverse-base decision rule
was frozen before this launch: Rome must win both one-site reverse edits and
the two-site edit must restore Paris at ratio 0.6--1.4 for
`QUORUM_REPLICATES_REVERSE_BASE`; Paris winning both one-site edits yields
`PARIS_PRIOR_REPLICATES_REVERSE_BASE`; all other eligible outcomes are mixed.
The cell reuses the previously null-validated L2 direction and adds no layer,
coefficient, or direction search.

Tokenizer-only mounted-model preflight (must print `PREFLIGHT_PASS` before the
weight-loading launch):

```bash
.venv/bin/python kernel/run_kaggle.py run delta_preprint_v2_preflight \
  --config '{"model_path":"/kaggle/input/**/14b-instruct-awq/**/config.json","n_world":30}' \
  --model-source qwen-lm/qwen2.5/transformers/14b-instruct-awq/1 \
  --kernel-slug cm-preprint-headline-qwen14b-v2-preflight \
  --slug-suffix qwen14b-headline-v2-preflight --max-wait 1800 --poll 30 \
  --accelerator NvidiaTeslaT4
```

```bash
.venv/bin/python kernel/run_kaggle.py run delta_preprint_battery \
  --config '{"model_path":"/kaggle/input/**/14b-instruct-awq/**/config.json","model_key":"qwen14b_headline","quantization":"awq","seeds":[],"n_world":30,"anchor_null":99,"run_probe":true,"probe_reps":6,"run_quorum":true,"skip":["matrix","entity"]}' \
  --model-source qwen-lm/qwen2.5/transformers/14b-instruct-awq/1 \
  --kernel-slug cm-preprint-headline-qwen14b-v2 \
  --slug-suffix qwen14b-headline-v2 --max-wait 14400 --poll 45 \
  --accelerator NvidiaTeslaT4
```

## 4. Later frozen confirmations and closeout

The DeepSeek-Llama, Gemma, multi-token locus, and final Paper 1 closeout runs
use checked-in JSON configs under `kernel/configs/`; these configs, rather than
copied command strings, are the launch records. Their immutable outputs and
SHA-256 hashes are indexed by
`paper_token_anchored/generated/evidence_manifest.json`. Runtime hardware and
package provenance are recorded in
`paper_token_anchored/generated/runtime_provenance.json`.

The final closeout protocol is `PAPER1_CLOSEOUT_PROTOCOL.md`. It contains the
prespecified pass/fail rules for the exact edited-anchor-only trajectory and
the held-out naturalized surfaces. After its artifact is archived, the full
evidence package is regenerated and checked with:

```bash
python paper_token_anchored/rebuild_evidence.py
python paper_token_anchored/verify_preprint.py
```

## Stopping rule

Do not redesign failed cells during these runs. Pull outputs, classify each
cell as pass/fail/ineligible under the frozen protocol, and update the paper
from artifacts. The final closeout has no rescue prompt, layer, surface, or
model: either its frozen gates pass or the observed boundary is reported and
Paper 1 experimentation stops.
