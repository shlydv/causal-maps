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

## 3. Qwen2.5-14B headline mechanism run

This run skips already-covered workspace/entity widening and spends the one
model load on the exhaustive anchor census, full-depth checkpoint trajectory,
and grouped cross-surface probe audit.

```bash
.venv/bin/python kernel/run_kaggle.py run delta_preprint_battery \
  --config '{"model_path":"/kaggle/input/**/14b-instruct-awq/**/config.json","model_key":"qwen14b_headline","quantization":"awq","seeds":[],"n_world":30,"anchor_null":99,"run_probe":true,"probe_reps":6,"skip":["matrix","entity"]}' \
  --model-source qwen-lm/qwen2.5/transformers/14b-instruct-awq/1 \
  --kernel-slug cm-preprint-headline-qwen14b-v1 \
  --slug-suffix qwen14b-headline-v1 --max-wait 14400 --poll 45 \
  --accelerator NvidiaTeslaT4
```

## Stopping rule

Do not redesign failed cells during these runs. Pull outputs, classify each
cell as pass/fail/ineligible under the frozen protocol, and update the paper
from artifacts. Llama/Gemma confirmation is launched only after these three
outputs establish which exact headline cell is eligible to freeze across the
new family.
