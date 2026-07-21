"""Cross-fitted native-trajectory and causal-mediation gate.

Frozen design: NATIVE_TRAJECTORY_PROTOCOL.md.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import variable_pairs
from .logutil import log
from .model_utils import (get_decoder_layers, input_device,
                          load_model_and_tokenizer, logit_diff)
from .nulls import permutation_pvalue
from .patching import _split_output
from .tensorize import tensorize_pairs

INJECT_LAYER = 2
MEDIATION_LAYER = 8
TRAJECTORY_LAYERS = (2, 4, 8, 14)
N_NULL = 100
EPS = 1e-8


def _subset_tensor(x, idx):
    return x[torch.as_tensor(idx, dtype=torch.long)]


def _cos_rows(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=1, eps=EPS)


def _error_rows(candidate, native):
    return ((candidate.float() - native.float()).norm(dim=1)
            / native.float().norm(dim=1).clamp(min=EPS))


@torch.no_grad()
def _forward(model, ids, am, positions, capture_layers=(),
             add=None, patch=None):
    """Forward with optional add/patch operations and sparse CPU captures.

    add: (layer, position, [D] or [B,D]), or a list of such additions.
    patch: (layer, position, [B,D])
    """
    layers = get_decoder_layers(model)
    handles = []
    cache = {}

    if add is not None:
        additions = add if isinstance(add, list) else [add]
        for layer_idx, position, delta in additions:
            def add_hook(module, inp, out, position=position, delta=delta):
                hs, rebuild = _split_output(out)
                hs = hs.clone()
                d = delta.to(device=hs.device, dtype=hs.dtype)
                hs[:, position, :] = hs[:, position, :] + d
                return rebuild(hs)

            handles.append(layers[layer_idx].register_forward_hook(add_hook))

    if patch is not None:
        layer_idx, position, value = patch

        def patch_hook(module, inp, out):
            hs, rebuild = _split_output(out)
            hs = hs.clone()
            hs[:, position, :] = value.to(device=hs.device, dtype=hs.dtype)
            return rebuild(hs)

        handles.append(layers[layer_idx].register_forward_hook(patch_hook))

    def capture_hook(layer_idx):
        def hook(module, inp, out):
            hs, _ = _split_output(out)
            cache[layer_idx] = hs[:, positions, :].detach().float().cpu()
        return hook

    for layer_idx in capture_layers:
        handles.append(layers[layer_idx].register_forward_hook(
            capture_hook(layer_idx)))

    try:
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
        logits = out.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits, cache


def _ld(logits, pos_ids, neg_ids):
    return logit_diff(logits, pos_ids.cpu(), neg_ids.cpu()).float()


def _value_directions(native_disp_l2, metas, train_idx):
    by_value = {}
    for i in train_idx:
        by_value.setdefault(metas[i]["val_cf"], []).append(native_disp_l2[i])
    return {value: torch.stack(rows).mean(0) for value, rows in by_value.items()}


def _row_directions(direction_map, metas, idx, wrong=False):
    values = sorted(direction_map)
    next_value = {value: values[(j + 1) % len(values)]
                  for j, value in enumerate(values)}
    rows = []
    for i in idx:
        value = metas[i]["val_cf"]
        rows.append(direction_map[next_value[value] if wrong else value])
    return torch.stack(rows)


def _random_like_rows(delta, generator):
    random = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
    random = random / random.norm(dim=1, keepdim=True).clamp(min=EPS)
    return random * delta.float().norm(dim=1, keepdim=True)


def _append_layer_metrics(store, clean_cache, cf_cache, add_cache):
    for layer_idx in TRAJECTORY_LAYERS:
        for site_idx, site in enumerate(("val_slot", "last")):
            native = cf_cache[layer_idx][:, site_idx] - clean_cache[layer_idx][:, site_idx]
            induced = add_cache[layer_idx][:, site_idx] - clean_cache[layer_idx][:, site_idx]
            key = f"L{layer_idx}_{site}"
            store.setdefault(key, {"cos": [], "error": [], "norm_ratio": []})
            store[key]["cos"].extend(_cos_rows(induced, native).tolist())
            store[key]["error"].extend(_error_rows(induced, native).tolist())
            ratio = (add_cache[layer_idx][:, site_idx].norm(dim=1)
                     / cf_cache[layer_idx][:, site_idx].norm(dim=1).clamp(min=EPS))
            store[key]["norm_ratio"].extend(ratio.tolist())


@torch.no_grad()
def run_delta_trajectory(model_path, out_dir, quantization="8bit",
                         device_map=None, seed=0, n_null=N_NULL):
    if n_null < 100:
        raise ValueError("native-trajectory p<.01 gate requires n_null >= 100")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    dev = input_device(model)
    ci = batch["clean"]["input_ids"].to(dev)
    cam = batch["clean"]["attention_mask"].to(dev)
    fi = batch["cf"]["input_ids"].to(dev)
    fam = batch["cf"]["attention_mask"].to(dev)
    pos_ids = batch["pos_ids"]
    neg_ids = batch["neg_ids"]
    val_pos = int(batch["anchors"]["val_slot"])
    last_pos = int(batch["S"] - 1)
    positions = (val_pos, last_pos)
    templates = batch["templates"]
    metas = batch["metas"]

    log(f"delta_trajectory: n={len(templates)} inject=L{INJECT_LAYER} "
        f"mediate=L{MEDIATION_LAYER} positions={positions} n_null={n_null}")

    clean_logits, clean_cache = _forward(
        model, ci, cam, positions, TRAJECTORY_LAYERS)
    cf_logits, cf_cache = _forward(
        model, fi, fam, positions, TRAJECTORY_LAYERS)
    clean_ld = _ld(clean_logits, pos_ids, neg_ids)
    cf_ld = _ld(cf_logits, pos_ids, neg_ids)
    clean_acc = float((clean_logits.argmax(-1) == neg_ids).float().mean())
    cf_acc = float((cf_logits.argmax(-1) == pos_ids).float().mean())
    g0 = clean_acc >= 0.80 and cf_acc >= 0.80
    log(f"G0 clean_acc={clean_acc:.0%} cf_acc={cf_acc:.0%} pass={g0}")

    base_result = {
        "stage": "delta_trajectory",
        "model_path": model_path,
        "inject_layer": INJECT_LAYER,
        "mediation_layer": MEDIATION_LAYER,
        "trajectory_layers": list(TRAJECTORY_LAYERS),
        "val_slot": val_pos,
        "last": last_pos,
        "n_null": int(n_null),
        "g0": {"clean_acc": clean_acc, "cf_acc": cf_acc, "pass": bool(g0)},
    }
    if not g0:
        base_result["verdict"] = "TRAJECTORY_INELICITABLE"
        path = os.path.join(out_dir, "results_delta_trajectory.json")
        with open(path, "w") as handle:
            json.dump(base_result, handle, indent=2)
        log("VERDICT: TRAJECTORY_INELICITABLE")
        return base_result

    by_template = {}
    for i, template in enumerate(templates):
        by_template.setdefault(template, []).append(i)
    names = sorted(by_template)
    native_disp_l2 = (cf_cache[INJECT_LAYER][:, 0]
                      - clean_cache[INJECT_LAYER][:, 0])
    generator = torch.Generator().manual_seed(seed + 1729)

    trajectory = {}
    add_effect_rows = []
    wrong_cos_val = []
    patch_add_effect_rows = []
    patch_cf_effect_rows = []
    blocked_effect_rows = []
    fold_rows = []
    null_output = np.zeros(n_null, dtype=np.float64)
    null_cos_val = np.zeros(n_null, dtype=np.float64)
    null_cos_last = np.zeros(n_null, dtype=np.float64)

    for fold, heldout in enumerate(names):
        test_idx = by_template[heldout]
        train_idx = [i for name in names if name != heldout
                     for i in by_template[name]]
        directions = _value_directions(native_disp_l2, metas, train_idx)
        delta = _row_directions(directions, metas, test_idx)
        wrong = _row_directions(directions, metas, test_idx, wrong=True)
        ids = _subset_tensor(batch["clean"]["input_ids"], test_idx).to(dev)
        am = _subset_tensor(batch["clean"]["attention_mask"], test_idx).to(dev)
        pids = _subset_tensor(pos_ids, test_idx)
        nids = _subset_tensor(neg_ids, test_idx)
        base_ld = clean_ld[test_idx]
        clean_fold = {layer: clean_cache[layer][test_idx]
                      for layer in TRAJECTORY_LAYERS}
        cf_fold = {layer: cf_cache[layer][test_idx]
                   for layer in TRAJECTORY_LAYERS}

        add_logits, add_cache = _forward(
            model, ids, am, positions, TRAJECTORY_LAYERS,
            add=(INJECT_LAYER, val_pos, delta))
        add_effect = _ld(add_logits, pids, nids) - base_ld
        add_effect_rows.extend(add_effect.tolist())
        _append_layer_metrics(trajectory, clean_fold, cf_fold, add_cache)

        wrong_logits, wrong_cache = _forward(
            model, ids, am, positions, (MEDIATION_LAYER,),
            add=(INJECT_LAYER, val_pos, wrong))
        del wrong_logits
        native_l8_val = (cf_fold[MEDIATION_LAYER][:, 0]
                         - clean_fold[MEDIATION_LAYER][:, 0])
        wrong_l8_val = (wrong_cache[MEDIATION_LAYER][:, 0]
                        - clean_fold[MEDIATION_LAYER][:, 0])
        wrong_cos_val.extend(_cos_rows(wrong_l8_val, native_l8_val).tolist())

        patch_add_logits, _ = _forward(
            model, ids, am, positions,
            patch=(MEDIATION_LAYER, val_pos,
                   add_cache[MEDIATION_LAYER][:, 0]))
        patch_cf_logits, _ = _forward(
            model, ids, am, positions,
            patch=(MEDIATION_LAYER, val_pos,
                   cf_fold[MEDIATION_LAYER][:, 0]))
        blocked_logits, _ = _forward(
            model, ids, am, positions,
            add=(INJECT_LAYER, val_pos, delta),
            patch=(MEDIATION_LAYER, val_pos,
                   clean_fold[MEDIATION_LAYER][:, 0]))
        patch_add_effect_rows.extend(
            (_ld(patch_add_logits, pids, nids) - base_ld).tolist())
        patch_cf_effect_rows.extend(
            (_ld(patch_cf_logits, pids, nids) - base_ld).tolist())
        blocked_effect_rows.extend(
            (_ld(blocked_logits, pids, nids) - base_ld).tolist())

        for null_idx in range(n_null):
            random_delta = _random_like_rows(delta, generator)
            random_logits, random_cache = _forward(
                model, ids, am, positions, (MEDIATION_LAYER,),
                add=(INJECT_LAYER, val_pos, random_delta))
            random_effect = _ld(random_logits, pids, nids) - base_ld
            random_val = (random_cache[MEDIATION_LAYER][:, 0]
                          - clean_fold[MEDIATION_LAYER][:, 0])
            random_last = (random_cache[MEDIATION_LAYER][:, 1]
                           - clean_fold[MEDIATION_LAYER][:, 1])
            native_last = (cf_fold[MEDIATION_LAYER][:, 1]
                           - clean_fold[MEDIATION_LAYER][:, 1])
            null_output[null_idx] += float(random_effect.mean()) / len(names)
            null_cos_val[null_idx] += float(
                _cos_rows(random_val, native_l8_val).mean()) / len(names)
            null_cos_last[null_idx] += float(
                _cos_rows(random_last, native_last).mean()) / len(names)

        fold_rows.append({
            "heldout": heldout,
            "n": len(test_idx),
            "mean_add_effect": float(add_effect.mean()),
            "mean_l8_val_cos": float(np.mean(
                trajectory["L8_val_slot"]["cos"][-len(test_idx):])),
        })
        log(f"fold {fold+1}/{len(names)} {heldout}: "
            f"effect={float(add_effect.mean()):+.3f} "
            f"L8cos={fold_rows[-1]['mean_l8_val_cos']:+.3f}")

    metric_summary = {}
    for key, values in trajectory.items():
        metric_summary[key] = {
            metric: float(np.mean(rows)) for metric, rows in values.items()
        }

    output_effect = float(np.mean(add_effect_rows))
    output_p = permutation_pvalue(output_effect, null_output, "greater")
    l8_val_cos = metric_summary["L8_val_slot"]["cos"]
    l8_last_cos = metric_summary["L8_last"]["cos"]
    val_cos_p = permutation_pvalue(l8_val_cos, null_cos_val, "greater")
    last_cos_p = permutation_pvalue(l8_last_cos, null_cos_last, "greater")
    wrong_val_cos = float(np.mean(wrong_cos_val))
    error_l2 = metric_summary["L2_val_slot"]["error"]
    error_l8 = metric_summary["L8_val_slot"]["error"]
    norm_ratio = metric_summary["L8_val_slot"]["norm_ratio"]

    patch_add = float(np.mean(patch_add_effect_rows))
    patch_cf = float(np.mean(patch_cf_effect_rows))
    blocked = float(np.mean(blocked_effect_rows))
    patch_ratio = patch_add / patch_cf if patch_cf > EPS else float("-inf")
    block_fraction = ((output_effect - blocked) / output_effect
                      if output_effect > EPS else float("-inf"))

    gates = {
        "O1": bool(output_effect > 0 and output_p < 0.01),
        "A1": bool(l8_val_cos >= 0.50 and val_cos_p < 0.01
                   and l8_val_cos > wrong_val_cos),
        "A2": bool(error_l8 <= 0.75 and error_l8 <= 0.80 * error_l2),
        "Q1": bool(l8_last_cos >= 0.30 and last_cos_p < 0.01),
        "M1": bool(patch_add > 0 and patch_cf > 0 and patch_ratio >= 0.50),
        "M2": bool(block_fraction >= 0.50),
        "D1": bool(0.80 <= norm_ratio <= 1.20),
    }
    if not gates["O1"]:
        verdict = "CONTROL_NULL"
    elif (gates["A1"] and gates["A2"] and gates["Q1"] and gates["M1"]
          and gates["M2"] and gates["D1"]):
        verdict = "CONTROL_GENERATES_NATIVE_STATE"
    elif (gates["A1"] and gates["M1"] and gates["M2"] and gates["D1"]):
        verdict = "CONTROL_NATIVE_LIKE_NO_CONVERGENCE"
    else:
        verdict = "CONTROL_ALTERNATE_PATH"

    result = {
        **base_result,
        "folds": fold_rows,
        "trajectory": metric_summary,
        "primary": {
            "output_effect": output_effect,
            "output_null_mean": float(null_output.mean()),
            "output_null_p95": float(np.percentile(null_output, 95)),
            "output_p": float(output_p),
            "l8_val_cos": l8_val_cos,
            "l8_val_cos_wrong": wrong_val_cos,
            "l8_val_cos_null_mean": float(null_cos_val.mean()),
            "l8_val_cos_p": float(val_cos_p),
            "l8_last_cos": l8_last_cos,
            "l8_last_cos_null_mean": float(null_cos_last.mean()),
            "l8_last_cos_p": float(last_cos_p),
            "error_l2_val": error_l2,
            "error_l8_val": error_l8,
            "l8_val_norm_ratio": norm_ratio,
            "patch_add_effect": patch_add,
            "patch_cf_effect": patch_cf,
            "patch_ratio": float(patch_ratio),
            "blocked_add_effect": blocked,
            "block_fraction": float(block_fraction),
        },
        "gates": gates,
        "verdict": verdict,
    }
    path = os.path.join(out_dir, "results_delta_trajectory.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
