"""Fixed early-window backup-formation replication on reversed binding order.

Frozen design: BINDING_BACKUP_REVERSED_CONFIRMATION_PROTOCOL.md.
"""
import json
import os

import torch

from .delta_binding_backup_formation import (
    LATE_READ_LAYERS, _condition, _recovery, _window_metrics)
from .delta_binding_slot_broadcast import _groups, _layout_trials
from .delta_binding_slot_bridge import PATCH_LAYER, _capture_default, _summary
from .delta_operator import (
    DONOR_NAMES, INJECT_LAYER, _encode_uniform, _single_text, _values)
from .delta_trajectory import EPS, _forward
from .logutil import log
from .model_utils import get_decoder_layers, input_device, load_model_and_tokenizer

EARLY_LAYERS = (3, 4, 5, 6, 7, 8)
PROTOCOL_VERSION = "2026-07-13-v1"


@torch.no_grad()
def run_delta_binding_backup_reversed_confirmation(
        model_path, out_dir, quantization="8bit", device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    layers = get_decoder_layers(model)
    if len(layers) <= max(LATE_READ_LAYERS):
        raise ValueError(f"model lacks reversed confirmation layer {max(LATE_READ_LAYERS)}")
    values = _values(tok)
    rows = _layout_trials(values)
    dev = input_device(model)

    donor_rows, donor_values = [], []
    for name in DONOR_NAMES:
        for value in values:
            donor_rows.append(_single_text(tok, name, value))
            donor_values.append(value)
    donor_ids, donor_am, donor_pos = _encode_uniform(tok, donor_rows)
    _, donor_cache = _forward(
        model, donor_ids.to(dev), donor_am.to(dev), (donor_pos,),
        (INJECT_LAYER,))
    donor_hidden = donor_cache[INJECT_LAYER][:, 0]
    prototypes = {
        value: donor_hidden[[i for i, observed in enumerate(donor_values)
                             if observed == value]].mean(0)
        for value in values
    }
    groups = _groups(tok, rows, prototypes, dev)
    for group in groups:
        group["slot"] = group["own_pos"]
        group["other_slot"] = group["other_pos"]

    log("delta_binding_backup_reversed_confirmation: "
        f"rows={len(rows)} early={EARLY_LAYERS} late={LATE_READ_LAYERS}")
    records = _capture_default(model, groups)
    baseline = _summary(records)
    ratio = (baseline["add_effect"] / baseline["natural_effect"]
             if baseline["natural_effect"] > EPS else None)
    baseline["add_to_natural_effect_ratio"] = ratio
    g0 = bool(
        baseline["clean_acc"] >= .80 and baseline["natural_acc"] >= .80
        and baseline["natural_effect"] > EPS and baseline["add_effect"] > EPS
        and baseline["add_positive_fraction"] >= .80 and ratio is not None
        and .70 <= ratio <= 1.30)
    result = {
        "stage": "delta_binding_backup_reversed_confirmation",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "prompt_layout": "reversed_Y_then_X",
        "n_trials": len(rows),
        "early_layers": list(EARLY_LAYERS),
        "late_read_layers": list(LATE_READ_LAYERS),
        "baseline": baseline,
        "gates": {"G0": g0},
    }
    if not g0:
        result["verdict"] = "REVERSED_LAYOUT_BACKUP_INELICITABLE"
    else:
        patched = _condition(model, records)
        patched_late = _condition(model, records, late_block=True)
        base_recovery = {
            effect: _recovery(patched_late, patched, effect)
            for effect in ("natural", "add")}
        g1 = all(value >= 5.0 for value in base_recovery.values())
        result.update({
            "post_l20_overwrite": patched,
            "post_l20_overwrite_plus_late_block": patched_late,
            "base_late_block_recovery": base_recovery,
            "gates": {"G0": g0, "G1": g1},
        })
        if not g1:
            result["verdict"] = "REVERSED_LAYOUT_BACKUP_INELICITABLE"
        else:
            own_without = _condition(
                model, records, EARLY_LAYERS, "slot", late_block=False)
            own_with = _condition(
                model, records, EARLY_LAYERS, "slot", late_block=True)
            other_without = _condition(
                model, records, EARLY_LAYERS, "other_slot", late_block=False)
            other_with = _condition(
                model, records, EARLY_LAYERS, "other_slot", late_block=True)
            metrics = _window_metrics(
                own_without, own_with, other_without, other_with)
            replicated = all(metrics[effect]["shared_formation"]
                             for effect in ("natural", "add"))
            result.update({
                "early_own_without_late": own_without,
                "early_own_with_late": own_with,
                "early_other_control_without_late": other_without,
                "early_other_control_with_late": other_with,
                "early_metrics": metrics,
                "early_prediction_replicated": replicated,
            })
            result["verdict"] = (
                "REVERSED_LAYOUT_SHARED_EARLY_BACKUP_REPLICATES"
                if replicated else "REVERSED_LAYOUT_EARLY_BACKUP_NOT_REPLICATED")
            log(f"early replicated={replicated} natural={metrics['natural']} "
                f"add={metrics['add']}")
    path = os.path.join(
        out_dir, "results_delta_binding_backup_reversed_confirmation.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
