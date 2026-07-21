"""Difference-in-differences causal timeline for recoverable binding backups.

Frozen design: BINDING_BACKUP_FORMATION_PROTOCOL.md.
"""
import json
import os

import numpy as np
import torch

from .delta_binding_component_convergence import _group_data, _split_trials
from .delta_binding_slot_bridge import PATCH_LAYER, _capture_default, _summary
from .delta_binding_slot_broadcast import _forward_broadcast
from .delta_operator import (
    DONOR_NAMES, INJECT_LAYER, _build_multi_group, _encode_uniform,
    _single_text, _trials, _values)
from .delta_trajectory import EPS, _forward
from .logutil import log
from .model_utils import get_decoder_layers, input_device, load_model_and_tokenizer

LATE_READ_LAYERS = (21, 22, 23, 24, 25, 26)
FORMATION_WINDOWS = {
    "early": tuple(range(3, 9)),
    "middle": tuple(range(9, 15)),
    "late": tuple(range(15, 21)),
}
PROTOCOL_VERSION = "2026-07-13-v1"


@torch.no_grad()
def _condition(model, records, early_layers=(), early_slot_key="slot",
               late_block=False):
    """Run P plus optional early block and optional fixed own-slot late block."""
    rows = []
    for record in records:
        group = record["group"]
        block_specs = [(layer, group[early_slot_key]) for layer in early_layers]
        if late_block:
            block_specs.extend((layer, group["slot"]) for layer in LATE_READ_LAYERS)
        patch = (PATCH_LAYER, group["slot"], record["clean_l20_slot"])
        clean_logits = _forward_broadcast(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            blocked_layer_keys=block_specs)
        natural_logits = _forward_broadcast(
            model, group["natural_ids"], group["natural_am"], group["slot"],
            patch=patch, blocked_layer_keys=block_specs)
        add_logits = _forward_broadcast(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            direction=group["direction"], patch=patch,
            blocked_layer_keys=block_specs)
        rows.append({"group": group, "clean": clean_logits,
                     "natural": natural_logits, "add": add_logits})
    return _summary(rows)


def _recovery(with_late, without_late, effect):
    return float(with_late[f"{effect}_effect"]
                 - without_late[f"{effect}_effect"])


def _window_metrics(own_without, own_with, other_without, other_with):
    metrics = {}
    for effect in ("natural", "add"):
        recovery_own = _recovery(own_with, own_without, effect)
        recovery_other = _recovery(other_with, other_without, effect)
        difference = recovery_other - recovery_own
        fraction = (difference / recovery_other
                    if recovery_other > EPS else None)
        shared = bool(
            recovery_other >= 5.0 and recovery_own >= 0.0
            and fraction is not None and fraction >= .50)
        partial = bool(
            recovery_other >= 5.0 and recovery_own >= 0.0
            and fraction is not None and fraction >= .25)
        metrics[effect] = {
            "recovery_own": recovery_own,
            "recovery_other_control": recovery_other,
            "difference_in_differences": difference,
            "fraction_of_controlled_recovery_prevented": fraction,
            "shared_formation": shared,
            "partial_formation": partial,
        }
    return metrics


@torch.no_grad()
def run_delta_binding_backup_formation(
        model_path, out_dir, quantization="8bit", device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    layers = get_decoder_layers(model)
    if len(layers) <= max(LATE_READ_LAYERS):
        raise ValueError(f"model lacks late backup layer {max(LATE_READ_LAYERS)}")
    values = _values(tok)
    rows = _trials(values)
    _discovery_rows, heldout_rows = _split_trials(rows)
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
    groups = _group_data(model, tok, heldout_rows, prototypes, dev)
    for group in groups:
        _selected, _ci, _cam, _ni, _nam, own, other = _build_multi_group(
            tok, heldout_rows, group["query"])
        if own != group["slot"]:
            raise ValueError("backup formation own-slot alignment failed")
        group["other_slot"] = int(other)

    log("delta_binding_backup_formation: "
        f"heldout={len(heldout_rows)} windows={FORMATION_WINDOWS} "
        f"late={LATE_READ_LAYERS}")
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
        "stage": "delta_binding_backup_formation",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "heldout_offsets": [5, 7],
        "patch_layer": PATCH_LAYER,
        "late_read_layers": list(LATE_READ_LAYERS),
        "formation_windows": {name: list(window)
                              for name, window in FORMATION_WINDOWS.items()},
        "baseline": baseline,
        "gates": {"G0": g0},
    }
    if not g0:
        result["verdict"] = "BACKUP_FORMATION_INELICITABLE"
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
            result["verdict"] = "BACKUP_FORMATION_INELICITABLE"
        else:
            windows = {}
            for name, window in FORMATION_WINDOWS.items():
                own_without = _condition(
                    model, records, window, "slot", late_block=False)
                own_with = _condition(
                    model, records, window, "slot", late_block=True)
                other_without = _condition(
                    model, records, window, "other_slot", late_block=False)
                other_with = _condition(
                    model, records, window, "other_slot", late_block=True)
                metrics = _window_metrics(
                    own_without, own_with, other_without, other_with)
                shared = all(metrics[effect]["shared_formation"]
                             for effect in ("natural", "add"))
                partial = all(metrics[effect]["partial_formation"]
                              for effect in ("natural", "add"))
                windows[name] = {
                    "layers": list(window),
                    "own_without_late": own_without,
                    "own_with_late": own_with,
                    "other_control_without_late": other_without,
                    "other_control_with_late": other_with,
                    "metrics": metrics,
                    "shared_formation_window": shared,
                    "partial_formation_window": partial,
                }
                log(f"window={name} shared={shared} partial={partial} "
                    f"natural={metrics['natural']} add={metrics['add']}")
            shared_windows = [name for name, row in windows.items()
                              if row["shared_formation_window"]]
            partial_windows = [name for name, row in windows.items()
                               if row["partial_formation_window"]]
            result["windows"] = windows
            result["shared_windows"] = shared_windows
            result["partial_windows"] = partial_windows
            if len(shared_windows) == 1:
                result["verdict"] = "LOCALIZED_SHARED_BACKUP_FORMATION"
            elif len(shared_windows) >= 2:
                result["verdict"] = "MULTIPHASE_SHARED_BACKUP_FORMATION"
            else:
                result["verdict"] = "PARTIAL_OR_UNRESOLVED_BACKUP_FORMATION"
    path = os.path.join(out_dir, "results_delta_binding_backup_formation.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
