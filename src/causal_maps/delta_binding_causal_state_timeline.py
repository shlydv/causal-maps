"""Full-state causal interchange timeline for the affine binding operator.

Frozen design: BINDING_CAUSAL_STATE_TIMELINE_PROTOCOL.md.  The intervention
states are actual matched CLEAN/NATURAL/ADD residuals, not learned directions.
"""
import json
import os

import numpy as np
import torch

from .delta_binding_component_convergence import _group_data, _split_trials
from .delta_operator import (
    DONOR_NAMES, INJECT_LAYER, MEDIATION_LAYER, _encode_uniform,
    _single_text, _trials, _values)
from .delta_trajectory import EPS, _cos_rows, _error_rows, _forward, _ld
from .logutil import log
from .model_utils import get_decoder_layers, input_device, load_model_and_tokenizer

LAYERS = (2, 4, 8, 12, 16, 20, 26)
EARLY_LAYERS = (2, 4, 8)
PROTOCOL_VERSION = "2026-07-13-v1"


def _effect(clean_logits, changed_logits, pos_ids, neg_ids):
    return _ld(changed_logits, pos_ids, neg_ids) - _ld(
        clean_logits, pos_ids, neg_ids)


def _safe_ratio(numerator, denominator):
    if abs(denominator) <= EPS:
        return None
    return float(numerator / denominator)


@torch.no_grad()
def _capture_conditions(model, groups):
    """Capture all preregistered sites for matched CLEAN/NATURAL/ADD runs."""
    observations = []
    natural_rows, add_rows, clean_hits, natural_hits, add_positive = [], [], [], [], []
    for group in groups:
        positions = (group["slot"], group["last"])
        clean_logits, clean_cache = _forward(
            model, group["clean_ids"], group["clean_am"], positions, LAYERS)
        natural_logits, natural_cache = _forward(
            model, group["natural_ids"], group["natural_am"], positions, LAYERS)
        add_logits, add_cache = _forward(
            model, group["clean_ids"], group["clean_am"], positions, LAYERS,
            add=(INJECT_LAYER, group["slot"], group["direction"]))
        natural = _effect(
            clean_logits, natural_logits, group["pos_ids"], group["neg_ids"])
        add = _effect(clean_logits, add_logits, group["pos_ids"], group["neg_ids"])
        observations.append({
            "group": group,
            "clean_logits": clean_logits,
            "clean_cache": clean_cache,
            "natural_cache": natural_cache,
            "add_cache": add_cache,
            "natural_rows": natural,
            "add_rows": add,
        })
        natural_rows.extend(natural.tolist())
        add_rows.extend(add.tolist())
        clean_hits.extend((clean_logits.argmax(-1) == group["neg_ids"]).tolist())
        natural_hits.extend((natural_logits.argmax(-1) == group["pos_ids"]).tolist())
        add_positive.extend((add > 0).tolist())
    return {
        "observations": observations,
        "natural_effect": float(np.mean(natural_rows)),
        "add_effect": float(np.mean(add_rows)),
        "clean_acc": float(np.mean(clean_hits)),
        "natural_acc": float(np.mean(natural_hits)),
        "add_positive_fraction": float(np.mean(add_positive)),
    }


@torch.no_grad()
def _site_metrics(model, observations, layer, site_index, base):
    """Measure six matched full-state replacements at one fixed site."""
    natural_patch_rows, add_patch_rows = [], []
    natural_into_add_rows, add_into_natural_rows = [], []
    clean_into_add_rows, clean_into_natural_rows = [], []
    geometry_cos, geometry_error = [], []
    for observation in observations:
        group = observation["group"]
        position = group["slot"] if site_index == 0 else group["last"]
        clean_state = observation["clean_cache"][layer][:, site_index]
        natural_state = observation["natural_cache"][layer][:, site_index]
        add_state = observation["add_cache"][layer][:, site_index]
        natural_disp = natural_state - clean_state
        add_disp = add_state - clean_state
        geometry_cos.extend(_cos_rows(add_disp, natural_disp).tolist())
        geometry_error.extend(_error_rows(add_disp, natural_disp).tolist())

        natural_patch_logits, _ = _forward(
            model, group["clean_ids"], group["clean_am"],
            (group["slot"], group["last"]),
            patch=(layer, position, natural_state))
        add_patch_logits, _ = _forward(
            model, group["clean_ids"], group["clean_am"],
            (group["slot"], group["last"]),
            patch=(layer, position, add_state))
        natural_into_add_logits, _ = _forward(
            model, group["clean_ids"], group["clean_am"],
            (group["slot"], group["last"]),
            add=(INJECT_LAYER, group["slot"], group["direction"]),
            patch=(layer, position, natural_state))
        add_into_natural_logits, _ = _forward(
            model, group["natural_ids"], group["natural_am"],
            (group["slot"], group["last"]),
            patch=(layer, position, add_state))
        clean_into_add_logits, _ = _forward(
            model, group["clean_ids"], group["clean_am"],
            (group["slot"], group["last"]),
            add=(INJECT_LAYER, group["slot"], group["direction"]),
            patch=(layer, position, clean_state))
        clean_into_natural_logits, _ = _forward(
            model, group["natural_ids"], group["natural_am"],
            (group["slot"], group["last"]),
            patch=(layer, position, clean_state))

        args = (observation["clean_logits"], group["pos_ids"], group["neg_ids"])
        natural_patch_rows.extend(_effect(args[0], natural_patch_logits, *args[1:]).tolist())
        add_patch_rows.extend(_effect(args[0], add_patch_logits, *args[1:]).tolist())
        natural_into_add_rows.extend(
            _effect(args[0], natural_into_add_logits, *args[1:]).tolist())
        add_into_natural_rows.extend(
            _effect(args[0], add_into_natural_logits, *args[1:]).tolist())
        clean_into_add_rows.extend(
            _effect(args[0], clean_into_add_logits, *args[1:]).tolist())
        clean_into_natural_rows.extend(
            _effect(args[0], clean_into_natural_logits, *args[1:]).tolist())

    natural_patch = float(np.mean(natural_patch_rows))
    add_patch = float(np.mean(add_patch_rows))
    natural_into_add = float(np.mean(natural_into_add_rows))
    add_into_natural = float(np.mean(add_into_natural_rows))
    clean_into_add = float(np.mean(clean_into_add_rows))
    clean_into_natural = float(np.mean(clean_into_natural_rows))
    natural_scale = abs(base["natural_effect"])
    add_block = _safe_ratio(base["add_effect"] - clean_into_add, base["add_effect"])
    natural_block = _safe_ratio(
        base["natural_effect"] - clean_into_natural, base["natural_effect"])
    add_swap_deviation = _safe_ratio(
        abs(natural_into_add - base["add_effect"]), natural_scale)
    natural_swap_deviation = _safe_ratio(
        abs(add_into_natural - base["natural_effect"]), natural_scale)
    patch_ratio = _safe_ratio(add_patch, natural_patch)
    interchangeable = bool(
        natural_patch > 0 and add_patch > 0
        and patch_ratio is not None and .70 <= patch_ratio <= 1.30
        and add_swap_deviation is not None and add_swap_deviation <= .10
        and natural_swap_deviation is not None and natural_swap_deviation <= .10
        and add_block is not None and add_block >= .50
        and natural_block is not None and natural_block >= .50)
    return {
        "geometry": {
            "add_natural_cos": float(np.mean(geometry_cos)),
            "add_natural_error": float(np.mean(geometry_error)),
        },
        "sufficiency": {
            "natural_state_into_clean_effect": natural_patch,
            "add_state_into_clean_effect": add_patch,
            "add_to_natural_patch_ratio": patch_ratio,
        },
        "swaps": {
            "natural_state_into_add_effect": natural_into_add,
            "add_state_into_natural_effect": add_into_natural,
            "add_swap_deviation": add_swap_deviation,
            "natural_swap_deviation": natural_swap_deviation,
        },
        "necessity": {
            "clean_state_into_add_effect": clean_into_add,
            "clean_state_into_natural_effect": clean_into_natural,
            "add_block_fraction": add_block,
            "natural_block_fraction": natural_block,
        },
        "causally_interchangeable": interchangeable,
    }


def _write_inelicitable(out_dir, model_path, quantization, seed, base):
    result = {
        "stage": "delta_binding_causal_state_timeline",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "heldout_base": base,
        "gates": {"G0": False},
        "verdict": "BINDING_TIMELINE_INELICITABLE",
    }
    path = os.path.join(out_dir, "results_delta_binding_causal_state_timeline.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log("VERDICT: BINDING_TIMELINE_INELICITABLE")
    return result


@torch.no_grad()
def run_delta_binding_causal_state_timeline(
        model_path, out_dir, quantization="8bit", device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    layers = get_decoder_layers(model)
    if len(layers) <= max(LAYERS):
        raise ValueError(f"model lacks frozen timeline layer {max(LAYERS)}")
    dev = input_device(model)
    values = _values(tok)
    rows = _trials(values)
    _discovery_rows, heldout_rows = _split_trials(rows)

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
    log("delta_binding_causal_state_timeline: "
        f"heldout={len(heldout_rows)} layers={LAYERS}")
    captured = _capture_conditions(model, groups)
    base = {key: captured[key] for key in (
        "natural_effect", "add_effect", "clean_acc", "natural_acc",
        "add_positive_fraction")}
    effect_ratio = _safe_ratio(base["add_effect"], base["natural_effect"])
    base["add_to_natural_effect_ratio"] = effect_ratio
    g0 = bool(
        base["clean_acc"] >= .80 and base["natural_acc"] >= .80
        and base["natural_effect"] > EPS and base["add_effect"] > EPS
        and base["add_positive_fraction"] >= .80
        and effect_ratio is not None and .70 <= effect_ratio <= 1.30)
    if not g0:
        return _write_inelicitable(out_dir, model_path, quantization, seed, base)

    timeline = []
    for layer in LAYERS:
        for site_index, site_name in enumerate(("queried_value_slot", "final_readout")):
            metrics = _site_metrics(
                model, captured["observations"], layer, site_index, base)
            timeline.append({"layer": layer, "site": site_name, **metrics})
            log(f"layer={layer} site={site_name} "
                f"interchangeable={metrics['causally_interchangeable']} "
                f"cos={metrics['geometry']['add_natural_cos']:.3f}")

    final_shared = [row["layer"] for row in timeline
                    if row["site"] == "final_readout"
                    and row["causally_interchangeable"]]
    slot_shared = [row["layer"] for row in timeline
                   if row["site"] == "queried_value_slot"
                   and row["causally_interchangeable"]]
    early_final = [layer for layer in final_shared if layer in EARLY_LAYERS]
    delayed_final = [layer for layer in final_shared if layer not in EARLY_LAYERS]
    if early_final:
        verdict = "EARLY_SHARED_CAUSAL_STATE"
    elif delayed_final:
        verdict = "DELAYED_SHARED_CAUSAL_STATE"
    elif slot_shared:
        verdict = "SLOT_ONLY_SHARED_STATE"
    else:
        verdict = "ALTERNATIVE_PATHS_OR_UNRESOLVED"
    result = {
        "stage": "delta_binding_causal_state_timeline",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "inject_layer": INJECT_LAYER,
        "layers": list(LAYERS),
        "n_trials": len(rows),
        "n_heldout": len(heldout_rows),
        "heldout_base": base,
        "timeline": timeline,
        "gates": {"G0": g0},
        "final_readout_shared_layers": final_shared,
        "queried_slot_shared_layers": slot_shared,
        "verdict": verdict,
    }
    path = os.path.join(out_dir, "results_delta_binding_causal_state_timeline.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {verdict}")
    return result
