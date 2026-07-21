"""Audit the relation between L20 state overwrites and late attention blocks.

Frozen design: BINDING_SLOT_BRIDGE_PROTOCOL.md.
"""
import json
import os

import numpy as np
import torch

from .delta_binding_component_convergence import _group_data, _split_trials
from .delta_binding_slot_broadcast import _forward_broadcast
from .delta_operator import (
    DONOR_NAMES, INJECT_LAYER, _build_multi_group, _encode_uniform,
    _single_text, _trials, _values)
from .delta_trajectory import EPS, _forward, _ld
from .logutil import log
from .model_utils import get_decoder_layers, input_device, load_model_and_tokenizer

PATCH_LAYER = 20
POST_PATCH_LAYERS = (21, 22, 23, 24, 25, 26)
PROTOCOL_VERSION = "2026-07-13-v1"


def _effect(clean_logits, changed_logits, pos_ids, neg_ids):
    return _ld(changed_logits, pos_ids, neg_ids) - _ld(
        clean_logits, pos_ids, neg_ids)


def _summary(rows):
    natural_rows, add_rows, clean_hits, natural_hits, add_positive = [], [], [], [], []
    for row in rows:
        group = row["group"]
        natural = _effect(row["clean"], row["natural"],
                          group["pos_ids"], group["neg_ids"])
        add = _effect(row["clean"], row["add"],
                      group["pos_ids"], group["neg_ids"])
        natural_rows.extend(natural.tolist())
        add_rows.extend(add.tolist())
        clean_hits.extend((row["clean"].argmax(-1) == group["neg_ids"]).tolist())
        natural_hits.extend((row["natural"].argmax(-1) == group["pos_ids"]).tolist())
        add_positive.extend((add > 0).tolist())
    return {
        "natural_effect": float(np.mean(natural_rows)),
        "add_effect": float(np.mean(add_rows)),
        "clean_acc": float(np.mean(clean_hits)),
        "natural_acc": float(np.mean(natural_hits)),
        "add_positive_fraction": float(np.mean(add_positive)),
    }


def _loss(base, changed, key):
    if base[key] <= EPS:
        return None
    return float((base[key] - changed[key]) / base[key])


def _losses(base, changed):
    return {
        "natural": _loss(base, changed, "natural_effect"),
        "add": _loss(base, changed, "add_effect"),
    }


@torch.no_grad()
def _capture_default(model, groups):
    """Default forwards plus the exact post-L20 matched CLEAN slot state."""
    records = []
    for group in groups:
        position = group["slot"]
        clean_logits, clean_cache = _forward(
            model, group["clean_ids"], group["clean_am"], (position,),
            (PATCH_LAYER,))
        natural_logits, _ = _forward(
            model, group["natural_ids"], group["natural_am"], (position,),
            (PATCH_LAYER,))
        add_logits, _ = _forward(
            model, group["clean_ids"], group["clean_am"], (position,),
            (), add=(INJECT_LAYER, position, group["direction"]))
        records.append({
            "group": group,
            "clean": clean_logits,
            "natural": natural_logits,
            "add": add_logits,
            "clean_l20_slot": clean_cache[PATCH_LAYER][:, 0],
        })
    return records


@torch.no_grad()
def _state_overwrite_default(model, records):
    rows = []
    for record in records:
        group, clean_state = record["group"], record["clean_l20_slot"]
        position = group["slot"]
        natural_logits, _ = _forward(
            model, group["natural_ids"], group["natural_am"], (position,),
            patch=(PATCH_LAYER, position, clean_state))
        add_logits, _ = _forward(
            model, group["clean_ids"], group["clean_am"], (position,),
            add=(INJECT_LAYER, position, group["direction"]),
            patch=(PATCH_LAYER, position, clean_state))
        rows.append({"group": group, "clean": record["clean"],
                     "natural": natural_logits, "add": add_logits})
    return _summary(rows)


@torch.no_grad()
def _custom_condition(model, records, blocked_layers=(), slot_key="slot",
                      overwrite=False):
    """One matched custom-mask condition; optional post-L20 CLEAN overwrite."""
    rows = []
    for record in records:
        group = record["group"]
        key_position = group[slot_key]
        patch = ((PATCH_LAYER, group["slot"], record["clean_l20_slot"])
                 if overwrite else None)
        clean_logits = _forward_broadcast(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            blocked_layers=blocked_layers, key_position=key_position)
        natural_logits = _forward_broadcast(
            model, group["natural_ids"], group["natural_am"], group["slot"],
            patch=patch, blocked_layers=blocked_layers, key_position=key_position)
        add_logits = _forward_broadcast(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            direction=group["direction"], patch=patch,
            blocked_layers=blocked_layers, key_position=key_position)
        rows.append({"group": group, "clean": clean_logits,
                     "natural": natural_logits, "add": add_logits})
    return _summary(rows)


def _within_fraction(reference, observed, fraction=.05):
    if abs(reference) <= EPS:
        return False
    return abs(observed - reference) / abs(reference) <= fraction


@torch.no_grad()
def run_delta_binding_slot_bridge(
        model_path, out_dir, quantization="8bit", device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    layers = get_decoder_layers(model)
    if len(layers) <= max(POST_PATCH_LAYERS):
        raise ValueError(f"model lacks bridge layer {max(POST_PATCH_LAYERS)}")
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
            raise ValueError("bridge own-slot alignment failed")
        group["other_slot"] = int(other)

    log("delta_binding_slot_bridge: "
        f"heldout={len(heldout_rows)} patch=L{PATCH_LAYER} "
        f"outgoing_layers={POST_PATCH_LAYERS}")
    records = _capture_default(model, groups)
    default_base = _summary(records)
    ratio = (default_base["add_effect"] / default_base["natural_effect"]
             if default_base["natural_effect"] > EPS else None)
    default_base["add_to_natural_effect_ratio"] = ratio
    g0 = bool(
        default_base["clean_acc"] >= .80 and default_base["natural_acc"] >= .80
        and default_base["natural_effect"] > EPS and default_base["add_effect"] > EPS
        and default_base["add_positive_fraction"] >= .80 and ratio is not None
        and .70 <= ratio <= 1.30)
    result = {
        "stage": "delta_binding_slot_bridge",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "heldout_offsets": [5, 7],
        "patch_layer": PATCH_LAYER,
        "outgoing_layers": list(POST_PATCH_LAYERS),
        "default_baseline": default_base,
        "gates": {"G0": g0},
    }
    if not g0:
        result["verdict"] = "BRIDGE_INELICITABLE"
    else:
        overwrite_default = _state_overwrite_default(model, records)
        custom_base = _custom_condition(model, records)
        overwrite_custom = _custom_condition(model, records, overwrite=True)
        own_block = _custom_condition(
            model, records, POST_PATCH_LAYERS, "slot")
        other_block = _custom_condition(
            model, records, POST_PATCH_LAYERS, "other_slot")
        combined = _custom_condition(
            model, records, POST_PATCH_LAYERS, "slot", overwrite=True)
        mask_equivalent = all(
            _within_fraction(default_base[key], custom_base[key])
            for key in ("natural_effect", "add_effect"))
        overwrite_default_losses = _losses(default_base, overwrite_default)
        overwrite_custom_losses = _losses(custom_base, overwrite_custom)
        overwrite_reproduced = all(
            value is not None and value >= .80
            for value in (*overwrite_default_losses.values(),
                          *overwrite_custom_losses.values()))
        own_losses = _losses(custom_base, own_block)
        other_losses = _losses(custom_base, other_block)
        gaps = {key: (own_losses[key] - other_losses[key]
                      if own_losses[key] is not None and other_losses[key] is not None
                      else None) for key in ("natural", "add")}
        reconciled = bool(
            mask_equivalent and overwrite_reproduced
            and all(own_losses[key] is not None and own_losses[key] >= .80
                    and gaps[key] is not None and gaps[key] >= .50
                    for key in ("natural", "add")))
        result.update({
            "default_post_l20_overwrite": overwrite_default,
            "custom_mask_baseline": custom_base,
            "custom_post_l20_overwrite": overwrite_custom,
            "own_slot_outgoing_block": own_block,
            "other_slot_outgoing_block": other_block,
            "combined_overwrite_and_own_block": combined,
            "metrics": {
                "mask_equivalent": mask_equivalent,
                "default_overwrite_losses": overwrite_default_losses,
                "custom_overwrite_losses": overwrite_custom_losses,
                "overwrite_reproduced": overwrite_reproduced,
                "own_block_losses": own_losses,
                "other_block_losses": other_losses,
                "own_minus_other_block_loss": gaps,
                "combined_losses": _losses(custom_base, combined),
                "outgoing_channel_reconciled": reconciled,
            },
        })
        if not mask_equivalent:
            result["verdict"] = "MASK_SEMANTICS_UNVERIFIED"
        elif not overwrite_reproduced:
            result["verdict"] = "TIMELINE_OVERWRITE_NOT_REPRODUCED"
        elif reconciled:
            result["verdict"] = "OUTGOING_CHANNEL_RECONCILED"
        else:
            result["verdict"] = "PATCH_MASK_DISSOCIATION"
        log(f"mask_equivalent={mask_equivalent} "
            f"overwrite_default={overwrite_default_losses} "
            f"overwrite_custom={overwrite_custom_losses}")
        log(f"outgoing_own={own_losses} outgoing_gaps={gaps} "
            f"combined={result['metrics']['combined_losses']}")
    path = os.path.join(out_dir, "results_delta_binding_slot_bridge.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
