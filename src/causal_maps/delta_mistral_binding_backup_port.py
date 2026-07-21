"""Single-window causal backup port from Qwen to Mistral-7B.

This is intentionally a constrained replication.  Mistral's L2 affine
operator was selected on discovery substitutions and confirmed on held-out
substitutions by ``delta_binding_cross_model_gate``.  Here we test one
architecture-normalized late overwrite and one architecture-normalized early
outgoing-access window.  There is no layer or head search.
"""
import json
import os

import numpy as np
import torch

from .delta_binding_component_convergence import _group_data
from .delta_binding_slot_bridge import _summary
from .delta_binding_cross_model_gate import _split_rows, _tokenizer_valid_values
from .delta_operator import (
    DONOR_NAMES, _build_multi_group, _encode_uniform, _single_text, _trials)
from .delta_trajectory import EPS, _forward
from .logutil import log
from .model_utils import get_decoder_layers, input_device, load_model_and_tokenizer
from .patching import _split_output


# Qwen indices are normalized by its final valid layer index (27) and rounded
# once for Mistral-7B's final valid index (31). These are fixed before running.
INJECT_LAYER = 2
PATCH_LAYER = 23       # round(20 / 27 * 31)
LATE_READ_LAYERS = tuple(range(24, 31))   # round(21 / 27 * 31) ... round(26 / 27 * 31)
EARLY_FORMATION_LAYERS = tuple(range(3, 10))  # normalized Qwen L3-L8
PROTOCOL_VERSION = "2026-07-14-v1"


def _mistral_port_layers(n_layers):
    """Validate this frozen port is used only on the intended depth."""
    if n_layers != 32:
        raise ValueError(
            f"Mistral backup port is frozen for 32 layers, got {n_layers}")
    return {
        "inject_layer": INJECT_LAYER,
        "patch_layer": PATCH_LAYER,
        "late_read_layers": LATE_READ_LAYERS,
        "early_formation_layers": EARLY_FORMATION_LAYERS,
    }


def _causal_mask(input_ids, attention_mask, dtype):
    """Explicit 4-D causal mask, forcing a mask visible to Mistral hooks."""
    batch, sequence = input_ids.shape
    mask = torch.zeros((batch, 1, sequence, sequence), dtype=dtype,
                       device=input_ids.device)
    future = torch.triu(torch.ones((sequence, sequence), dtype=torch.bool,
                                   device=input_ids.device), diagonal=1)
    minimum = torch.finfo(dtype).min
    mask.masked_fill_(future[None, None], minimum)
    mask.masked_fill_(~attention_mask.to(torch.bool)[:, None, None, :], minimum)
    return mask


@torch.no_grad()
def _forward_mistral_masked(model, input_ids, attention_mask, inject_position,
                            direction=None, patch=None, blocked_layer_keys=()):
    """Mistral-native 4-D mask forward with optional ADD, overwrite, edges.

    Unlike the Qwen broadcast helper, Mistral receives the tensor directly;
    its model creates/propagates that causal mask to every decoder layer.  The
    pre-hooks then remove only later-query -> specified-key edges.
    """
    layers = get_decoder_layers(model)
    sequence = int(input_ids.shape[1])
    specs = {(int(layer), int(key)) for layer, key in blocked_layer_keys}
    if any(not 0 <= layer < len(layers) for layer, _key in specs):
        raise ValueError(f"invalid blocked layer: {sorted(specs)}")
    if any(not 0 <= key < sequence for _layer, key in specs):
        raise ValueError(f"invalid blocked key: {sorted(specs)}")
    handles = []
    if direction is not None:
        def add_hook(_module, _inputs, output):
            hidden, rebuild = _split_output(output)
            edited = hidden.clone()
            delta = direction.to(device=edited.device, dtype=edited.dtype)
            if delta.ndim == 1:
                delta = delta.unsqueeze(0).expand(edited.shape[0], -1)
            if delta.shape != (edited.shape[0], edited.shape[-1]):
                raise ValueError("Mistral ADD direction shape mismatch")
            edited[:, inject_position] += delta
            return rebuild(edited)
        handles.append(layers[INJECT_LAYER].register_forward_hook(add_hook))
    if patch is not None:
        patch_layer, patch_position, patch_value = patch
        def patch_hook(_module, _inputs, output):
            hidden, rebuild = _split_output(output)
            edited = hidden.clone()
            value = patch_value.to(device=edited.device, dtype=edited.dtype)
            if value.shape != (edited.shape[0], edited.shape[-1]):
                raise ValueError("Mistral overwrite state shape mismatch")
            edited[:, patch_position] = value
            return rebuild(edited)
        handles.append(layers[patch_layer].register_forward_hook(patch_hook))
    by_layer = {}
    for layer, key in specs:
        by_layer.setdefault(layer, set()).add(key)
    num_heads = int(model.config.num_attention_heads)

    def edge_hook(keys):
        def hook(_module, args, kwargs):
            mask = kwargs.get("attention_mask")
            if mask is None or mask.ndim != 4:
                raise ValueError(
                    "Mistral masked port expected a 4-D attention mask in self_attn")
            if mask.shape[-2] < sequence or any(mask.shape[-1] <= key for key in keys):
                raise ValueError(f"Mistral attention mask too short: {tuple(mask.shape)}")
            if mask.shape[1] not in (1, num_heads):
                raise ValueError(f"unexpected Mistral mask heads: {mask.shape[1]}")
            edited = mask.expand(mask.shape[0], num_heads, mask.shape[-2],
                                 mask.shape[-1]).clone()
            for key in keys:
                edited[:, :, key + 1:, key] = torch.finfo(edited.dtype).min
            kwargs["attention_mask"] = edited
            return args, kwargs
        return hook

    try:
        for layer, keys in by_layer.items():
            handles.append(layers[layer].self_attn.register_forward_pre_hook(
                edge_hook(tuple(sorted(keys))), with_kwargs=True))
        mask = _causal_mask(input_ids, attention_mask,
                            model.get_input_embeddings().weight.dtype)
        output = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        return output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def _recovery(with_late, without_late, effect):
    return float(with_late[f"{effect}_effect"] - without_late[f"{effect}_effect"])


def _formation_metrics(own_without, own_with, other_without, other_with):
    result = {}
    for effect in ("natural", "add"):
        own = _recovery(own_with, own_without, effect)
        other = _recovery(other_with, other_without, effect)
        difference = other - own
        fraction = difference / other if other > EPS else None
        result[effect] = {
            "recovery_own": own,
            "recovery_other_control": other,
            "difference_in_differences": difference,
            "fraction_of_controlled_recovery_prevented": fraction,
            "shared_early_formation": bool(
                other >= 5.0 and own >= 0.0 and fraction is not None and fraction >= .50),
        }
    return result


@torch.no_grad()
def _capture_default(model, groups):
    records = []
    for group in groups:
        slot = group["slot"]
        clean_logits, cache = _forward(
            model, group["clean_ids"], group["clean_am"], (slot,), (PATCH_LAYER,))
        natural_logits, _ = _forward(
            model, group["natural_ids"], group["natural_am"], (slot,))
        add_logits, _ = _forward(
            model, group["clean_ids"], group["clean_am"], (slot,), (),
            add=(INJECT_LAYER, slot, group["direction"]))
        records.append({"group": group, "clean": clean_logits,
                        "natural": natural_logits, "add": add_logits,
                        "clean_patch_slot": cache[PATCH_LAYER][:, 0]})
    return records


@torch.no_grad()
def _condition(model, records, early_slot_key=None, late_block=False):
    rows = []
    for record in records:
        group = record["group"]
        specs = []
        if early_slot_key is not None:
            specs.extend((layer, group[early_slot_key])
                         for layer in EARLY_FORMATION_LAYERS)
        if late_block:
            specs.extend((layer, group["slot"]) for layer in LATE_READ_LAYERS)
        patch = (PATCH_LAYER, group["slot"], record["clean_patch_slot"])
        clean = _forward_mistral_masked(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            blocked_layer_keys=specs)
        natural = _forward_mistral_masked(
            model, group["natural_ids"], group["natural_am"], group["slot"],
            patch=patch, blocked_layer_keys=specs)
        add = _forward_mistral_masked(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            direction=group["direction"], patch=patch, blocked_layer_keys=specs)
        rows.append({"group": group, "clean": clean,
                     "natural": natural, "add": add})
    return _summary(rows)


@torch.no_grad()
def _masked_baseline(model, records):
    """CLEAN/NATURAL/ADD under the explicit native Mistral mask, no overwrite."""
    rows = []
    for record in records:
        group = record["group"]
        clean = _forward_mistral_masked(
            model, group["clean_ids"], group["clean_am"], group["slot"])
        natural = _forward_mistral_masked(
            model, group["natural_ids"], group["natural_am"], group["slot"])
        add = _forward_mistral_masked(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            direction=group["direction"])
        rows.append({"group": group, "clean": clean,
                     "natural": natural, "add": add})
    return _summary(rows)


def _within_fraction(reference, observed, fraction=.05):
    return bool(abs(reference) > EPS and abs(observed - reference) / abs(reference) <= fraction)


@torch.no_grad()
def run_delta_mistral_binding_backup_port(
        model_path, out_dir, quantization="8bit", device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    layer_config = _mistral_port_layers(len(get_decoder_layers(model)))
    values, excluded_values = _tokenizer_valid_values(tok)
    if len(values) < 8:
        result = {"stage": "delta_mistral_binding_backup_port", "model_path": model_path,
                  "protocol_version": PROTOCOL_VERSION,
                  "tokenizer_valid_values": values,
                  "tokenizer_excluded_values": excluded_values,
                  "verdict": "MISTRAL_BACKUP_PORT_TOKENIZATION_INELIGIBLE"}
        return _write(out_dir, result)
    rows = _trials(values)
    _discovery, heldout = _split_rows(rows)
    dev = input_device(model)
    donor_rows, donor_values = [], []
    for name in DONOR_NAMES:
        for value in values:
            donor_rows.append(_single_text(tok, name, value))
            donor_values.append(value)
    donor_ids, donor_am, donor_pos = _encode_uniform(tok, donor_rows)
    _, donor_cache = _forward(
        model, donor_ids.to(dev), donor_am.to(dev), (donor_pos,), (INJECT_LAYER,))
    donor_hidden = donor_cache[INJECT_LAYER][:, 0]
    prototypes = {value: donor_hidden[[i for i, seen in enumerate(donor_values)
                                       if seen == value]].mean(0)
                  for value in values}
    groups = _group_data(model, tok, heldout, prototypes, dev)
    for group in groups:
        _selected, _ci, _cam, _ni, _nam, own, other = _build_multi_group(
            tok, heldout, group["query"])
        if own != group["slot"]:
            raise ValueError("Mistral own-slot alignment failed")
        group["other_slot"] = int(other)

    log("delta_mistral_binding_backup_port: "
        f"heldout={len(heldout)} patch=L{PATCH_LAYER} "
        f"late={LATE_READ_LAYERS} early={EARLY_FORMATION_LAYERS}")
    records = _capture_default(model, groups)
    baseline = _summary(records)
    ratio = (baseline["add_effect"] / baseline["natural_effect"]
             if baseline["natural_effect"] > EPS else None)
    baseline["add_to_natural_effect_ratio"] = ratio
    g0 = bool(baseline["clean_acc"] >= .80 and baseline["natural_acc"] >= .80
              and baseline["natural_effect"] > EPS and baseline["add_effect"] > EPS
              and baseline["add_positive_fraction"] >= .80 and ratio is not None
              and .70 <= ratio <= 1.30)
    result = {"stage": "delta_mistral_binding_backup_port", "model_path": model_path,
              "quantization": quantization, "seed": int(seed),
              "protocol_version": PROTOCOL_VERSION, "heldout_offsets": [5, 7],
              "tokenizer_valid_values": values,
              "tokenizer_excluded_values": excluded_values,
              **layer_config, "baseline": baseline, "gates": {"G0": g0}}
    if not g0:
        result["verdict"] = "MISTRAL_BACKUP_PORT_INELICITABLE"
        return _write(out_dir, result)

    masked_baseline = _masked_baseline(model, records)
    mask_equivalent = all(_within_fraction(baseline[key], masked_baseline[key])
                          for key in ("natural_effect", "add_effect"))
    overwrite = _condition(model, records)
    late_released = _condition(model, records, late_block=True)
    base_recovery = {effect: _recovery(late_released, overwrite, effect)
                     for effect in ("natural", "add")}
    g1 = bool(mask_equivalent and all(value >= 5.0 for value in base_recovery.values()))
    result.update({"masked_baseline": masked_baseline, "post_l23_overwrite": overwrite,
                   "post_l23_overwrite_plus_late_block": late_released,
                   "base_late_block_recovery": base_recovery,
                   "gates": {"G0": g0, "mask_equivalent": mask_equivalent, "G1": g1}})
    if not g1:
        result["verdict"] = "MISTRAL_BACKUP_PORT_INELICITABLE"
        return _write(out_dir, result)

    own_without = _condition(model, records, "slot", late_block=False)
    own_with = _condition(model, records, "slot", late_block=True)
    other_without = _condition(model, records, "other_slot", late_block=False)
    other_with = _condition(model, records, "other_slot", late_block=True)
    metrics = _formation_metrics(own_without, own_with, other_without, other_with)
    replicated = all(metrics[effect]["shared_early_formation"]
                     for effect in ("natural", "add"))
    result.update({"early_own_without_late": own_without,
                   "early_own_with_late": own_with,
                   "early_other_control_without_late": other_without,
                   "early_other_control_with_late": other_with,
                   "early_formation_metrics": metrics,
                   "verdict": ("MISTRAL_SHARED_EARLY_BACKUP_REPLICATES"
                               if replicated else "MISTRAL_EARLY_BACKUP_NOT_REPLICATED")})
    log(f"early metrics={metrics}")
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_mistral_binding_backup_port.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
