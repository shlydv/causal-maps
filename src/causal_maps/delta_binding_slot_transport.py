"""Fresh-trial all-head attention-edge test for binding slot-to-readout transport.

Frozen design: BINDING_SLOT_TRANSPORT_PROTOCOL.md.
"""
import json
import os

import numpy as np
import torch

from .delta_operator import (
    DONOR_NAMES, INJECT_LAYER, _build_multi_group, _directions,
    _encode_uniform, _single_text, _trials, _values)
from .delta_trajectory import EPS, _ld
from .logutil import log
from .model_utils import (
    get_decoder_layers, input_device, load_model_and_tokenizer, single_token_id)
from .patching import _split_output

TRANSPORT_LAYERS = (20, 21, 22, 23, 24, 25, 26)
FRESH_OFFSETS = (2, 4, 6, 8)
PROTOCOL_VERSION = "2026-07-13-v1"


def _fresh_trials(values):
    """Deterministic mappings disjoint from the operator offsets 1/3/5/7."""
    rows = []
    count = len(values)
    for source_idx, source in enumerate(values):
        for index, offset in enumerate(FRESH_OFFSETS):
            target_idx = (source_idx + offset) % count
            target = values[target_idx]
            wrong_idx = (target_idx + 1) % count
            while wrong_idx in (source_idx, target_idx):
                wrong_idx = (wrong_idx + 1) % count
            distractor_idx = (target_idx + 3) % count
            while distractor_idx in (source_idx, target_idx, wrong_idx):
                distractor_idx = (distractor_idx + 1) % count
            rows.append({
                "source": source,
                "target": target,
                "wrong": values[wrong_idx],
                "distractor": values[distractor_idx],
                "query": "X" if index % 2 == 0 else "Y",
                "offset": offset,
            })
    return rows


def _groups(tok, rows, prototypes, device):
    groups = []
    for query in ("X", "Y"):
        selected, clean_ids, clean_am, natural_ids, natural_am, own_pos, other_pos = (
            _build_multi_group(tok, rows, query))
        groups.append({
            "query": query,
            "rows": selected,
            "clean_ids": clean_ids.to(device),
            "clean_am": clean_am.to(device),
            "natural_ids": natural_ids.to(device),
            "natural_am": natural_am.to(device),
            "own_pos": int(own_pos),
            "other_pos": int(other_pos),
            "last": int(clean_ids.shape[1] - 1),
            "direction": _directions(prototypes, selected),
            "pos_ids": torch.tensor(
                [single_token_id(tok, row["target"]) for row in selected]),
            "neg_ids": torch.tensor(
                [single_token_id(tok, row["source"]) for row in selected]),
        })
    return groups


@torch.no_grad()
def _forward_edge(model, input_ids, attention_mask, inject_position,
                  direction=None, blocked_layers=(), key_position=None):
    """Run with optional L2 ADD and final-query→key all-head edge blocks."""
    layers = get_decoder_layers(model)
    blocked_layers = tuple(sorted(set(blocked_layers)))
    if any(layer < 0 or layer >= len(layers) for layer in blocked_layers):
        raise ValueError(f"invalid edge layers: {blocked_layers}")
    final_position = int(input_ids.shape[1] - 1)
    if key_position is None:
        raise ValueError("key_position is required")
    if not 0 <= key_position <= final_position:
        raise ValueError(f"invalid key position: {key_position}")
    num_heads = int(model.config.num_attention_heads)
    handles = []

    if direction is not None:
        def add_hook(_module, _inputs, output):
            hidden, rebuild = _split_output(output)
            edited = hidden.clone()
            addition = direction.to(device=edited.device, dtype=edited.dtype)
            if addition.ndim == 1:
                addition = addition.unsqueeze(0).expand(edited.shape[0], -1)
            if addition.shape != (edited.shape[0], edited.shape[-1]):
                raise ValueError("binding direction shape does not match hidden state")
            edited[:, inject_position] += addition
            return rebuild(edited)
        handles.append(layers[INJECT_LAYER].register_forward_hook(add_hook))

    def edge_hook(_module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.ndim != 4:
            raise ValueError("expected Qwen 4D additive attention mask")
        if mask.shape[-2] <= final_position or mask.shape[-1] <= key_position:
            raise ValueError(f"attention mask too short: {tuple(mask.shape)}")
        if mask.shape[1] not in (1, num_heads):
            raise ValueError(f"unexpected mask head dimension: {mask.shape[1]}")
        edited = mask.expand(
            mask.shape[0], num_heads, mask.shape[-2], mask.shape[-1]).clone()
        edited[:, :, final_position, key_position] = torch.finfo(edited.dtype).min
        kwargs["attention_mask"] = edited
        return args, kwargs

    try:
        for layer in blocked_layers:
            handles.append(layers[layer].self_attn.register_forward_pre_hook(
                edge_hook, with_kwargs=True))
        batch, sequence = input_ids.shape
        dtype = model.get_input_embeddings().weight.dtype
        mask = torch.zeros(
            (batch, 1, sequence, sequence), dtype=dtype, device=input_ids.device)
        future = torch.triu(torch.ones(
            (sequence, sequence), dtype=torch.bool, device=input_ids.device),
            diagonal=1)
        mask.masked_fill_(future.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)
        mask.masked_fill_(
            ~attention_mask.to(torch.bool)[:, None, None, :], torch.finfo(dtype).min)
        output = model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            use_cache=False, logits_to_keep=1)
        return output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def _effect(clean_logits, changed_logits, pos_ids, neg_ids):
    return _ld(changed_logits, pos_ids, neg_ids) - _ld(
        clean_logits, pos_ids, neg_ids)


@torch.no_grad()
def _measure(model, groups, blocked_layers=(), slot="own"):
    """Matched CLEAN/NATURAL/ADD effects under one edge condition."""
    natural_rows, add_rows, clean_hits, natural_hits, add_positive = [], [], [], [], []
    for group in groups:
        key_position = group[f"{slot}_pos"]
        clean_logits = _forward_edge(
            model, group["clean_ids"], group["clean_am"], group["own_pos"],
            blocked_layers=blocked_layers, key_position=key_position)
        natural_logits = _forward_edge(
            model, group["natural_ids"], group["natural_am"], group["own_pos"],
            blocked_layers=blocked_layers, key_position=key_position)
        add_logits = _forward_edge(
            model, group["clean_ids"], group["clean_am"], group["own_pos"],
            direction=group["direction"], blocked_layers=blocked_layers,
            key_position=key_position)
        natural = _effect(
            clean_logits, natural_logits, group["pos_ids"], group["neg_ids"])
        add = _effect(clean_logits, add_logits, group["pos_ids"], group["neg_ids"])
        natural_rows.extend(natural.tolist())
        add_rows.extend(add.tolist())
        clean_hits.extend((clean_logits.argmax(-1) == group["neg_ids"]).tolist())
        natural_hits.extend((natural_logits.argmax(-1) == group["pos_ids"]).tolist())
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


def _condition(base, own, other, layers):
    own_natural = _loss(base, own, "natural_effect")
    own_add = _loss(base, own, "add_effect")
    other_natural = _loss(base, other, "natural_effect")
    other_add = _loss(base, other, "add_effect")
    own_control_natural_gap = (
        own_natural - other_natural
        if own_natural is not None and other_natural is not None else None)
    own_control_add_gap = (
        own_add - other_add
        if own_add is not None and other_add is not None else None)
    shared = bool(
        own_natural is not None and own_add is not None
        and own_control_natural_gap is not None and own_control_add_gap is not None
        and own_natural >= .50 and own_add >= .50
        and own_control_natural_gap >= .25 and own_control_add_gap >= .25)
    natural_only = bool(
        own_natural is not None and own_control_natural_gap is not None
        and own_natural >= .50 and own_control_natural_gap >= .25)
    add_only = bool(
        own_add is not None and own_control_add_gap is not None
        and own_add >= .50 and own_control_add_gap >= .25)
    return {
        "blocked_layers": list(layers),
        "own": own,
        "other_slot_control": other,
        "own_losses": {"natural": own_natural, "add": own_add},
        "own_minus_other_loss": {
            "natural": own_control_natural_gap, "add": own_control_add_gap},
        "shared_transport": shared,
        "natural_only": natural_only,
        "add_only": add_only,
    }


def _write_inelicitable(out_dir, model_path, quantization, seed, base):
    result = {
        "stage": "delta_binding_slot_transport",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "fresh_offsets": list(FRESH_OFFSETS),
        "base": base,
        "gates": {"G0": False},
        "verdict": "SLOT_TRANSPORT_INELICITABLE",
    }
    path = os.path.join(out_dir, "results_delta_binding_slot_transport.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log("VERDICT: SLOT_TRANSPORT_INELICITABLE")
    return result


@torch.no_grad()
def run_delta_binding_slot_transport(
        model_path, out_dir, quantization="8bit", device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    layers = get_decoder_layers(model)
    if len(layers) <= max(TRANSPORT_LAYERS):
        raise ValueError(f"model lacks transport layer {max(TRANSPORT_LAYERS)}")
    values = _values(tok)
    rows = _fresh_trials(values)
    if len(rows) != 40:
        raise ValueError(f"expected 40 fresh rows, got {len(rows)}")
    if set(row["offset"] for row in rows) != set(FRESH_OFFSETS):
        raise ValueError("fresh offsets malformed")
    dev = input_device(model)

    donor_rows, donor_values = [], []
    for name in DONOR_NAMES:
        for value in values:
            donor_rows.append(_single_text(tok, name, value))
            donor_values.append(value)
    donor_ids, donor_am, donor_pos = _encode_uniform(tok, donor_rows)
    from .delta_trajectory import _forward
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
    log("delta_binding_slot_transport: "
        f"fresh_rows={len(rows)} layers={TRANSPORT_LAYERS}")
    base = _measure(model, groups, slot="own")
    ratio = (base["add_effect"] / base["natural_effect"]
             if base["natural_effect"] > EPS else None)
    base["add_to_natural_effect_ratio"] = ratio
    g0 = bool(
        base["clean_acc"] >= .80 and base["natural_acc"] >= .80
        and base["natural_effect"] > EPS and base["add_effect"] > EPS
        and base["add_positive_fraction"] >= .80
        and ratio is not None and .70 <= ratio <= 1.30)
    if not g0:
        return _write_inelicitable(out_dir, model_path, quantization, seed, base)

    individual = []
    for layer in TRANSPORT_LAYERS:
        own = _measure(model, groups, (layer,), "own")
        other = _measure(model, groups, (layer,), "other")
        record = _condition(base, own, other, (layer,))
        individual.append(record)
        log(f"layer={layer} own_losses={record['own_losses']} "
            f"gaps={record['own_minus_other_loss']} "
            f"shared={record['shared_transport']}")
    cumulative_own = _measure(model, groups, TRANSPORT_LAYERS, "own")
    cumulative_other = _measure(model, groups, TRANSPORT_LAYERS, "other")
    cumulative = _condition(
        base, cumulative_own, cumulative_other, TRANSPORT_LAYERS)
    log(f"cumulative own_losses={cumulative['own_losses']} "
        f"gaps={cumulative['own_minus_other_loss']} "
        f"shared={cumulative['shared_transport']}")

    direct_layers = [record["blocked_layers"][0] for record in individual
                     if record["shared_transport"]]
    one_sided = any(record["natural_only"] != record["add_only"]
                    for record in [*individual, cumulative])
    if direct_layers:
        verdict = "SHARED_DIRECT_SLOT_ATTENTION_PATH"
    elif cumulative["shared_transport"]:
        verdict = "SHARED_DISTRIBUTED_SLOT_ATTENTION_PATH"
    elif one_sided:
        verdict = "DIVERGENT_SLOT_ATTENTION_TRANSPORT"
    else:
        verdict = "NONDIRECT_OR_UNRESOLVED_SLOT_TRANSPORT"
    result = {
        "stage": "delta_binding_slot_transport",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "inject_layer": INJECT_LAYER,
        "fresh_offsets": list(FRESH_OFFSETS),
        "transport_layers": list(TRANSPORT_LAYERS),
        "n_trials": len(rows),
        "base": base,
        "individual_layers": individual,
        "cumulative": cumulative,
        "gates": {"G0": g0},
        "shared_direct_layers": direct_layers,
        "verdict": verdict,
    }
    path = os.path.join(out_dir, "results_delta_binding_slot_transport.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {verdict}")
    return result
