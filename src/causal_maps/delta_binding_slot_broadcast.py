"""Held-out-layout distributed receiver broadcast test for binding.

Frozen design: BINDING_SLOT_BROADCAST_PROTOCOL.md.
"""
import json
import os

import numpy as np
import torch

from .delta_operator import (
    DONOR_NAMES, INJECT_LAYER, _directions, _encode_uniform, _single_text,
    _values)
from .delta_trajectory import EPS, _forward, _ld
from .logutil import log
from .model_utils import (
    get_decoder_layers, input_device, load_model_and_tokenizer, single_token_id)
from .patching import _split_output

BROADCAST_LAYERS = (20, 21, 22, 23, 24, 25, 26)
LAYOUT_OFFSETS = tuple(range(1, 9))
PROTOCOL_VERSION = "2026-07-13-v1"


def _layout_trials(values):
    """Balanced mappings for the held-out reversed-order prompt layout."""
    rows = []
    count = len(values)
    for source_idx, source in enumerate(values):
        for index, offset in enumerate(LAYOUT_OFFSETS):
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


def _reversed_two_var_text(tok, x, y, x_value, y_value, query):
    """Two bindings in Y-then-X order, returning X and Y value anchors."""
    user = (f"Let {y} = {y_value}. Let {x} = {x_value}. "
            f"What is the value of {query}?")
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True)
    text = prefix + f"{query} ="
    x_marker = f"Let {x} = "
    y_marker = f"Let {y} = "
    x_offset = text.find(x_marker) + len(x_marker)
    y_offset = text.find(y_marker) + len(y_marker)
    if text.find(x_marker) < 0 or text.find(y_marker) < 0:
        raise ValueError("reversed binding anchors not found")
    return text, x_offset, y_offset


def _groups(tok, rows, prototypes, device):
    groups = []
    for query in ("X", "Y"):
        selected = [row for row in rows if row["query"] == query]
        clean_rows, natural_rows, x_positions, y_positions = [], [], [], []
        for row in selected:
            if query == "X":
                clean_args = (row["source"], row["distractor"])
                natural_args = (row["target"], row["distractor"])
            else:
                clean_args = (row["distractor"], row["source"])
                natural_args = (row["distractor"], row["target"])
            clean, clean_x, clean_y = _reversed_two_var_text(
                tok, "X", "Y", *clean_args, query)
            natural, natural_x, natural_y = _reversed_two_var_text(
                tok, "X", "Y", *natural_args, query)
            own_clean = clean_x if query == "X" else clean_y
            own_natural = natural_x if query == "X" else natural_y
            clean_rows.append((clean, own_clean))
            natural_rows.append((natural, own_natural))
            x_positions.append(clean_x)
            y_positions.append(clean_y)
        clean_ids, clean_am, own_pos = _encode_uniform(tok, clean_rows)
        natural_ids, natural_am, natural_own_pos = _encode_uniform(
            tok, natural_rows)
        if clean_ids.shape != natural_ids.shape or own_pos != natural_own_pos:
            raise ValueError("reversed clean/natural alignment failed")
        from .tensorize import _anchor_token_index
        x_tokens = [_anchor_token_index(tok, row[0], pos)
                    for row, pos in zip(clean_rows, x_positions)]
        y_tokens = [_anchor_token_index(tok, row[0], pos)
                    for row, pos in zip(clean_rows, y_positions)]
        if len(set(x_tokens)) != 1 or len(set(y_tokens)) != 1:
            raise ValueError("reversed slot positions are not uniform")
        other_pos = y_tokens[0] if query == "X" else x_tokens[0]
        groups.append({
            "query": query,
            "rows": selected,
            "clean_ids": clean_ids.to(device),
            "clean_am": clean_am.to(device),
            "natural_ids": natural_ids.to(device),
            "natural_am": natural_am.to(device),
            "own_pos": int(own_pos),
            "other_pos": int(other_pos),
            "direction": _directions(prototypes, selected),
            "pos_ids": torch.tensor(
                [single_token_id(tok, row["target"]) for row in selected]),
            "neg_ids": torch.tensor(
                [single_token_id(tok, row["source"]) for row in selected]),
        })
    return groups


@torch.no_grad()
def _forward_broadcast(model, input_ids, attention_mask, inject_position,
                       direction=None, patch=None, blocked_layers=(),
                       key_position=None, blocked_layer_keys=()):
    """Optional ADD/state patch plus all-later-queries-to-key edge blocks.

    ``blocked_layers`` uses the one shared ``key_position`` shorthand.
    ``blocked_layer_keys`` is an explicit iterable of ``(layer, key)`` pairs
    for interventions that require different controlled keys in different
    windows. Duplicate layer/key pairs are harmless; two distinct keys at one
    layer are both blocked.
    """
    layers = get_decoder_layers(model)
    blocked_layers = tuple(sorted(set(blocked_layers)))
    sequence = int(input_ids.shape[1])
    if blocked_layers and (key_position is None
                           or not 0 <= key_position < sequence):
        raise ValueError(f"invalid broadcast key position: {key_position}")
    edge_specs = {(layer, key_position) for layer in blocked_layers}
    edge_specs.update((int(layer), int(key)) for layer, key in blocked_layer_keys)
    if any(layer < 0 or layer >= len(layers) for layer, _key in edge_specs):
        raise ValueError(f"invalid broadcast layers: {sorted(edge_specs)}")
    if any(not 0 <= key < sequence for _layer, key in edge_specs):
        raise ValueError(f"invalid broadcast key positions: {sorted(edge_specs)}")
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

    if patch is not None:
        patch_layer, patch_position, patch_value = patch
        if not 0 <= patch_layer < len(layers):
            raise ValueError(f"invalid patch layer: {patch_layer}")
        if not 0 <= patch_position < sequence:
            raise ValueError(f"invalid patch position: {patch_position}")

        def patch_hook(_module, _inputs, output):
            hidden, rebuild = _split_output(output)
            edited = hidden.clone()
            replacement = patch_value.to(device=edited.device, dtype=edited.dtype)
            if replacement.shape != (edited.shape[0], edited.shape[-1]):
                raise ValueError("patch state shape does not match hidden state")
            edited[:, patch_position] = replacement
            return rebuild(edited)
        handles.append(layers[patch_layer].register_forward_hook(patch_hook))

    def broadcast_hook(keys):
        def hook(_module, args, kwargs):
            mask = kwargs.get("attention_mask")
            if mask is None or mask.ndim != 4:
                raise ValueError("expected Qwen 4D additive attention mask")
            if mask.shape[-2] < sequence or any(mask.shape[-1] <= key for key in keys):
                raise ValueError(f"attention mask too short: {tuple(mask.shape)}")
            if mask.shape[1] not in (1, num_heads):
                raise ValueError(f"unexpected mask head dimension: {mask.shape[1]}")
            edited = mask.expand(
                mask.shape[0], num_heads, mask.shape[-2], mask.shape[-1]).clone()
            for key in keys:
                edited[:, :, key + 1:, key] = torch.finfo(edited.dtype).min
            kwargs["attention_mask"] = edited
            return args, kwargs
        return hook

    try:
        by_layer = {}
        for layer, key in edge_specs:
            by_layer.setdefault(layer, set()).add(key)
        for layer, keys in by_layer.items():
            handles.append(layers[layer].self_attn.register_forward_pre_hook(
                broadcast_hook(tuple(sorted(keys))), with_kwargs=True))
        batch = input_ids.shape[0]
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
    """CLEAN/NATURAL/ADD effects under one matched broadcast intervention."""
    natural_rows, add_rows, clean_hits, natural_hits, add_positive = [], [], [], [], []
    for group in groups:
        key_position = group[f"{slot}_pos"]
        clean_logits = _forward_broadcast(
            model, group["clean_ids"], group["clean_am"], group["own_pos"],
            blocked_layers=blocked_layers, key_position=key_position)
        natural_logits = _forward_broadcast(
            model, group["natural_ids"], group["natural_am"], group["own_pos"],
            blocked_layers=blocked_layers, key_position=key_position)
        add_logits = _forward_broadcast(
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


def _condition(base, own, other):
    own_natural = _loss(base, own, "natural_effect")
    own_add = _loss(base, own, "add_effect")
    control_natural = _loss(base, other, "natural_effect")
    control_add = _loss(base, other, "add_effect")
    gap_natural = (own_natural - control_natural if own_natural is not None
                   and control_natural is not None else None)
    gap_add = (own_add - control_add if own_add is not None
               and control_add is not None else None)
    essential = bool(
        own_natural is not None and own_add is not None and gap_natural is not None
        and gap_add is not None and own_natural >= .80 and own_add >= .80
        and gap_natural >= .50 and gap_add >= .50)
    partial = bool(
        own_natural is not None and own_add is not None and gap_natural is not None
        and gap_add is not None and own_natural >= .50 and own_add >= .50
        and gap_natural >= .25 and gap_add >= .25)
    return {
        "own": own,
        "other_slot_control": other,
        "own_losses": {"natural": own_natural, "add": own_add},
        "own_minus_other_loss": {"natural": gap_natural, "add": gap_add},
        "essential_shared_broadcast": essential,
        "partial_shared_broadcast": partial,
    }


@torch.no_grad()
def run_delta_binding_slot_broadcast(
        model_path, out_dir, quantization="8bit", device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    layers = get_decoder_layers(model)
    if len(layers) <= max(BROADCAST_LAYERS):
        raise ValueError(f"model lacks broadcast layer {max(BROADCAST_LAYERS)}")
    values = _values(tok)
    rows = _layout_trials(values)
    if len(rows) != len(values) * len(LAYOUT_OFFSETS):
        raise ValueError("broadcast trial count malformed")
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
    log("delta_binding_slot_broadcast: "
        f"heldout_layout=reversed rows={len(rows)} layers={BROADCAST_LAYERS}")
    base = _measure(model, groups)
    ratio = (base["add_effect"] / base["natural_effect"]
             if base["natural_effect"] > EPS else None)
    base["add_to_natural_effect_ratio"] = ratio
    g0 = bool(
        base["clean_acc"] >= .80 and base["natural_acc"] >= .80
        and base["natural_effect"] > EPS and base["add_effect"] > EPS
        and base["add_positive_fraction"] >= .80 and ratio is not None
        and .70 <= ratio <= 1.30)
    result = {
        "stage": "delta_binding_slot_broadcast",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "inject_layer": INJECT_LAYER,
        "prompt_layout": "reversed_Y_then_X",
        "layout_offsets": list(LAYOUT_OFFSETS),
        "broadcast_layers": list(BROADCAST_LAYERS),
        "n_trials": len(rows),
        "base": base,
        "gates": {"G0": g0},
    }
    if not g0:
        result["verdict"] = "BROADCAST_INELICITABLE"
    else:
        own = _measure(model, groups, BROADCAST_LAYERS, "own")
        other = _measure(model, groups, BROADCAST_LAYERS, "other")
        condition = _condition(base, own, other)
        result["cumulative_broadcast"] = condition
        if condition["essential_shared_broadcast"]:
            result["verdict"] = "SHARED_BROADCAST_ESSENTIAL"
        elif condition["partial_shared_broadcast"]:
            result["verdict"] = "SHARED_BROADCAST_PARTIAL"
        else:
            result["verdict"] = "DIVERGENT_OR_UNRESOLVED_BROADCAST"
        log(f"cumulative own_losses={condition['own_losses']} "
            f"gaps={condition['own_minus_other_loss']} "
            f"essential={condition['essential_shared_broadcast']} "
            f"partial={condition['partial_shared_broadcast']}")
    path = os.path.join(out_dir, "results_delta_binding_slot_broadcast.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
