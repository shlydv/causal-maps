"""Cross-skill approximate natural-reachability atlas.

Frozen design: REACHABILITY_ATLAS_PROTOCOL.md.
"""
import json
import os

import numpy as np
import torch

from .delta_instruction import (_data_text, _encode_pool as _encode_instruction,
                                _instr_text, _payloads)
from .delta_select import (SELECT_TEMPLATES, _encode_pool as _encode_select,
                           _pair_pool, _render)
from .delta_trajectory import (_cos_rows, _error_rows, _forward, _ld,
                               _random_like_rows)
from .delta_transform import (COMPUTED_TEMPLATES, _computed_text,
                              _encode_pool as _encode_transform)
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer, single_token_id
from .nulls import permutation_pvalue

N_NULL = 100
EPS = 1e-8


def _mean(rows):
    return float(np.mean(rows))


@torch.no_grad()
def _extract_direction(model, clean_ids, clean_am, cf_ids, cf_am,
                       layer, clean_pos, cf_pos):
    _, clean_cache = _forward(
        model, clean_ids, clean_am, (clean_pos,), (layer,))
    _, cf_cache = _forward(
        model, cf_ids, cf_am, (cf_pos,), (layer,))
    return (cf_cache[layer][:, 0] - clean_cache[layer][:, 0]).mean(0)


@torch.no_grad()
def _evaluate_cell(model, name, clean_ids, clean_am, cf_ids, cf_am,
                   pos_ids, neg_ids, direction, inject_layer, checkpoint_layer,
                   clean_site, cf_site, behav_gate, generator, n_null=N_NULL):
    clean_last = int(clean_ids.shape[1] - 1)
    cf_last = int(cf_ids.shape[1] - 1)
    capture_layers = tuple(sorted({inject_layer, checkpoint_layer}))

    clean_logits, clean_cache = _forward(
        model, clean_ids, clean_am, (clean_site, clean_last), capture_layers)
    cf_logits, cf_cache = _forward(
        model, cf_ids, cf_am, (cf_site, cf_last), capture_layers)
    clean_acc = float((clean_logits.argmax(-1) == neg_ids.cpu()).float().mean())
    cf_acc = float((cf_logits.argmax(-1) == pos_ids.cpu()).float().mean())
    g0 = clean_acc >= behav_gate and cf_acc >= behav_gate
    base = {
        "name": name,
        "inject_layer": inject_layer,
        "checkpoint_layer": checkpoint_layer,
        "n": int(clean_ids.shape[0]),
        "behavior": {
            "clean_acc": clean_acc, "cf_acc": cf_acc,
            "threshold": behav_gate, "pass": bool(g0),
        },
    }
    if not g0:
        return {**base, "gates": {"G0": False}, "verdict": "INELICITABLE"}

    batch_direction = direction.unsqueeze(0).expand(clean_ids.shape[0], -1)
    clean_ld = _ld(clean_logits, pos_ids, neg_ids)
    natural_effect_rows = _ld(cf_logits, pos_ids, neg_ids) - clean_ld

    add_logits, add_cache = _forward(
        model, clean_ids, clean_am, (clean_site, clean_last), capture_layers,
        add=(inject_layer, clean_site, batch_direction))
    add_effect_rows = _ld(add_logits, pos_ids, neg_ids) - clean_ld

    anti_logits, anti_cache = _forward(
        model, clean_ids, clean_am, (clean_site, clean_last), capture_layers,
        add=(inject_layer, clean_site, -batch_direction))
    anti_effect_rows = _ld(anti_logits, pos_ids, neg_ids) - clean_ld

    native_inject = (
        cf_cache[inject_layer][:, 0] - clean_cache[inject_layer][:, 0])
    add_inject = (
        add_cache[inject_layer][:, 0] - clean_cache[inject_layer][:, 0])
    anti_inject = (
        anti_cache[inject_layer][:, 0] - clean_cache[inject_layer][:, 0])
    native_query = (
        cf_cache[checkpoint_layer][:, 1]
        - clean_cache[checkpoint_layer][:, 1])
    add_query = (
        add_cache[checkpoint_layer][:, 1]
        - clean_cache[checkpoint_layer][:, 1])
    inject_cos_rows = _cos_rows(add_inject, native_inject)
    inject_error_rows = _error_rows(add_inject, native_inject)
    anti_cos_rows = _cos_rows(anti_inject, native_inject)
    query_cos_rows = _cos_rows(add_query, native_query)
    query_error_rows = _error_rows(add_query, native_query)

    patch_add_logits, _ = _forward(
        model, clean_ids, clean_am, (clean_site, clean_last),
        patch=(checkpoint_layer, clean_site,
               add_cache[checkpoint_layer][:, 0]))
    patch_cf_logits, _ = _forward(
        model, clean_ids, clean_am, (clean_site, clean_last),
        patch=(checkpoint_layer, clean_site,
               cf_cache[checkpoint_layer][:, 0]))
    blocked_add_logits, _ = _forward(
        model, clean_ids, clean_am, (clean_site, clean_last),
        add=(inject_layer, clean_site, batch_direction),
        patch=(checkpoint_layer, clean_site,
               clean_cache[checkpoint_layer][:, 0]))
    blocked_cf_logits, _ = _forward(
        model, cf_ids, cf_am, (cf_site, cf_last),
        patch=(checkpoint_layer, cf_site,
               clean_cache[checkpoint_layer][:, 0]))
    patch_add_rows = _ld(patch_add_logits, pos_ids, neg_ids) - clean_ld
    patch_cf_rows = _ld(patch_cf_logits, pos_ids, neg_ids) - clean_ld
    blocked_add_rows = _ld(blocked_add_logits, pos_ids, neg_ids) - clean_ld
    blocked_cf_rows = _ld(blocked_cf_logits, pos_ids, neg_ids) - clean_ld

    null_output = np.zeros(n_null, dtype=np.float64)
    null_inject_cos = np.zeros(n_null, dtype=np.float64)
    null_query_cos = np.zeros(n_null, dtype=np.float64)
    for null_idx in range(n_null):
        random = _random_like_rows(batch_direction, generator)
        random_logits, random_cache = _forward(
            model, clean_ids, clean_am, (clean_site, clean_last),
            capture_layers, add=(inject_layer, clean_site, random))
        random_effect = _ld(random_logits, pos_ids, neg_ids) - clean_ld
        random_inject = (
            random_cache[inject_layer][:, 0]
            - clean_cache[inject_layer][:, 0])
        random_query = (
            random_cache[checkpoint_layer][:, 1]
            - clean_cache[checkpoint_layer][:, 1])
        null_output[null_idx] = float(random_effect.mean())
        null_inject_cos[null_idx] = float(
            _cos_rows(random_inject, native_inject).mean())
        null_query_cos[null_idx] = float(
            _cos_rows(random_query, native_query).mean())

    output_effect = float(add_effect_rows.mean())
    natural_effect = float(natural_effect_rows.mean())
    output_ratio = output_effect / natural_effect if natural_effect > EPS else -np.inf
    patch_add = float(patch_add_rows.mean())
    patch_cf = float(patch_cf_rows.mean())
    patch_ratio = patch_add / patch_cf if patch_cf > EPS else -np.inf
    blocked_add = float(blocked_add_rows.mean())
    blocked_cf = float(blocked_cf_rows.mean())
    add_block_fraction = (
        (output_effect - blocked_add) / output_effect
        if output_effect > EPS else -np.inf)
    cf_block_fraction = (
        (natural_effect - blocked_cf) / natural_effect
        if natural_effect > EPS else -np.inf)
    block_gap = abs(add_block_fraction - cf_block_fraction)
    inject_cos = float(inject_cos_rows.mean())
    inject_error = float(inject_error_rows.mean())
    query_cos = float(query_cos_rows.mean())
    query_error = float(query_error_rows.mean())
    anti_cos = float(anti_cos_rows.mean())
    positive_fraction = float((add_effect_rows > 0).float().mean())
    inject_norm_ratio = float(
        (add_cache[inject_layer][:, 0].norm(dim=1)
         / cf_cache[inject_layer][:, 0].norm(dim=1).clamp(min=EPS)).mean())
    checkpoint_norm_ratio = float(
        (add_cache[checkpoint_layer][:, 0].norm(dim=1)
         / cf_cache[checkpoint_layer][:, 0].norm(dim=1).clamp(min=EPS)).mean())

    inject_p = permutation_pvalue(inject_cos, null_inject_cos, "greater")
    query_p = permutation_pvalue(query_cos, null_query_cos, "greater")
    output_p = permutation_pvalue(output_effect, null_output, "greater")
    gates = {
        "G0": True,
        "A1": bool(inject_cos >= 0.80 and inject_error <= 0.60
                   and inject_p < 0.01 and inject_cos > anti_cos),
        "Q1": bool(query_cos >= 0.50 and query_error <= 0.80
                   and query_p < 0.01),
        "O1": bool(output_effect > 0 and output_p < 0.01
                   and 0.70 <= output_ratio <= 1.30
                   and positive_fraction >= 0.80),
        "M1": bool(patch_add > 0 and patch_cf > 0
                   and 0.70 <= patch_ratio <= 1.30),
        "M2": bool(add_block_fraction >= 0.70
                   and cf_block_fraction >= 0.70 and block_gap <= 0.20),
        "D1": bool(0.80 <= inject_norm_ratio <= 1.20
                   and 0.80 <= checkpoint_norm_ratio <= 1.20),
    }
    if all(gates.values()):
        verdict = "NATURAL_REACHABLE"
    elif gates["O1"]:
        verdict = "OUTPUT_EQUIVALENT_ONLY"
    else:
        verdict = "CONTROL_NULL"

    rows = []
    for i in range(clean_ids.shape[0]):
        rows.append({
            "natural_effect": float(natural_effect_rows[i]),
            "add_effect": float(add_effect_rows[i]),
            "anti_effect": float(anti_effect_rows[i]),
            "inject_cos": float(inject_cos_rows[i]),
            "inject_error": float(inject_error_rows[i]),
            "anti_inject_cos": float(anti_cos_rows[i]),
            "query_cos": float(query_cos_rows[i]),
            "query_error": float(query_error_rows[i]),
            "patch_add_effect": float(patch_add_rows[i]),
            "patch_cf_effect": float(patch_cf_rows[i]),
            "blocked_add_effect": float(blocked_add_rows[i]),
            "blocked_cf_effect": float(blocked_cf_rows[i]),
        })
    return {
        **base,
        "direction_norm": float(direction.norm()),
        "metrics": {
            "inject_cos": inject_cos,
            "inject_error": inject_error,
            "anti_inject_cos": anti_cos,
            "inject_p": float(inject_p),
            "query_cos": query_cos,
            "query_error": query_error,
            "query_p": float(query_p),
            "output_effect": output_effect,
            "natural_effect": natural_effect,
            "output_ratio": float(output_ratio),
            "positive_fraction": positive_fraction,
            "output_p": float(output_p),
            "anti_output_effect": float(anti_effect_rows.mean()),
            "patch_add_effect": patch_add,
            "patch_cf_effect": patch_cf,
            "patch_ratio": float(patch_ratio),
            "blocked_add_effect": blocked_add,
            "blocked_cf_effect": blocked_cf,
            "add_block_fraction": float(add_block_fraction),
            "cf_block_fraction": float(cf_block_fraction),
            "block_fraction_gap": float(block_gap),
            "inject_norm_ratio": inject_norm_ratio,
            "checkpoint_norm_ratio": checkpoint_norm_ratio,
        },
        "gates": gates,
        "rows": rows,
        "null_draws": {
            "output_effect": null_output.tolist(),
            "inject_cos": null_inject_cos.tolist(),
            "query_cos": null_query_cos.tolist(),
        },
        "verdict": verdict,
    }


def _select_data(tok, dev, seed):
    tmpl = next(row for row in SELECT_TEMPLATES if row["name"] == "value_of")
    rng = np.random.default_rng(seed)
    pairs = _pair_pool(tok, tmpl, rng, 16)
    if len(pairs) != 16:
        raise ValueError(f"select expected 16 pairs, got {len(pairs)}")
    train, test = pairs[:8], pairs[8:]

    def build(rows, flag):
        texts, keys, flags = [], [], []
        for va, vb in rows:
            text, key, f = _render(tok, tmpl, va, vb, flag)
            texts.append(text); keys.append(key); flags.append(f)
        return _encode_select(tok, texts, keys, flags, dev)

    tr0, tram0, trpos = build(train, 0)
    tr1, tram1, trpos1 = build(train, 1)
    te0, team0, tepos = build(test, 0)
    te1, team1, tepos1 = build(test, 1)
    if trpos != trpos1 or tepos != tepos1:
        raise ValueError("select clean/cf flag positions differ")
    pos_ids = torch.tensor([single_token_id(tok, va) for va, _ in test])
    neg_ids = torch.tensor([single_token_id(tok, vb) for _, vb in test])
    return (tr0, tram0, tr1, tram1, trpos,
            te0, team0, te1, team1, tepos, pos_ids, neg_ids)


def _transform_data(tok, dev, seed):
    tmpl = next(row for row in COMPUTED_TEMPLATES
                if row["name"] == "direct_sum")
    pairs = [(a, b) for a in range(1, 9) for b in range(1, 8)
             if 3 <= a + b <= 8]
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    cut = (2 * len(pairs)) // 3
    train, test = pairs[:cut], pairs[cut:]

    def build(rows, increment):
        texts = [_computed_text(tok, a, b + increment, tmpl)
                 for a, b in rows]
        return _encode_transform(tok, texts, dev)

    tr0, tram0 = build(train, 0)
    tr1, tram1 = build(train, 1)
    te0, team0 = build(test, 0)
    te1, team1 = build(test, 1)
    trpos0, trpos1 = tr0.shape[1] - 1, tr1.shape[1] - 1
    tepos0, tepos1 = te0.shape[1] - 1, te1.shape[1] - 1
    pos_ids = torch.tensor([
        single_token_id(tok, str(a + b + 1), leading_space=False)
        for a, b in test])
    neg_ids = torch.tensor([
        single_token_id(tok, str(a + b), leading_space=False)
        for a, b in test])
    return (tr0, tram0, tr1, tram1, trpos0, trpos1,
            te0, team0, te1, team1, tepos0, tepos1, pos_ids, neg_ids)


def _instruction_data(tok, dev, seed):
    words = _payloads(tok)
    rng = np.random.default_rng(seed)
    rng.shuffle(words)
    train, test = words[:8], words[8:16]
    trd, tramd = _encode_instruction(
        tok, [_data_text(tok, word) for word in train], dev)
    tri, trami = _encode_instruction(
        tok, [_instr_text(tok, word) for word in train], dev)
    ted, teamd = _encode_instruction(
        tok, [_data_text(tok, word) for word in test], dev)
    tei, teami = _encode_instruction(
        tok, [_instr_text(tok, word) for word in test], dev)
    pos_ids = torch.tensor([
        single_token_id(tok, word, leading_space=False) for word in test])
    out_id = single_token_id(tok, "Output", leading_space=False)
    neg_ids = torch.tensor([out_id] * len(test))
    return (trd, tramd, tri, trami, trd.shape[1] - 1, tri.shape[1] - 1,
            ted, teamd, tei, teami, ted.shape[1] - 1, tei.shape[1] - 1,
            pos_ids, neg_ids)


@torch.no_grad()
def run_delta_reachability(model_path, out_dir, quantization="8bit",
                           device_map=None, seed=0, n_null=N_NULL):
    if n_null < 100:
        raise ValueError("reachability p<.01 gates require n_null >= 100")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    generator = torch.Generator().manual_seed(seed + 9017)
    cells = {}

    data = _select_data(tok, dev, seed)
    (tr0, tram0, tr1, tram1, trpos,
     te0, team0, te1, team1, tepos, pos_ids, neg_ids) = data
    direction = _extract_direction(
        model, tr0, tram0, tr1, tram1, 2, trpos, trpos)
    cells["select"] = _evaluate_cell(
        model, "select", te0, team0, te1, team1, pos_ids, neg_ids,
        direction, 2, 8, tepos, tepos, 0.80, generator, n_null)
    log(f"atlas select: {cells['select']['verdict']} "
        f"{cells['select'].get('gates')}")

    data = _transform_data(tok, dev, seed)
    (tr0, tram0, tr1, tram1, trpos0, trpos1,
     te0, team0, te1, team1, tepos0, tepos1, pos_ids, neg_ids) = data
    direction = _extract_direction(
        model, tr0, tram0, tr1, tram1, 20, trpos0, trpos1)
    cells["transform"] = _evaluate_cell(
        model, "transform", te0, team0, te1, team1, pos_ids, neg_ids,
        direction, 20, 26, tepos0, tepos1, 0.80, generator, n_null)
    log(f"atlas transform: {cells['transform']['verdict']} "
        f"{cells['transform'].get('gates')}")

    data = _instruction_data(tok, dev, seed)
    (tr0, tram0, tr1, tram1, trpos0, trpos1,
     te0, team0, te1, team1, tepos0, tepos1, pos_ids, neg_ids) = data
    direction = _extract_direction(
        model, tr0, tram0, tr1, tram1, 20, trpos0, trpos1)
    cells["instruction"] = _evaluate_cell(
        model, "instruction", te0, team0, te1, team1, pos_ids, neg_ids,
        direction, 20, 26, tepos0, tepos1, 0.70, generator, n_null)
    log(f"atlas instruction: {cells['instruction']['verdict']} "
        f"{cells['instruction'].get('gates')}")

    reachable = [name for name, cell in cells.items()
                 if cell["verdict"] == "NATURAL_REACHABLE"]
    if len(reachable) >= 2:
        verdict = "COUNTERFACTUAL_OPERATORS_GENERALIZE"
    elif reachable == ["select"]:
        verdict = "BINDING_ROUTING_REACHABILITY"
    elif not reachable:
        verdict = "BINDING_SPECIFIC_REACHABILITY"
    else:
        verdict = "MIXED_REACHABILITY"
    result = {
        "stage": "delta_reachability",
        "model_path": model_path,
        "n_null": int(n_null),
        "store_reference": {
            "operator": "AFFINE_COUNTERFACTUAL_OPERATOR",
            "content": "CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR",
        },
        "cells": cells,
        "reachable_cells": reachable,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_reachability.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"VERDICT: {verdict} reachable={reachable}")
    return result
