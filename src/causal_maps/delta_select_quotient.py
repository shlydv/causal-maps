"""Causal interchange test for geometrically divergent SELECT query states."""
import json
import os

import numpy as np
import torch

from .delta_reachability import _extract_direction, _select_data
from .delta_trajectory import _cos_rows, _error_rows, _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

INJECT_LAYER = 2
LAYERS = (4, 8, 12, 14, 16, 20, 26)
EPS = 1e-8


@torch.no_grad()
def run_delta_select_quotient(model_path, out_dir, quantization="8bit",
                              device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    data = _select_data(tok, dev, seed)
    (tr0, tram0, tr1, tram1, trpos,
     te0, team0, te1, team1, tepos, pos_ids, neg_ids) = data
    direction = _extract_direction(
        model, tr0, tram0, tr1, tram1,
        INJECT_LAYER, trpos, trpos)
    batch_direction = direction.unsqueeze(0).expand(te0.shape[0], -1)
    last = int(te0.shape[1] - 1)
    capture_layers = (INJECT_LAYER, *LAYERS)

    clean_logits, clean_cache = _forward(
        model, te0, team0, (tepos, last), capture_layers)
    natural_logits, natural_cache = _forward(
        model, te1, team1, (tepos, last), capture_layers)
    add_logits, add_cache = _forward(
        model, te0, team0, (tepos, last), capture_layers,
        add=(INJECT_LAYER, tepos, batch_direction))
    clean_ld = _ld(clean_logits, pos_ids, neg_ids)
    natural_rows = _ld(natural_logits, pos_ids, neg_ids) - clean_ld
    add_rows = _ld(add_logits, pos_ids, neg_ids) - clean_ld
    natural_effect = float(natural_rows.mean())
    add_effect = float(add_rows.mean())
    output_ratio = add_effect / natural_effect if natural_effect > EPS else -np.inf
    behavior = {
        "clean_acc": float(
            (clean_logits.argmax(-1) == neg_ids).float().mean()),
        "natural_acc": float(
            (natural_logits.argmax(-1) == pos_ids).float().mean()),
        "natural_effect": natural_effect,
        "add_effect": add_effect,
        "output_ratio": float(output_ratio),
    }
    g0 = (behavior["clean_acc"] >= 0.80
          and behavior["natural_acc"] >= 0.80
          and 0.90 <= output_ratio <= 1.10)
    if not g0:
        result = {
            "stage": "delta_select_quotient",
            "model_path": model_path,
            "behavior": behavior,
            "gates": {"G0": False},
            "verdict": "SELECT_INELICITABLE",
        }
        with open(os.path.join(
                out_dir, "results_delta_select_quotient.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        log(f"VERDICT: SELECT_INELICITABLE behavior={behavior}")
        return result

    per_layer = []
    for layer in LAYERS:
        native_query = (
            natural_cache[layer][:, 1] - clean_cache[layer][:, 1])
        add_query = add_cache[layer][:, 1] - clean_cache[layer][:, 1]
        cos_rows = _cos_rows(add_query, native_query)
        error_rows = _error_rows(add_query, native_query)
        cosine = float(cos_rows.mean())
        error = float(error_rows.mean())

        patch_add_logits, _ = _forward(
            model, te0, team0, (tepos, last),
            patch=(layer, last, add_cache[layer][:, 1]))
        patch_nat_logits, _ = _forward(
            model, te0, team0, (tepos, last),
            patch=(layer, last, natural_cache[layer][:, 1]))
        swap_add_nat_logits, _ = _forward(
            model, te0, team0, (tepos, last),
            add=(INJECT_LAYER, tepos, batch_direction),
            patch=(layer, last, natural_cache[layer][:, 1]))
        swap_nat_add_logits, _ = _forward(
            model, te1, team1, (tepos, last),
            patch=(layer, last, add_cache[layer][:, 1]))
        block_add_logits, _ = _forward(
            model, te0, team0, (tepos, last),
            add=(INJECT_LAYER, tepos, batch_direction),
            patch=(layer, last, clean_cache[layer][:, 1]))
        block_nat_logits, _ = _forward(
            model, te1, team1, (tepos, last),
            patch=(layer, last, clean_cache[layer][:, 1]))

        patch_add_rows = _ld(patch_add_logits, pos_ids, neg_ids) - clean_ld
        patch_nat_rows = _ld(patch_nat_logits, pos_ids, neg_ids) - clean_ld
        swap_add_nat_rows = (
            _ld(swap_add_nat_logits, pos_ids, neg_ids) - clean_ld)
        swap_nat_add_rows = (
            _ld(swap_nat_add_logits, pos_ids, neg_ids) - clean_ld)
        block_add_rows = _ld(block_add_logits, pos_ids, neg_ids) - clean_ld
        block_nat_rows = _ld(block_nat_logits, pos_ids, neg_ids) - clean_ld
        patch_add = float(patch_add_rows.mean())
        patch_nat = float(patch_nat_rows.mean())
        patch_ratio = patch_add / patch_nat if patch_nat > EPS else -np.inf
        swap_add_nat = float(swap_add_nat_rows.mean())
        swap_nat_add = float(swap_nat_add_rows.mean())
        swap_add_deviation = abs(swap_add_nat - add_effect) / abs(natural_effect)
        swap_nat_deviation = abs(swap_nat_add - natural_effect) / abs(natural_effect)
        blocked_add = float(block_add_rows.mean())
        blocked_nat = float(block_nat_rows.mean())
        add_block_fraction = (
            (add_effect - blocked_add) / add_effect
            if add_effect > EPS else -np.inf)
        nat_block_fraction = (
            (natural_effect - blocked_nat) / natural_effect
            if natural_effect > EPS else -np.inf)
        discrepancy = cosine < 0.50 or error > 0.80
        causal_equivalent = bool(
            discrepancy
            and patch_add > 0 and patch_nat > 0
            and 0.80 <= patch_ratio <= 1.20
            and swap_add_deviation <= 0.10
            and swap_nat_deviation <= 0.10
            and add_block_fraction >= 0.50
            and nat_block_fraction >= 0.50)
        delayed_convergence = bool(
            layer > 8 and cosine >= 0.80 and error <= 0.60)
        per_layer.append({
            "layer": layer,
            "cosine": cosine,
            "error": error,
            "geometric_discrepancy": bool(discrepancy),
            "patch_add_effect": patch_add,
            "patch_natural_effect": patch_nat,
            "patch_ratio": float(patch_ratio),
            "swap_add_to_natural_effect": swap_add_nat,
            "swap_natural_to_add_effect": swap_nat_add,
            "swap_add_deviation": float(swap_add_deviation),
            "swap_natural_deviation": float(swap_nat_deviation),
            "blocked_add_effect": blocked_add,
            "blocked_natural_effect": blocked_nat,
            "add_block_fraction": float(add_block_fraction),
            "natural_block_fraction": float(nat_block_fraction),
            "causal_equivalent": causal_equivalent,
            "delayed_convergence": delayed_convergence,
            "rows": [{
                "cosine": float(cos_rows[i]),
                "error": float(error_rows[i]),
                "patch_add_effect": float(patch_add_rows[i]),
                "patch_natural_effect": float(patch_nat_rows[i]),
                "swap_add_to_natural_effect": float(swap_add_nat_rows[i]),
                "swap_natural_to_add_effect": float(swap_nat_add_rows[i]),
                "blocked_add_effect": float(block_add_rows[i]),
                "blocked_natural_effect": float(block_nat_rows[i]),
            } for i in range(te0.shape[0])],
        })

    l8 = next(row for row in per_layer if row["layer"] == 8)
    d0 = bool(l8["geometric_discrepancy"])
    quotient_layers = [
        row["layer"] for row in per_layer if row["causal_equivalent"]]
    convergence_layers = [
        row["layer"] for row in per_layer if row["delayed_convergence"]]
    if not d0:
        verdict = "DISCREPANCY_NOT_REPLICATED"
    elif quotient_layers:
        verdict = "CAUSAL_QUOTIENT_EQUIVALENCE"
    elif convergence_layers:
        verdict = "DELAYED_NATURAL_CONVERGENCE"
    else:
        verdict = "PARALLEL_OR_UNRESOLVED_PATHS"
    result = {
        "stage": "delta_select_quotient",
        "model_path": model_path,
        "layers": list(LAYERS),
        "direction_norm": float(direction.norm()),
        "behavior": behavior,
        "gates": {"G0": True, "D0": d0},
        "causal_equivalence_layers": quotient_layers,
        "delayed_convergence_layers": convergence_layers,
        "per_layer": per_layer,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_select_quotient.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"L8 discrepancy cos={l8['cosine']:.3f} error={l8['error']:.3f}")
    log(f"quotient_layers={quotient_layers} convergence_layers={convergence_layers}")
    log(f"VERDICT: {verdict}")
    return result
