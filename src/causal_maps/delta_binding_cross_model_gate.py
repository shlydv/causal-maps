"""Cross-model behavioral replication gate for the binding affine operator.

This deliberately asks a small question before porting the longer causal
timeline: can a tokenizer-valid set of binding substitutions be reproduced by
an affine residual addition at an *early queried value slot* in another model?
Injection depth is selected on discovery substitutions and evaluated only on
held-out substitutions.  It makes no claim that a positive result shares the
same circuit as Qwen.
"""
import json
import os

import numpy as np
import torch

from . import variable_pairs
from .delta_operator import (DONOR_NAMES, _build_multi_group, _directions,
                             _encode_uniform, _single_text, _trials)
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import (input_device, load_model_and_tokenizer,
                          single_token_id)


MIN_VALUES = 8
DEPTH_FRACTIONS = (0.06, 0.12, 0.20)
PROTOCOL_VERSION = "2026-07-14-v1"


def _candidate_layers(n_layers):
    """Three pre-registered early depths, normalized to architecture depth."""
    if n_layers < 8:
        raise ValueError(f"need at least 8 layers, got {n_layers}")
    layers = sorted({max(1, min(n_layers - 2, round((n_layers - 1) * fraction)))
                     for fraction in DEPTH_FRACTIONS})
    if len(layers) != len(DEPTH_FRACTIONS):
        raise ValueError(f"early-depth candidates collapsed for {n_layers} layers")
    return tuple(layers)


def _tokenizer_valid_values(tok):
    valid, excluded = [], []
    for _source, value in variable_pairs._VALUE_PAIRS:
        try:
            single_token_id(tok, value)
            valid.append(value)
        except ValueError:
            excluded.append(value)
    return valid, excluded


def _split_rows(rows):
    """Within-source offset 1/3 discovery and 5/7 held-out split."""
    if len(rows) % 4:
        raise ValueError(f"trials must have four rows per source, got {len(rows)}")
    discovery, test = [], []
    for start in range(0, len(rows), 4):
        discovery.extend(rows[start:start + 2])
        test.extend(rows[start + 2:start + 4])
    for name, split in (("discovery", discovery), ("test", test)):
        counts = {query: sum(row["query"] == query for row in split)
                  for query in ("X", "Y")}
        expected = len(rows) // 4
        if counts != {"X": expected, "Y": expected}:
            raise ValueError(f"{name} query balance invalid: {counts}")
    return discovery, test


def _prototype_map(model, tok, dev, values, layer):
    donor_rows, donor_values = [], []
    for name in DONOR_NAMES:
        for value in values:
            donor_rows.append(_single_text(tok, name, value))
            donor_values.append(value)
    ids, am, position = _encode_uniform(tok, donor_rows)
    _logits, cache = _forward(model, ids.to(dev), am.to(dev), (position,), (layer,))
    hidden = cache[layer][:, 0]
    return {value: hidden[[i for i, seen in enumerate(donor_values)
                           if seen == value]].mean(0)
            for value in values}, int(position)


def _aggregate(values):
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def _evaluate_layer(model, tok, dev, prototypes, rows, layer):
    """Measure target, wrong-direction, and wrong-slot effects for one depth."""
    output = {key: [] for key in (
        "natural_effect_rows", "add_effect_rows", "wrong_effect_rows",
        "other_slot_effect_rows", "clean_hits", "natural_hits")}
    for query in ("X", "Y"):
        group, ci, cam, fi, fam, own_pos, other_pos = _build_multi_group(
            tok, rows, query)
        ci, cam, fi, fam = ci.to(dev), cam.to(dev), fi.to(dev), fam.to(dev)
        last = int(ci.shape[1] - 1)
        pos_ids = torch.tensor([single_token_id(tok, row["target"])
                                for row in group])
        neg_ids = torch.tensor([single_token_id(tok, row["source"])
                                for row in group])
        target_direction = _directions(prototypes, group, key="target")
        wrong_direction = _directions(prototypes, group, key="wrong")
        clean_logits, _ = _forward(model, ci, cam, (own_pos, last))
        natural_logits, _ = _forward(model, fi, fam, (own_pos, last))
        add_logits, _ = _forward(
            model, ci, cam, (own_pos, last), add=(layer, own_pos, target_direction))
        wrong_logits, _ = _forward(
            model, ci, cam, (own_pos, last), add=(layer, own_pos, wrong_direction))
        other_logits, _ = _forward(
            model, ci, cam, (own_pos, last), add=(layer, other_pos, target_direction))
        clean_ld = _ld(clean_logits, pos_ids, neg_ids)
        output["natural_effect_rows"].extend((_ld(natural_logits, pos_ids, neg_ids) - clean_ld).tolist())
        output["add_effect_rows"].extend((_ld(add_logits, pos_ids, neg_ids) - clean_ld).tolist())
        output["wrong_effect_rows"].extend((_ld(wrong_logits, pos_ids, neg_ids) - clean_ld).tolist())
        output["other_slot_effect_rows"].extend((_ld(other_logits, pos_ids, neg_ids) - clean_ld).tolist())
        output["clean_hits"].extend((clean_logits.argmax(-1) == neg_ids).tolist())
        output["natural_hits"].extend((natural_logits.argmax(-1) == pos_ids).tolist())
    natural = _aggregate(output["natural_effect_rows"])
    add = _aggregate(output["add_effect_rows"])
    result = {
        "layer": int(layer),
        "n_rows": len(rows),
        "clean_acc": _aggregate(output["clean_hits"]),
        "natural_acc": _aggregate(output["natural_hits"]),
        "natural_effect": natural,
        "add_effect": add,
        "wrong_direction_effect": _aggregate(output["wrong_effect_rows"]),
        "other_slot_effect": _aggregate(output["other_slot_effect_rows"]),
        "positive_add_fraction": _aggregate(
            [item > 0 for item in output["add_effect_rows"]]),
        "effect_ratio": float(add / natural) if natural > 0 else float("nan"),
    }
    return result


def _discovery_viable(metrics):
    """Selection gate; intentionally weaker than the held-out confirmation."""
    ratio = metrics["effect_ratio"]
    return bool(
        metrics["clean_acc"] >= 0.70
        and metrics["natural_acc"] >= 0.70
        and metrics["natural_effect"] > 0
        and metrics["add_effect"] > 0
        and metrics["positive_add_fraction"] >= 0.70
        and 0.50 <= ratio <= 1.50
        and metrics["add_effect"] > metrics["wrong_direction_effect"] + 0.10)


def _heldout_confirmed(metrics):
    ratio = metrics["effect_ratio"]
    return bool(
        metrics["clean_acc"] >= 0.80
        and metrics["natural_acc"] >= 0.80
        and metrics["natural_effect"] > 0
        and metrics["add_effect"] > 0
        and metrics["positive_add_fraction"] >= 0.80
        and 0.70 <= ratio <= 1.30
        and metrics["add_effect"] > metrics["wrong_direction_effect"] + 0.10
        and metrics["add_effect"] > metrics["other_slot_effect"] + 0.10)


@torch.no_grad()
def run_delta_binding_cross_model_gate(model_path, out_dir, quantization="8bit",
                                       device_map=None, seed=0,
                                       trust_remote_code=False):
    del seed  # Deterministic design; retained for runner compatibility.
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization,
        trust_remote_code=trust_remote_code)
    dev = input_device(model)
    values, excluded = _tokenizer_valid_values(tok)
    base = {
        "stage": "delta_binding_cross_model_gate",
        "protocol_version": PROTOCOL_VERSION,
        "model_path": model_path,
        "trust_remote_code": bool(trust_remote_code),
        "num_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "tokenizer_valid_values": values,
        "tokenizer_excluded_values": excluded,
    }
    if len(values) < MIN_VALUES:
        base["verdict"] = "CROSS_MODEL_TOKENIZATION_INELICITABLE"
        _write(out_dir, base)
        log(f"VERDICT: {base['verdict']} valid_values={len(values)}")
        return base

    rows = _trials(values)
    discovery_rows, test_rows = _split_rows(rows)
    layers = _candidate_layers(int(model.config.num_hidden_layers))
    log("delta_binding_cross_model_gate: "
        f"values={len(values)} discovery={len(discovery_rows)} heldout={len(test_rows)} "
        f"candidate_layers={layers}")
    discovery = []
    for layer in layers:
        prototypes, donor_position = _prototype_map(model, tok, dev, values, layer)
        metrics = _evaluate_layer(model, tok, dev, prototypes, discovery_rows, layer)
        metrics["donor_position"] = donor_position
        metrics["viable"] = _discovery_viable(metrics)
        discovery.append(metrics)
        log(f"discovery L{layer}: natural={metrics['natural_effect']:+.3f} "
            f"add={metrics['add_effect']:+.3f} ratio={metrics['effect_ratio']:.3f} "
            f"pos={metrics['positive_add_fraction']:.2f} viable={metrics['viable']}")
    viable = [row for row in discovery if row["viable"]]
    base.update({"candidate_layers": list(layers), "discovery": discovery})
    if not viable:
        base["verdict"] = "CROSS_MODEL_OPERATOR_NOT_ELICITABLE"
        _write(out_dir, base)
        log(f"VERDICT: {base['verdict']}")
        return base
    selected = max(viable, key=lambda row: (row["add_effect"], -row["layer"]))
    layer = selected["layer"]
    prototypes, donor_position = _prototype_map(model, tok, dev, values, layer)
    heldout = _evaluate_layer(model, tok, dev, prototypes, test_rows, layer)
    heldout["donor_position"] = donor_position
    heldout["confirmed"] = _heldout_confirmed(heldout)
    base.update({"selected_layer": layer, "selection_rule": "largest viable discovery add_effect",
                 "heldout": heldout,
                 "verdict": ("CROSS_MODEL_AFFINE_OPERATOR_CONFIRMED"
                             if heldout["confirmed"]
                             else "CROSS_MODEL_OPERATOR_NOT_CONFIRMED")})
    _write(out_dir, base)
    log(f"heldout L{layer}: natural={heldout['natural_effect']:+.3f} "
        f"add={heldout['add_effect']:+.3f} ratio={heldout['effect_ratio']:.3f} "
        f"confirmed={heldout['confirmed']}")
    log(f"VERDICT: {base['verdict']}")
    return base


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_binding_cross_model_gate.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
