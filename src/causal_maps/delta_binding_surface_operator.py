"""Fixed cross-surface affine binding-operator replication.

The original grammar is ``Let X = value``.  This stage uses only the
out-of-distribution mapping grammar ``X maps to value`` while retaining the
two-address binding structure, held-out substitutions, L2 ADD, and matched
wrong-direction / other-slot controls.  It tests recurrence of the operator
phenomenon across surface forms, not identity of a vector learned in the old
grammar.
"""
import json
import os

import numpy as np
import torch

from .delta_binding_cross_model_gate import _split_rows, _tokenizer_valid_values
from .delta_operator import DONOR_NAMES, _directions, _encode_uniform, _trials
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer, single_token_id
from .tensorize import _anchor_token_index


INJECT_LAYER = 2
PROTOCOL_VERSION = "2026-07-14-v2"
PROMPT_FAMILY = "mapping_table_value_completion_v2"


def _surface_single_text(tok, name, value):
    user = f"In the table, {name} maps to {value}. What does {name} map to?"
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True)
    # The answer token must be the very next token. The v1 free-form
    # completion ("It maps to") made an article such as "the" the greedy
    # next token, invalidating the behavioral gate before any causal claim.
    text = prefix + f"{name} ="
    marker = f"{name} maps to "
    offset = text.find(marker) + len(marker)
    if text.find(marker) < 0:
        raise ValueError("mapping donor anchor not found")
    return text, offset


def _surface_two_text(tok, x, y, x_value, y_value, query):
    user = (f"In the table, {x} maps to {x_value}; {y} maps to {y_value}. "
            f"What does {query} map to?")
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True)
    text = prefix + f"{query} ="
    x_marker, y_marker = f"{x} maps to ", f"{y} maps to "
    if text.find(x_marker) < 0 or text.find(y_marker) < 0:
        raise ValueError("mapping two-binding anchor not found")
    return (text, text.find(x_marker) + len(x_marker),
            text.find(y_marker) + len(y_marker))


def _build_mapping_group(tok, rows, query):
    selected = [row for row in rows if row["query"] == query]
    clean_rows, natural_rows, x_positions, y_positions = [], [], [], []
    for row in selected:
        if query == "X":
            clean_args = (row["source"], row["distractor"])
            natural_args = (row["target"], row["distractor"])
        else:
            clean_args = (row["distractor"], row["source"])
            natural_args = (row["distractor"], row["target"])
        clean, clean_x, clean_y = _surface_two_text(
            tok, "X", "Y", *clean_args, query)
        natural, natural_x, natural_y = _surface_two_text(
            tok, "X", "Y", *natural_args, query)
        own_clean = clean_x if query == "X" else clean_y
        own_natural = natural_x if query == "X" else natural_y
        clean_rows.append((clean, own_clean))
        natural_rows.append((natural, own_natural))
        x_positions.append(_anchor_token_index(tok, clean, clean_x))
        y_positions.append(_anchor_token_index(tok, clean, clean_y))
    clean_ids, clean_am, own_position = _encode_uniform(tok, clean_rows)
    natural_ids, natural_am, natural_own = _encode_uniform(tok, natural_rows)
    if clean_ids.shape != natural_ids.shape or own_position != natural_own:
        raise ValueError("mapping clean/natural alignment failed")
    if None in x_positions or None in y_positions:
        raise ValueError("mapping slot anchor is unstable")
    if len(set(x_positions)) != 1 or len(set(y_positions)) != 1:
        raise ValueError("mapping slot anchors are non-uniform")
    other_position = y_positions[0] if query == "X" else x_positions[0]
    return (selected, clean_ids, clean_am, natural_ids, natural_am,
            int(own_position), int(other_position))


def _prototype_map(model, tok, dev, values):
    rows, observed = [], []
    for name in DONOR_NAMES:
        for value in values:
            rows.append(_surface_single_text(tok, name, value))
            observed.append(value)
    ids, am, position = _encode_uniform(tok, rows)
    _logits, cache = _forward(
        model, ids.to(dev), am.to(dev), (position,), (INJECT_LAYER,))
    hidden = cache[INJECT_LAYER][:, 0]
    return ({value: hidden[[i for i, seen in enumerate(observed)
                            if seen == value]].mean(0)
             for value in values}, int(position))


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def _evaluate(model, tok, dev, prototypes, rows):
    data = {key: [] for key in ("natural", "add", "wrong", "other",
                                 "clean_hits", "natural_hits")}
    for query in ("X", "Y"):
        group, ci, cam, ni, nam, own, other = _build_mapping_group(tok, rows, query)
        ci, cam, ni, nam = ci.to(dev), cam.to(dev), ni.to(dev), nam.to(dev)
        last = int(ci.shape[1] - 1)
        pos_ids = torch.tensor([single_token_id(tok, row["target"])
                                for row in group])
        neg_ids = torch.tensor([single_token_id(tok, row["source"])
                                for row in group])
        direction = _directions(prototypes, group, key="target")
        wrong_direction = _directions(prototypes, group, key="wrong")
        clean_logits, _ = _forward(model, ci, cam, (own, last))
        natural_logits, _ = _forward(model, ni, nam, (own, last))
        add_logits, _ = _forward(
            model, ci, cam, (own, last), add=(INJECT_LAYER, own, direction))
        wrong_logits, _ = _forward(
            model, ci, cam, (own, last), add=(INJECT_LAYER, own, wrong_direction))
        other_logits, _ = _forward(
            model, ci, cam, (own, last), add=(INJECT_LAYER, other, direction))
        clean_ld = _ld(clean_logits, pos_ids, neg_ids)
        data["natural"].extend((_ld(natural_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["add"].extend((_ld(add_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["wrong"].extend((_ld(wrong_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["other"].extend((_ld(other_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["clean_hits"].extend((clean_logits.argmax(-1) == neg_ids).tolist())
        data["natural_hits"].extend((natural_logits.argmax(-1) == pos_ids).tolist())
    natural, add = _mean(data["natural"]), _mean(data["add"])
    return {
        "n_rows": len(rows),
        "clean_acc": _mean(data["clean_hits"]),
        "natural_acc": _mean(data["natural_hits"]),
        "natural_effect": natural,
        "add_effect": add,
        "wrong_direction_effect": _mean(data["wrong"]),
        "other_slot_effect": _mean(data["other"]),
        "positive_add_fraction": _mean([row > 0 for row in data["add"]]),
        "effect_ratio": float(add / natural) if natural > 0 else None,
    }


def _confirmed(metrics):
    ratio = metrics["effect_ratio"]
    return bool(metrics["clean_acc"] >= .80 and metrics["natural_acc"] >= .80
                and metrics["natural_effect"] > 0 and metrics["add_effect"] > 0
                and metrics["positive_add_fraction"] >= .80
                and ratio is not None and .70 <= ratio <= 1.30
                and metrics["add_effect"] > metrics["wrong_direction_effect"] + .10
                and metrics["add_effect"] > metrics["other_slot_effect"] + .10)


@torch.no_grad()
def run_delta_binding_surface_operator(model_path, out_dir, quantization="8bit",
                                       device_map=None, seed=0):
    del seed
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    values, excluded = _tokenizer_valid_values(tok)
    result = {"stage": "delta_binding_surface_operator",
              "protocol_version": PROTOCOL_VERSION, "prompt_family": PROMPT_FAMILY,
              "model_path": model_path, "inject_layer": INJECT_LAYER,
              "tokenizer_valid_values": values, "tokenizer_excluded_values": excluded,
              "heldout_offsets": [5, 7]}
    if len(values) < 8:
        result["verdict"] = "SURFACE_OPERATOR_TOKENIZATION_INELIGIBLE"
        return _write(out_dir, result)
    _discovery, heldout = _split_rows(_trials(values))
    prototypes, donor_position = _prototype_map(model, tok, dev, values)
    log("delta_binding_surface_operator: "
        f"family={PROMPT_FAMILY} values={len(values)} heldout={len(heldout)} "
        f"inject=L{INJECT_LAYER}")
    metrics = _evaluate(model, tok, dev, prototypes, heldout)
    metrics["donor_position"] = donor_position
    result.update({"heldout": metrics,
                   "verdict": ("SURFACE_OPERATOR_CONFIRMED"
                               if _confirmed(metrics)
                               else "SURFACE_OPERATOR_NOT_CONFIRMED")})
    log(f"heldout: natural={metrics['natural_effect']:+.3f} "
        f"add={metrics['add_effect']:+.3f} ratio={metrics['effect_ratio']:.3f} "
        f"wrong={metrics['wrong_direction_effect']:+.3f} "
        f"other={metrics['other_slot_effect']:+.3f}")
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_binding_surface_operator.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
