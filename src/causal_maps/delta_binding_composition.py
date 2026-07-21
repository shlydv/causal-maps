"""Simultaneous, address-specific composition of literal binding controllers."""
import json
import os

import numpy as np
import torch

from .delta_binding_cross_model_gate import _tokenizer_valid_values
from .delta_binding_cross_surface_transfer import _let_prototype_map
from .delta_operator import _directions, _encode_uniform
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer, single_token_id
from .tensorize import _anchor_token_index


INJECT_LAYER = 2
MIN_VALUES = 8
PROTOCOL_VERSION = "2026-07-14-v1"
PROMPT_FAMILY = "mapping_table_three_binding_v1"


def _three_text(tok, x_value, y_value, z_value, query):
    user = (f"In the table, X maps to {x_value}; Y maps to {y_value}; "
            f"Z maps to {z_value}. What does {query} map to?")
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True)
    text = prefix + f"{query} ="
    offsets = []
    for name in ("X", "Y", "Z"):
        marker = f"{name} maps to "
        start = text.find(marker)
        if start < 0:
            raise ValueError(f"three-binding anchor missing for {name}")
        offsets.append(start + len(marker))
    return text, tuple(offsets)


def _composition_rows(values):
    """Fresh three-binding arrangements; offsets 5/7 are held-out pair changes."""
    rows, n = [], len(values)
    for i in range(n):
        row = {
            "source_x": values[i], "source_y": values[(i + 1) % n],
            "source_z": values[(i + 2) % n],
            "target_x": values[(i + 5) % n], "target_y": values[(i + 7) % n],
        }
        if len({row["source_x"], row["source_y"], row["source_z"],
                row["target_x"], row["target_y"]}) < 5:
            raise ValueError("composition row values must be distinct")
        rows.extend([{**row, "query": query} for query in ("X", "Y")])
    return rows


def _build_group(tok, rows, query):
    selected = [row for row in rows if row["query"] == query]
    clean_rows, natural_rows, positions = [], [], [[], [], []]
    for row in selected:
        clean, clean_offsets = _three_text(
            tok, row["source_x"], row["source_y"], row["source_z"], query)
        natural, natural_offsets = _three_text(
            tok, row["target_x"], row["target_y"], row["source_z"], query)
        query_idx = 0 if query == "X" else 1
        clean_rows.append((clean, clean_offsets[query_idx]))
        natural_rows.append((natural, natural_offsets[query_idx]))
        for idx, offset in enumerate(clean_offsets):
            positions[idx].append(_anchor_token_index(tok, clean, offset))
    ci, cam, own = _encode_uniform(tok, clean_rows)
    ni, nam, natural_own = _encode_uniform(tok, natural_rows)
    if ci.shape != ni.shape or own != natural_own:
        raise ValueError("three-binding clean/natural alignment failed")
    if any(None in row or len(set(row)) != 1 for row in positions):
        raise ValueError("three-binding anchors are not uniform")
    return selected, ci, cam, ni, nam, tuple(int(row[0]) for row in positions)


def _query_rows(group, query):
    own, other = ("x", "y") if query == "X" else ("y", "x")
    return [{"source": row[f"source_{own}"], "target": row[f"target_{own}"]}
            for row in group]


def _mean(items):
    return float(np.mean(items)) if items else float("nan")


@torch.no_grad()
def _evaluate(model, tok, dev, prototypes, rows):
    data = {key: [] for key in ("natural", "solo", "joint", "swapped",
                                 "clean_hits", "natural_hits", "crosstalk")}
    for query in ("X", "Y"):
        group, ci, cam, ni, nam, positions = _build_group(tok, rows, query)
        ci, cam, ni, nam = ci.to(dev), cam.to(dev), ni.to(dev), nam.to(dev)
        x_pos, y_pos, _z_pos = positions
        own_pos = x_pos if query == "X" else y_pos
        last = int(ci.shape[1] - 1)
        own_rows = _query_rows(group, query)
        x_rows = [{"source": r["source_x"], "target": r["target_x"]} for r in group]
        y_rows = [{"source": r["source_y"], "target": r["target_y"]} for r in group]
        x_delta = _directions(prototypes, x_rows, key="target")
        y_delta = _directions(prototypes, y_rows, key="target")
        pos_ids = torch.tensor([single_token_id(tok, row["target"])
                                for row in own_rows])
        neg_ids = torch.tensor([single_token_id(tok, row["source"])
                                for row in own_rows])
        clean_logits, _ = _forward(model, ci, cam, (own_pos, last))
        natural_logits, _ = _forward(model, ni, nam, (own_pos, last))
        solo_delta = x_delta if query == "X" else y_delta
        solo_logits, _ = _forward(
            model, ci, cam, (own_pos, last), add=(INJECT_LAYER, own_pos, solo_delta))
        joint_logits, _ = _forward(
            model, ci, cam, (own_pos, last), add=[
                (INJECT_LAYER, x_pos, x_delta), (INJECT_LAYER, y_pos, y_delta)])
        swapped_logits, _ = _forward(
            model, ci, cam, (own_pos, last), add=[
                (INJECT_LAYER, x_pos, y_delta), (INJECT_LAYER, y_pos, x_delta)])
        clean_ld = _ld(clean_logits, pos_ids, neg_ids)
        natural = _ld(natural_logits, pos_ids, neg_ids) - clean_ld
        solo = _ld(solo_logits, pos_ids, neg_ids) - clean_ld
        joint = _ld(joint_logits, pos_ids, neg_ids) - clean_ld
        swapped = _ld(swapped_logits, pos_ids, neg_ids) - clean_ld
        data["natural"].extend(natural.tolist())
        data["solo"].extend(solo.tolist())
        data["joint"].extend(joint.tolist())
        data["swapped"].extend(swapped.tolist())
        data["crosstalk"].extend((joint - solo).tolist())
        data["clean_hits"].extend((clean_logits.argmax(-1) == neg_ids).tolist())
        data["natural_hits"].extend((natural_logits.argmax(-1) == pos_ids).tolist())
    natural, solo, joint = (_mean(data[key]) for key in ("natural", "solo", "joint"))
    return {
        "n_rows": len(rows), "clean_acc": _mean(data["clean_hits"]),
        "natural_acc": _mean(data["natural_hits"]), "natural_effect": natural,
        "own_only_effect": solo, "joint_effect": joint,
        "swapped_joint_effect": _mean(data["swapped"]),
        "mean_joint_minus_own_only": _mean(data["crosstalk"]),
        "positive_joint_fraction": _mean([row > 0 for row in data["joint"]]),
        "joint_to_natural_ratio": float(joint / natural) if natural > 0 else None,
        "joint_to_own_only_ratio": float(joint / solo) if solo > 0 else None,
    }


def _confirmed(metrics):
    natural_ratio, solo_ratio = (metrics["joint_to_natural_ratio"],
                                  metrics["joint_to_own_only_ratio"])
    return bool(
        metrics["clean_acc"] >= .80 and metrics["natural_acc"] >= .80
        and metrics["natural_effect"] > 0 and metrics["joint_effect"] > 0
        and metrics["positive_joint_fraction"] >= .80
        and natural_ratio is not None and .70 <= natural_ratio <= 1.30
        and solo_ratio is not None and .70 <= solo_ratio <= 1.30
        and abs(metrics["mean_joint_minus_own_only"])
        <= .15 * metrics["natural_effect"]
        and metrics["joint_effect"] > metrics["swapped_joint_effect"] + .10)


@torch.no_grad()
def run_delta_binding_composition(model_path, out_dir, quantization="8bit",
                                  device_map=None, seed=0):
    del seed
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    values, excluded = _tokenizer_valid_values(tok)
    result = {"stage": "delta_binding_composition", "protocol_version": PROTOCOL_VERSION,
              "model_path": model_path, "inject_layer": INJECT_LAYER,
              "prompt_family": PROMPT_FAMILY, "tokenizer_valid_values": values,
              "tokenizer_excluded_values": excluded,
              "rule": "two raw Let-derived L2 deltas installed simultaneously at X and Y"}
    if len(values) < MIN_VALUES:
        result["verdict"] = "BINDING_COMPOSITION_TOKENIZATION_INELIGIBLE"
        return _write(out_dir, result)
    rows = _composition_rows(values)
    prototypes, donor_position = _let_prototype_map(model, tok, dev, values)
    log(f"delta_binding_composition: values={len(values)} rows={len(rows)} "
        f"inject=L{INJECT_LAYER} simultaneous_writes=2")
    metrics = _evaluate(model, tok, dev, prototypes, rows)
    metrics["let_donor_position"] = donor_position
    metrics["confirmed"] = _confirmed(metrics)
    result.update({"heldout": metrics,
                   "verdict": ("COMPOSITIONAL_BINDING_CONTROLLER_CONFIRMED"
                               if metrics["confirmed"]
                               else "COMPOSITIONAL_BINDING_CONTROLLER_NOT_CONFIRMED")})
    log(f"heldout: natural={metrics['natural_effect']:+.3f} "
        f"own={metrics['own_only_effect']:+.3f} joint={metrics['joint_effect']:+.3f} "
        f"joint/natural={metrics['joint_to_natural_ratio']} "
        f"joint/own={metrics['joint_to_own_only_ratio']} "
        f"swap={metrics['swapped_joint_effect']:+.3f} "
        f"crosstalk={metrics['mean_joint_minus_own_only']:+.3f}")
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_binding_composition.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
