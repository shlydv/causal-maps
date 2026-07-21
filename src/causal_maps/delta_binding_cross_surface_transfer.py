"""Strict literal controller transfer across binding surface forms.

The prior surface replication re-derived value directions inside the mapping
grammar.  This stage asks the substantially stronger question: does the raw
L2 direction learned from ``Let X = value`` change a value in the mapping
grammar *without* any rescaling, alignment, or template-specific fitting?
"""
import json
import os

import numpy as np
import torch

from .delta_binding_cross_model_gate import _split_rows, _tokenizer_valid_values
from .delta_binding_surface_operator import _build_mapping_group, _prototype_map
from .delta_operator import DONOR_NAMES, _directions, _encode_uniform, _single_text, _trials
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer, single_token_id


INJECT_LAYER = 2
MIN_VALUES = 8
PROTOCOL_VERSION = "2026-07-14-v1"
SOURCE_FAMILY = "let_value_completion"
TARGET_FAMILY = "mapping_table_value_completion_v2"


def _let_prototype_map(model, tok, dev, values):
    """Value prototypes from the original grammar only, at the fixed L2."""
    rows, observed = [], []
    for name in DONOR_NAMES:
        for value in values:
            rows.append(_single_text(tok, name, value))
            observed.append(value)
    ids, am, position = _encode_uniform(tok, rows)
    _logits, cache = _forward(
        model, ids.to(dev), am.to(dev), (position,), (INJECT_LAYER,))
    hidden = cache[INJECT_LAYER][:, 0]
    return ({value: hidden[[i for i, seen in enumerate(observed)
                            if seen == value]].mean(0)
             for value in values}, int(position))


def _mean(rows):
    return float(np.mean(rows)) if rows else float("nan")


@torch.no_grad()
def _evaluate(model, tok, dev, let_prototypes, mapping_prototypes, rows):
    """Evaluate a frozen literal transfer alongside a mapping-native baseline."""
    data = {key: [] for key in (
        "natural", "native", "transfer", "wrong", "other",
        "clean_hits", "natural_hits")}
    for query in ("X", "Y"):
        group, ci, cam, ni, nam, own, other = _build_mapping_group(tok, rows, query)
        ci, cam, ni, nam = ci.to(dev), cam.to(dev), ni.to(dev), nam.to(dev)
        last = int(ci.shape[1] - 1)
        pos_ids = torch.tensor([single_token_id(tok, row["target"])
                                for row in group])
        neg_ids = torch.tensor([single_token_id(tok, row["source"])
                                for row in group])
        let_direction = _directions(let_prototypes, group, key="target")
        let_wrong_direction = _directions(let_prototypes, group, key="wrong")
        native_direction = _directions(mapping_prototypes, group, key="target")
        clean_logits, _ = _forward(model, ci, cam, (own, last))
        natural_logits, _ = _forward(model, ni, nam, (own, last))
        native_logits, _ = _forward(
            model, ci, cam, (own, last),
            add=(INJECT_LAYER, own, native_direction))
        transfer_logits, _ = _forward(
            model, ci, cam, (own, last),
            add=(INJECT_LAYER, own, let_direction))
        wrong_logits, _ = _forward(
            model, ci, cam, (own, last),
            add=(INJECT_LAYER, own, let_wrong_direction))
        other_logits, _ = _forward(
            model, ci, cam, (own, last),
            add=(INJECT_LAYER, other, let_direction))
        clean_ld = _ld(clean_logits, pos_ids, neg_ids)
        data["natural"].extend((_ld(natural_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["native"].extend((_ld(native_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["transfer"].extend((_ld(transfer_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["wrong"].extend((_ld(wrong_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["other"].extend((_ld(other_logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["clean_hits"].extend((clean_logits.argmax(-1) == neg_ids).tolist())
        data["natural_hits"].extend((natural_logits.argmax(-1) == pos_ids).tolist())
    natural, native, transfer = (_mean(data[key]) for key in
                                 ("natural", "native", "transfer"))
    return {
        "n_rows": len(rows),
        "clean_acc": _mean(data["clean_hits"]),
        "natural_acc": _mean(data["natural_hits"]),
        "natural_effect": natural,
        "mapping_native_add_effect": native,
        "literal_transfer_add_effect": transfer,
        "literal_wrong_direction_effect": _mean(data["wrong"]),
        "literal_other_slot_effect": _mean(data["other"]),
        "positive_literal_transfer_fraction": _mean(
            [row > 0 for row in data["transfer"]]),
        "literal_to_natural_ratio": float(transfer / natural) if natural > 0 else None,
        "literal_to_native_ratio": float(transfer / native) if native > 0 else None,
    }


def _mapping_baseline_confirmed(metrics):
    """Re-check that this model/prompt instance has the expected native effect."""
    ratio = (float(metrics["mapping_native_add_effect"] / metrics["natural_effect"])
             if metrics["natural_effect"] > 0 else None)
    return bool(
        metrics["clean_acc"] >= .80 and metrics["natural_acc"] >= .80
        and metrics["natural_effect"] > 0
        and metrics["mapping_native_add_effect"] > 0
        and ratio is not None and .70 <= ratio <= 1.30)


def _literal_transfer_confirmed(metrics):
    natural_ratio = metrics["literal_to_natural_ratio"]
    native_ratio = metrics["literal_to_native_ratio"]
    return bool(
        _mapping_baseline_confirmed(metrics)
        and metrics["literal_transfer_add_effect"] > 0
        and metrics["positive_literal_transfer_fraction"] >= .80
        and natural_ratio is not None and .70 <= natural_ratio <= 1.30
        and native_ratio is not None and .70 <= native_ratio <= 1.30
        and metrics["literal_transfer_add_effect"]
        > metrics["literal_wrong_direction_effect"] + .10
        and metrics["literal_transfer_add_effect"]
        > metrics["literal_other_slot_effect"] + .10)


@torch.no_grad()
def run_delta_binding_cross_surface_transfer(model_path, out_dir,
                                             quantization="8bit",
                                             device_map=None, seed=0):
    del seed
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    values, excluded = _tokenizer_valid_values(tok)
    result = {
        "stage": "delta_binding_cross_surface_transfer",
        "protocol_version": PROTOCOL_VERSION,
        "model_path": model_path,
        "inject_layer": INJECT_LAYER,
        "source_family": SOURCE_FAMILY,
        "target_family": TARGET_FAMILY,
        "tokenizer_valid_values": values,
        "tokenizer_excluded_values": excluded,
        "heldout_offsets": [5, 7],
        "transfer_rule": "raw let prototype(target)-prototype(source); no rescaling or alignment",
    }
    if len(values) < MIN_VALUES:
        result["verdict"] = "LITERAL_CROSS_SURFACE_TRANSFER_TOKENIZATION_INELIGIBLE"
        return _write(out_dir, result)
    _discovery, heldout = _split_rows(_trials(values))
    let_prototypes, let_donor_position = _let_prototype_map(model, tok, dev, values)
    mapping_prototypes, mapping_donor_position = _prototype_map(model, tok, dev, values)
    log("delta_binding_cross_surface_transfer: "
        f"values={len(values)} heldout={len(heldout)} inject=L{INJECT_LAYER} "
        "source=let target=mapping raw_delta=True")
    metrics = _evaluate(
        model, tok, dev, let_prototypes, mapping_prototypes, heldout)
    metrics["let_donor_position"] = let_donor_position
    metrics["mapping_donor_position"] = mapping_donor_position
    metrics["mapping_baseline_confirmed"] = _mapping_baseline_confirmed(metrics)
    metrics["literal_transfer_confirmed"] = _literal_transfer_confirmed(metrics)
    result.update({
        "heldout": metrics,
        "verdict": ("LITERAL_CROSS_SURFACE_CONTROLLER_CONFIRMED"
                    if metrics["literal_transfer_confirmed"]
                    else "LITERAL_CROSS_SURFACE_CONTROLLER_NOT_CONFIRMED"),
    })
    log(f"heldout: natural={metrics['natural_effect']:+.3f} "
        f"native={metrics['mapping_native_add_effect']:+.3f} "
        f"literal={metrics['literal_transfer_add_effect']:+.3f} "
        f"literal/natural={metrics['literal_to_natural_ratio']} "
        f"literal/native={metrics['literal_to_native_ratio']} "
        f"wrong={metrics['literal_wrong_direction_effect']:+.3f} "
        f"other={metrics['literal_other_slot_effect']:+.3f}")
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_binding_cross_surface_transfer.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
