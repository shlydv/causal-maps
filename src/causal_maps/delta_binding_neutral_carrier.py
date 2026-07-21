"""Competing-explanation test: binding controller versus neutral lexical state."""
import json
import os

import numpy as np
import torch

from .delta_binding_cross_model_gate import _split_rows, _tokenizer_valid_values
from .delta_binding_cross_surface_transfer import _let_prototype_map
from .delta_binding_surface_operator import _build_mapping_group
from .delta_operator import _directions, _encode_uniform, _trials
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer, single_token_id


INJECT_LAYER = 2
MIN_VALUES = 8
PROTOCOL_VERSION = "2026-07-14-v1"
NEUTRAL_CARRIER = "Here is a token: VALUE."


def _neutral_text(tok, value):
    user = f"Here is a token: {value}."
    text = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True)
    marker = "Here is a token: "
    start = text.find(marker)
    if start < 0:
        raise ValueError("neutral carrier anchor not found")
    return text, start + len(marker)


def _neutral_prototype_map(model, tok, dev, values):
    rows = [_neutral_text(tok, value) for value in values]
    ids, am, position = _encode_uniform(tok, rows)
    _logits, cache = _forward(
        model, ids.to(dev), am.to(dev), (position,), (INJECT_LAYER,))
    hidden = cache[INJECT_LAYER][:, 0]
    return {value: hidden[idx] for idx, value in enumerate(values)}, int(position)


def _mean(items):
    return float(np.mean(items)) if items else float("nan")


@torch.no_grad()
def _evaluate(model, tok, dev, let_prototypes, neutral_prototypes, rows):
    data = {key: [] for key in ("natural", "let", "neutral", "wrong", "other",
                                 "clean_hits", "natural_hits")}
    for query in ("X", "Y"):
        group, ci, cam, ni, nam, own, other = _build_mapping_group(tok, rows, query)
        ci, cam, ni, nam = ci.to(dev), cam.to(dev), ni.to(dev), nam.to(dev)
        last = int(ci.shape[1] - 1)
        pos_ids = torch.tensor([single_token_id(tok, row["target"]) for row in group])
        neg_ids = torch.tensor([single_token_id(tok, row["source"]) for row in group])
        let_delta = _directions(let_prototypes, group, key="target")
        neutral_delta = _directions(neutral_prototypes, group, key="target")
        neutral_wrong = _directions(neutral_prototypes, group, key="wrong")
        clean_logits, _ = _forward(model, ci, cam, (own, last))
        natural_logits, _ = _forward(model, ni, nam, (own, last))
        let_logits, _ = _forward(
            model, ci, cam, (own, last), add=(INJECT_LAYER, own, let_delta))
        neutral_logits, _ = _forward(
            model, ci, cam, (own, last), add=(INJECT_LAYER, own, neutral_delta))
        wrong_logits, _ = _forward(
            model, ci, cam, (own, last), add=(INJECT_LAYER, own, neutral_wrong))
        other_logits, _ = _forward(
            model, ci, cam, (own, last), add=(INJECT_LAYER, other, neutral_delta))
        clean_ld = _ld(clean_logits, pos_ids, neg_ids)
        for key, logits in (("natural", natural_logits), ("let", let_logits),
                            ("neutral", neutral_logits), ("wrong", wrong_logits),
                            ("other", other_logits)):
            data[key].extend((_ld(logits, pos_ids, neg_ids) - clean_ld).tolist())
        data["clean_hits"].extend((clean_logits.argmax(-1) == neg_ids).tolist())
        data["natural_hits"].extend((natural_logits.argmax(-1) == pos_ids).tolist())
    natural, let, neutral = (_mean(data[key]) for key in ("natural", "let", "neutral"))
    return {
        "n_rows": len(rows), "clean_acc": _mean(data["clean_hits"]),
        "natural_acc": _mean(data["natural_hits"]), "natural_effect": natural,
        "let_controller_effect": let, "neutral_carrier_effect": neutral,
        "neutral_wrong_direction_effect": _mean(data["wrong"]),
        "neutral_other_slot_effect": _mean(data["other"]),
        "positive_neutral_fraction": _mean([row > 0 for row in data["neutral"]]),
        "neutral_to_natural_ratio": float(neutral / natural) if natural > 0 else None,
        "neutral_to_let_ratio": float(neutral / let) if let > 0 else None,
    }


def _neutral_matches(metrics):
    nr, lr = metrics["neutral_to_natural_ratio"], metrics["neutral_to_let_ratio"]
    return bool(
        metrics["clean_acc"] >= .80 and metrics["natural_acc"] >= .80
        and metrics["let_controller_effect"] > 0 and metrics["neutral_carrier_effect"] > 0
        and metrics["positive_neutral_fraction"] >= .80
        and nr is not None and .70 <= nr <= 1.30
        and lr is not None and .70 <= lr <= 1.30
        and metrics["neutral_carrier_effect"]
        > metrics["neutral_wrong_direction_effect"] + .10
        and metrics["neutral_carrier_effect"]
        > metrics["neutral_other_slot_effect"] + .10)


@torch.no_grad()
def run_delta_binding_neutral_carrier(model_path, out_dir, quantization="8bit",
                                      device_map=None, seed=0):
    del seed
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    values, excluded = _tokenizer_valid_values(tok)
    result = {"stage": "delta_binding_neutral_carrier", "protocol_version": PROTOCOL_VERSION,
              "model_path": model_path, "inject_layer": INJECT_LAYER,
              "neutral_carrier": NEUTRAL_CARRIER, "tokenizer_valid_values": values,
              "tokenizer_excluded_values": excluded, "heldout_offsets": [5, 7],
              "rule": "raw neutral token-state deltas; no fitting or alignment"}
    if len(values) < MIN_VALUES:
        result["verdict"] = "NEUTRAL_CARRIER_TOKENIZATION_INELIGIBLE"
        return _write(out_dir, result)
    _discovery, heldout = _split_rows(_trials(values))
    let_prototypes, let_position = _let_prototype_map(model, tok, dev, values)
    neutral_prototypes, neutral_position = _neutral_prototype_map(model, tok, dev, values)
    log(f"delta_binding_neutral_carrier: values={len(values)} heldout={len(heldout)} "
        f"inject=L{INJECT_LAYER} carrier={NEUTRAL_CARRIER!r}")
    metrics = _evaluate(model, tok, dev, let_prototypes, neutral_prototypes, heldout)
    metrics["let_donor_position"] = let_position
    metrics["neutral_donor_position"] = neutral_position
    metrics["neutral_matches"] = _neutral_matches(metrics)
    result.update({"heldout": metrics,
                   "verdict": ("NEUTRAL_CARRIER_MATCHES_LITERAL_CONTROLLER"
                               if metrics["neutral_matches"]
                               else "NEUTRAL_CARRIER_DOES_NOT_MATCH_LITERAL_CONTROLLER")})
    log(f"heldout: natural={metrics['natural_effect']:+.3f} "
        f"let={metrics['let_controller_effect']:+.3f} "
        f"neutral={metrics['neutral_carrier_effect']:+.3f} "
        f"neutral/natural={metrics['neutral_to_natural_ratio']} "
        f"neutral/let={metrics['neutral_to_let_ratio']} "
        f"wrong={metrics['neutral_wrong_direction_effect']:+.3f} "
        f"other={metrics['neutral_other_slot_effect']:+.3f}")
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_binding_neutral_carrier.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
