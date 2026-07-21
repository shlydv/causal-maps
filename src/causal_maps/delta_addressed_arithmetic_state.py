"""Addressed operand-state editing with arithmetic consequence propagation."""
import json
import os

import numpy as np
import torch

from .delta_binding_neutral_carrier import _neutral_text
from .delta_operator import _directions, _encode_uniform
from .delta_reasoning_controller import _continuation_token_id
from .delta_reasoning_screen import _arithmetic_prompt
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer


INJECT_LAYER = 2
WORKSPACE_LAYERS = (4, 8, 12, 16, 20, 26)
PROTOCOL_VERSION = "2026-07-14-v2"


def _rows():
    return [{"source": a, "target": a + 2, "wrong": a + 1, "b": b}
            for b in (1, 2, 3) for a in (1, 2, 3, 4)]


def _batch(tok, rows, target=False):
    prompt_rows, answers = [], []
    for row in rows:
        a = row["target"] if target else row["source"]
        text = _arithmetic_prompt(tok, {"a": a, "b": row["b"]}, "add")
        marker = f"First number: {a}"
        start = text.find(marker)
        if start < 0:
            raise ValueError("arithmetic operand anchor missing")
        prompt_rows.append((text, start + len("First number: ")))
        answers.append(str(a + row["b"]))
    ids, am, position = _encode_uniform(tok, prompt_rows)
    answer_ids = torch.tensor([
        _continuation_token_id(tok, text, answer)
        for (text, _offset), answer in zip(prompt_rows, answers)])
    return ids, am, position, answer_ids


def _digit_prototype_map(model, tok, dev, digits):
    """Capture the digit token, not Qwen's separate preceding-space token."""
    from .tensorize import _anchor_token_index

    texts = [_neutral_text(tok, digit) for digit in digits]
    encoded, positions = [], []
    for text, offset in texts:
        ids = tok.encode(text, add_special_tokens=False)
        anchor = _anchor_token_index(tok, text, offset)
        candidates = [idx for idx in range(anchor, min(anchor + 3, len(ids)))
                      if tok.decode([ids[idx]]).strip() == text[offset:offset + 1]]
        if len(candidates) != 1:
            raise ValueError(f"digit token not uniquely located near anchor: {candidates}")
        encoded.append(ids)
        positions.append(candidates[0])
    if len({len(ids) for ids in encoded}) != 1 or len(set(positions)) != 1:
        raise ValueError("digit neutral donors are not uniform")
    ids = torch.tensor(encoded, dtype=torch.long)
    am = torch.ones_like(ids)
    position = positions[0]
    _logits, cache = _forward(
        model, ids.to(dev), am.to(dev), (position,), (INJECT_LAYER,))
    hidden = cache[INJECT_LAYER][:, 0]
    return {digit: hidden[idx] for idx, digit in enumerate(digits)}, int(position)


def _summary(logits, clean_ld, target_ids, source_ids, natural_effect):
    rows = _ld(logits, target_ids, source_ids) - clean_ld
    effect = float(rows.mean())
    return {"effect": effect,
            "effect_ratio": float(effect / natural_effect) if natural_effect > 0 else None,
            "target_acc": float((logits.argmax(-1) == target_ids).float().mean()),
            "positive_fraction": float((rows > 0).float().mean())}


@torch.no_grad()
def run_delta_addressed_arithmetic_state(model_path, out_dir,
                                          quantization="8bit", device_map=None,
                                          seed=0, workspace_layers=None,
                                          confirmation=False):
    del seed
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    rows = _rows()
    ci, cam, operand_pos, source_ids = _batch(tok, rows, target=False)
    ni, nam, natural_operand_pos, target_ids = _batch(tok, rows, target=True)
    if ci.shape != ni.shape or operand_pos != natural_operand_pos:
        raise ValueError("operand clean/natural alignment failed")
    changed_positions = []
    for clean_row, natural_row in zip(ci, ni):
        changed = (clean_row != natural_row).nonzero().flatten().tolist()
        if len(changed) != 1:
            raise ValueError(f"expected one changed operand token, got {changed}")
        changed_positions.append(changed[0])
    if len(set(changed_positions)) != 1:
        raise ValueError(f"operand token positions are nonuniform: {changed_positions}")
    operand_token_pos = changed_positions[0]
    ci, cam, ni, nam = ci.to(dev), cam.to(dev), ni.to(dev), nam.to(dev)
    requested_layers = (tuple(workspace_layers) if workspace_layers is not None
                        else WORKSPACE_LAYERS)
    layers = tuple(layer for layer in requested_layers
                   if layer < int(model.config.num_hidden_layers))
    last = int(ci.shape[1] - 1)
    clean_logits, clean_cache = _forward(model, ci, cam, (last,), layers)
    natural_logits, natural_cache = _forward(model, ni, nam, (last,), layers)
    clean_acc = float((clean_logits.argmax(-1) == source_ids).float().mean())
    natural_acc = float((natural_logits.argmax(-1) == target_ids).float().mean())
    result = {"stage": ("delta_addressed_arithmetic_confirmation"
                        if confirmation else "delta_addressed_arithmetic_state"),
              "protocol_version": PROTOCOL_VERSION, "model_path": model_path,
              "inject_layer": INJECT_LAYER, "workspace_layers": list(layers),
              "n_rows": len(rows), "operand_anchor_position": operand_pos,
              "operand_token_position": operand_token_pos,
              "last_position": last,
              "status": "frozen_cross_model_confirmation" if confirmation
              else "discovery_only",
              "behavior": {"clean_acc": clean_acc, "natural_acc": natural_acc,
                           "pass": bool(clean_acc >= .90 and natural_acc >= .90)}}
    log(f"addressed_arithmetic_state: rows={len(rows)} G0={clean_acc:.0%}/"
        f"{natural_acc:.0%} operand_token={operand_token_pos} last={last}")
    if not result["behavior"]["pass"]:
        result["verdict"] = "ADDRESSED_ARITHMETIC_BEHAVIORALLY_INELIGIBLE"
        return _write(out_dir, result)

    clean_ld = _ld(clean_logits, target_ids, source_ids)
    natural_effect = float(
        (_ld(natural_logits, target_ids, source_ids) - clean_ld).mean())
    digits = [str(value) for value in range(1, 7)]
    prototypes, neutral_position = _digit_prototype_map(model, tok, dev, digits)
    target_rows = [{"source": str(row["source"]), "target": str(row["target"])}
                   for row in rows]
    wrong_rows = [{"source": str(row["source"]), "target": str(row["wrong"])}
                  for row in rows]
    neutral_delta = _directions(prototypes, target_rows)
    wrong_delta = _directions(prototypes, wrong_rows)
    embedding = model.get_input_embeddings().weight.detach().float().cpu()
    embed_delta = torch.stack([
        embedding[int(ni[idx, operand_token_pos])]
        - embedding[int(ci[idx, operand_token_pos])]
        for idx in range(len(rows))])
    neutral_logits, _ = _forward(
        model, ci, cam, (last,),
        add=(INJECT_LAYER, operand_token_pos, neutral_delta))
    wrong_logits, _ = _forward(
        model, ci, cam, (last,),
        add=(INJECT_LAYER, operand_token_pos, wrong_delta))
    embed_logits, _ = _forward(
        model, ci, cam, (last,),
        add=(INJECT_LAYER, operand_token_pos, embed_delta))
    neutral = _summary(neutral_logits, clean_ld, target_ids, source_ids, natural_effect)
    wrong = _summary(wrong_logits, clean_ld, target_ids, source_ids, natural_effect)
    embed = _summary(embed_logits, clean_ld, target_ids, source_ids, natural_effect)
    consequence_pass = bool(
        neutral["target_acc"] >= .80 and neutral["positive_fraction"] >= .80
        and neutral["effect_ratio"] is not None
        and .70 <= neutral["effect_ratio"] <= 1.30
        and neutral["effect"] > wrong["effect"] + .10)
    embedding_equivalent = bool(
        embed["target_acc"] >= .80 and embed["positive_fraction"] >= .80
        and embed["effect_ratio"] is not None
        and .70 <= embed["effect_ratio"] <= 1.30)
    result["consequence"] = {
        "natural_effect": natural_effect, "neutral": neutral, "wrong": wrong,
        "embedding": embed, "neutral_pass": consequence_pass,
        "embedding_equivalent": embedding_equivalent,
        "neutral_donor_position": neutral_position}

    workspace = []
    for layer in layers:
        cf_into_clean, _ = _forward(
            model, ci, cam, (last,), patch=(layer, last, natural_cache[layer][:, 0]))
        clean_into_cf, _ = _forward(
            model, ni, nam, (last,), patch=(layer, last, clean_cache[layer][:, 0]))
        forward = _summary(
            cf_into_clean, clean_ld, target_ids, source_ids, natural_effect)
        reverse_acc = float((clean_into_cf.argmax(-1) == source_ids).float().mean())
        interchangeable = bool(
            forward["target_acc"] >= .80 and forward["effect_ratio"] is not None
            and .70 <= forward["effect_ratio"] <= 1.30 and reverse_acc >= .80)
        workspace.append({"layer": layer, "cf_into_clean": forward,
                          "clean_into_cf_source_acc": reverse_acc,
                          "interchangeable": interchangeable})
        log(f"workspace L{layer}: target={forward['target_acc']:.0%} "
            f"ratio={forward['effect_ratio']} reverse={reverse_acc:.0%} "
            f"interchangeable={interchangeable}")
    earliest = next((item["layer"] for item in workspace
                     if item["interchangeable"]), None)
    result["workspace_discovery"] = workspace
    result["earliest_interchangeable_workspace_layer"] = earliest
    if confirmation:
        result["verdict"] = (
            "ADDRESSED_ARITHMETIC_CROSS_MODEL_CONFIRMED"
            if consequence_pass and earliest is not None else
            "ADDRESSED_ARITHMETIC_CROSS_MODEL_NOT_CONFIRMED")
    else:
        result["verdict"] = (
            "ADDRESSED_ARITHMETIC_WRITE_WITH_WORKSPACE"
            if consequence_pass and earliest is not None else
            "ADDRESSED_ARITHMETIC_WRITE_ONLY" if consequence_pass else
            "ADDRESSED_ARITHMETIC_CONSEQUENCE_NULL")
    log(f"consequence: natural={natural_effect:+.3f} neutral={neutral} "
        f"embed={embed} wrong={wrong} earliest_workspace={earliest}")
    return _write(out_dir, result)


def _write(out_dir, result):
    name = ("results_delta_addressed_arithmetic_confirmation.json"
            if result["stage"] == "delta_addressed_arithmetic_confirmation"
            else "results_delta_addressed_arithmetic_state.json")
    path = os.path.join(out_dir, name)
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
