"""Discovery ladder from an addressed lexical write to a two-hop consequence."""
import json
import os

import numpy as np
import torch

from .delta_binding_cross_model_gate import _tokenizer_valid_values
from .delta_binding_neutral_carrier import _neutral_prototype_map
from .delta_operator import _directions, _encode_uniform
from .delta_reasoning_controller import _continuation_token_id
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer


INJECT_LAYER = 2
WORKSPACE_LAYERS = (4, 8, 12, 16, 20, 26)
PROTOCOL_VERSION = "2026-07-14-v1"
MIN_VALUES = 9


def _rows(values):
    rows, n = [], len(values)
    for i in range(n):
        cycle = [values[(i + j) % n] for j in range(9)]
        if len(set(cycle)) != 9:
            raise ValueError("reasoning ladder requires nine distinct values")
        rows.append({
            "source_start": cycle[0], "source_mid": cycle[1],
            "source_answer": cycle[2], "target_start": cycle[3],
            "target_mid": cycle[4], "target_answer": cycle[5],
            "wrong_start": cycle[6], "wrong_mid": cycle[7],
            "wrong_answer": cycle[8],
        })
    return rows


def _prompt(tok, row, start):
    rules = (
        f"{row['source_start']} -> {row['source_mid']}; "
        f"{row['source_mid']} -> {row['source_answer']}; "
        f"{row['target_start']} -> {row['target_mid']}; "
        f"{row['target_mid']} -> {row['target_answer']}; "
        f"{row['wrong_start']} -> {row['wrong_mid']}; "
        f"{row['wrong_mid']} -> {row['wrong_answer']}"
    )
    user = (f"Rules: {rules}. Start: {start}. Apply exactly two transitions. "
            "Return only the final value.")
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True)
    # Keep the model's native generation boundary. This makes the gold value
    # a stable one-token continuation instead of risking token merger with a
    # hand-written answer primer.
    text = prefix
    marker = f"Start: {start}"
    offset = text.rfind(marker)
    if offset < 0:
        raise ValueError("start anchor missing")
    return text, offset + len("Start: ")


def _batch(tok, rows, target=False):
    prompts = []
    for row in rows:
        start = row["target_start"] if target else row["source_start"]
        prompts.append(_prompt(tok, row, start))
    ids, am, start_pos = _encode_uniform(tok, prompts)
    answers = [row["target_answer"] if target else row["source_answer"]
               for row in rows]
    answer_ids = torch.tensor([
        _continuation_token_id(tok, text, " " + answer)
        for (text, _offset), answer in zip(prompts, answers)])
    return ids, am, start_pos, answer_ids


def _mean(items):
    return float(np.mean(items)) if items else float("nan")


def _summary(logits, clean_ld, pos_ids, neg_ids, natural_effect):
    rows = _ld(logits, pos_ids, neg_ids) - clean_ld
    effect = float(rows.mean())
    return {
        "effect": effect,
        "effect_ratio": float(effect / natural_effect) if natural_effect > 0 else None,
        "target_acc": float((logits.argmax(-1) == pos_ids).float().mean()),
        "positive_fraction": float((rows > 0).float().mean()),
    }


@torch.no_grad()
def run_delta_addressed_reasoning_ladder(model_path, out_dir,
                                         quantization="8bit", device_map=None,
                                         seed=0):
    del seed
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    values, excluded = _tokenizer_valid_values(tok)
    layers = tuple(layer for layer in WORKSPACE_LAYERS
                   if layer < int(model.config.num_hidden_layers))
    result = {
        "stage": "delta_addressed_reasoning_ladder",
        "protocol_version": PROTOCOL_VERSION, "model_path": model_path,
        "inject_layer": INJECT_LAYER, "workspace_layers": list(layers),
        "tokenizer_valid_values": values, "tokenizer_excluded_values": excluded,
        "status": "discovery_only",
    }
    if len(values) < MIN_VALUES:
        result["verdict"] = "REASONING_LADDER_TOKENIZATION_INELIGIBLE"
        return _write(out_dir, result)
    rows = _rows(values)
    ci, cam, start_pos, source_ids = _batch(tok, rows, target=False)
    ni, nam, natural_start_pos, target_ids = _batch(tok, rows, target=True)
    if ci.shape != ni.shape or start_pos != natural_start_pos:
        raise ValueError("clean/natural reasoning prompts are not aligned")
    ci, cam, ni, nam = ci.to(dev), cam.to(dev), ni.to(dev), nam.to(dev)
    last = int(ci.shape[1] - 1)
    clean_logits, clean_cache = _forward(model, ci, cam, (last,), layers)
    natural_logits, natural_cache = _forward(model, ni, nam, (last,), layers)
    clean_acc = float((clean_logits.argmax(-1) == source_ids).float().mean())
    natural_acc = float((natural_logits.argmax(-1) == target_ids).float().mean())
    g0 = clean_acc >= .80 and natural_acc >= .80
    result["behavior"] = {"n_rows": len(rows), "clean_acc": clean_acc,
                          "natural_acc": natural_acc, "pass": bool(g0)}
    log(f"addressed_reasoning_ladder: rows={len(rows)} prompt_len={ci.shape[1]} "
        f"start={start_pos} last={last} G0={clean_acc:.0%}/{natural_acc:.0%}")
    if not g0:
        result["verdict"] = "REASONING_LADDER_BEHAVIORALLY_INELIGIBLE"
        return _write(out_dir, result)

    clean_ld = _ld(clean_logits, target_ids, source_ids)
    natural_rows = _ld(natural_logits, target_ids, source_ids) - clean_ld
    natural_effect = float(natural_rows.mean())
    prototypes, neutral_pos = _neutral_prototype_map(model, tok, dev, values)
    target_delta = _directions(prototypes, [
        {"source": row["source_start"], "target": row["target_start"]}
        for row in rows])
    wrong_delta = _directions(prototypes, [
        {"source": row["source_start"], "target": row["wrong_start"]}
        for row in rows])
    embedding = model.get_input_embeddings().weight.detach().float().cpu()
    embed_delta = torch.stack([
        embedding[int(ni[idx, start_pos])] - embedding[int(ci[idx, start_pos])]
        for idx in range(len(rows))])
    neutral_logits, _ = _forward(
        model, ci, cam, (last,), add=(INJECT_LAYER, start_pos, target_delta))
    wrong_logits, _ = _forward(
        model, ci, cam, (last,), add=(INJECT_LAYER, start_pos, wrong_delta))
    embed_logits, _ = _forward(
        model, ci, cam, (last,), add=(INJECT_LAYER, start_pos, embed_delta))
    neutral = _summary(
        neutral_logits, clean_ld, target_ids, source_ids, natural_effect)
    wrong = _summary(wrong_logits, clean_ld, target_ids, source_ids, natural_effect)
    embed = _summary(embed_logits, clean_ld, target_ids, source_ids, natural_effect)
    g1 = bool(neutral["target_acc"] >= .80
              and neutral["positive_fraction"] >= .80
              and neutral["effect_ratio"] is not None
              and .70 <= neutral["effect_ratio"] <= 1.30
              and neutral["effect"] > wrong["effect"] + .10)
    embed_equivalent = bool(embed["target_acc"] >= .80
                            and embed["positive_fraction"] >= .80
                            and embed["effect_ratio"] is not None
                            and .70 <= embed["effect_ratio"] <= 1.30)
    result["consequence"] = {
        "natural_effect": natural_effect, "neutral": neutral, "wrong": wrong,
        "embedding": embed, "neutral_pass": g1,
        "embedding_equivalent": embed_equivalent,
        "neutral_donor_position": neutral_pos,
    }

    workspace = []
    for layer in layers:
        cf_into_clean, _ = _forward(
            model, ci, cam, (last,),
            patch=(layer, last, natural_cache[layer][:, 0]))
        clean_into_cf, _ = _forward(
            model, ni, nam, (last,),
            patch=(layer, last, clean_cache[layer][:, 0]))
        forward = _summary(
            cf_into_clean, clean_ld, target_ids, source_ids, natural_effect)
        reverse_acc = float(
            (clean_into_cf.argmax(-1) == source_ids).float().mean())
        interchangeable = bool(
            forward["target_acc"] >= .80 and forward["effect_ratio"] is not None
            and .70 <= forward["effect_ratio"] <= 1.30 and reverse_acc >= .80)
        workspace.append({"layer": layer, "cf_into_clean": forward,
                          "clean_into_cf_source_acc": reverse_acc,
                          "interchangeable": interchangeable})
        log(f"workspace L{layer}: target={forward['target_acc']:.0%} "
            f"ratio={forward['effect_ratio']} reverse={reverse_acc:.0%} "
            f"interchangeable={interchangeable}")
    result["workspace_discovery"] = workspace
    earliest = next((row["layer"] for row in workspace
                     if row["interchangeable"]), None)
    result["earliest_interchangeable_workspace_layer"] = earliest
    result["verdict"] = (
        "TWO_HOP_ADDRESSED_WRITE_WITH_WORKSPACE"
        if g1 and earliest is not None else
        "TWO_HOP_ADDRESSED_WRITE_ONLY" if g1 else
        "TWO_HOP_CONSEQUENCE_CONTROL_NULL")
    log(f"consequence: natural={natural_effect:+.3f} neutral={neutral} "
        f"embed={embed} wrong={wrong} earliest_workspace={earliest}")
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_addressed_reasoning_ladder.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
