"""Behavior-only screen for a fresh Qwen-14B replication bucket."""
from __future__ import annotations

import json
import os

import torch

from .delta_anchor_write import TARGET, _resolve
from .delta_distributed_label_transplant import _aligned_batches
from .delta_preprint_battery import _compatible_world_rows
from .delta_structured_workspace import LOCATIONS, _counterfactual
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

PROTOCOL_VERSION = "2026-07-24-p2-qwen14b-behavior-screen-v1"
PROTOCOL_SHA256 = (
    "F6EB5BBB8A5DB609B44CD669D1AECC30D0049CD43E173A8175AE85D30CD13E33")
CANDIDATE_START = 0
DIAGNOSTIC_START = 15
SPLIT_N = 15
MIN_ELIGIBLE = 8


@torch.no_grad()
def _baseline(model, batch):
    return model(
        input_ids=batch["ids"], attention_mask=batch["am"],
        use_cache=False).logits[:, -1, :].detach().float().cpu()


def _predictions(logits, batch, expected):
    pool = torch.tensor([batch["amap"][x] for x in LOCATIONS])
    scores = logits[:, pool]
    chosen = scores.argmax(-1)
    rows = []
    for index in range(scores.shape[0]):
        gold_index = LOCATIONS.index(expected[index])
        other = torch.cat((
            scores[index, :gold_index],
            scores[index, gold_index + 1:]))
        rows.append({
            "predicted": LOCATIONS[int(chosen[index])],
            "expected": expected[index],
            "correct": bool(int(chosen[index]) == gold_index),
            "gold_margin": float(
                scores[index, gold_index] - other.max()),
        })
    return rows


@torch.no_grad()
def _screen_split(model, tok, dev, rows, compatible_indices):
    natural_rows = _counterfactual(rows, {"ac": TARGET})
    pairs, differing, groups, _candidates, marker, readout = (
        _aligned_batches(tok, dev, rows))
    output = []
    for local_index in range(len(rows)):
        output.append({
            "compatible_position": int(compatible_indices[local_index]),
            "world": rows[local_index],
            "belief_clean": None,
            "belief_natural": None,
            "search_clean": None,
            "search_natural": None,
        })
    for arm_index, arm_name, expected_rows in (
            (0, "clean", rows),
            (1, "natural", natural_rows)):
        belief_batch, search_batch = pairs[arm_index]
        belief = _predictions(
            _baseline(model, belief_batch), belief_batch,
            [row["ac"] for row in expected_rows])
        search = _predictions(
            _baseline(model, search_batch), search_batch,
            [row["ac"] for row in expected_rows])
        for local_index in range(len(rows)):
            output[local_index][f"belief_{arm_name}"] = belief[local_index]
            output[local_index][f"search_{arm_name}"] = search[local_index]
    for row in output:
        row["eligible"] = bool(all(
            row[cell]["correct"] for cell in (
                "belief_clean", "belief_natural",
                "search_clean", "search_natural")))
    return {
        "alignment": {
            "marker_position": marker,
            "readout_position": readout,
            "differing_positions": differing,
            "instruction_positions": groups[0],
            "answer_prefix_positions": groups[1],
        },
        "rows": output,
        "accuracies": {
            cell: sum(row[cell]["correct"] for row in output) / len(output)
            for cell in (
                "belief_clean", "belief_natural",
                "search_clean", "search_natural")
        },
    }


@torch.no_grad()
def run_delta_replication_behavior_screen(
        model_path, out_dir,
        model_key="qwen14b_behavior_screen",
        quantization="awq", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != 30:
        raise ValueError("v1 requires exactly 30 compatible worlds")

    try:
        candidate = _screen_split(
            model, tok, dev,
            rows[CANDIDATE_START:CANDIDATE_START + SPLIT_N],
            indices[CANDIDATE_START:CANDIDATE_START + SPLIT_N])
        diagnostic = _screen_split(
            model, tok, dev,
            rows[DIAGNOSTIC_START:DIAGNOSTIC_START + SPLIT_N],
            indices[DIAGNOSTIC_START:DIAGNOSTIC_START + SPLIT_N])
    except ValueError as exc:
        result = {
            "stage": "delta_replication_behavior_screen",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "alignment_error": str(exc),
            "verdict": "TOKEN_ALIGNMENT_INVALID",
        }
    else:
        eligible_positions = [
            row["compatible_position"]
            for row in candidate["rows"] if row["eligible"]
        ]
        result = {
            "stage": "delta_replication_behavior_screen",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "model_path": model_path,
            "quantization": quantization,
            "selection_rule": (
                "all candidate rows correct in BELIEF/SEARCH x "
                "clean/natural; preserve order"),
            "minimum_eligible": MIN_ELIGIBLE,
            "candidate": candidate,
            "diagnostic": diagnostic,
            "eligible_compatible_positions": eligible_positions,
            "eligible_candidate_count": len(eligible_positions),
            "verdict": (
                "ELIGIBLE_BUCKET_AVAILABLE"
                if len(eligible_positions) >= MIN_ELIGIBLE
                else "ELIGIBLE_BUCKET_TOO_SMALL"),
        }
    path = os.path.join(
        out_dir,
        f"results_delta_replication_behavior_screen_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"REPLICATION BEHAVIOR SCREEN verdict={result['verdict']} "
        f"eligible={result.get('eligible_candidate_count')} "
        f"artifact={path}")
    return result
