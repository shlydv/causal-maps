"""Neutral prefix-token ladder for causal depth substitution."""
from __future__ import annotations

import json
import os

import torch

from .delta_anchor_write import _resolve
from .delta_depth_replication import _first_passing
from .delta_operation_handoff_depth import (
    CAPTURE_LAYERS,
    LAYERS,
    _evaluate_sites,
    _full_sites,
)
from .delta_preprint_battery import _compatible_world_rows
from .delta_semantic_command_factor import (
    BELIEF_QUESTION,
    SEARCH_QUESTION,
)
from .delta_source_head_mediation import _public_task, _task_context
from .delta_sparse_transport import _attention_geometry
from .delta_structured_workspace import QUERY
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-token-length-ladder-v1"
PROTOCOL_SHA256 = (
    "89CC11BDF9F5EC13AB55A6E13A82F60BA820FF685206FA65FE5B32987ACE2867")
QUESTION_NAMES = ("belief", "search")
QUESTIONS = {
    "belief": BELIEF_QUESTION,
    "search": SEARCH_QUESTION,
}
FILLER_COUNTS = (0, 1, 2, 3, 4)


def _command(count):
    return " ".join(["X"] * int(count) + ["SEARCH"])


TASK_GRID = {
    question: {
        count: f"length_{question}_{count}"
        for count in FILLER_COUNTS
    }
    for question in QUESTION_NAMES
}
ALL_TASKS = tuple(
    TASK_GRID[question][count]
    for question in QUESTION_NAMES
    for count in FILLER_COUNTS
)
QUERY.update({
    TASK_GRID[question][count]: (
        QUESTIONS[question], _command(count), "ac")
    for question in QUESTION_NAMES
    for count in FILLER_COUNTS
})


def _summary(tasks, prefix):
    return {
        question: {
            str(count): {
                "readout_position": int(
                    tasks[TASK_GRID[question][count]][
                        "readout_position"]),
                "l24_minimum_mediation": float(
                    prefix[TASK_GRID[question][count]]["24"][
                        "mediation"]["minimum_fraction"]),
                "first_passing_prefix": _first_passing(
                    prefix[TASK_GRID[question][count]]),
            }
            for count in FILLER_COUNTS
        }
        for question in QUESTION_NAMES
    }


def _verdict(tasks, prefix):
    if not all(tasks[name]["eligible"] for name in ALL_TASKS):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in ALL_TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    summary = _summary(tasks, prefix)
    positions_valid = all(
        all(right > left for left, right in zip(
            [
                summary[question][str(count)]["readout_position"]
                for count in FILLER_COUNTS
            ][:-1],
            [
                summary[question][str(count)]["readout_position"]
                for count in FILLER_COUNTS
            ][1:],
        ))
        for question in QUESTION_NAMES
    )
    if not positions_valid:
        return "TOKENIZATION_INVALID"
    first = [
        summary[question][str(count)]["first_passing_prefix"]
        for question in QUESTION_NAMES
        for count in FILLER_COUNTS
    ]
    if any(value is None for value in first):
        return "DEPTH_UNRESOLVED"
    monotone = {}
    rise = {}
    earlier = {}
    for question in QUESTION_NAMES:
        l24 = [
            summary[question][str(count)]["l24_minimum_mediation"]
            for count in FILLER_COUNTS
        ]
        monotone[question] = all(
            right + 1e-9 >= left
            for left, right in zip(l24[:-1], l24[1:]))
        rise[question] = l24[-1] - l24[0] >= 0.05 - 1e-9
        earlier[question] = (
            summary[question][str(FILLER_COUNTS[-1])][
                "first_passing_prefix"]
            < summary[question][str(FILLER_COUNTS[0])][
                "first_passing_prefix"]
        )
    if all(monotone.values()) and any(
            rise[q] and earlier[q] for q in QUESTION_NAMES):
        return "TOKEN_LENGTH_DEPTH_SUBSTITUTION"
    if all(monotone.values()) and any(rise.values()):
        return "CONTINUOUS_LENGTH_EFFECT"
    return "NO_MONOTONE_LENGTH_EFFECT"


@torch.no_grad()
def run_delta_token_length_ladder(
        model_path, out_dir,
        model_key="qwen7b_token_length_ladder",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != 30:
        raise ValueError("v1 requires all 30 compatible worlds")
    if max(LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("ladder layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)

    hb = Heartbeat(
        len(ALL_TASKS) * (3 + len(LAYERS)),
        "token_length_ladder", every_sec=30, out_dir=out_dir)
    tasks = {}
    prefix = {}
    for query in ALL_TASKS:
        context = _task_context(
            model, tok, dev, rows, query,
            CAPTURE_LAYERS, head_dim, surface="narrative")
        tasks[query] = _public_task(context)
        hb.step(extra=f"{query}/baseline")
        hb.step(extra=f"{query}/source")
        hb.step(extra=f"{query}/base_path")
        prefix[query] = {}
        for layer in LAYERS:
            selected = tuple(x for x in LAYERS if x <= layer)
            cell = _evaluate_sites(
                model, context, _full_sites(selected, n_heads), head_dim)
            prefix[query][str(layer)] = cell
            hb.step(extra=(
                f"{query}/prefixL{layer}="
                f"{cell['mediation']['pass']}"))
    hb.done()

    summary = _summary(tasks, prefix)
    verdict = _verdict(tasks, prefix)
    result = {
        "stage": "delta_token_length_ladder",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "narrative",
        "questions": list(QUESTION_NAMES),
        "filler_counts": list(FILLER_COUNTS),
        "commands": {
            str(count): _command(count)
            for count in FILLER_COUNTS
        },
        "task_grid": {
            question: {
                str(count): TASK_GRID[question][count]
                for count in FILLER_COUNTS
            }
            for question in QUESTION_NAMES
        },
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
        },
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "tasks": tasks,
        "cumulative_prefix": prefix,
        "length_summary": summary,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_token_length_ladder_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"TOKEN LENGTH LADDER verdict={verdict} "
        f"summary={summary} artifact={path}")
    return result

