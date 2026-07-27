"""Position-matched output-label diagnostic for causal handoff depth."""
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
    TELL_QUESTION,
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

PROTOCOL_VERSION = "2026-07-24-p2-position-matched-label-v1"
PROTOCOL_SHA256 = (
    "9BC8ACCADF3289B0718738137F6DB966CD4CC37F205201A1A4B810EF0A9433C0")
QUESTION_NAMES = ("belief", "tell", "search")
QUESTIONS = {
    "belief": BELIEF_QUESTION,
    "tell": TELL_QUESTION,
    "search": SEARCH_QUESTION,
}
LABEL_NAMES = ("belief", "tell", "search")
COMMANDS = {
    "belief": "BELIEF",
    "tell": "X TELL",
    "search": "X X SEARCH",
}
TASK_GRID = {
    question: {
        label: f"matched_{question}_{label}"
        for label in LABEL_NAMES
    }
    for question in QUESTION_NAMES
}
ALL_TASKS = tuple(
    TASK_GRID[question][label]
    for question in QUESTION_NAMES
    for label in LABEL_NAMES
)
QUERY.update({
    TASK_GRID[question][label]: (
        QUESTIONS[question], COMMANDS[label], "ac")
    for question in QUESTION_NAMES
    for label in LABEL_NAMES
})


def _summary(tasks, prefix):
    cells = {
        question: {
            label: {
                "readout_position": int(
                    tasks[TASK_GRID[question][label]][
                        "readout_position"]),
                "l24_minimum_mediation": float(
                    prefix[TASK_GRID[question][label]]["24"][
                        "mediation"]["minimum_fraction"]),
                "first_passing_prefix": _first_passing(
                    prefix[TASK_GRID[question][label]]),
            }
            for label in LABEL_NAMES
        }
        for question in QUESTION_NAMES
    }
    label_means = {
        label: sum(
            cells[question][label]["l24_minimum_mediation"]
            for question in QUESTION_NAMES
        ) / len(QUESTION_NAMES)
        for label in LABEL_NAMES
    }
    return {"cells": cells, "label_mean_l24": label_means}


def _verdict(tasks, prefix):
    if not all(tasks[name]["eligible"] for name in ALL_TASKS):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in ALL_TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    summary = _summary(tasks, prefix)
    cells = summary["cells"]
    if not all(
            len({
                cells[question][label]["readout_position"]
                for label in LABEL_NAMES
            }) == 1
            for question in QUESTION_NAMES):
        return "POSITION_MATCH_INVALID"
    first = [
        cells[question][label]["first_passing_prefix"]
        for question in QUESTION_NAMES
        for label in LABEL_NAMES
    ]
    if any(value is None for value in first):
        return "DEPTH_UNRESOLVED"
    means = list(summary["label_mean_l24"].values())
    continuous = max(means) - min(means) >= 0.05 - 1e-9
    categorical = any(
        len({
            cells[question][label]["first_passing_prefix"]
            for label in LABEL_NAMES
        }) > 1
        for question in QUESTION_NAMES
    )
    if continuous and categorical:
        return "POSITION_MATCHED_LABEL_EFFECT"
    if continuous:
        return "CONTINUOUS_POSITION_MATCHED_LABEL_EFFECT"
    return "NO_POSITION_MATCHED_LABEL_EFFECT"


@torch.no_grad()
def run_delta_position_matched_label(
        model_path, out_dir,
        model_key="qwen7b_position_matched_label",
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
        raise ValueError("matched-label layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)

    hb = Heartbeat(
        len(ALL_TASKS) * (3 + len(LAYERS)),
        "position_matched_label", every_sec=30, out_dir=out_dir)
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
        "stage": "delta_position_matched_label",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "narrative",
        "questions": list(QUESTION_NAMES),
        "labels": list(LABEL_NAMES),
        "commands": COMMANDS,
        "task_grid": TASK_GRID,
        "query_contracts": {
            query: {
                "question": QUERY[query][0],
                "command": QUERY[query][1],
                "field": QUERY[query][2],
            }
            for query in ALL_TASKS
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
        "matched_label_summary": summary,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_position_matched_label_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"POSITION MATCHED LABEL verdict={verdict} "
        f"summary={summary} artifact={path}")
    return result

