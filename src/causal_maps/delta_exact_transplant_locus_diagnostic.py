"""Final diagnostic: exact operation-state transport across layer and locus."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _anchor_position, _resolve
from .delta_cross_domain_controller import _generic_accuracy
from .delta_heterogeneous_family_screen import (
    FAMILY_SPECS,
    VALUES,
    _family_alignment,
)
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output


PROTOCOL_VERSION = "2026-07-26-p2-exact-transplant-locus-v1"
FAMILIES = ("maximum_score", "two_hop_pointer")
PATCH_LAYERS = (12, 16, 20, 21, 22, 23, 24, 26)
CHECKPOINT_LAYERS = (24, 27)
N_WORLD = 12
RANDOM_SEED = 220039

MINIMUM_MEAN_PROGRESS = 0.25
MAXIMUM_MEDIAN_DISTANCE_RATIO = 0.90
MINIMUM_POSITIVE_ROWS = 18
MINIMUM_ANSWER_ACCURACY = 0.80


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "purpose": (
        "Distinguish a wrong fixed causal locus from direction-asymmetric "
        "or incomplete route control, then close the current experiment "
        "loop without rescue runs."),
    "families": list(FAMILIES),
    "family_roles": {
        "maximum_score": "previous bidirectional reference",
        "two_hop_pointer": "previous one-directional failure",
    },
    "rows": {
        "n": N_WORLD,
        "construction": (
            "Two distractor variants for each of the six directed color "
            "pairs absent from the previous 50-pair experiments."),
        "previously_unused_source_target_pairs": True,
    },
    "patch_layers": list(PATCH_LAYERS),
    "checkpoint_layers": list(CHECKPOINT_LAYERS),
    "position_groups": [
        "answer_prefix_3",
        "instruction_3",
        "all_differing_6",
        "identical_control_3",
        "identical_control_6",
    ],
    "intervention": (
        "At a frozen layer, replace the selected origin-operation states "
        "with the exact opposite-operation states from the same history "
        "and row. Test BELIEF-to-SEARCH and SEARCH-to-BELIEF separately."),
    "outcomes": {
        "checkpoint_progress": (
            "Projection of patched-minus-origin checkpoint state onto "
            "target-minus-origin, divided by target-minus-origin energy."),
        "checkpoint_distance_ratio": (
            "Patched-to-target distance divided by origin-to-target "
            "distance."),
        "content_preservation": "value-token accuracy after intervention",
    },
    "cell_pass": {
        "minimum_mean_progress_each_direction":
            MINIMUM_MEAN_PROGRESS,
        "maximum_median_distance_ratio_each_direction":
            MAXIMUM_MEDIAN_DISTANCE_RATIO,
        "minimum_positive_rows_each_direction":
            MINIMUM_POSITIVE_ROWS,
        "minimum_answer_accuracy_each_direction":
            MINIMUM_ANSWER_ACCURACY,
    },
    "stopping_rule": (
        "Interpret the frozen verdict and close this experimental loop. "
        "No prompt, layer, threshold, or position revision follows."),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _unused_rows():
    """Twelve histories from the six directed pairs not used previously."""
    pairs = [
        (source_index, (source_index - 1) % len(VALUES))
        for source_index in range(2, len(VALUES))
    ]
    rows = []
    for variant in range(2):
        for source_index, target_index in pairs:
            source = VALUES[source_index]
            target = VALUES[target_index]
            remaining = [
                VALUES[(source_index + offset) % len(VALUES)]
                for offset in range(1, len(VALUES) + 1)
                if VALUES[(source_index + offset) % len(VALUES)]
                not in (source, target)
            ]
            d1 = remaining[(2 * variant) % len(remaining)]
            d2 = remaining[(2 * variant + 1) % len(remaining)]
            rows.append({
                "row_index": len(rows),
                "pair_variant": variant,
                "source": source,
                "target": target,
                "state": source,
                "d1": d1,
                "d2": d2,
            })
    if len(rows) != N_WORLD:
        raise AssertionError("frozen diagnostic row count changed")
    return rows


def _position_groups(alignment, seed):
    instruction = list(alignment["instruction_positions"])
    answer = list(alignment["answer_positions"])
    differing = sorted(instruction + answer)
    clean_belief, clean_search = alignment["batches"][0]
    natural_belief, natural_search = alignment["batches"][1]
    batches = (
        clean_belief, clean_search, natural_belief, natural_search)
    readout = int(alignment["readout"])
    anchors = {
        _anchor_position(clean_belief, natural_belief),
        _anchor_position(clean_search, natural_search),
    }
    candidates = []
    for position in range(8, readout):
        if position in set(differing) | anchors:
            continue
        reference = batches[0]["ids"][:, position]
        if all(torch.equal(
                reference, batch["ids"][:, position])
                for batch in batches[1:]):
            candidates.append(position)
    # Prefer causally downstream identical tokens after the instruction label.
    downstream = [
        position for position in candidates
        if instruction[-1] < position < answer[0]
    ]
    pool = downstream if len(downstream) >= 6 else candidates
    if len(pool) < 6:
        raise ValueError("fewer than six matched identical-token positions")
    rng = np.random.default_rng(int(seed))
    identical6 = sorted(
        int(value) for value in rng.choice(
            pool, size=6, replace=False))
    return {
        "answer_prefix_3": answer,
        "instruction_3": instruction,
        "all_differing_6": differing,
        "identical_control_3": identical6[:3],
        "identical_control_6": identical6,
    }


@torch.no_grad()
def _capture_baseline(model, batch, layer, union_positions):
    blocks = get_decoder_layers(model)
    cache = {}
    handles = []

    def source_hook(_module, _args, output):
        states, _rebuild = _split_output(output)
        cache["source"] = (
            states[:, union_positions, :].detach().float().cpu())

    def checkpoint_hook(checkpoint):
        def hook(_module, _args, output):
            states, _rebuild = _split_output(output)
            cache[f"checkpoint_{checkpoint}"] = (
                states[:, -1, :].detach().float().cpu())
        return hook

    handles.append(blocks[int(layer)].register_forward_hook(source_hook))
    for checkpoint in CHECKPOINT_LAYERS:
        handles.append(blocks[int(checkpoint)].register_forward_hook(
            checkpoint_hook(checkpoint)))
    try:
        output = model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
        cache["logits"] = output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return cache


@torch.no_grad()
def _run_exact_patch(
        model, batch, layer, positions, target_values):
    blocks = get_decoder_layers(model)
    cache = {}
    handles = []

    def patch_hook(_module, _args, output):
        states, rebuild = _split_output(output)
        states = states.clone()
        states[:, positions, :] = target_values.to(
            device=states.device, dtype=states.dtype)
        return rebuild(states)

    def checkpoint_hook(checkpoint):
        def hook(_module, _args, output):
            states, _rebuild = _split_output(output)
            cache[f"checkpoint_{checkpoint}"] = (
                states[:, -1, :].detach().float().cpu())
        return hook

    # Registration order ensures that a checkpoint at the patch layer sees
    # the patched output.
    handles.append(blocks[int(layer)].register_forward_hook(patch_hook))
    for checkpoint in CHECKPOINT_LAYERS:
        if checkpoint <= int(layer):
            continue
        handles.append(blocks[int(checkpoint)].register_forward_hook(
            checkpoint_hook(checkpoint)))
    try:
        output = model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
        cache["logits"] = output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return cache


def _row_transport(origin, target, patched):
    direction = target.float() - origin.float()
    displacement = patched.float() - origin.float()
    norm2 = direction.square().sum(dim=-1).clamp_min(1e-8)
    progress = (displacement * direction).sum(dim=-1) / norm2
    distance_ratio = (
        (patched.float() - target.float()).norm(dim=-1)
        / direction.norm(dim=-1).clamp_min(1e-8))
    return progress.tolist(), distance_ratio.tolist()


def _direction_summary(progress, distance_ratio, accuracies):
    return {
        "n_rows": len(progress),
        "mean_progress": float(np.mean(progress)),
        "median_progress": float(np.median(progress)),
        "positive_rows": int(sum(value > 0.0 for value in progress)),
        "progress_rows": [float(value) for value in progress],
        "mean_distance_ratio": float(np.mean(distance_ratio)),
        "median_distance_ratio": float(np.median(distance_ratio)),
        "distance_ratio_rows": [
            float(value) for value in distance_ratio],
        "minimum_answer_accuracy": float(min(accuracies)),
        "answer_accuracy_by_history": [
            float(value) for value in accuracies],
    }


def _direction_pass(summary):
    return bool(
        summary["mean_progress"]
        >= MINIMUM_MEAN_PROGRESS - 1e-9
        and summary["median_distance_ratio"]
        <= MAXIMUM_MEDIAN_DISTANCE_RATIO + 1e-9
        and summary["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and summary["minimum_answer_accuracy"]
        >= MINIMUM_ANSWER_ACCURACY - 1e-9)


def _cell_score(cell):
    return min(
        cell["belief_to_search"]["mean_progress"],
        cell["search_to_belief"]["mean_progress"])


def _family_summary(cells):
    experimental = [
        cell for cell in cells
        if not cell["position_group"].startswith("identical_control")]
    controls = [
        cell for cell in cells
        if cell["position_group"].startswith("identical_control")]
    pass_cells = [cell for cell in experimental if cell["pass"]]
    control_pass_cells = [cell for cell in controls if cell["pass"]]
    one_direction_cells = [
        cell for cell in experimental
        if cell["belief_to_search_pass"]
        != cell["search_to_belief_pass"]
    ]
    best = max(experimental, key=_cell_score)
    return {
        "pass_cells": [
            {
                "layer": cell["layer"],
                "checkpoint": cell["checkpoint"],
                "position_group": cell["position_group"],
                "score": _cell_score(cell),
            }
            for cell in pass_cells
        ],
        "control_pass_cells": [
            {
                "layer": cell["layer"],
                "checkpoint": cell["checkpoint"],
                "position_group": cell["position_group"],
                "score": _cell_score(cell),
            }
            for cell in control_pass_cells
        ],
        "one_direction_cell_count": len(one_direction_cells),
        "best_cell": {
            "layer": best["layer"],
            "checkpoint": best["checkpoint"],
            "position_group": best["position_group"],
            "score": _cell_score(best),
            "belief_to_search_pass": best["belief_to_search_pass"],
            "search_to_belief_pass": best["search_to_belief_pass"],
            "belief_to_search_mean_progress":
                best["belief_to_search"]["mean_progress"],
            "search_to_belief_mean_progress":
                best["search_to_belief"]["mean_progress"],
        },
    }


def _verdict(summaries):
    reference = summaries["maximum_score"]
    target = summaries["two_hop_pointer"]
    if not reference["pass_cells"]:
        return "REFERENCE_FAMILY_NOT_REPRODUCED"
    if (
            reference["control_pass_cells"]
            or target["control_pass_cells"]):
        return "IDENTICAL_POSITION_CONTROL_FAILED"
    if target["pass_cells"]:
        target_l24 = any(
            cell["checkpoint"] == 24
            for cell in target["pass_cells"])
        target_l27 = any(
            cell["checkpoint"] == 27
            for cell in target["pass_cells"])
        target_answer = any(
            cell["position_group"] == "answer_prefix_3"
            for cell in target["pass_cells"])
        target_all = any(
            cell["position_group"] == "all_differing_6"
            for cell in target["pass_cells"])
        if target_l27 and not target_l24:
            return "L24_ASSAY_INCOMPLETE"
        if target_all and not target_answer:
            return "DISTRIBUTED_CONTROL_BEYOND_ANSWER_PREFIX"
        if (
                target["best_cell"]["layer"]
                != reference["best_cell"]["layer"]
                or target["best_cell"]["position_group"]
                != reference["best_cell"]["position_group"]):
            return "COMPUTATION_DEPENDENT_CAUSAL_LOCUS"
        return "EXACT_TRANSPLANT_ROUTE_RESCUED"
    if target["one_direction_cell_count"] > 0:
        return "DIRECTION_ASYMMETRIC_CONTROL"
    return "NO_EXACT_TRANSPLANT_CAUSAL_ROUTE"


@torch.no_grad()
def run_delta_exact_transplant_locus_diagnostic(
        model_path, out_dir,
        model_key="qwen7b_exact_transplant_locus_diagnostic",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=12):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != N_WORLD:
        raise ValueError("v1 is frozen to exactly 12 histories")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(CHECKPOINT_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("frozen checkpoint layer is absent")
    rows = _unused_rows()

    prepared = {}
    for family_index, family in enumerate(FAMILIES):
        spec = {**FAMILY_SPECS[family], "values": VALUES}
        alignment = _family_alignment(tok, dev, rows, spec)
        groups = _position_groups(
            alignment, RANDOM_SEED + 1009 * family_index)
        prepared[family] = {
            "spec": spec,
            "alignment": alignment,
            "position_groups": groups,
        }

    total_cells = (
        len(FAMILIES) * len(PATCH_LAYERS) * 5)
    # Four baseline forwards plus four patched forwards per cell/layer.
    heartbeat = Heartbeat(
        total_cells * 4,
        "exact_transplant_locus_diagnostic",
        every_sec=30, out_dir=out_dir)
    results = {}
    for family, data in prepared.items():
        alignment = data["alignment"]
        groups = data["position_groups"]
        union_positions = sorted({
            position
            for positions in groups.values()
            for position in positions
        })
        union_index = {
            position: index
            for index, position in enumerate(union_positions)
        }
        cells = []
        for layer in PATCH_LAYERS:
            baselines = {}
            for history_index, (belief, search) in enumerate(
                    alignment["batches"]):
                for operation, batch in (
                        ("belief", belief), ("search", search)):
                    baselines[(history_index, operation)] = (
                        _capture_baseline(
                            model, batch, layer, union_positions))
            for group_name, positions in groups.items():
                direction_rows = {
                    "belief_to_search": {
                        "progress": [],
                        "distance": [],
                        "accuracy": [],
                    },
                    "search_to_belief": {
                        "progress": [],
                        "distance": [],
                        "accuracy": [],
                    },
                }
                offsets = [union_index[position] for position in positions]
                for history_index, (belief, search) in enumerate(
                        alignment["batches"]):
                    expected = (
                        [row["source"] for row in rows]
                        if history_index == 0
                        else [row["target"] for row in rows])
                    for origin_operation, target_operation, batch, name in (
                            ("belief", "search", belief,
                             "belief_to_search"),
                            ("search", "belief", search,
                             "search_to_belief")):
                        target_values = baselines[
                            (history_index, target_operation)
                        ]["source"][:, offsets, :]
                        patched = _run_exact_patch(
                            model, batch, layer, positions, target_values)
                        accuracy = _generic_accuracy(
                            patched["logits"], batch, expected, VALUES)
                        direction_rows[name]["accuracy"].append(accuracy)
                        for checkpoint in CHECKPOINT_LAYERS:
                            if checkpoint <= layer:
                                continue
                            origin_state = baselines[
                                (history_index, origin_operation)
                            ][f"checkpoint_{checkpoint}"]
                            target_state = baselines[
                                (history_index, target_operation)
                            ][f"checkpoint_{checkpoint}"]
                            progress, distance = _row_transport(
                                origin_state, target_state,
                                patched[f"checkpoint_{checkpoint}"])
                            direction_rows[name].setdefault(
                                f"progress_{checkpoint}", []).extend(
                                    progress)
                            direction_rows[name].setdefault(
                                f"distance_{checkpoint}", []).extend(
                                    distance)
                        heartbeat.step(
                            extra=(
                                f"{family}/L{layer}/{group_name}/{name}"))

                for checkpoint in CHECKPOINT_LAYERS:
                    if checkpoint <= layer:
                        continue
                    belief_to_search = _direction_summary(
                        direction_rows["belief_to_search"][
                            f"progress_{checkpoint}"],
                        direction_rows["belief_to_search"][
                            f"distance_{checkpoint}"],
                        direction_rows["belief_to_search"]["accuracy"])
                    search_to_belief = _direction_summary(
                        direction_rows["search_to_belief"][
                            f"progress_{checkpoint}"],
                        direction_rows["search_to_belief"][
                            f"distance_{checkpoint}"],
                        direction_rows["search_to_belief"]["accuracy"])
                    b_pass = _direction_pass(belief_to_search)
                    s_pass = _direction_pass(search_to_belief)
                    cells.append({
                        "layer": int(layer),
                        "checkpoint": int(checkpoint),
                        "position_group": group_name,
                        "positions": list(positions),
                        "belief_to_search": belief_to_search,
                        "search_to_belief": search_to_belief,
                        "belief_to_search_pass": b_pass,
                        "search_to_belief_pass": s_pass,
                        "pass": bool(b_pass and s_pass),
                    })
        results[family] = {
            "position_groups": groups,
            "cells": cells,
        }
    heartbeat.done()

    summaries = {
        family: _family_summary(value["cells"])
        for family, value in results.items()
    }
    verdict = _verdict(summaries)
    result = {
        "stage": "delta_exact_transplant_locus_diagnostic",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "rows": rows,
        "results": results,
        "family_summaries": summaries,
        "verdict": verdict,
        "loop_closed": True,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_exact_transplant_locus_diagnostic_"
        f"{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"EXACT-TRANSPLANT-LOCUS verdict={verdict} "
        f"maximum_best={summaries['maximum_score']['best_cell']} "
        f"twohop_best={summaries['two_hop_pointer']['best_cell']} "
        f"artifact={path}")
    return result
