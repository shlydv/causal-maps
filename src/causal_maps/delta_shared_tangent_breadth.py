"""Prospective breadth screen for the frozen shared causal tangent."""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_cross_domain_controller import (
    _generic_accuracy,
    _generic_evaluate_sites,
    _public_generic_task,
)
from .delta_exact_transplant_locus_diagnostic import (
    CHECKPOINT_LAYERS,
    _capture_baseline,
    _direction_summary,
    _row_transport,
    _run_exact_patch,
)
from .delta_heterogeneous_family_screen import (
    FAMILY_SPECS,
    VALUES,
    _family_batch,
    _validate_history_change,
)
from .delta_lexical_class import TASK_GRID, _padding_plan
from .delta_operation_handoff_depth import _full_sites
from .delta_preprint_battery import _compatible_world_rows
from .delta_prospective_causal_sensitivity import (
    CONTROL_LAYERS,
    _lean_generic_context,
    _prospective_rows,
)
from .delta_sparse_transport import _attention_geometry
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)


PROTOCOL_VERSION = "2026-07-27-p2-shared-tangent-breadth-v1"
FROZEN_PREDICTION_NPZ_SHA256 = (
    "9F67BDD81BA74EABF47BBA6E74ACA6FCEE512AE7155603A30517A793C5E97CF2")
SOURCE_LAYER = 21
PRIMARY_CHECKPOINT = 27
PATCH_WIDTH = 3
TEST_N = 8
N_RANDOM = 19
ROUTE_RANDOM_INDICES = (0, 1, 2)
RANDOM_SEED = 728431
FAMILIES = (
    "latest_update",
    "key_value_lookup",
    "conditional_selection",
    "temporal_slot",
)
PANELS = ("anchor", "synonym")
DIRECTIONS = ("belief_to_search", "search_to_belief")

MINIMUM_EXACT_PROGRESS = 0.50
MINIMUM_ANCHOR_MEAN = 0.20
MINIMUM_SYNONYM_MEAN = 0.15
MINIMUM_POSITIVE_CELLS = 14
MINIMUM_POSITIVE_ROW_FRACTION = 0.80
MINIMUM_VALUE_ACCURACY = 0.80
MINIMUM_RANDOM_MARGIN = 0.05
MAXIMUM_RANDOM_P = 0.05
MAXIMUM_INSTRUCTION_FRACTION = 0.50

MINIMUM_ROUTE_GAP = 0.010
MINIMUM_EXACT_ROUTE_PROGRESS = 0.25
MINIMUM_RESOLVED_ROUTE_CELLS = 8
MINIMUM_SHARED_ROUTE_MEAN = 0.15
MINIMUM_SHARED_ROUTE_POSITIVE_FRACTION = 0.75
MINIMUM_ROUTE_CONTROL_MARGIN = 0.03


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "status": "width screen; not claim establishing",
    "hypothesis": (
        "A training-only shared layer-21 answer-prefix tangent learned on "
        "four computations causally generalizes across unseen computation "
        "families and synonymous lexical coordinates. It must move both "
        "the processed downstream state and an independent late-circuit "
        "content-propagation readout."),
    "frozen_intervention_artifact_sha256":
        FROZEN_PREDICTION_NPZ_SHA256,
    "families": list(FAMILIES),
    "panels": {
        "anchor": ["BELIEF", "X X SEARCH"],
        "synonym": [
            "position-matched THINK command",
            "position-matched FIND command",
        ],
    },
    "rows": {
        "n_directed_pairs": TEST_N,
        "selection": (
            "first eight prospective-row pairs; disjoint from the held-out "
            "inverse experiment's rows 20-49"),
        "histories_per_pair": 2,
    },
    "locus": {
        "source_layer": SOURCE_LAYER,
        "positions": "last three answer-prefix command tokens",
        "checkpoints": list(CHECKPOINT_LAYERS),
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "direct_identity_removed": True,
    },
    "primary_state_arms": [
        "frozen shared tangent",
        "sign-reversed shared tangent",
        "instruction-position shared tangent",
        "exact matched-state oracle",
        f"{N_RANDOM} random directions in the same frozen rank-four basis",
    ],
    "independent_consequence": {
        "measure": (
            "normalized movement of layer-22-through-24 content-intervention "
            "mediation toward the opposite operation"),
        "arms": [
            "frozen shared tangent",
            "sign-reversed shared tangent",
            "instruction-position shared tangent",
            "exact matched-state oracle",
            *[
                f"frozen random rank-four direction {index}"
                for index in ROUTE_RANDOM_INDICES
            ],
        ],
    },
    "state_gate": {
        "minimum_exact_progress_every_cell": MINIMUM_EXACT_PROGRESS,
        "minimum_anchor_panel_mean": MINIMUM_ANCHOR_MEAN,
        "minimum_synonym_panel_mean": MINIMUM_SYNONYM_MEAN,
        "minimum_positive_cells_of_16": MINIMUM_POSITIVE_CELLS,
        "minimum_positive_row_fraction": MINIMUM_POSITIVE_ROW_FRACTION,
        "minimum_value_accuracy": MINIMUM_VALUE_ACCURACY,
        "minimum_margin_over_best_random": MINIMUM_RANDOM_MARGIN,
        "maximum_add_one_random_p": MAXIMUM_RANDOM_P,
        "maximum_instruction_fraction_of_selected":
            MAXIMUM_INSTRUCTION_FRACTION,
        "negative_aggregate_must_be_nonpositive": True,
    },
    "consequence_gate": {
        "minimum_original_route_gap": MINIMUM_ROUTE_GAP,
        "minimum_exact_route_progress": MINIMUM_EXACT_ROUTE_PROGRESS,
        "minimum_resolved_cells": MINIMUM_RESOLVED_ROUTE_CELLS,
        "shared_arm_functional_on_every_resolved_cell": True,
        "minimum_shared_mean_progress": MINIMUM_SHARED_ROUTE_MEAN,
        "minimum_positive_fraction":
            MINIMUM_SHARED_ROUTE_POSITIVE_FRACTION,
        "minimum_margin_over_best_frozen_random":
            MINIMUM_ROUTE_CONTROL_MARGIN,
        "both_panels_must_have_positive_mean": True,
    },
    "stopping_rule": (
        "No family, wording, row, vector, layer, position, random seed, "
        "metric or threshold changes follow the output. A state-only pass "
        "does not license a controller claim; a broad two-outcome pass only "
        "licenses deeper confirmation."),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _artifact_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest().upper()


def _tensor_sha256(value):
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _resolve_artifact(path_pattern):
    hits = sorted(glob.glob(path_pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"frozen prediction artifact not found: {path_pattern}")
    exact = [
        path for path in hits
        if os.path.basename(path)
        == "heldout_inverse_predictions_qwen7b_heldout_inverse_control.npz"
    ]
    path = exact[0] if exact else hits[0]
    actual = _artifact_sha256(path)
    if actual != FROZEN_PREDICTION_NPZ_SHA256:
        raise ValueError(
            f"frozen prediction artifact hash mismatch: {actual}")
    return path


def _load_frozen_interventions(path_pattern):
    path = _resolve_artifact(path_pattern)
    with np.load(path) as archive:
        basis = torch.from_numpy(np.array(archive["basis"])).float()
        coefficients = {}
        for direction in DIRECTIONS:
            key = f"minimum_score_{direction}_shared_inverse"
            coefficients[direction] = torch.from_numpy(
                np.array(archive[key])).float()
            comparison = torch.from_numpy(np.array(
                archive[
                    f"set_intersection_{direction}_shared_inverse"]
            )).float()
            if not torch.equal(coefficients[direction], comparison):
                raise AssertionError(
                    f"shared coefficients vary by held-out family: "
                    f"{direction}")
    if basis.shape[0] != 4:
        raise ValueError("frozen basis is not rank four")
    deltas = {
        direction: (
            coefficients[direction] @ basis
        ).reshape(1, PATCH_WIDTH, -1)
        for direction in DIRECTIONS
    }
    return {
        "path": path,
        "artifact_sha256": _artifact_sha256(path),
        "basis": basis,
        "coefficients": coefficients,
        "deltas": deltas,
        "basis_sha256": _tensor_sha256(basis),
        "coefficient_sha256": {
            direction: _tensor_sha256(coefficients[direction])
            for direction in DIRECTIONS
        },
        "delta_sha256": {
            direction: _tensor_sha256(deltas[direction])
            for direction in DIRECTIONS
        },
    }


def _command_pair(tok):
    rows, _indices = _compatible_world_rows(
        tok, torch.device("cpu"), 30)
    _target, padding_plan, _tables = _padding_plan(tok, rows[0])
    if padding_plan is None:
        raise ValueError("position-matched lexical padding plan is absent")
    synonym = (
        padding_plan[TASK_GRID["epistemic"]["THINK"]]["command"],
        padding_plan[TASK_GRID["search"]["FIND"]]["command"],
    )
    return {
        "anchor": ("BELIEF", "X X SEARCH"),
        "synonym": synonym,
    }, padding_plan


def _contiguous_groups(positions):
    groups = []
    for position in positions:
        if not groups or position != groups[-1][-1] + 1:
            groups.append([position])
        else:
            groups[-1].append(position)
    return groups


def _panel_alignment(tok, dev, rows, spec, commands):
    natural_rows = [
        {**row, "state": row["target"]} for row in rows
    ]
    batches = []
    reference_mask = None
    marker = None
    for history_rows in (rows, natural_rows):
        belief = _family_batch(
            tok, history_rows, spec, commands[0], dev)
        search = _family_batch(
            tok, history_rows, spec, commands[1], dev)
        if belief["ids"].shape != search["ids"].shape:
            raise ValueError("panel operation shapes differ")
        if belief["marker"] != search["marker"]:
            raise ValueError("panel operation marker positions differ")
        difference = belief["ids"] != search["ids"]
        if not bool((difference == difference[0:1]).all()):
            raise ValueError("panel command mask varies by row")
        mask = difference[0].detach().cpu()
        if reference_mask is not None and not torch.equal(
                reference_mask, mask):
            raise ValueError("panel command mask varies by history")
        reference_mask = mask
        if marker is not None and marker != int(belief["marker"]):
            raise ValueError("panel marker varies by history")
        marker = int(belief["marker"])
        batches.append((belief, search))
    _validate_history_change(batches[0][0], batches[1][0])
    _validate_history_change(batches[0][1], batches[1][1])

    groups = _contiguous_groups(torch.nonzero(
        reference_mask, as_tuple=False).flatten().tolist())
    if len(groups) != 2:
        raise ValueError(
            f"expected instruction and answer command groups, got {groups}")
    readout = int(batches[0][0]["ids"].shape[1] - 1)
    answer_positions = list(range(
        groups[-1][-1] - PATCH_WIDTH + 1, groups[-1][-1] + 1))
    instruction_positions = list(range(
        groups[0][-1] - PATCH_WIDTH + 1, groups[0][-1] + 1))
    if min(answer_positions + instruction_positions) < 0:
        raise ValueError("a command window starts before the prompt")
    if set(answer_positions) & set(instruction_positions):
        raise ValueError("instruction and answer command windows overlap")
    if answer_positions != list(range(
            readout - PATCH_WIDTH + 1, readout + 1)):
        raise ValueError("answer command is not the final three tokens")
    return {
        "batches": batches,
        "marker": marker,
        "readout": readout,
        "instruction_positions": instruction_positions,
        "answer_positions": answer_positions,
        "difference_groups": groups,
    }


def _random_basis_deltas(basis, reference, direction_index):
    generator = torch.Generator().manual_seed(
        RANDOM_SEED + int(direction_index) * 100003)
    controls = []
    reference_norm = reference.norm().clamp_min(1e-8)
    for _index in range(N_RANDOM):
        coefficients = torch.randn(
            1, basis.shape[0], generator=generator)
        value = (
            coefficients @ basis
        ).reshape(1, PATCH_WIDTH, -1)
        value = value * (
            reference_norm / value.norm().clamp_min(1e-8))
        controls.append(value)
    return controls


def _history_expected(rows, history):
    key = "source" if int(history) == 0 else "target"
    return [row[key] for row in rows]


def _processed(checkpoint, source):
    return checkpoint.float() - source[:, -1, :].float()


def _empty_accumulator():
    return {
        checkpoint: {
            "progress": [],
            "distance": [],
        }
        for checkpoint in CHECKPOINT_LAYERS
    }


def _append_transport(accumulator, origin, target, patched, patched_direct):
    for checkpoint in CHECKPOINT_LAYERS:
        origin_processed = _processed(
            origin[f"checkpoint_{checkpoint}"], origin["source"])
        target_processed = _processed(
            target[f"checkpoint_{checkpoint}"], target["source"])
        patched_processed = (
            patched[f"checkpoint_{checkpoint}"] - patched_direct)
        progress, distance = _row_transport(
            origin_processed, target_processed, patched_processed)
        accumulator[checkpoint]["progress"].extend(progress)
        accumulator[checkpoint]["distance"].extend(distance)


def _arm_summary(accumulator, accuracies):
    return {
        str(checkpoint): _direction_summary(
            accumulator[checkpoint]["progress"],
            accumulator[checkpoint]["distance"],
            accuracies)
        for checkpoint in CHECKPOINT_LAYERS
    }


def _sequence_patch(states, positions, delta=None, target=None):
    patches = []
    for history in range(2):
        if target is not None:
            values = target[history]["source"]
        else:
            values = (
                states[history]["source"]
                + delta.expand(states[history]["source"].shape[0], -1, -1))
        patches.append((positions, values))
    return tuple(patches)


def _route_value(cell):
    value = cell["mediation"]["minimum_fraction"]
    return None if value is None else float(value)


def _route_progress(origin, target, patched):
    if any(value is None for value in (origin, target, patched)):
        return None
    gap = float(target) - float(origin)
    if abs(gap) < MINIMUM_ROUTE_GAP - 1e-12:
        return None
    return (float(patched) - float(origin)) / gap


def _tail_probability(observed, random_values):
    exceed = sum(
        float(value) >= float(observed) for value in random_values)
    return (1.0 + exceed) / (1.0 + len(random_values)), exceed


def _aggregate_state(cells, arm):
    selected = [
        cell["state_arms"][arm][str(PRIMARY_CHECKPOINT)]
        for cell in cells.values()
    ]
    means = [float(value["mean_progress"]) for value in selected]
    row_progress = [
        float(row)
        for value in selected for row in value["progress_rows"]
    ]
    panels = {
        panel: float(np.mean([
            float(cell["state_arms"][arm][
                str(PRIMARY_CHECKPOINT)]["mean_progress"])
            for cell in cells.values()
            if cell["panel"] == panel
        ]))
        for panel in PANELS
    }
    return {
        "mean_progress": float(np.mean(means)),
        "minimum_cell_progress": float(min(means)),
        "positive_cells": int(sum(value > 0.0 for value in means)),
        "positive_row_fraction": float(np.mean([
            value > 0.0 for value in row_progress])),
        "minimum_answer_accuracy": float(min(
            value["minimum_answer_accuracy"] for value in selected)),
        "panel_mean_progress": panels,
        "cell_mean_progress": means,
    }


def _aggregate_route(cells, arm, resolved_keys):
    values = [
        cells[key]["route_arms"][arm]["normalized_progress"]
        for key in resolved_keys
        if cells[key]["route_arms"][arm]["normalized_progress"] is not None
    ]
    panels = {}
    for panel in PANELS:
        panel_values = [
            cells[key]["route_arms"][arm]["normalized_progress"]
            for key in resolved_keys
            if cells[key]["panel"] == panel
            and cells[key]["route_arms"][arm][
                "normalized_progress"] is not None
        ]
        panels[panel] = (
            float(np.mean(panel_values)) if panel_values else None)
    return {
        "n_cells": len(values),
        "functional_cells": int(sum(
            bool(cells[key]["route_arms"][arm]["functional"])
            for key in resolved_keys)),
        "mean_progress": (
            float(np.mean(values)) if values else None),
        "positive_fraction": (
            float(np.mean([value > 0.0 for value in values]))
            if values else None),
        "panel_mean_progress": panels,
        "cell_progress": [float(value) for value in values],
    }


def _adjudicate(cells):
    exact_state = _aggregate_state(cells, "exact_state_oracle")
    shared = _aggregate_state(cells, "shared")
    negative = _aggregate_state(cells, "negative")
    instruction = _aggregate_state(cells, "instruction")
    random_aggregates = [
        _aggregate_state(cells, f"random_{index:02d}")
        for index in range(N_RANDOM)
    ]
    selected_score = min(
        shared["panel_mean_progress"].values())
    random_scores = [
        min(value["panel_mean_progress"].values())
        for value in random_aggregates
    ]
    random_p, random_exceed = _tail_probability(
        selected_score, random_scores)
    best_random = max(random_scores)
    state_pass = bool(
        exact_state["minimum_cell_progress"]
        >= MINIMUM_EXACT_PROGRESS - 1e-9
        and shared["panel_mean_progress"]["anchor"]
        >= MINIMUM_ANCHOR_MEAN - 1e-9
        and shared["panel_mean_progress"]["synonym"]
        >= MINIMUM_SYNONYM_MEAN - 1e-9
        and shared["positive_cells"] >= MINIMUM_POSITIVE_CELLS
        and shared["positive_row_fraction"]
        >= MINIMUM_POSITIVE_ROW_FRACTION - 1e-9
        and shared["minimum_answer_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
        and selected_score - best_random
        >= MINIMUM_RANDOM_MARGIN - 1e-9
        and random_p <= MAXIMUM_RANDOM_P + 1e-12
        and instruction["mean_progress"]
        < MAXIMUM_INSTRUCTION_FRACTION
        * shared["mean_progress"] + 1e-12
        and negative["mean_progress"] <= 1e-9)

    resolved_keys = [
        key for key, cell in cells.items()
        if (
            cell["route_arms"]["exact_state_oracle"][
                "functional"]
            and cell["route_arms"]["exact_state_oracle"][
                "normalized_progress"] is not None
            and cell["route_arms"]["exact_state_oracle"][
                "normalized_progress"]
            >= MINIMUM_EXACT_ROUTE_PROGRESS - 1e-9)
    ]
    shared_route = _aggregate_route(cells, "shared", resolved_keys)
    negative_route = _aggregate_route(cells, "negative", resolved_keys)
    instruction_route = _aggregate_route(
        cells, "instruction", resolved_keys)
    route_random = {
        f"random_{index:02d}": _aggregate_route(
            cells, f"random_{index:02d}", resolved_keys)
        for index in ROUTE_RANDOM_INDICES
    }
    route_random_means = [
        value["mean_progress"] for value in route_random.values()
        if value["mean_progress"] is not None
    ]
    route_best_random = (
        max(route_random_means) if route_random_means else None)
    route_panel_positive = bool(all(
        value is not None and value > 0.0
        for value in shared_route["panel_mean_progress"].values()))
    consequence_pass = bool(
        len(resolved_keys) >= MINIMUM_RESOLVED_ROUTE_CELLS
        and shared_route["functional_cells"] == len(resolved_keys)
        and shared_route["mean_progress"] is not None
        and shared_route["mean_progress"]
        >= MINIMUM_SHARED_ROUTE_MEAN - 1e-9
        and shared_route["positive_fraction"] is not None
        and shared_route["positive_fraction"]
        >= MINIMUM_SHARED_ROUTE_POSITIVE_FRACTION - 1e-9
        and route_panel_positive
        and route_best_random is not None
        and shared_route["mean_progress"] - route_best_random
        >= MINIMUM_ROUTE_CONTROL_MARGIN - 1e-9
        and (
            negative_route["mean_progress"] is None
            or negative_route["mean_progress"]
            < shared_route["mean_progress"])
        and (
            instruction_route["mean_progress"] is None
            or instruction_route["mean_progress"]
            < shared_route["mean_progress"]))
    if state_pass and consequence_pass:
        verdict = "BROAD_SHARED_CAUSAL_TANGENT"
    elif state_pass:
        verdict = "SHARED_STATE_TANGENT_WITHOUT_CAUSAL_PROPAGATION"
    elif len(resolved_keys) < MINIMUM_RESOLVED_ROUTE_CELLS:
        verdict = "STATE_BREADTH_FAILED_AND_ROUTE_ASSAY_UNRESOLVED"
    else:
        verdict = "NO_SHARED_TANGENT_BREADTH"
    return {
        "state": {
            "exact": exact_state,
            "shared": shared,
            "negative": negative,
            "instruction": instruction,
            "random_aggregates": random_aggregates,
            "selected_score": selected_score,
            "best_random_score": best_random,
            "margin_over_best_random": selected_score - best_random,
            "random_empirical_p": random_p,
            "random_exceed_count": random_exceed,
            "pass": state_pass,
        },
        "consequence": {
            "resolved_cells": resolved_keys,
            "shared": shared_route,
            "negative": negative_route,
            "instruction": instruction_route,
            "random": route_random,
            "best_random_mean": route_best_random,
            "pass": consequence_pass,
        },
        "verdict": verdict,
    }


def _self_check():
    synthetic = {}
    for family in FAMILIES:
        for panel in PANELS:
            for direction in DIRECTIONS:
                key = f"{family}/{panel}/{direction}"
                state_arms = {}
                for arm, progress in (
                        ("exact_state_oracle", 0.8),
                        ("shared", 0.3),
                        ("negative", -0.1),
                        ("instruction", 0.05)):
                    state_arms[arm] = {
                        str(checkpoint): {
                            "mean_progress": progress,
                            "minimum_answer_accuracy": 1.0,
                            "progress_rows": [progress] * (2 * TEST_N),
                        }
                        for checkpoint in CHECKPOINT_LAYERS
                    }
                for index in range(N_RANDOM):
                    progress = 0.05 + 0.001 * index
                    state_arms[f"random_{index:02d}"] = {
                        str(checkpoint): {
                            "mean_progress": progress,
                            "minimum_answer_accuracy": 1.0,
                            "progress_rows": [progress] * (2 * TEST_N),
                        }
                        for checkpoint in CHECKPOINT_LAYERS
                    }
                route_arms = {
                    "exact_state_oracle": {
                        "functional": True,
                        "normalized_progress": 0.8,
                    },
                    "shared": {
                        "functional": True,
                        "normalized_progress": 0.3,
                    },
                    "negative": {
                        "functional": True,
                        "normalized_progress": -0.1,
                    },
                    "instruction": {
                        "functional": True,
                        "normalized_progress": 0.02,
                    },
                }
                for index in ROUTE_RANDOM_INDICES:
                    route_arms[f"random_{index:02d}"] = {
                        "functional": True,
                        "normalized_progress": 0.05,
                    }
                synthetic[key] = {
                    "family": family,
                    "panel": panel,
                    "direction": direction,
                    "state_arms": state_arms,
                    "route_arms": route_arms,
                }
    result = _adjudicate(synthetic)
    if result["verdict"] != "BROAD_SHARED_CAUSAL_TANGENT":
        raise AssertionError("positive breadth adjudication self-check failed")
    synthetic[next(iter(synthetic))]["state_arms"]["shared"]["27"][
        "mean_progress"] = -2.0
    negative_result = _adjudicate(synthetic)
    if negative_result["state"]["pass"]:
        raise AssertionError("negative breadth adjudication self-check failed")
    return {
        "positive_adjudication_check": True,
        "negative_adjudication_check": True,
        "panel_cell_count_check": len(synthetic) == 16,
        "pass": True,
    }


@torch.no_grad()
def run_delta_shared_tangent_breadth(
        model_path, controller_archive_path, out_dir,
        model_key="qwen7b_shared_tangent_breadth",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=TEST_N, self_test_only=False):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TEST_N:
        raise ValueError("v1 is frozen to exactly eight directed pairs")
    self_check = _self_check()
    frozen = _load_frozen_interventions(controller_archive_path)
    frozen_public = {
        key: value for key, value in frozen.items()
        if key not in ("basis", "coefficients", "deltas")
    }
    if self_test_only:
        result = {
            "stage": "delta_shared_tangent_breadth",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "self_check": self_check,
            "frozen_intervention": frozen_public,
            "verdict": "SELF_CHECK_PASS",
        }
        path = os.path.join(
            out_dir, "delta_shared_tangent_breadth_self_check.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log(
            "SHARED-TANGENT-BREADTH self-check pass "
            f"protocol={PROTOCOL_SHA256} "
            f"artifact={frozen['artifact_sha256']}")
        return result

    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if PRIMARY_CHECKPOINT >= model_num_hidden_layers(model):
        raise ValueError("frozen checkpoint is absent")
    n_heads, head_dim = _attention_geometry(model)
    l24_sites = _full_sites((22, 23, 24), n_heads)
    rows = _prospective_rows(VALUES)[:TEST_N]
    commands, padding_plan = _command_pair(tok)

    prepared = {}
    for family in FAMILIES:
        prepared[family] = {}
        for panel in PANELS:
            prepared[family][panel] = _panel_alignment(
                tok, dev, rows, FAMILY_SPECS[family], commands[panel])

    random_deltas = {
        direction: _random_basis_deltas(
            frozen["basis"], frozen["deltas"][direction],
            DIRECTIONS.index(direction))
        for direction in DIRECTIONS
    }
    state_arm_names = [
        "shared",
        "negative",
        "instruction",
        "exact_state_oracle",
        *[f"random_{index:02d}" for index in range(N_RANDOM)],
    ]
    route_arm_names = [
        "shared",
        "negative",
        "instruction",
        "exact_state_oracle",
        *[f"random_{index:02d}" for index in ROUTE_RANDOM_INDICES],
    ]
    state_total = (
        len(FAMILIES) * len(PANELS)
        * (8 + len(DIRECTIONS) * 2 * len(state_arm_names)))
    route_total = (
        len(FAMILIES) * len(PANELS)
        * (
            2 * 4
            + len(DIRECTIONS) * len(route_arm_names) * 4))
    heartbeat = Heartbeat(
        state_total + route_total,
        "shared_tangent_breadth",
        every_sec=30, out_dir=out_dir)
    cells = {}

    for family in FAMILIES:
        for panel in PANELS:
            alignment = prepared[family][panel]
            baselines = {
                "belief": [],
                "search": [],
                "belief_instruction": [],
                "search_instruction": [],
            }
            for history, pair in enumerate(alignment["batches"]):
                for operation, index in (("belief", 0), ("search", 1)):
                    baselines[operation].append(_capture_baseline(
                        model, pair[index], SOURCE_LAYER,
                        alignment["answer_positions"]))
                    heartbeat.step(
                        extra=(
                            f"{family}/{panel}/{operation}/h{history}/"
                            "answer_baseline"))
                    baselines[f"{operation}_instruction"].append(
                        _capture_baseline(
                            model, pair[index], SOURCE_LAYER,
                            alignment["instruction_positions"]))
                    heartbeat.step(
                        extra=(
                            f"{family}/{panel}/{operation}/h{history}/"
                            "instruction_baseline"))

            for direction in DIRECTIONS:
                origin_operation, target_operation = (
                    ("belief", "search")
                    if direction == "belief_to_search"
                    else ("search", "belief"))
                origin_index = 0 if origin_operation == "belief" else 1
                accumulators = {
                    arm: _empty_accumulator() for arm in state_arm_names
                }
                accuracies = {arm: [] for arm in state_arm_names}
                delta = frozen["deltas"][direction]
                for history, pair in enumerate(alignment["batches"]):
                    origin_batch = pair[origin_index]
                    origin = baselines[origin_operation][history]
                    target = baselines[target_operation][history]
                    expected = _history_expected(rows, history)
                    for arm in state_arm_names:
                        if arm == "shared":
                            arm_delta = delta
                            positions = alignment["answer_positions"]
                            source = origin["source"]
                            target_for_metric = target
                        elif arm == "negative":
                            arm_delta = -delta
                            positions = alignment["answer_positions"]
                            source = origin["source"]
                            target_for_metric = target
                        elif arm == "instruction":
                            arm_delta = delta
                            positions = alignment["instruction_positions"]
                            source = baselines[
                                f"{origin_operation}_instruction"][history][
                                    "source"]
                            target_for_metric = target
                        elif arm == "exact_state_oracle":
                            arm_delta = (
                                target["source"] - origin["source"])
                            positions = alignment["answer_positions"]
                            source = origin["source"]
                            target_for_metric = target
                        else:
                            random_index = int(arm.split("_")[-1])
                            arm_delta = random_deltas[
                                direction][random_index]
                            positions = alignment["answer_positions"]
                            source = origin["source"]
                            target_for_metric = target
                        expanded = arm_delta.expand(
                            source.shape[0], -1, -1)
                        patched = _run_exact_patch(
                            model, origin_batch, SOURCE_LAYER,
                            positions, source + expanded)
                        accuracies[arm].append(float(_generic_accuracy(
                            patched["logits"], origin_batch,
                            expected, VALUES)))
                        if arm == "instruction":
                            patched_direct = (
                                origin["source"][:, -1, :])
                        else:
                            patched_direct = (
                                origin["source"][:, -1, :]
                                + expanded[:, -1, :])
                        _append_transport(
                            accumulators[arm], origin,
                            target_for_metric, patched, patched_direct)
                        heartbeat.step(
                            extra=(
                                f"{family}/{panel}/{direction}/h{history}/"
                                f"{arm}"))
                key = f"{family}/{panel}/{direction}"
                cells[key] = {
                    "family": family,
                    "panel": panel,
                    "direction": direction,
                    "state_arms": {
                        arm: _arm_summary(
                            accumulators[arm], accuracies[arm])
                        for arm in state_arm_names
                    },
                    "route_arms": {},
                }

            # Independent consequence assay. Baseline operation contexts are
            # shared by the two directions.
            route_baseline = {}
            for operation, index in (("belief", 0), ("search", 1)):
                clean = alignment["batches"][0][index]
                natural = alignment["batches"][1][index]
                context = _lean_generic_context(
                    model, clean, natural, list(VALUES),
                    [row["source"] for row in rows],
                    [row["target"] for row in rows],
                    CONTROL_LAYERS, head_dim)
                route_baseline[operation] = {
                    "context": context,
                    "public": _public_generic_task(context),
                    "cell": _generic_evaluate_sites(
                        model, context, l24_sites, head_dim),
                }
                for phase in ("context", "source", "L24", "public"):
                    heartbeat.step(
                        extra=f"{family}/{panel}/{operation}/route/{phase}")

            for direction in DIRECTIONS:
                origin_operation, target_operation = (
                    ("belief", "search")
                    if direction == "belief_to_search"
                    else ("search", "belief"))
                origin_index = 0 if origin_operation == "belief" else 1
                origin_value = _route_value(
                    route_baseline[origin_operation]["cell"])
                target_value = _route_value(
                    route_baseline[target_operation]["cell"])
                cell_key = f"{family}/{panel}/{direction}"
                for arm in route_arm_names:
                    if arm == "exact_state_oracle":
                        patch = _sequence_patch(
                            baselines[origin_operation],
                            alignment["answer_positions"],
                            target=baselines[target_operation])
                    elif arm == "instruction":
                        patch = _sequence_patch(
                            baselines[
                                f"{origin_operation}_instruction"],
                            alignment["instruction_positions"],
                            delta=frozen["deltas"][direction])
                    elif arm == "negative":
                        patch = _sequence_patch(
                            baselines[origin_operation],
                            alignment["answer_positions"],
                            delta=-frozen["deltas"][direction])
                    elif arm == "shared":
                        patch = _sequence_patch(
                            baselines[origin_operation],
                            alignment["answer_positions"],
                            delta=frozen["deltas"][direction])
                    else:
                        random_index = int(arm.split("_")[-1])
                        patch = _sequence_patch(
                            baselines[origin_operation],
                            alignment["answer_positions"],
                            delta=random_deltas[
                                direction][random_index])
                    clean = alignment["batches"][0][origin_index]
                    natural = alignment["batches"][1][origin_index]
                    context = _lean_generic_context(
                        model, clean, natural, list(VALUES),
                        [row["source"] for row in rows],
                        [row["target"] for row in rows],
                        CONTROL_LAYERS, head_dim,
                        sequence_patch=patch)
                    public = _public_generic_task(context)
                    route_cell = _generic_evaluate_sites(
                        model, context, l24_sites, head_dim)
                    patched_value = _route_value(route_cell)
                    functional = bool(
                        public["eligible"]
                        and public["source_intervention"]["sufficient"])
                    cells[cell_key]["route_arms"][arm] = {
                        "origin_value": origin_value,
                        "target_value": target_value,
                        "patched_value": patched_value,
                        "original_gap": (
                            None if (
                                origin_value is None
                                or target_value is None)
                            else target_value - origin_value),
                        "normalized_progress": _route_progress(
                            origin_value, target_value, patched_value),
                        "functional": functional,
                        "task": public,
                        "l24_cell": route_cell,
                    }
                    for phase in ("context", "source", "L24", "public"):
                        heartbeat.step(
                            extra=(
                                f"{family}/{panel}/{direction}/route/"
                                f"{arm}/{phase}"))
    heartbeat.done()

    adjudication = _adjudicate(cells)
    result = {
        "stage": "delta_shared_tangent_breadth",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "self_check": self_check,
        "frozen_intervention": frozen_public,
        "commands": commands,
        "padding_plan": padding_plan,
        "rows": rows,
        "alignment": {
            family: {
                panel: {
                    key: value
                    for key, value in prepared[family][panel].items()
                    if key != "batches"
                }
                for panel in PANELS
            }
            for family in FAMILIES
        },
        "random_delta_sha256": {
            direction: [
                _tensor_sha256(value)
                for value in random_deltas[direction]
            ]
            for direction in DIRECTIONS
        },
        "cells": cells,
        "adjudication": adjudication,
        "verdict": adjudication["verdict"],
    }
    path = os.path.join(
        out_dir,
        f"results_delta_shared_tangent_breadth_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"SHARED-TANGENT-BREADTH verdict={result['verdict']} "
        f"state={adjudication['state']['pass']} "
        f"consequence={adjudication['consequence']['pass']} "
        f"artifact={path}")
    return result
