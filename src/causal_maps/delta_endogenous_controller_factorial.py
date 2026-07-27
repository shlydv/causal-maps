"""Frozen endogenous-faithfulness and P x R interaction experiment.

This experiment distinguishes a naturally used controller coordinate from an
effective but off-manifold injected shortcut.  It reuses the independently
validated color residual R and its orthogonal high-energy projection P, then
compares additive sufficiency, endogenous equalization, their factorial
interaction, and exact natural answer-prefix interchange.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_content_cancelled_controller import (
    EPS,
    _fixed_patch,
    _movement,
    _random_position_sets,
    _summary,
    _world_mediation,
)
from .delta_controller_matrix import (
    _controller_from_alignment,
    _fresh_domain_rows,
    _location_alignment,
)
from .delta_cross_domain_controller import (
    DOMAIN_SPECS,
    _domain_alignment,
    _domain_rows,
    _generic_curve,
    _generic_evaluate_sites,
    _generic_task_context,
    _public_generic_task,
)
from .delta_distributed_label_transplant import _capture_full_l21
from .delta_leave_color_out_shared import (
    SIGN_MINIMUM,
    SIGN_P_MAX,
    TEST_N,
    _controller_metadata,
    _fresh_color_rows,
    _original_gate,
    _shared_controllers,
    _sign_tail,
    _statistical_world_movement,
)
from .delta_operation_handoff_depth import CAPTURE_LAYERS, LAYERS, _full_sites
from .delta_preprint_battery import _compatible_world_rows
from .delta_residual_only_confirmation import _fresh_color_rows_v3
from .delta_shared_adapter_decomposition import (
    _color_decomposition,
    _fresh_color_rows_v2,
    _orthogonal_norm_matched_residuals,
)
from .delta_sparse_transport import _attention_geometry
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output


PROTOCOL_VERSION = "2026-07-26-p2-endogenous-controller-factorial-v1"
CARRIER_SENTENCE = (
    "A sealed envelope rested on a nearby table throughout."
)
DONOR_N = 15
TRAJECTORY_LAYERS = tuple(range(21, 28))
INTERACTION_SIGN_MINIMUM = 24
MINIMUM_ORIGINAL_GAP = 0.03
MINIMUM_NECESSITY_FRACTION = 0.50
MINIMUM_NATURALIZED_SUFFICIENCY = 0.50
MINIMUM_ABSOLUTE_INTERACTION_FRACTION = 0.15
MAXIMUM_TRAJECTORY_DISTANCE_RATIO = 1.0
MAXIMUM_EQUALIZED_SEPARATION_RATIO = 0.65
NUMERICAL_TOLERANCE = 1e-5
CONTROL_DIRECTION_SEED = 71027
CONTROL_POSITION_SEED = 71039

PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "domain": "color_state",
    "evaluation_rows": 30,
    "donor_rows_per_domain": 15,
    "carrier_sentence": CARRIER_SENTENCE,
    "source_layer": 21,
    "route_layers": list(LAYERS),
    "trajectory_layers": list(TRAJECTORY_LAYERS),
    "arms": [
        "original",
        "residual_additive_switch",
        "projection_equalization",
        "residual_equalization",
        "joint_projection_residual_equalization",
        "natural_answer_prefix_interchange",
    ],
    "scale_free_gates": {
        "minimum_original_gap": MINIMUM_ORIGINAL_GAP,
        "minimum_residual_necessity_fraction": MINIMUM_NECESSITY_FRACTION,
        "minimum_residual_naturalized_sufficiency":
            MINIMUM_NATURALIZED_SUFFICIENCY,
        "minimum_absolute_interaction_fraction":
            MINIMUM_ABSOLUTE_INTERACTION_FRACTION,
        "minimum_interaction_consistent_worlds": INTERACTION_SIGN_MINIMUM,
        "maximum_trajectory_distance_ratio":
            MAXIMUM_TRAJECTORY_DISTANCE_RATIO,
        "maximum_equalized_separation_ratio":
            MAXIMUM_EQUALIZED_SEPARATION_RATIO,
        "minimum_world_fraction_for_trajectory_sign": 0.90,
    },
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _carrier_spec():
    spec = dict(DOMAIN_SPECS["color_state"])
    spec["rule"] = f"{spec['rule']} {CARRIER_SENTENCE}"
    return spec


def _fresh_color_rows_v4():
    """Construct 30 histories disjoint from every previous color evaluation."""
    values = DOMAIN_SPECS["color_state"]["values"]
    prior_rows = (
        _domain_rows(values)
        + _fresh_domain_rows(values)
        + _fresh_color_rows()
        + _fresh_color_rows_v2()
        + _fresh_color_rows_v3()
    )
    used = {
        (state, row["d1"], row["d2"])
        for row in prior_rows
        for state in (row["source"], row["target"])
    }
    # Frozen before the run. Source-target pairs are unique; source and target
    # marginals occur three or four times each.
    frozen = (
        ("red", "white", "green", "pink"),
        ("blue", "brown", "gray", "black"),
        ("green", "blue", "white", "gray"),
        ("black", "green", "pink", "red"),
        ("white", "pink", "green", "red"),
        ("brown", "black", "gray", "pink"),
        ("pink", "red", "white", "green"),
        ("gray", "blue", "brown", "black"),
        ("red", "gray", "black", "green"),
        ("blue", "pink", "black", "green"),
        ("green", "red", "pink", "white"),
        ("black", "white", "pink", "brown"),
        ("white", "brown", "red", "gray"),
        ("brown", "gray", "black", "blue"),
        ("pink", "black", "red", "brown"),
        ("gray", "green", "white", "pink"),
        ("red", "blue", "white", "pink"),
        ("blue", "black", "brown", "white"),
        ("green", "white", "gray", "pink"),
        ("black", "gray", "blue", "brown"),
        ("white", "green", "red", "pink"),
        ("brown", "pink", "blue", "red"),
        ("pink", "brown", "red", "black"),
        ("gray", "red", "white", "black"),
        ("red", "green", "brown", "white"),
        ("blue", "red", "black", "pink"),
        ("green", "black", "brown", "blue"),
        ("black", "blue", "gray", "brown"),
        ("white", "gray", "blue", "red"),
        ("brown", "white", "gray", "green"),
    )
    rows = [
        {
            "row_index": index,
            "source": source,
            "target": target,
            "state": source,
            "d1": d1,
            "d2": d2,
        }
        for index, (source, target, d1, d2) in enumerate(frozen)
    ]
    if len({(row["source"], row["target"]) for row in rows}) != TEST_N:
        raise AssertionError("v4 source-target pairs are not unique")
    if any(
            (state, row["d1"], row["d2"]) in used
            for row in rows for state in (row["source"], row["target"])):
        raise AssertionError("v4 prompt histories overlap prior evaluations")
    signatures = {
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }
    if len(signatures) != 2 * TEST_N:
        raise AssertionError("v4 clean/natural histories are not unique")
    return rows


def _nested_max(rows):
    return float(max(
        max(float(value) for value in row)
        for row in rows
    ))


def _equalization_patch_pair_axes(
        belief_states, search_states, positions, named_axes):
    """Remove the named sequence-level coordinates while preserving midpoint.

    The three answer-prefix positions form one structured vector.  Coefficients
    are therefore solved on the flattened 3 x d state, matching the scalar dose
    used by the validated additive controller.  This deliberately avoids the
    previous necessity test's more permissive three-independent-coefficient
    intervention.
    """
    if len(belief_states) != len(search_states):
        raise ValueError("BELIEF/SEARCH state batches differ")
    axes = {
        name: value.detach().double().cpu()
        for name, value in named_axes.items()
    }
    if not axes:
        raise ValueError("at least one equalization axis is required")
    expected_shape = (
        len(positions), int(belief_states[0].shape[-1]))
    if any(tuple(axis.shape) != expected_shape for axis in axes.values()):
        raise ValueError("equalization axis has the wrong shape")
    names = list(axes)
    axis_matrix = torch.stack(
        [axes[name].flatten() for name in names], dim=0)
    gram = axis_matrix @ axis_matrix.T
    if int(torch.linalg.matrix_rank(gram)) != len(names):
        raise ValueError("equalization axes are linearly dependent")
    gram_inverse = torch.linalg.inv(gram)
    pairwise_cosines = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            left = axes[left_name].flatten()
            right = axes[right_name].flatten()
            cosine = torch.dot(left, right) / (
                left.norm() * right.norm()).clamp_min(EPS)
            pairwise_cosines[
                f"{left_name}__{right_name}"] = float(cosine)

    belief_patches = []
    search_patches = []
    midpoint_errors = []
    coordinate_errors = {name: [] for name in names}
    orthogonal_errors = []
    coefficients = {name: [] for name in names}
    for belief, search in zip(belief_states, search_states):
        b = belief[:, positions, :].double().cpu()
        s = search[:, positions, :].double().cpu()
        difference = b - s
        difference_flat = difference.flatten(start_dim=1)
        coefficient_matrix = (
            difference_flat @ axis_matrix.T @ gram_inverse)
        removed = (coefficient_matrix @ axis_matrix).reshape_as(difference)
        for axis_index, name in enumerate(names):
            coefficients[name].extend(
                coefficient_matrix[:, axis_index].tolist())
        new_b = (b - 0.5 * removed).float()
        new_s = (s + 0.5 * removed).float()
        actual_b = new_b.double()
        actual_s = new_s.double()
        actual_difference = actual_b - actual_s
        actual_flat = actual_difference.flatten(start_dim=1)
        for axis_index, name in enumerate(names):
            coordinate = (
                actual_flat @ axis_matrix[axis_index]
                / gram[axis_index, axis_index])
            coordinate_errors[name].extend(
                coordinate.abs().unsqueeze(-1).tolist())
        expected_orthogonal = difference - removed
        relative_orthogonal_error = (
            (actual_difference - expected_orthogonal).norm(dim=-1)
            / difference.norm(dim=-1).clamp_min(EPS)
        )
        midpoint_error = (
            0.5 * (actual_b + actual_s) - 0.5 * (b + s)
        ).abs().amax(dim=-1)
        orthogonal_errors.extend(relative_orthogonal_error.tolist())
        midpoint_errors.extend(midpoint_error.tolist())
        belief_patches.append((list(positions), new_b))
        search_patches.append((list(positions), new_s))

    invariants = {
        "axes": names,
        "pairwise_position_cosines": pairwise_cosines,
        "maximum_midpoint_absolute_error": _nested_max(midpoint_errors),
        "maximum_axis_coordinate_difference": {
            name: _nested_max(values)
            for name, values in coordinate_errors.items()
        },
        "maximum_relative_orthogonal_component_error":
            _nested_max(orthogonal_errors),
        "coefficient_ranges": {
            name: [
                float(min(values)),
                float(max(values)),
            ]
            for name, values in coefficients.items()
        },
    }
    invariants["pass"] = bool(
        invariants["maximum_midpoint_absolute_error"]
        <= NUMERICAL_TOLERANCE
        and max(invariants[
            "maximum_axis_coordinate_difference"].values())
        <= NUMERICAL_TOLERANCE
        and invariants["maximum_relative_orthogonal_component_error"]
        <= NUMERICAL_TOLERANCE
    )
    return tuple(belief_patches), tuple(search_patches), invariants


def _natural_transplant_pair(
        belief_states, search_states, positions):
    belief_to_search = tuple(
        (list(positions), value[:, positions, :].clone())
        for value in search_states
    )
    search_to_belief = tuple(
        (list(positions), value[:, positions, :].clone())
        for value in belief_states
    )
    return belief_to_search, search_to_belief


@torch.no_grad()
def _capture_trajectory(
        model, batch, source_position, readout_position,
        sequence_patch=None, layers=TRAJECTORY_LAYERS, source_layer=21):
    """Capture post-block source/readout states, applying the L21 patch once."""
    blocks = get_decoder_layers(model)
    selected = tuple(sorted({int(layer) for layer in layers}))
    if int(source_layer) not in selected:
        raise ValueError("trajectory must include the intervention layer")
    cache = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _args, output):
            states, rebuild = _split_output(output)
            changed = False
            if int(layer) == int(source_layer) and sequence_patch is not None:
                positions, values = sequence_patch
                states = states.clone()
                states[:, positions, :] = values.to(
                    device=states.device, dtype=states.dtype)
                changed = True
            cache[int(layer)] = {
                "source": states[:, int(source_position), :]
                    .detach().float().cpu(),
                "readout": states[:, int(readout_position), :]
                    .detach().float().cpu(),
            }
            if changed:
                return rebuild(states)
        return hook

    for layer in selected:
        handles.append(
            blocks[layer].register_forward_hook(make_hook(layer)))
    try:
        model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in selected if layer not in cache]
    if missing:
        raise RuntimeError(f"trajectory layers missing: {missing}")
    return cache


def _trajectory_direction_metrics(original, opposite, intervened):
    """Movement toward the natural opposite-operation trajectory."""
    result = {}
    for history in ("clean", "natural"):
        result[history] = {}
        for layer in TRAJECTORY_LAYERS:
            start = original[history][layer]["readout"].double()
            target = opposite[history][layer]["readout"].double()
            value = intervened[history][layer]["readout"].double()
            direction = target - start
            movement = value - start
            norm2 = direction.square().sum(-1).clamp_min(EPS)
            progress = (movement * direction).sum(-1) / norm2
            distance_ratio = (
                (value - target).norm(dim=-1)
                / direction.norm(dim=-1).clamp_min(EPS)
            )
            source_change = (
                intervened[history][layer]["source"].double()
                - original[history][layer]["source"].double()
            ).norm(dim=-1)
            result[history][str(layer)] = {
                "mean_progress": float(progress.mean()),
                "median_progress": float(progress.median()),
                "positive_progress_fraction": float(
                    (progress > 0).double().mean()),
                "mean_distance_ratio": float(distance_ratio.mean()),
                "median_distance_ratio": float(distance_ratio.median()),
                "mean_source_change_norm": float(source_change.mean()),
                "maximum_source_change_norm": float(source_change.max()),
            }
    return result


def _trajectory_pair_metrics(original, intervened):
    """Remaining B-S separation after a paired equalization intervention."""
    result = {}
    for history in ("clean", "natural"):
        result[history] = {}
        for layer in TRAJECTORY_LAYERS:
            original_gap = (
                original["belief"][history][layer]["readout"].double()
                - original["search"][history][layer]["readout"].double()
            )
            remaining_gap = (
                intervened["belief"][history][layer]["readout"].double()
                - intervened["search"][history][layer]["readout"].double()
            )
            ratio = (
                remaining_gap.norm(dim=-1)
                / original_gap.norm(dim=-1).clamp_min(EPS)
            )
            result[history][str(layer)] = {
                "mean_separation_ratio": float(ratio.mean()),
                "median_separation_ratio": float(ratio.median()),
                "reduced_separation_fraction": float(
                    (ratio < 1.0).double().mean()),
            }
    return result


def _gap_statistics(original_tasks, original_curves,
                    arm_tasks, arm_curves):
    original_belief = _world_mediation(
        original_tasks["belief"], original_curves["belief"]["24"])
    original_search = _world_mediation(
        original_tasks["search"], original_curves["search"]["24"])
    arm_belief = _world_mediation(
        arm_tasks["belief"], arm_curves["belief"]["24"])
    arm_search = _world_mediation(
        arm_tasks["search"], arm_curves["search"]["24"])
    original_aggregate = float(
        original_curves["belief"]["24"]["mediation"]["minimum_fraction"]
        - original_curves["search"]["24"]["mediation"]["minimum_fraction"])
    remaining_aggregate = float(
        arm_curves["belief"]["24"]["mediation"]["minimum_fraction"]
        - arm_curves["search"]["24"]["mediation"]["minimum_fraction"])
    reduction = original_aggregate - remaining_aggregate
    rows = []
    reductions = []
    for index, (ob, os_, ab, ass) in enumerate(zip(
            original_belief, original_search, arm_belief, arm_search)):
        if any(value is None for value in (ob, os_, ab, ass)):
            rows.append({
                "world_offset": index,
                "gap_reduction": None,
                "predicted_sign": False,
            })
            continue
        original_gap = float(ob) - float(os_)
        arm_gap = float(ab) - float(ass)
        row_reduction = original_gap - arm_gap
        reductions.append(row_reduction)
        rows.append({
            "world_offset": index,
            "original_gap": original_gap,
            "remaining_gap": arm_gap,
            "gap_reduction": row_reduction,
            "predicted_sign": bool(row_reduction > 0.0),
        })
    successes = sum(value > 0.0 for value in reductions)
    valid = len(reductions) == TEST_N
    return {
        "original_gap": original_aggregate,
        "remaining_gap": remaining_aggregate,
        "gap_reduction": reduction,
        "gap_reduction_fraction": (
            reduction / original_aggregate
            if abs(original_aggregate) > EPS else None),
        "rows": rows,
        "valid_worlds": len(reductions),
        "positive_reductions": successes,
        "exact_one_sided_sign_p": (
            _sign_tail(successes, TEST_N) if valid else None),
        "statistical_pass": bool(
            valid
            and successes >= SIGN_MINIMUM
            and _sign_tail(successes, TEST_N) <= SIGN_P_MAX),
    }


def _interaction_statistics(projection, residual, joint):
    rows = []
    values = []
    for p_row, r_row, j_row in zip(
            projection["rows"], residual["rows"], joint["rows"]):
        parts = (
            p_row.get("gap_reduction"),
            r_row.get("gap_reduction"),
            j_row.get("gap_reduction"),
        )
        if any(value is None for value in parts):
            rows.append({
                "world_offset": p_row["world_offset"],
                "interaction": None,
            })
            continue
        value = float(parts[2] - parts[0] - parts[1])
        values.append(value)
        rows.append({
            "world_offset": p_row["world_offset"],
            "interaction": value,
        })
    aggregate = float(
        joint["gap_reduction"]
        - projection["gap_reduction"]
        - residual["gap_reduction"])
    original_gap = float(joint["original_gap"])
    sign = 1.0 if aggregate >= 0.0 else -1.0
    consistent = sum(sign * value > 0.0 for value in values)
    valid = len(values) == TEST_N
    sign_p = _sign_tail(consistent, TEST_N) if valid else None
    fraction = (
        aggregate / original_gap
        if abs(original_gap) > EPS else None)
    return {
        "definition": "joint_reduction - projection_reduction - residual_reduction",
        "aggregate": aggregate,
        "fraction_of_original_gap": fraction,
        "aggregate_sign": "synergistic" if aggregate >= 0.0 else "redundant",
        "consistent_worlds": consistent,
        "exact_one_sided_sign_p": sign_p,
        "rows": rows,
        "stable": bool(
            valid
            and fraction is not None
            and abs(fraction) >= MINIMUM_ABSOLUTE_INTERACTION_FRACTION
            and consistent >= INTERACTION_SIGN_MINIMUM
            and sign_p is not None
            and sign_p <= 0.01),
    }


@torch.no_grad()
def run_delta_endogenous_controller_factorial(
        model_path, out_dir,
        model_key="qwen7b_endogenous_controller_factorial",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TEST_N:
        raise ValueError("v1 is frozen to exactly 30 evaluation histories")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(TRAJECTORY_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("factorial trajectory layers are absent")
    n_heads, head_dim = _attention_geometry(model)

    spec = _carrier_spec()
    location_rows, location_indices = _compatible_world_rows(
        tok, torch.device("cpu"), 30)
    if len(location_rows) != 30:
        raise ValueError("all 30 compatible location worlds are required")
    ownership_rows = _fresh_domain_rows(
        DOMAIN_SPECS["ownership"]["values"])[:DONOR_N]
    color_donor_rows = _fresh_domain_rows(
        DOMAIN_SPECS["color_state"]["values"])[:DONOR_N]
    donor_alignments = {
        "location": _location_alignment(
            tok, dev, location_rows[:DONOR_N]),
        "ownership": _domain_alignment(
            tok, dev, ownership_rows, DOMAIN_SPECS["ownership"]),
        "color": _domain_alignment(
            tok, dev, color_donor_rows, DOMAIN_SPECS["color_state"]),
    }
    donor_controllers = {
        name: _controller_from_alignment(model, alignment)[0]
        for name, alignment in donor_alignments.items()
    }
    shared_pc1, _donor_mean, shared_geometry = _shared_controllers(
        donor_controllers["location"], donor_controllers["ownership"])
    projection, residual, decomposition = _color_decomposition(
        shared_pc1, donor_controllers["color"])

    controller_path = os.path.join(
        out_dir, f"factorial_controllers_{model_key}.npz")
    np.savez(
        controller_path,
        shared_pc1=shared_pc1.numpy(),
        color_projection=projection.numpy(),
        color_residual=residual.numpy())
    with open(controller_path, "rb") as handle:
        archive_sha = hashlib.sha256(handle.read()).hexdigest().upper()

    evaluation_rows = _fresh_color_rows_v4()
    alignment = _domain_alignment(tok, dev, evaluation_rows, spec)
    states = {"belief": [], "search": []}
    for belief_batch, search_batch in alignment["batches"]:
        states["belief"].append(_capture_full_l21(model, belief_batch))
        states["search"].append(_capture_full_l21(model, search_batch))
    positions = alignment["answer_positions"]

    r_belief, r_search, r_invariants = _equalization_patch_pair_axes(
        states["belief"], states["search"], positions, {"R": residual})
    p_belief, p_search, p_invariants = _equalization_patch_pair_axes(
        states["belief"], states["search"], positions, {"P": projection})
    pr_belief, pr_search, pr_invariants = _equalization_patch_pair_axes(
        states["belief"], states["search"], positions,
        {"P": projection, "R": residual})
    natural_belief, natural_search = _natural_transplant_pair(
        states["belief"], states["search"], positions)
    patches = {
        "original": {"belief": None, "search": None},
        "residual_additive_switch": {
            "belief": _fixed_patch(
                states["belief"], positions, residual, -1.0),
            "search": _fixed_patch(
                states["search"], positions, residual, +1.0),
        },
        "projection_equalization": {
            "belief": p_belief, "search": p_search},
        "residual_equalization": {
            "belief": r_belief, "search": r_search},
        "joint_equalization": {
            "belief": pr_belief, "search": pr_search},
        "natural_prefix_interchange": {
            "belief": natural_belief, "search": natural_search},
    }

    source = [row["source"] for row in evaluation_rows]
    target = [row["target"] for row in evaluation_rows]
    clean_belief, clean_search = alignment["batches"][0]
    natural_history_belief, natural_history_search = alignment["batches"][1]

    def operation_batches(operation):
        if operation == "belief":
            return clean_belief, natural_history_belief
        return clean_search, natural_history_search

    def build_context(operation, sequence_patch, layers=CAPTURE_LAYERS):
        clean, natural_history = operation_batches(operation)
        return _generic_task_context(
            model, clean, natural_history, list(spec["values"]),
            source, target, layers, head_dim,
            sequence_patch=sequence_patch)

    total_full = len(patches) * 2 * (3 + len(LAYERS))
    total_trajectory = len(patches) * 2 * 2
    hb = Heartbeat(
        total_full + total_trajectory,
        "endogenous_controller_factorial",
        every_sec=30, out_dir=out_dir)
    arm_results = {}
    trajectories = {}
    for arm_name, operation_patches in patches.items():
        arm_results[arm_name] = {
            "tasks": {}, "cumulative_prefix": {}, "summaries": {}}
        trajectories[arm_name] = {}
        for operation in ("belief", "search"):
            context = build_context(
                operation, operation_patches[operation])
            arm_results[arm_name]["tasks"][operation] = (
                _public_generic_task(context))
            for phase in ("baseline", "source", "base_path"):
                hb.step(extra=f"{arm_name}/{operation}/{phase}")
            curve = {}
            for layer, cell in _generic_curve(
                    model, context, n_heads, head_dim).items():
                curve[layer] = cell
                hb.step(extra=f"{arm_name}/{operation}/prefixL{layer}")
            arm_results[arm_name]["cumulative_prefix"][operation] = curve
            arm_results[arm_name]["summaries"][operation] = _summary(curve)

            trajectories[arm_name][operation] = {}
            clean, natural_history = operation_batches(operation)
            patch_pair = operation_patches[operation]
            for history_name, batch, patch in (
                    ("clean", clean,
                     patch_pair[0] if patch_pair is not None else None),
                    ("natural", natural_history,
                     patch_pair[1] if patch_pair is not None else None)):
                trajectories[arm_name][operation][history_name] = (
                    _capture_trajectory(
                        model, batch, context["source_position"],
                        context["readout_position"], patch))
                hb.step(extra=(
                    f"{arm_name}/{operation}/{history_name}/trajectory"))
    hb.done()

    original_tasks = arm_results["original"]["tasks"]
    original_curves = arm_results["original"]["cumulative_prefix"]
    original_summaries = arm_results["original"]["summaries"]
    original_summary = {
        "belief": original_summaries["belief"],
        "search": original_summaries["search"],
    }
    original_movement = _movement(
        original_summary,
        original_summary["belief"]["l24_minimum_mediation"],
        original_summary["search"]["l24_minimum_mediation"])
    original_gate = _original_gate(
        {
            "belief_original": original_tasks["belief"],
            "search_original": original_tasks["search"],
        },
        {
            "belief_original": original_summaries["belief"],
            "search_original": original_summaries["search"],
        },
        original_movement)

    for arm_name in (
            "residual_additive_switch",
            "natural_prefix_interchange"):
        arm = arm_results[arm_name]
        movement = _movement(
            original_summary,
            arm["summaries"]["belief"]["l24_minimum_mediation"],
            arm["summaries"]["search"]["l24_minimum_mediation"])
        world = _statistical_world_movement(
            _world_mediation(
                original_tasks["belief"], original_curves["belief"]["24"]),
            _world_mediation(
                original_tasks["search"], original_curves["search"]["24"]),
            _world_mediation(
                arm["tasks"]["belief"],
                arm["cumulative_prefix"]["belief"]["24"]),
            _world_mediation(
                arm["tasks"]["search"],
                arm["cumulative_prefix"]["search"]["24"]))
        arm["movement"] = movement
        arm["per_world_movement"] = world
        arm["trajectory_toward_opposite"] = {
            "belief_to_search": _trajectory_direction_metrics(
                trajectories["original"]["belief"],
                trajectories["original"]["search"],
                trajectories[arm_name]["belief"]),
            "search_to_belief": _trajectory_direction_metrics(
                trajectories["original"]["search"],
                trajectories["original"]["belief"],
                trajectories[arm_name]["search"]),
        }

    for arm_name in (
            "projection_equalization",
            "residual_equalization",
            "joint_equalization"):
        arm = arm_results[arm_name]
        arm["gap_statistics"] = _gap_statistics(
            original_tasks, original_curves,
            arm["tasks"], arm["cumulative_prefix"])
        arm["trajectory_separation"] = _trajectory_pair_metrics(
            trajectories["original"], trajectories[arm_name])

    interaction = _interaction_statistics(
        arm_results["projection_equalization"]["gap_statistics"],
        arm_results["residual_equalization"]["gap_statistics"],
        arm_results["joint_equalization"]["gap_statistics"])

    residual_switch = arm_results["residual_additive_switch"]
    natural_switch = arm_results["natural_prefix_interchange"]
    naturalized = {}
    for direction in (
            "belief_to_search_movement", "search_to_belief_movement"):
        denominator = natural_switch["movement"][direction]
        naturalized[direction] = (
            residual_switch["movement"][direction] / denominator
            if denominator > EPS else None)
    naturalized["minimum_ratio"] = min(
        value for value in naturalized.values()
        if value is not None)

    # Lightweight sanity controls. Statistical null distributions were already
    # frozen and passed in the independent residual-only confirmation.
    random_direction = _orthogonal_norm_matched_residuals(
        shared_pc1, residual, n_random=1,
        seed=CONTROL_DIRECTION_SEED)[0]
    random_positions = _random_position_sets(
        alignment["random_candidates"], n_random=1,
        seed=CONTROL_POSITION_SEED)[0]
    control_specs = {
        "instruction_position": (
            residual, alignment["instruction_positions"]),
        "orthogonal_direction": (
            random_direction, alignment["answer_positions"]),
        "matched_random_position": (
            residual, random_positions),
    }
    control_hb = Heartbeat(
        len(control_specs) * 2 * 4,
        "endogenous_controller_factorial_controls",
        every_sec=30, out_dir=out_dir)
    controls = {}
    l24_sites = _full_sites((22, 23, 24), n_heads)
    for name, (direction, control_positions) in control_specs.items():
        belief_patch, search_patch, invariants = (
            _equalization_patch_pair_axes(
                states["belief"], states["search"],
                control_positions, {"axis": direction}))
        tasks = {}
        curves = {}
        for operation, patch in (
                ("belief", belief_patch), ("search", search_patch)):
            context = build_context(
                operation, patch, layers=(21, 22, 23, 24))
            tasks[operation] = _public_generic_task(context)
            for phase in ("baseline", "source", "base_path"):
                control_hb.step(extra=f"{name}/{operation}/{phase}")
            curves[operation] = {
                "24": _generic_evaluate_sites(
                    model, context, l24_sites, head_dim)}
            control_hb.step(extra=f"{name}/{operation}/L24")
        controls[name] = {
            "positions": list(control_positions),
            "invariants": invariants,
            "functional": bool(all(
                task["eligible"]
                and task["source_intervention"]["sufficient"]
                for task in tasks.values())),
            "gap_statistics": _gap_statistics(
                original_tasks, original_curves, tasks, curves),
        }
    control_hb.done()

    all_tasks = [
        task
        for arm in arm_results.values()
        for task in arm["tasks"].values()
    ]
    functional = bool(all(
        task["eligible"] and task["source_intervention"]["sufficient"]
        for task in all_tasks))
    residual_necessity = arm_results[
        "residual_equalization"]["gap_statistics"]
    residual_sufficiency = bool(
        naturalized["minimum_ratio"]
        >= MINIMUM_NATURALIZED_SUFFICIENCY
        and residual_switch[
            "per_world_movement"]["statistical_uniformity_pass"])
    necessity_pass = bool(
        residual_necessity["gap_reduction_fraction"] is not None
        and residual_necessity["gap_reduction_fraction"]
        >= MINIMUM_NECESSITY_FRACTION
        and residual_necessity["statistical_pass"])

    l24 = "24"
    residual_trajectory = residual_switch["trajectory_toward_opposite"]
    trajectory_convergence = bool(
        all(
            residual_trajectory[direction][history][l24][
                "median_progress"] > 0.0
            and residual_trajectory[direction][history][l24][
                "positive_progress_fraction"] >= 0.90
            and residual_trajectory[direction][history][l24][
                "median_distance_ratio"]
            < MAXIMUM_TRAJECTORY_DISTANCE_RATIO
            for direction in ("belief_to_search", "search_to_belief")
            for history in ("clean", "natural")
        )
        and all(
            arm_results["residual_equalization"][
                "trajectory_separation"][history][l24][
                    "median_separation_ratio"]
            < MAXIMUM_EQUALIZED_SEPARATION_RATIO
            and arm_results["residual_equalization"][
                "trajectory_separation"][history][l24][
                    "reduced_separation_fraction"] >= 0.90
            for history in ("clean", "natural")
        )
    )
    source_invariance_diagnostics = {}
    for arm_name in patches:
        if arm_name == "original":
            continue
        source_invariance_diagnostics[arm_name] = {}
        for operation in ("belief", "search"):
            source_invariance_diagnostics[arm_name][operation] = {}
            for history in ("clean", "natural"):
                difference = (
                    trajectories[arm_name][operation][history][21]["source"]
                    - trajectories["original"][operation][history][21][
                        "source"]
                ).double().norm(dim=-1)
                source_invariance_diagnostics[arm_name][operation][history] = {
                    "mean_change_norm": float(difference.mean()),
                    "maximum_change_norm": float(difference.max()),
                }
    source_invariance = bool(all(
        cell["maximum_change_norm"] <= NUMERICAL_TOLERANCE
        for arm in source_invariance_diagnostics.values()
        for operation in arm.values()
        for cell in operation.values()
    ))
    control_specificity = bool(all(
        cell["functional"]
        and cell["gap_statistics"]["gap_reduction_fraction"]
            < residual_necessity["gap_reduction_fraction"]
        for cell in controls.values()
    ))

    if original_gate != "ELIGIBLE":
        verdict = original_gate
    elif not functional:
        verdict = "INTERVENTION_BEHAVIOR_OR_SOURCE_INELIGIBLE"
    elif not residual_sufficiency:
        verdict = "RESIDUAL_SUFFICIENCY_NOT_NATURALIZED"
    elif not necessity_pass:
        verdict = "ENDOGENOUS_RESIDUAL_NOT_NECESSARY"
    elif not trajectory_convergence:
        verdict = "ROUTE_CHANGE_WITHOUT_NATURAL_TRAJECTORY_CONVERGENCE"
    elif not source_invariance:
        verdict = "SOURCE_STATE_NOT_INVARIANT"
    elif not control_specificity:
        verdict = "ENDOGENOUS_EFFECT_NONSPECIFIC"
    elif not interaction["stable"]:
        verdict = "FAITHFUL_RESIDUAL_WITHOUT_STABLE_PXR_INTERACTION"
    else:
        verdict = "ENDOGENOUS_CONTROLLER_INTERACTION"

    result = {
        "stage": "delta_endogenous_controller_factorial",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "A low-energy answer-prefix coordinate is both sufficient and "
            "endogenously necessary for route selection, follows the natural "
            "opposite-command trajectory, and interacts reproducibly with "
            "the dominant shared controller geometry."),
        "carrier_template": {
            "sentence": CARRIER_SENTENCE,
            "evaluation_rows": evaluation_rows,
            "template_absent_from_prior_evaluations": True,
            "histories_absent_from_prior_evaluations": True,
        },
        "splits": {
            "location_donor_indices": location_indices[:DONOR_N],
            "ownership_donor_rows": ownership_rows,
            "color_donor_rows": color_donor_rows,
            "evaluation_excluded_from_construction": True,
        },
        "construction": {
            "shared_geometry": shared_geometry,
            "decomposition": decomposition,
        },
        "controllers": {
            "shared_pc1": _controller_metadata(shared_pc1),
            "color_projection": _controller_metadata(projection),
            "color_residual": _controller_metadata(residual),
        },
        "controller_archive": {
            "artifact": os.path.basename(controller_path),
            "sha256": archive_sha,
        },
        "patch_invariants": {
            "projection_equalization": p_invariants,
            "residual_equalization": r_invariants,
            "joint_equalization": pr_invariants,
        },
        "original_gate": original_gate,
        "naturalized_residual_sufficiency": naturalized,
        "interaction": interaction,
        "source_state_invariance": source_invariance_diagnostics,
        "controls": controls,
        "decision_components": {
            "functional": functional,
            "residual_sufficiency": residual_sufficiency,
            "residual_necessity": necessity_pass,
            "trajectory_convergence": trajectory_convergence,
            "source_invariance": source_invariance,
            "control_specificity": control_specificity,
            "stable_pxr_interaction": interaction["stable"],
        },
        "arms": arm_results,
        "layers": list(LAYERS),
        "trajectory_layers": list(TRAJECTORY_LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_endogenous_controller_factorial_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"ENDOGENOUS-FACTORIAL verdict={verdict} gate={original_gate} "
        f"R_suff={residual_sufficiency} "
        f"R_nec={residual_necessity['gap_reduction_fraction']:+.3f} "
        f"PxR={interaction['fraction_of_original_gap']:+.3f} "
        f"traj={trajectory_convergence} specific={control_specificity} "
        f"artifact={path}")
    return result
