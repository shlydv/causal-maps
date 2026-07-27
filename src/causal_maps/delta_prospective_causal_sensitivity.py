"""Leave-one-domain-out prediction of causal answer-prefix coordinates."""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import torch

from .delta_anchor_write import _anchor_position, _resolve
from .delta_causal_rank_spectrum import (
    _component,
    _controller_basis,
    _natural_patch_pair,
    _positive_worlds,
)
from .delta_content_cancelled_controller import (
    _fixed_patch,
    _functional_score,
    _movement,
    _norm_matched_directions,
    _random_position_sets,
    _world_mediation,
    _world_movements,
)
from .delta_controller_matrix import (
    _controller_from_alignment,
    _evaluation_states,
)
from .delta_cross_domain_controller import (
    _domain_alignment,
    _generic_accuracy,
    _generic_cell,
    _generic_evaluate_sites,
    _public_generic_task,
)
from .delta_endogenous_controller_factorial import _carrier_spec
from .delta_operation_handoff_depth import _full_sites
from .delta_sparse_transport import _attention_geometry, _o_proj
from .delta_source_head_mediation import (
    _capture_source_heads,
    _run_intervention,
)
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output


PROTOCOL_VERSION = "2026-07-26-p2-prospective-causal-sensitivity-v1"
DOMAINS = (
    "color_state",
    "material_state",
    "animal_state",
    "shape_state",
)
DONOR_N = 15
CALIBRATION_N = 5
TEST_N = 15
CONFIRMATION_N = 15
TOTAL_N = DONOR_N + CALIBRATION_N + TEST_N + CONFIRMATION_N
SOURCE_LAYER = 21
ROUTE_LAYER = 24
CONTROL_LAYERS = (21, 22, 23, 24)
N_RANDOM = 3
RANDOM_SEED = 94001
MINIMUM_ORIGINAL_GAP = 0.03
MINIMUM_REFERENCE_SCORE = 0.025
MINIMUM_PREDICTED_SCORE = 0.02
MINIMUM_WORLD_FRACTION = 0.80
MINIMUM_RECOVERY_OF_BEST = 0.75
MINIMUM_POOLED_SPEARMAN = 0.50
MINIMUM_LOW_ENERGY_MARGIN = 0.005


NOVEL_DOMAIN_SPECS = {
    "material_state": {
        "values": (
            "wood", "steel", "glass", "stone",
            "paper", "cloth", "clay", "gold",
        ),
        "rule": (
            "A technician's private material record is the last material "
            "that technician personally inspected. Replacements made while "
            "the technician is absent do not update the private record."
        ),
        "story": (
            "Rina personally inspected the primary sample made of {state}, "
            "then left. While Rina was absent, the primary sample was "
            "replaced by {d1}. Separately, the backup sample was made of "
            "{d2}."
        ),
        "question": (
            "According to Rina's own information, what material does she "
            "believe the primary sample is made of?"
        ),
        "answer_description": "material word",
    },
    "animal_state": {
        "values": (
            "cat", "dog", "bird", "fish",
            "horse", "lion", "bear", "wolf",
        ),
        "rule": (
            "A ranger's private animal record is the last animal that ranger "
            "personally observed. Later substitutions made while the ranger "
            "is absent do not update the private record."
        ),
        "story": (
            "Rina personally observed the primary enclosure containing a "
            "{state}, then left. While Rina was absent, that animal was "
            "replaced by a {d1}. Separately, the backup enclosure contained "
            "a {d2}."
        ),
        "question": (
            "According to Rina's own information, what animal does she "
            "believe is in the primary enclosure?"
        ),
        "answer_description": "animal word",
    },
    "shape_state": {
        "values": (
            "circle", "square", "triangle", "oval",
            "star", "cube", "cone", "sphere",
        ),
        "rule": (
            "A designer's private shape record is the last shape that "
            "designer personally observed. Changes made while the designer "
            "is absent do not update the private record."
        ),
        "story": (
            "Rina personally observed the primary display showing a "
            "{state}, then left. While Rina was absent, the display changed "
            "to a {d1}. Separately, the backup display showed a {d2}."
        ),
        "question": (
            "According to Rina's own information, what shape does she "
            "believe the primary display shows?"
        ),
        "answer_description": "shape word",
    },
}


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "hypothesis": (
        "Donor-domain natural command contrasts span a compact causal "
        "control space, and a held-out domain's active donor coordinate is "
        "predicted by its downstream Jacobian rather than SVD energy."),
    "domains": list(DOMAINS),
    "domain_specs": {
        "color_state": _carrier_spec(),
        **NOVEL_DOMAIN_SPECS,
    },
    "splits": {
        "donor_n": DONOR_N,
        "non_interventional_calibration_n": CALIBRATION_N,
        "prospective_core_holdout_n": TEST_N,
        "reserved_random_null_confirmation_n": CONFIRMATION_N,
        "row_construction": (
            "First 50 cyclic directed source-target pairs, with the first "
            "lexicographic pair of distractors giving globally unique "
            "clean/counterfactual history signatures."),
    },
    "frozen_layers": {
        "source_and_prefix": SOURCE_LAYER,
        "route_readout": ROUTE_LAYER,
        "route_mediation": list(CONTROL_LAYERS),
    },
    "basis": (
        "Per-fold uncentered SVD of content-cancelled BELIEF-minus-SEARCH "
        "controllers from all other frozen domains only."),
    "selection": (
        "For target projection v_k=<D_t,u_k>u_k, score_k is the minimum of "
        "the BELIEF and SEARCH directional derivatives of a frozen layer-24 "
        "route readout along v_k. No target activation is changed."),
    "causal_arms": [
        "zero/original",
        "exact natural answer-prefix interchange",
        "every donor-basis target projection",
        "full donor-span reconstruction",
        "norm-matched raw donor mean",
        "instruction-position predicted-axis control",
        "matched identical-token-position predicted-axis control",
        f"{N_RANDOM} norm-matched random smoke controls",
    ],
    "random_seed": RANDOM_SEED,
    "decision": {
        "minimum_original_gap": MINIMUM_ORIGINAL_GAP,
        "minimum_reference_score": MINIMUM_REFERENCE_SCORE,
        "minimum_predicted_score": MINIMUM_PREDICTED_SCORE,
        "minimum_world_fraction": MINIMUM_WORLD_FRACTION,
        "minimum_recovery_of_best_axis": MINIMUM_RECOVERY_OF_BEST,
        "minimum_pooled_spearman": MINIMUM_POOLED_SPEARMAN,
        "minimum_low_energy_margin": MINIMUM_LOW_ENERGY_MARGIN,
    },
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _domain_specs():
    return {
        "color_state": _carrier_spec(),
        **NOVEL_DOMAIN_SPECS,
    }


def _prospective_rows(values, n_rows=TOTAL_N):
    """Frozen unique directed source-target histories for every domain."""
    if len(values) != 8 or int(n_rows) != TOTAL_N:
        raise ValueError("v1 requires eight values and exactly 50 rows")
    pairs = [
        (source_index, target_index)
        for shift in range(1, len(values))
        for source_index in range(len(values))
        for target_index in [(source_index + shift) % len(values)]
    ][:int(n_rows)]
    rows = []
    used_signatures = set()
    for index, (source_index, target_index) in enumerate(pairs):
        source = values[source_index]
        target = values[target_index]
        remaining = [
            value for value in values if value not in (source, target)
        ]
        distractors = next(
            (
                (d1, d2)
                for d1 in remaining
                for d2 in remaining
                if d1 != d2
                and (source, d1, d2) not in used_signatures
                and (target, d1, d2) not in used_signatures
            ),
            None)
        if distractors is None:
            raise AssertionError(
                f"no unique distractors remain for row {index}")
        d1, d2 = distractors
        used_signatures.update({
            (source, d1, d2),
            (target, d1, d2),
        })
        rows.append({
            "row_index": index,
            "source": source,
            "target": target,
            "state": source,
            "d1": d1,
            "d2": d2,
        })
    if len({(row["source"], row["target"]) for row in rows}) != n_rows:
        raise AssertionError("source-target pairs are not unique")
    signatures = {
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }
    if len(signatures) != 2 * n_rows:
        raise AssertionError("clean/natural histories are not unique")
    return rows


def _capture_route_states(model, batch, readout_position):
    cache = {}

    def hook(_module, args):
        cache["state"] = (
            args[0][:, int(readout_position), :].detach().float().cpu())

    handle = _o_proj(model, ROUTE_LAYER).register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(
                input_ids=batch["ids"], attention_mask=batch["am"],
                use_cache=False)
    finally:
        handle.remove()
    if "state" not in cache:
        raise RuntimeError("layer-24 route state was not captured")
    return cache["state"]


def _route_readout(model, alignment):
    """Natural target-domain BELIEF-minus-SEARCH layer-24 readout."""
    differences = []
    for belief, search in alignment["batches"]:
        belief_state = _capture_route_states(
            model, belief, alignment["readout"])
        search_state = _capture_route_states(
            model, search, alignment["readout"])
        differences.extend(belief_state - search_state)
    direction = torch.stack(differences).mean(dim=0)
    norm = float(direction.norm())
    if norm <= 1e-8:
        raise ValueError("target route readout has zero norm")
    return direction / norm, {
        "norm_before_unit_normalization": norm,
        "n_calibration_states": len(differences),
    }


def _prefix_route_gradient(
        model, batch, positions, readout_position, route_direction):
    """Gradient of a fixed route readout at the unchanged prefix state."""
    cache = {}
    blocks = get_decoder_layers(model)

    def source_hook(_module, _args, output):
        states, rebuild = _split_output(output)
        leaf = (
            states[:, positions, :].detach().clone().requires_grad_(True))
        unchanged = states.detach().clone()
        unchanged[:, positions, :] = leaf
        cache["leaf"] = leaf
        return rebuild(unchanged)

    def route_hook(_module, args):
        cache["route"] = args[0][:, int(readout_position), :]

    handles = [
        blocks[SOURCE_LAYER].register_forward_hook(source_hook),
        _o_proj(model, ROUTE_LAYER).register_forward_pre_hook(route_hook),
    ]
    try:
        with torch.enable_grad():
            model(
                input_ids=batch["ids"], attention_mask=batch["am"],
                use_cache=False)
            direction = route_direction.to(
                device=cache["route"].device,
                dtype=cache["route"].dtype)
            scalar = (
                cache["route"] * direction.unsqueeze(0)
            ).sum(dim=-1).mean()
            gradient = torch.autograd.grad(
                scalar, cache["leaf"], retain_graph=False,
                create_graph=False)[0]
    finally:
        for handle in handles:
            handle.remove()
    return gradient.detach().float().cpu(), float(scalar.detach().cpu())


def _target_gradients(model, alignment, route_direction):
    by_operation = {"belief": [], "search": []}
    readout_values = {"belief": [], "search": []}
    for belief, search in alignment["batches"]:
        for operation, batch in (
                ("belief", belief), ("search", search)):
            gradient, readout = _prefix_route_gradient(
                model, batch, alignment["answer_positions"],
                alignment["readout"], route_direction)
            by_operation[operation].append(gradient)
            readout_values[operation].append(readout)
    return {
        operation: torch.stack(values).mean(dim=0)
        for operation, values in by_operation.items()
    }, readout_values


def _axis_prediction(components, gradients):
    """Freeze axis order from non-interventional directional derivatives."""
    rows = []
    for axis, component in components.items():
        belief = float((gradients["belief"] * component).sum())
        search = float((gradients["search"] * component).sum())
        rows.append({
            "axis": int(axis),
            "belief_to_search_derivative": belief,
            "search_to_belief_derivative": search,
            "bidirectional_sensitivity": min(belief, search),
            "predicted_belief_to_search_sign": int(np.sign(belief)),
            "predicted_search_to_belief_sign": int(np.sign(search)),
            "coherent_positive": bool(belief > 0.0 and search > 0.0),
        })
    rows.sort(
        key=lambda row: (
            row["bidirectional_sensitivity"],
            -row["axis"]),
        reverse=True)
    return rows


def _scale_per_position(value, reference):
    scaled = value.detach().float().clone()
    source_norm = scaled.norm(dim=-1, keepdim=True)
    target_norm = reference.detach().float().norm(
        dim=-1, keepdim=True)
    if bool((source_norm <= 1e-8).any()):
        raise ValueError("raw donor mean contains a zero-norm position")
    return scaled / source_norm * target_norm


def _rankdata(values):
    """Average ranks for ties; smallest value receives rank one."""
    values = [float(value) for value in values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while (stop < len(order)
               and values[order[stop]] == values[order[start]]):
            stop += 1
        rank = 0.5 * (start + 1 + stop)
        for offset in range(start, stop):
            ranks[order[offset]] = rank
        start = stop
    return ranks


def _correlation(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if float(a.std()) <= 1e-12 or float(b.std()) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(left, right):
    return _correlation(_rankdata(left), _rankdata(right))


def _probe_accessibility(samples, basis):
    flat = samples.reshape(samples.shape[0], -1).float()
    projected = flat @ basis.T
    mean = projected.mean(dim=0).abs()
    std = projected.std(dim=0, unbiased=False).clamp_min(1e-8)
    return (mean / std).tolist()


def _patch_pair(states, positions, displacement):
    return {
        "belief": _fixed_patch(
            states["belief"], positions, displacement, -1.0),
        "search": _fixed_patch(
            states["search"], positions, displacement, +1.0),
    }


def _score_arm(arm, original):
    original_summary = {
        operation: {
            "l24_minimum_mediation": original["values"][operation]}
        for operation in ("belief", "search")
    }
    movement = _movement(
        original_summary,
        arm["values"]["belief"], arm["values"]["search"])
    world = _world_movements(
        _world_mediation(
            original["tasks"]["belief"],
            original["cells"]["belief"]),
        _world_mediation(
            original["tasks"]["search"],
            original["cells"]["search"]),
        _world_mediation(
            arm["tasks"]["belief"], arm["cells"]["belief"]),
        _world_mediation(
            arm["tasks"]["search"], arm["cells"]["search"]))
    functional = _functional_score(
        movement,
        [
            *original["tasks"].values(),
            *arm["tasks"].values(),
        ])
    return {
        "movement": movement,
        "per_world_movement": world,
        "positive_worlds": _positive_worlds(world),
        "functional": functional["functional"],
        "functional_bidirectional_score": (
            functional["functional_bidirectional_score"]),
    }


def _fold_adjudication(
        target, fold, prediction_rows, baseline_axes):
    original = fold["arms"]["original"]
    original_tasks = list(original["tasks"].values())
    eligible = bool(
        all(
            task["eligible"]
            and task["source_intervention"]["sufficient"]
            for task in original_tasks)
        and (
            original["values"]["belief"]
            - original["values"]["search"]
        ) >= MINIMUM_ORIGINAL_GAP - 1e-9)
    natural = fold["arms"]["natural_prefix_interchange"]
    reference = bool(
        natural["functional"]
        and natural["functional_bidirectional_score"]
        >= MINIMUM_REFERENCE_SCORE - 1e-9)

    selected_axis = int(prediction_rows[0]["axis"])
    predicted = fold["arms"][f"axis_{selected_axis:02d}"]
    axis_scores = {
        int(row["axis"]): fold["arms"][
            f"axis_{int(row['axis']):02d}"][
                "functional_bidirectional_score"]
        for row in prediction_rows
    }
    valid_axis_scores = [
        score for score in axis_scores.values()
        if score > -1e8
    ]
    best_score = max(valid_axis_scores) if valid_axis_scores else -1e9
    recovery = (
        predicted["functional_bidirectional_score"] / best_score
        if best_score > 1e-8 else None)
    world_minimum = int(math.ceil(
        MINIMUM_WORLD_FRACTION * int(target["n_world"])))

    random_arms = [
        fold["arms"][f"random_direction_{index:02d}"]
        for index in range(N_RANDOM)
    ]
    valid_random = [
        arm["functional_bidirectional_score"]
        for arm in random_arms if arm["functional"]
    ]
    exceed = sum(
        value >= predicted["functional_bidirectional_score"]
        for value in valid_random)
    empirical_p = (
        (1.0 + exceed) / (1.0 + len(valid_random))
        if valid_random else None)
    random_smoke_pass = bool(
        len(valid_random) == N_RANDOM and exceed == 0)

    instruction = fold["arms"]["instruction_position_control"]
    random_position = fold["arms"]["random_position_control"]
    control_ceiling = max(
        0.01,
        0.5 * predicted["functional_bidirectional_score"])
    locus_specific = bool(
        instruction["functional"]
        and random_position["functional"]
        and instruction["functional_bidirectional_score"] < control_ceiling
        and random_position[
            "functional_bidirectional_score"] < control_ceiling)

    valid_axes = [
        axis for axis in sorted(axis_scores)
        if axis_scores[axis] > -1e8
    ]
    prediction_by_axis = {
        int(row["axis"]): row for row in prediction_rows}
    sensitivity_values = [
        float(prediction_by_axis[axis]["bidirectional_sensitivity"])
        for axis in valid_axes
    ]
    actual_values = [
        float(axis_scores[axis]) for axis in valid_axes
    ]
    rank_correlation = _spearman(
        sensitivity_values, actual_values)
    pearson = _correlation(sensitivity_values, actual_values)

    energy_axis = int(baseline_axes["svd_energy"])
    energy_score = axis_scores[energy_axis]
    low_energy_win = bool(
        selected_axis != energy_axis
        and predicted["functional_bidirectional_score"]
        >= energy_score + MINIMUM_LOW_ENERGY_MARGIN - 1e-9)
    variance_competition_pass = bool(
        selected_axis == energy_axis or low_energy_win)
    predicted_pass = bool(
        predicted["functional"]
        and predicted["functional_bidirectional_score"]
        >= MINIMUM_PREDICTED_SCORE - 1e-9
        and predicted["movement"]["belief_to_search_movement"] > 0.0
        and predicted["movement"]["search_to_belief_movement"] > 0.0
        and predicted["positive_worlds"] >= world_minimum
        and recovery is not None
        and recovery >= MINIMUM_RECOVERY_OF_BEST - 1e-9
        and predicted["source_state_max_abs_change"] <= 1e-8)
    passed = bool(
        eligible and reference and predicted_pass
        and random_smoke_pass and locus_specific
        and variance_competition_pass)
    return {
        "eligible": eligible,
        "reference_pass": reference,
        "selected_axis": selected_axis,
        "selected_axis_energy_rank": selected_axis,
        "predicted_axis_pass": predicted_pass,
        "predicted_axis_score":
            predicted["functional_bidirectional_score"],
        "best_axis_score": best_score,
        "recovery_of_best_axis": recovery,
        "positive_worlds": predicted["positive_worlds"],
        "valid_random_controls": len(valid_random),
        "random_exceedances": exceed,
        "random_empirical_p": empirical_p,
        "random_smoke_pass": random_smoke_pass,
        "random_null_confirmation_required": True,
        "locus_specific": locus_specific,
        "axis_sensitivity_actual_spearman": rank_correlation,
        "axis_sensitivity_actual_pearson": pearson,
        "baseline_axes": baseline_axes,
        "baseline_actual_scores": {
            name: axis_scores[int(axis)]
            for name, axis in baseline_axes.items()
        },
        "low_energy_win_over_pc1": low_energy_win,
        "variance_competition_pass": variance_competition_pass,
        "fold_pass": passed,
    }


def _lean_generic_context(
        model, clean, natural, values, source, target,
        layers, head_dim, sequence_patch=None):
    """Eligibility/source context without the unused top-8 blockade pass."""
    if clean["ids"].shape != natural["ids"].shape:
        raise ValueError("unaligned generic clean/natural batch")
    source_position = _anchor_position(clean, natural)
    readout_position = int(clean["ids"].shape[1] - 1)
    clean_patch, natural_patch = (
        sequence_patch if sequence_patch is not None
        else (None, None))
    clean_logits, clean_source, clean_heads = _capture_source_heads(
        model, clean["ids"], clean["am"],
        source_position, readout_position, layers,
        sequence_patch=clean_patch, source_layer=SOURCE_LAYER)
    natural_logits, natural_source, natural_heads = _capture_source_heads(
        model, natural["ids"], natural["am"],
        source_position, readout_position, layers,
        sequence_patch=natural_patch, source_layer=SOURCE_LAYER)
    eligible = bool(min(
        _generic_accuracy(clean_logits, clean, source, values),
        _generic_accuracy(natural_logits, natural, target, values),
    ) >= 0.80)
    forward = _run_intervention(
        model, clean["ids"], clean["am"],
        source_position, natural_source, readout_position,
        (), (), head_dim, sequence_patch=clean_patch,
        source_layer=SOURCE_LAYER)
    reverse = _run_intervention(
        model, natural["ids"], natural["am"],
        source_position, clean_source, readout_position,
        (), (), head_dim, sequence_patch=natural_patch,
        source_layer=SOURCE_LAYER)
    source_cell = _generic_cell(
        clean_logits, natural_logits, forward, reverse,
        clean, source, target, values)
    return {
        "clean": clean,
        "natural": natural,
        "values": list(values),
        "source": list(source),
        "target": list(target),
        "source_position": source_position,
        "source_layer": SOURCE_LAYER,
        "readout_position": readout_position,
        "clean_sequence_patch": clean_patch,
        "natural_sequence_patch": natural_patch,
        "clean_logits": clean_logits,
        "natural_logits": natural_logits,
        "clean_source": clean_source,
        "natural_source": natural_source,
        "clean_heads": clean_heads,
        "natural_heads": natural_heads,
        "eligible": eligible,
        "g0_clean": float(_generic_accuracy(
            clean_logits, clean, source, values)),
        "g0_natural": float(_generic_accuracy(
            natural_logits, natural, target, values)),
        "source_intervention": source_cell,
    }


@torch.no_grad()
def _evaluate_arm(
        model, target, patches, l24_sites, head_dim, heartbeat, arm_name):
    tasks = {}
    cells = {}
    values = {}
    private_contexts = {}
    for operation in ("belief", "search"):
        batch_index = 0 if operation == "belief" else 1
        clean = target["alignment"]["batches"][0][batch_index]
        natural = target["alignment"]["batches"][1][batch_index]
        context = _lean_generic_context(
            model, clean, natural,
            list(target["spec"]["values"]),
            target["source"], target["target"],
            CONTROL_LAYERS, head_dim,
            sequence_patch=(
                None if patches is None else patches[operation]))
        private_contexts[operation] = context
        tasks[operation] = _public_generic_task(context)
        for phase in ("baseline", "source", "base_path"):
            heartbeat.step(
                extra=f"{target['name']}/{arm_name}/{operation}/{phase}")
        cells[operation] = _generic_evaluate_sites(
            model, context, l24_sites, head_dim)
        values[operation] = float(
            cells[operation]["mediation"]["minimum_fraction"])
        heartbeat.step(
            extra=f"{target['name']}/{arm_name}/{operation}/L24")
    return {
        "tasks": tasks,
        "cells": cells,
        "values": values,
        "_private_contexts": private_contexts,
    }


def run_delta_prospective_causal_sensitivity(
        model_path, out_dir,
        model_key="qwen7b_prospective_causal_sensitivity",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=50):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TOTAL_N:
        raise ValueError("v1 is frozen to exactly 50 worlds per domain")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if ROUTE_LAYER >= model_num_hidden_layers(model):
        raise ValueError("frozen route layer is absent")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    n_heads, head_dim = _attention_geometry(model)
    l24_sites = _full_sites((22, 23, 24), n_heads)

    specs = _domain_specs()
    prepared = {}
    alignment_errors = {}
    for name in DOMAINS:
        spec = specs[name]
        rows = _prospective_rows(spec["values"])
        split_rows = {
            "donor": rows[:DONOR_N],
            "calibration": rows[
                DONOR_N:DONOR_N + CALIBRATION_N],
            "test": rows[
                DONOR_N + CALIBRATION_N:
                DONOR_N + CALIBRATION_N + TEST_N],
            "confirmation": rows[
                DONOR_N + CALIBRATION_N + TEST_N:],
        }
        try:
            alignments = {
                split: _domain_alignment(tok, dev, values, spec)
                for split, values in split_rows.items()
            }
        except (AssertionError, ValueError) as exc:
            alignment_errors[name] = str(exc)
            continue
        positions = {
            tuple(alignment["answer_positions"])
            for alignment in alignments.values()
        }
        if len(positions) != 1:
            alignment_errors[name] = (
                "answer-prefix positions differ across splits")
            continue
        controller, samples = _controller_from_alignment(
            model, alignments["donor"])
        prepared[name] = {
            "name": name,
            "spec": spec,
            "rows": split_rows,
            "alignments": alignments,
            "controller": controller,
            "samples": samples,
        }
    if len(prepared) < 4:
        result = {
            "stage": "delta_prospective_causal_sensitivity",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "alignment_errors": alignment_errors,
            "prepared_domains": list(prepared),
            "verdict": "TOKENIZATION_OR_ALIGNMENT_INELIGIBLE",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_prospective_causal_sensitivity_"
            f"{model_key}.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2, default=float)
        return result

    # Construct every leave-one-domain-out basis and every target prediction.
    # No target activation is changed in this phase.
    folds = {}
    predictions_public = {}
    for target_name, target in prepared.items():
        donors = {
            name: data["controller"]
            for name, data in prepared.items()
            if name != target_name
        }
        donor_names, singular, basis, coefficients, geometry = (
            _controller_basis(donors))
        components = {}
        component_metadata = {}
        for axis in range(1, len(donors) + 1):
            value, metadata = _component(
                target["controller"], basis, [axis])
            components[axis] = value
            component_metadata[str(axis)] = metadata
        donor_span = sum(components.values())
        raw_mean = torch.stack(list(donors.values())).mean(dim=0)
        raw_mean = _scale_per_position(
            raw_mean, target["controller"])

        calibration = target["alignments"]["calibration"]
        route_direction, route_metadata = _route_readout(
            model, calibration)
        gradients, calibration_readouts = _target_gradients(
            model, calibration, route_direction)
        prediction_rows = _axis_prediction(components, gradients)
        selected_axis = int(prediction_rows[0]["axis"])
        probe = _probe_accessibility(target["samples"], basis)
        target_flat = target["controller"].flatten().float()
        target_coefficients = (target_flat @ basis.T).tolist()
        component_norms = [
            float(components[axis].norm())
            for axis in range(1, len(donors) + 1)
        ]
        baseline_axes = {
            "svd_energy": 1,
            "lowest_energy": len(donors),
            "coefficient_magnitude": int(
                np.argmax(np.abs(target_coefficients))) + 1,
            "vector_norm": int(np.argmax(component_norms)) + 1,
            "probe_accessibility": int(np.argmax(probe)) + 1,
            "cosine_similarity": int(
                np.argmax(np.abs(target_coefficients))) + 1,
        }
        folds[target_name] = {
            "donor_names": donor_names,
            "singular": singular,
            "basis": basis,
            "basis_coefficients": coefficients,
            "basis_geometry": geometry,
            "components": components,
            "component_metadata": component_metadata,
            "donor_span": donor_span,
            "raw_donor_mean": raw_mean,
            "route_direction": route_direction,
            "route_metadata": route_metadata,
            "gradients": gradients,
            "prediction_rows": prediction_rows,
            "selected_axis": selected_axis,
            "baseline_axes": baseline_axes,
            "probe_accessibility": probe,
            "target_coefficients": target_coefficients,
            "component_norms": component_norms,
            "calibration_readouts": calibration_readouts,
        }
        predictions_public[target_name] = {
            "donor_names": donor_names,
            "basis_geometry": geometry,
            "component_metadata": component_metadata,
            "route_metadata": route_metadata,
            "prediction_ranking": prediction_rows,
            "selected_axis": selected_axis,
            "baseline_axes": baseline_axes,
            "probe_accessibility": probe,
            "target_coefficients": target_coefficients,
            "component_norms": component_norms,
            "calibration_readouts": calibration_readouts,
        }

    prediction_artifact = {
        "protocol_sha256": PROTOCOL_SHA256,
        "statement": (
            "Frozen before any target answer-prefix causal intervention."),
        "predictions": predictions_public,
    }
    prediction_path = os.path.join(
        out_dir, f"prospective_predictions_{model_key}.json")
    with open(prediction_path, "w") as handle:
        json.dump(prediction_artifact, handle, indent=2, default=float)
    with open(prediction_path, "rb") as handle:
        prediction_sha = hashlib.sha256(
            handle.read()).hexdigest().upper()
    log(
        f"FROZEN prospective predictions sha256={prediction_sha} "
        + " ".join(
            f"{name}=axis{fold['selected_axis']}"
            for name, fold in folds.items()))

    # Causal holdout phase begins only after the prediction artifact is fixed.
    arms_per_fold = 4 + (len(prepared) - 1) + 2 + N_RANDOM
    heartbeat = Heartbeat(
        len(prepared) * arms_per_fold * 2 * 4,
        "prospective_causal_sensitivity",
        every_sec=30, out_dir=out_dir)
    causal_results = {}
    for domain_index, (target_name, target_data) in enumerate(
            prepared.items()):
        fold = folds[target_name]
        alignment = target_data["alignments"]["test"]
        states = _evaluation_states(model, alignment)
        target = {
            "name": target_name,
            "spec": target_data["spec"],
            "alignment": alignment,
            "states": states,
            "positions": alignment["answer_positions"],
            "source": [
                row["source"] for row in target_data["rows"]["test"]],
            "target": [
                row["target"] for row in target_data["rows"]["test"]],
            "n_world": TEST_N,
        }
        selected = fold["components"][fold["selected_axis"]]
        random_positions = _random_position_sets(
            alignment["random_candidates"], n_random=1,
            seed=RANDOM_SEED + 1009 * domain_index)[0]
        random_directions = _norm_matched_directions(
            selected, n_random=N_RANDOM,
            seed=RANDOM_SEED + 1009 * domain_index)

        arm_patches = {
            "original": None,
            "natural_prefix_interchange": _natural_patch_pair(
                states, target["positions"]),
            "donor_span_reconstruction": _patch_pair(
                states, target["positions"], fold["donor_span"]),
            "raw_donor_mean": _patch_pair(
                states, target["positions"], fold["raw_donor_mean"]),
        }
        for axis, component in fold["components"].items():
            arm_patches[f"axis_{axis:02d}"] = _patch_pair(
                states, target["positions"], component)
        arm_patches["instruction_position_control"] = _patch_pair(
            states, alignment["instruction_positions"], selected)
        arm_patches["random_position_control"] = _patch_pair(
            states, random_positions, selected)
        for index, direction in enumerate(random_directions):
            arm_patches[f"random_direction_{index:02d}"] = _patch_pair(
                states, target["positions"], direction)

        arm_results = {}
        original_sources = None
        for arm_name, patches in arm_patches.items():
            arm = _evaluate_arm(
                model, target, patches, l24_sites, head_dim,
                heartbeat, arm_name)
            private = arm.pop("_private_contexts")
            if arm_name == "original":
                original_sources = {
                    operation: {
                        "clean": private[
                            operation]["clean_source"].clone(),
                        "natural": private[
                            operation]["natural_source"].clone(),
                    }
                    for operation in ("belief", "search")
                }
                arm["source_state_max_abs_change"] = 0.0
            else:
                maximum = 0.0
                for operation in ("belief", "search"):
                    maximum = max(
                        maximum,
                        float((
                            private[operation]["clean_source"]
                            - original_sources[operation]["clean"]
                        ).abs().max()),
                        float((
                            private[operation]["natural_source"]
                            - original_sources[operation]["natural"]
                        ).abs().max()),
                    )
                arm["source_state_max_abs_change"] = maximum
            arm_results[arm_name] = arm

        original = arm_results["original"]
        for arm_name, arm in arm_results.items():
            if arm_name == "original":
                continue
            arm.update(_score_arm(arm, original))
        causal_results[target_name] = {
            "test_rows": target_data["rows"]["test"],
            "positions": {
                "answer_prefix": target["positions"],
                "instruction": alignment["instruction_positions"],
                "random_control": random_positions,
            },
            "arms": arm_results,
        }
    heartbeat.done()

    adjudication = {}
    pooled_predicted = []
    pooled_actual = []
    for target_name, target_data in prepared.items():
        fold_result = causal_results[target_name]
        fold_result_for_adjudication = {
            "arms": fold_result["arms"]}
        adjudication[target_name] = _fold_adjudication(
            {
                "n_world": TEST_N,
            },
            fold_result_for_adjudication,
            folds[target_name]["prediction_rows"],
            folds[target_name]["baseline_axes"])
        by_axis = {
            row["axis"]: row for row in
            folds[target_name]["prediction_rows"]
        }
        for axis in sorted(by_axis):
            actual = fold_result["arms"][f"axis_{axis:02d}"][
                "functional_bidirectional_score"]
            if actual > -1e8:
                pooled_predicted.append(
                    by_axis[axis]["bidirectional_sensitivity"])
                pooled_actual.append(actual)

    pooled_spearman = _spearman(
        pooled_predicted, pooled_actual)
    pooled_pearson = _correlation(
        pooled_predicted, pooled_actual)
    eligible = [
        name for name, value in adjudication.items()
        if value["eligible"]]
    references = [
        name for name, value in adjudication.items()
        if value["eligible"] and value["reference_pass"]]
    passed = [
        name for name, value in adjudication.items()
        if value["fold_pass"]]
    low_energy_wins = [
        name for name, value in adjudication.items()
        if value["fold_pass"] and value["low_energy_win_over_pc1"]]
    any_axis_effect = [
        name for name, value in adjudication.items()
        if value["best_axis_score"] >= MINIMUM_PREDICTED_SCORE - 1e-9]
    random_smoke_passed = [
        name for name, value in adjudication.items()
        if value["random_smoke_pass"]]

    if len(prepared) < 4:
        verdict = "TOKENIZATION_OR_ALIGNMENT_INELIGIBLE"
    elif len(eligible) < 4:
        verdict = "BEHAVIORALLY_INELIGIBLE"
    elif len(references) < 4:
        verdict = "NATURAL_REFERENCE_WEAK"
    elif (
            len(passed) >= 3
            and pooled_spearman is not None
            and pooled_spearman >= MINIMUM_POOLED_SPEARMAN
            and len(low_energy_wins) >= 2):
        verdict = "PROSPECTIVE_LOW_ENERGY_CAUSAL_GEOMETRY_CANDIDATE"
    elif (
            len(passed) >= 3
            and pooled_spearman is not None
            and pooled_spearman >= MINIMUM_POOLED_SPEARMAN):
        if all(
                adjudication[name]["selected_axis"] == 1
                for name in passed):
            verdict = "UNIVERSAL_HIGH_ENERGY_AXIS_CANDIDATE"
        else:
            verdict = "PROSPECTIVE_COMPACT_CAUSAL_GEOMETRY_CANDIDATE"
    elif len(any_axis_effect) >= 3 and len(passed) < 3:
        verdict = "CAUSAL_AXES_NOT_PROSPECTIVELY_PREDICTABLE"
    elif len(passed) >= 2:
        verdict = "PROSPECTIVE_CAUSAL_GEOMETRY_PARTIAL"
    else:
        verdict = "NO_TRANSFERABLE_DONOR_CAUSAL_CONTROL"

    basis_path = os.path.join(
        out_dir, f"prospective_fold_bases_{model_key}.npz")
    np.savez(
        basis_path,
        **{
            f"{name}_basis": fold["basis"].numpy()
            for name, fold in folds.items()
        },
        **{
            f"{name}_route_direction": fold[
                "route_direction"].numpy()
            for name, fold in folds.items()
        })
    with open(basis_path, "rb") as handle:
        basis_sha = hashlib.sha256(handle.read()).hexdigest().upper()

    result = {
        "stage": "delta_prospective_causal_sensitivity",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "A held-out domain's causally active donor-basis coordinate can "
            "be selected before target steering from the local downstream "
            "Jacobian, including low-variance coordinates."),
        "alignment_errors": alignment_errors,
        "splits": {
            name: data["rows"] for name, data in prepared.items()
        },
        "prediction_artifact": {
            "file": os.path.basename(prediction_path),
            "sha256": prediction_sha,
            "frozen_before_target_axis_interventions": True,
        },
        "basis_artifact": {
            "file": os.path.basename(basis_path),
            "sha256": basis_sha,
        },
        "predictions": predictions_public,
        "causal_results": causal_results,
        "adjudication": adjudication,
        "aggregate": {
            "eligible_domains": eligible,
            "reference_domains": references,
            "passing_folds": passed,
            "low_energy_wins": low_energy_wins,
            "domains_with_any_axis_effect": any_axis_effect,
            "random_smoke_passed_domains": random_smoke_passed,
            "random_null_confirmation_required": True,
            "pooled_sensitivity_actual_spearman": pooled_spearman,
            "pooled_sensitivity_actual_pearson": pooled_pearson,
        },
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_prospective_causal_sensitivity_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"PROSPECTIVE-CAUSAL-SENSITIVITY verdict={verdict} "
        f"eligible={len(eligible)} passed={len(passed)} "
        f"low_energy_wins={len(low_energy_wins)} "
        f"rho={pooled_spearman} artifact={path}")
    return result
