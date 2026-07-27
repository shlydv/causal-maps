"""Controller-to-circuit blockade and rescue on a causal holdout."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _anchor_position, _resolve
from .delta_content_cancelled_controller import (
    EPS,
    _fixed_patch,
    _movement,
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
    _generic_accuracy,
    _generic_cell,
)
from .delta_distributed_label_transplant import _capture_full_l21
from .delta_leave_color_out_shared import (
    _controller_metadata,
    _fresh_color_rows,
    _shared_controllers,
    _sign_tail,
)
from .delta_preprint_battery import _compatible_world_rows
from .delta_residual_only_confirmation import _fresh_color_rows_v3
from .delta_shared_adapter_decomposition import (
    _color_decomposition,
    _fresh_color_rows_v2,
)
from .delta_source_head_mediation import _mediation_pass
from .delta_sparse_transport import (
    G_ACC,
    _attention_geometry,
    _o_proj,
    _site_value,
)
from .delta_sparse_transport_confirmation import (
    CANDIDATE_LAYERS,
    FROZEN_TOP8,
)
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output

PROTOCOL_VERSION = "2026-07-26-p2-controller-circuit-epistasis-v2"
PROTOCOL_SHA256 = (
    "B6B0F793E0117C92A0D1D636EDDD0E663F0AE6D00AD23B54ECB2B5FBA5E6B9D2")
DONOR_N = 15
DISCOVERY_N = 30
HOLDOUT_N = 30
SELECT_K = 4
N_RANDOM = 19
RANDOM_SEED = 12721
SOURCE_LAYER = 21
GATE_LAYER = 22
MINIMUM_EFFECT_FRACTION = 0.50
MINIMUM_SIGN_SUCCESSES = 24
MAXIMUM_SIGN_P = 0.01
MAXIMUM_NULL_P = 0.05
SOURCE_INVARIANCE_TOLERANCE = 1e-5


def _fresh_color_rows_v4():
    """Validated discovery rows plus 30 histories absent from every prior set."""
    values = DOMAIN_SPECS["color_state"]["values"]
    discovery_rows = _fresh_color_rows_v3()
    prior_rows = (
        _domain_rows(values)
        + _fresh_domain_rows(values)
        + _fresh_color_rows()
        + _fresh_color_rows_v2()
        + discovery_rows
    )
    prior_signatures = {
        (state, row["d1"], row["d2"])
        for row in prior_rows
        for state in (row["source"], row["target"])
    }

    # A prompt signature is (state, d1, d2). For each fixed distractor pair,
    # pair all remaining unused states. These pairs form a deterministic
    # matching, so choosing any subset cannot reuse a rendered history.
    edge_pool = []
    for d1 in values:
        for d2 in values:
            if d1 == d2:
                continue
            available = [
                state for state in values
                if state not in (d1, d2)
                and (state, d1, d2) not in prior_signatures
            ]
            for source, target in zip(
                    available[::2], available[1::2]):
                edge_pool.append({
                    "left": source,
                    "right": target,
                    "d1": d1,
                    "d2": d2,
                })
    if len(edge_pool) < HOLDOUT_N:
        raise AssertionError(
            f"only {len(edge_pool)} disjoint unseen color edges remain")

    # Select and orient 30 edges while balancing source, target, and both
    # distractor marginals. Balancing uses no model activation or outcome.
    source_counts = {value: 0 for value in values}
    target_counts = {value: 0 for value in values}
    d1_counts = {value: 0 for value in values}
    d2_counts = {value: 0 for value in values}
    remaining_edges = list(enumerate(edge_pool))
    holdout_rows = []

    def projected_score(edge_index, edge, source, target):
        projected_source = {
            value: source_counts[value] + int(value == source)
            for value in values
        }
        projected_target = {
            value: target_counts[value] + int(value == target)
            for value in values
        }
        projected_d1 = {
            value: d1_counts[value] + int(value == edge["d1"])
            for value in values
        }
        projected_d2 = {
            value: d2_counts[value] + int(value == edge["d2"])
            for value in values
        }
        all_counts = (
            list(projected_source.values())
            + list(projected_target.values())
            + list(projected_d1.values())
            + list(projected_d2.values()))
        return (
            max(projected_source.values()),
            max(projected_target.values()),
            max(projected_d1.values()),
            max(projected_d2.values()),
            sum(value * value for value in all_counts),
            edge_index,
            values.index(source),
            values.index(target),
        )

    while len(holdout_rows) < HOLDOUT_N:
        options = []
        for edge_index, edge in remaining_edges:
            options.append((
                projected_score(
                    edge_index, edge, edge["left"], edge["right"]),
                edge_index, edge, edge["left"], edge["right"]))
            options.append((
                projected_score(
                    edge_index, edge, edge["right"], edge["left"]),
                edge_index, edge, edge["right"], edge["left"]))
        _score, chosen_index, edge, source, target = min(
            options, key=lambda item: item[0])
        holdout_rows.append({
            "row_index": DISCOVERY_N + len(holdout_rows),
            "source": source,
            "target": target,
            "state": source,
            "d1": edge["d1"],
            "d2": edge["d2"],
        })
        source_counts[source] += 1
        target_counts[target] += 1
        d1_counts[edge["d1"]] += 1
        d2_counts[edge["d2"]] += 1
        remaining_edges = [
            item for item in remaining_edges
            if item[0] != chosen_index
        ]

    holdout_signatures = {
        (state, row["d1"], row["d2"])
        for row in holdout_rows
        for state in (row["source"], row["target"])
    }
    if holdout_signatures & prior_signatures:
        raise AssertionError("causal holdout overlaps a preceding color set")
    rows = list(discovery_rows) + holdout_rows
    signatures = {
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }
    if len(signatures) != 2 * (DISCOVERY_N + HOLDOUT_N):
        raise AssertionError("fourth fresh rendered histories are not unique")
    return rows


def _random_head_sets(candidate_heads, selected_heads, n_random=N_RANDOM,
                      seed=RANDOM_SEED):
    pool = sorted(set(candidate_heads) - set(selected_heads))
    if len(pool) < SELECT_K:
        raise ValueError("too few nonselected layer-22 heads")
    rng = np.random.default_rng(int(seed))
    result = []
    seen = set()
    while len(result) < int(n_random):
        choice = tuple(sorted(
            int(value) for value in rng.choice(
                pool, size=SELECT_K, replace=False)))
        if choice not in seen:
            seen.add(choice)
            result.append(choice)
    return result


def _source_response(bundle, head, head_dim):
    start = int(head) * int(head_dim)
    stop = start + int(head_dim)
    forward = (
        bundle["forward"][:, start:stop]
        - bundle["clean"][:, start:stop])
    reverse = (
        bundle["reverse"][:, start:stop]
        - bundle["natural"][:, start:stop])
    return torch.cat([forward, reverse], dim=0).flatten().double()


def _discover_gate_heads(bundles, candidate_heads, head_dim,
                         select_k=SELECT_K):
    rows = []
    for head in candidate_heads:
        belief = _source_response(
            bundles["belief"], head, head_dim)
        search = _source_response(
            bundles["search"], head, head_dim)
        belief_steered = _source_response(
            bundles["belief_steered"], head, head_dim)
        search_steered = _source_response(
            bundles["search_steered"], head, head_dim)
        natural_gap = search - belief
        belief_delta = belief_steered - belief
        search_delta = search_steered - search
        gap_energy = float(torch.dot(natural_gap, natural_gap))
        gap_norm = float(natural_gap.norm())
        if gap_energy <= EPS:
            belief_projection = -1e9
            search_projection = -1e9
        else:
            belief_projection = float(
                torch.dot(belief_delta, natural_gap) / gap_energy)
            search_projection = float(
                torch.dot(search_delta, -natural_gap) / gap_energy)
        rows.append({
            "layer": GATE_LAYER,
            "head": int(head),
            "natural_gap_norm": gap_norm,
            "belief_to_search_projection": belief_projection,
            "search_to_belief_projection": search_projection,
            "bidirectional_projection_score": min(
                belief_projection, search_projection),
        })
    positive_norms = [
        row["natural_gap_norm"] for row in rows
        if row["natural_gap_norm"] > EPS
    ]
    median_norm = float(np.median(positive_norms)) if positive_norms else 0.0
    for row in rows:
        row["norm_eligible"] = bool(
            row["natural_gap_norm"] + EPS >= median_norm
            and row["natural_gap_norm"] > EPS)
        row["selection_score"] = (
            row["bidirectional_projection_score"]
            if row["norm_eligible"] else -1e9)
    ranked = sorted(
        rows,
        key=lambda row: (-row["selection_score"], row["head"]))
    eligible_positive = [
        row for row in ranked
        if row["norm_eligible"] and row["selection_score"] > 0.0
    ]
    selected = [
        int(row["head"])
        for row in eligible_positive[:int(select_k)]
    ]
    return {
        "candidate_rows": rows,
        "median_natural_gap_norm": median_norm,
        "ranked_heads": [
            {"head": row["head"],
             "selection_score": row["selection_score"]}
            for row in ranked
        ],
        "selected_heads": selected,
        "stable": bool(len(selected) == int(select_k)),
    }


def _head_value(full_value, head, head_dim):
    start = int(head) * int(head_dim)
    stop = start + int(head_dim)
    return full_value[:, start:stop]


@torch.no_grad()
def _run_route_pass(
        model, batch, source_position, readout_position, capture_layers,
        head_dim, source_value=None, sequence_patch=None,
        gate_heads=(), gate_value=None, transport_sites=(),
        transport_values=()):
    if (gate_value is None) != (len(gate_heads) == 0):
        raise ValueError("gate heads and gate value must be supplied together")
    if len(transport_sites) != len(transport_values):
        raise ValueError("transport sites and values differ")
    blocks = get_decoder_layers(model)
    source_cache = {}
    head_cache = {}
    handles = []

    def source_hook(_module, _args, output):
        states, rebuild = _split_output(output)
        changed = source_value is not None or sequence_patch is not None
        if changed:
            states = states.clone()
        if source_value is not None:
            states[:, int(source_position), :] = source_value.to(
                device=states.device, dtype=states.dtype)
        if sequence_patch is not None:
            positions, values = sequence_patch
            states[:, positions, :] = values.to(
                device=states.device, dtype=states.dtype)
        source_cache["state"] = (
            states[:, int(source_position), :].detach().float().cpu())
        if changed:
            return rebuild(states)

    grouped_transport = {}
    for site, value in zip(transport_sites, transport_values):
        grouped_transport.setdefault(int(site[0]), []).append(
            (int(site[1]), value))

    def head_hook(layer):
        def hook(_module, args):
            states = args[0]
            head_cache[int(layer)] = (
                states[:, int(readout_position), :].detach().float().cpu())
            gate_entries = (
                list(gate_heads) if int(layer) == GATE_LAYER else [])
            transport_entries = grouped_transport.get(int(layer), [])
            if not gate_entries and not transport_entries:
                return None
            updated = states.clone()
            for head in gate_entries:
                start = int(head) * int(head_dim)
                stop = start + int(head_dim)
                updated[:, int(readout_position), start:stop] = _head_value(
                    gate_value, head, head_dim).to(
                        device=states.device, dtype=states.dtype)
            for head, value in transport_entries:
                start = int(head) * int(head_dim)
                stop = start + int(head_dim)
                updated[:, int(readout_position), start:stop] = value.to(
                    device=states.device, dtype=states.dtype)
            return (updated,) + tuple(args[1:])
        return hook

    handles.append(
        blocks[int(SOURCE_LAYER)].register_forward_hook(source_hook))
    for layer in capture_layers:
        handles.append(_o_proj(model, layer).register_forward_pre_hook(
            head_hook(layer)))
    try:
        output = model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
        logits = output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if "state" not in source_cache:
        raise RuntimeError("layer-21 source state was not captured")
    missing = [
        layer for layer in capture_layers
        if int(layer) not in head_cache
    ]
    if missing:
        raise RuntimeError(f"head layers were not captured: {missing}")
    return logits, source_cache["state"], head_cache


@torch.no_grad()
def _evaluate_route(
        model, clean, natural, values, source, target,
        head_dim, sequence_patch=None, gate_heads=(),
        gate_bundle=None, heartbeat=None, label="route"):
    source_position = _anchor_position(clean, natural)
    readout_position = int(clean["ids"].shape[1] - 1)
    clean_sequence, natural_sequence = (
        sequence_patch if sequence_patch is not None
        else (None, None))

    def gate_value(pass_name):
        if not gate_heads:
            return None
        if gate_bundle is None or pass_name not in gate_bundle:
            raise ValueError(f"missing gate donor for {pass_name}")
        return gate_bundle[pass_name]

    clean_logits, clean_source, clean_heads = _run_route_pass(
        model, clean, source_position, readout_position,
        CANDIDATE_LAYERS, head_dim,
        sequence_patch=clean_sequence,
        gate_heads=gate_heads, gate_value=gate_value("clean"))
    if heartbeat:
        heartbeat.step(extra=f"{label}/clean")
    natural_logits, natural_source, natural_heads = _run_route_pass(
        model, natural, source_position, readout_position,
        CANDIDATE_LAYERS, head_dim,
        sequence_patch=natural_sequence,
        gate_heads=gate_heads, gate_value=gate_value("natural"))
    if heartbeat:
        heartbeat.step(extra=f"{label}/natural")

    forward_logits, _forward_source, forward_heads = _run_route_pass(
        model, clean, source_position, readout_position,
        CANDIDATE_LAYERS, head_dim,
        source_value=natural_source,
        sequence_patch=clean_sequence,
        gate_heads=gate_heads, gate_value=gate_value("forward"))
    if heartbeat:
        heartbeat.step(extra=f"{label}/forward")
    reverse_logits, _reverse_source, reverse_heads = _run_route_pass(
        model, natural, source_position, readout_position,
        CANDIDATE_LAYERS, head_dim,
        source_value=clean_source,
        sequence_patch=natural_sequence,
        gate_heads=gate_heads, gate_value=gate_value("reverse"))
    if heartbeat:
        heartbeat.step(extra=f"{label}/reverse")

    source_cell = _generic_cell(
        clean_logits, natural_logits, forward_logits, reverse_logits,
        clean, source, target, values)
    forward_blocked, _state, _heads = _run_route_pass(
        model, clean, source_position, readout_position,
        CANDIDATE_LAYERS, head_dim,
        source_value=natural_source,
        sequence_patch=clean_sequence,
        gate_heads=gate_heads, gate_value=gate_value("forward"),
        transport_sites=FROZEN_TOP8,
        transport_values=[
            _site_value(clean_heads, site, head_dim)
            for site in FROZEN_TOP8])
    if heartbeat:
        heartbeat.step(extra=f"{label}/forward_transport_block")
    reverse_blocked, _state, _heads = _run_route_pass(
        model, natural, source_position, readout_position,
        CANDIDATE_LAYERS, head_dim,
        source_value=clean_source,
        sequence_patch=natural_sequence,
        gate_heads=gate_heads, gate_value=gate_value("reverse"),
        transport_sites=FROZEN_TOP8,
        transport_values=[
            _site_value(natural_heads, site, head_dim)
            for site in FROZEN_TOP8])
    if heartbeat:
        heartbeat.step(extra=f"{label}/reverse_transport_block")
    blocked_cell = _generic_cell(
        clean_logits, natural_logits, forward_blocked, reverse_blocked,
        clean, source, target, values)
    mediation = _mediation_pass(
        source_cell, blocked_cell,
        _generic_accuracy(forward_blocked, clean, source, values),
        _generic_accuracy(reverse_blocked, natural, target, values))
    eligible = bool(min(
        _generic_accuracy(clean_logits, clean, source, values),
        _generic_accuracy(natural_logits, natural, target, values),
    ) >= G_ACC)
    public = {
        "eligible": eligible,
        "g0_clean": float(_generic_accuracy(
            clean_logits, clean, source, values)),
        "g0_natural": float(_generic_accuracy(
            natural_logits, natural, target, values)),
        "source_intervention": source_cell,
        "blocked_intervention": blocked_cell,
        "mediation": mediation,
    }
    return {
        "public": public,
        "route_score": float(mediation["minimum_fraction"]),
        "world_route": _world_mediation(public, {
            "blocked_intervention": blocked_cell,
        }),
        "source_states": {
            "clean": clean_source,
            "natural": natural_source,
        },
        "gate_bundle": {
            "clean": clean_heads[GATE_LAYER],
            "natural": natural_heads[GATE_LAYER],
            "forward": forward_heads[GATE_LAYER],
            "reverse": reverse_heads[GATE_LAYER],
        },
    }


def _functional(route):
    task = route["public"]
    return bool(
        task["eligible"]
        and task["source_intervention"]["sufficient"])


def _paired_sign(values):
    valid = [float(value) for value in values if value is not None]
    successes = sum(value > 0.0 for value in valid)
    p_value = (
        _sign_tail(successes, HOLDOUT_N)
        if len(valid) == HOLDOUT_N else None)
    return {
        "valid_worlds": len(valid),
        "successes": successes,
        "fraction": (
            successes / HOLDOUT_N
            if len(valid) == HOLDOUT_N else None),
        "exact_one_sided_sign_p": p_value,
        "range": [min(valid), max(valid)] if valid else None,
        "mean": float(np.mean(valid)) if valid else None,
        "pass": bool(
            len(valid) == HOLDOUT_N
            and successes >= MINIMUM_SIGN_SUCCESSES
            and p_value is not None
            and p_value <= MAXIMUM_SIGN_P + 1e-12),
    }


def _paired_rows(left, right, direction):
    rows = []
    values = []
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        value = None
        if left_value is not None and right_value is not None:
            value = float(direction) * (
                float(left_value) - float(right_value))
            values.append(value)
        rows.append({
            "world_offset": index,
            "effect": value,
            "predicted_sign": bool(
                value is not None and value > 0.0),
        })
    return rows, _paired_sign(values)


def _epistasis_metrics(original_belief, original_search,
                       steered_belief, steered_search,
                       blocked_belief, blocked_search,
                       rescued_belief, rescued_search):
    belief_movement = (
        original_belief["route_score"] - steered_belief["route_score"])
    search_movement = (
        steered_search["route_score"] - original_search["route_score"])
    belief_blockade = (
        blocked_belief["route_score"] - steered_belief["route_score"])
    search_blockade = (
        steered_search["route_score"] - blocked_search["route_score"])
    belief_rescue = (
        original_belief["route_score"] - rescued_belief["route_score"])
    search_rescue = (
        rescued_search["route_score"] - original_search["route_score"])

    def fraction(effect, movement):
        return float(effect / movement) if movement > EPS else -1e9

    belief_block_rows, belief_block_sign = _paired_rows(
        blocked_belief["world_route"],
        steered_belief["world_route"], +1.0)
    search_block_rows, search_block_sign = _paired_rows(
        steered_search["world_route"],
        blocked_search["world_route"], +1.0)
    belief_rescue_rows, belief_rescue_sign = _paired_rows(
        original_belief["world_route"],
        rescued_belief["world_route"], +1.0)
    search_rescue_rows, search_rescue_sign = _paired_rows(
        rescued_search["world_route"],
        original_search["world_route"], +1.0)
    all_routes = (
        blocked_belief, blocked_search,
        rescued_belief, rescued_search,
    )
    functional = all(_functional(route) for route in all_routes)
    blockade_fractions = {
        "belief_to_search": fraction(
            belief_blockade, belief_movement),
        "search_to_belief": fraction(
            search_blockade, search_movement),
    }
    rescue_fractions = {
        "belief_to_search": fraction(
            belief_rescue, belief_movement),
        "search_to_belief": fraction(
            search_rescue, search_movement),
    }
    blockade_score = min(blockade_fractions.values())
    rescue_score = min(rescue_fractions.values())
    blockade_pass = bool(
        functional
        and blockade_score >= MINIMUM_EFFECT_FRACTION - 1e-9
        and belief_block_sign["pass"]
        and search_block_sign["pass"])
    rescue_pass = bool(
        functional
        and rescue_score >= MINIMUM_EFFECT_FRACTION - 1e-9
        and belief_rescue_sign["pass"]
        and search_rescue_sign["pass"])
    return {
        "calibrated_controller_movement": {
            "belief_to_search": belief_movement,
            "search_to_belief": search_movement,
        },
        "blockade": {
            "absolute_effect": {
                "belief_to_search": belief_blockade,
                "search_to_belief": search_blockade,
            },
            "fraction_of_controller_movement": blockade_fractions,
            "bidirectional_fraction_score": blockade_score,
            "per_world": {
                "belief_to_search": {
                    "rows": belief_block_rows,
                    **belief_block_sign,
                },
                "search_to_belief": {
                    "rows": search_block_rows,
                    **search_block_sign,
                },
            },
            "pass": blockade_pass,
        },
        "rescue": {
            "absolute_effect": {
                "belief_to_search": belief_rescue,
                "search_to_belief": search_rescue,
            },
            "fraction_of_controller_movement": rescue_fractions,
            "bidirectional_fraction_score": rescue_score,
            "per_world": {
                "belief_to_search": {
                    "rows": belief_rescue_rows,
                    **belief_rescue_sign,
                },
                "search_to_belief": {
                    "rows": search_rescue_rows,
                    **search_rescue_sign,
                },
            },
            "pass": rescue_pass,
        },
        "all_contexts_functional": functional,
    }


def _tail_probability(observed, null_values):
    exceed = sum(
        float(value) >= float(observed)
        for value in null_values)
    return (1.0 + exceed) / (1.0 + len(null_values)), exceed


def _adjudicate(discovery_stable, calibration_pass, source_invariant,
                blockade_pass, rescue_pass,
                blockade_specific, rescue_specific):
    if not discovery_stable:
        return "GATE_HEAD_DISCOVERY_UNSTABLE"
    if not calibration_pass or not source_invariant:
        return "CONTROLLER_CALIBRATION_FAILED"
    if ((blockade_pass and not blockade_specific)
            or (rescue_pass and not rescue_specific)):
        return "CONTROLLER_EFFECT_DISTRIBUTED_OR_NONSPECIFIC"
    if (blockade_pass and blockade_specific
            and rescue_pass and rescue_specific):
        return "CONTROLLER_GATES_TRANSPORT_CIRCUIT"
    if blockade_pass and blockade_specific:
        return "CONTROLLER_CIRCUIT_BLOCKADE_ONLY"
    if rescue_pass and rescue_specific:
        return "CONTROLLER_CIRCUIT_RESCUE_ONLY"
    return "CONTROLLER_CIRCUIT_UNRESOLVED"


@torch.no_grad()
def run_delta_controller_circuit_epistasis(
        model_path, out_dir,
        model_key="qwen7b_controller_circuit_epistasis",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=60):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != DISCOVERY_N + HOLDOUT_N:
        raise ValueError("v1 is frozen to exactly 60 fresh histories")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(CANDIDATE_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("controller-circuit layers are absent")
    n_heads, head_dim = _attention_geometry(model)
    if any(head >= n_heads for _layer, head in FROZEN_TOP8):
        raise ValueError("a frozen transport head is absent")

    spec = DOMAIN_SPECS["color_state"]
    location_rows, location_indices = _compatible_world_rows(
        tok, torch.device("cpu"), 30)
    if len(location_rows) != 30:
        raise ValueError("all 30 compatible location worlds are required")
    ownership_rows = _fresh_domain_rows(
        DOMAIN_SPECS["ownership"]["values"])[:DONOR_N]
    color_donor_rows = _fresh_domain_rows(
        spec["values"])[:DONOR_N]
    donor_alignments = {
        "location": _location_alignment(
            tok, dev, location_rows[:DONOR_N]),
        "ownership": _domain_alignment(
            tok, dev, ownership_rows, DOMAIN_SPECS["ownership"]),
        "color": _domain_alignment(tok, dev, color_donor_rows, spec),
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
        out_dir, f"controller_circuit_{model_key}.npz")
    np.savez(
        controller_path,
        shared_pc1=shared_pc1.numpy(),
        color_projection=projection.numpy(),
        color_residual=residual.numpy())
    with open(controller_path, "rb") as handle:
        archive_sha = hashlib.sha256(handle.read()).hexdigest().upper()

    all_rows = _fresh_color_rows_v4()
    discovery_rows = all_rows[:DISCOVERY_N]
    holdout_rows = all_rows[DISCOVERY_N:]
    candidate_heads = [
        head for head in range(n_heads)
        if (GATE_LAYER, head) not in set(FROZEN_TOP8)
    ]
    total_evaluations = 4 + 4 + 4 + N_RANDOM * 4
    hb = Heartbeat(
        total_evaluations * 6,
        "controller_circuit_epistasis", every_sec=30, out_dir=out_dir)

    def split_contexts(rows, split_name):
        alignment = _domain_alignment(tok, dev, rows, spec)
        states = {"belief": [], "search": []}
        for belief_batch, search_batch in alignment["batches"]:
            states["belief"].append(
                _capture_full_l21(model, belief_batch))
            states["search"].append(
                _capture_full_l21(model, search_batch))
        clean_belief, clean_search = alignment["batches"][0]
        natural_belief, natural_search = alignment["batches"][1]
        source = [row["source"] for row in rows]
        target = [row["target"] for row in rows]

        def evaluate(operation, steered=False, gate_heads=(),
                     gate_bundle=None, label="route"):
            clean = (
                clean_belief if operation == "belief"
                else clean_search)
            natural = (
                natural_belief if operation == "belief"
                else natural_search)
            patch = None
            if steered:
                sign = -1.0 if operation == "belief" else +1.0
                patch = _fixed_patch(
                    states[operation],
                    alignment["answer_positions"],
                    residual, sign)
            return _evaluate_route(
                model, clean, natural, list(spec["values"]),
                source, target, head_dim,
                sequence_patch=patch,
                gate_heads=gate_heads, gate_bundle=gate_bundle,
                heartbeat=hb, label=f"{split_name}/{label}")

        return {
            "alignment": alignment,
            "states": states,
            "evaluate": evaluate,
        }

    discovery = split_contexts(discovery_rows, "discovery")
    discovery_raw = {
        "belief": discovery["evaluate"](
            "belief", label="belief"),
        "search": discovery["evaluate"](
            "search", label="search"),
        "belief_steered": discovery["evaluate"](
            "belief", steered=True, label="belief_steered"),
        "search_steered": discovery["evaluate"](
            "search", steered=True, label="search_steered"),
    }
    discovery_result = _discover_gate_heads(
        {
            name: route["gate_bundle"]
            for name, route in discovery_raw.items()
        },
        candidate_heads, head_dim)
    selected_heads = discovery_result["selected_heads"]

    holdout = split_contexts(holdout_rows, "holdout")
    holdout_raw = {
        "belief": holdout["evaluate"](
            "belief", label="belief"),
        "search": holdout["evaluate"](
            "search", label="search"),
        "belief_steered": holdout["evaluate"](
            "belief", steered=True, label="belief_steered"),
        "search_steered": holdout["evaluate"](
            "search", steered=True, label="search_steered"),
    }
    original = {
        "belief": {
            "l24_minimum_mediation":
                holdout_raw["belief"]["route_score"]},
        "search": {
            "l24_minimum_mediation":
                holdout_raw["search"]["route_score"]},
    }
    calibration_movement = _movement(
        original,
        holdout_raw["belief_steered"]["route_score"],
        holdout_raw["search_steered"]["route_score"])
    calibration_functional = all(
        _functional(route) for route in holdout_raw.values())
    calibration_pass = bool(
        discovery_result["stable"]
        and calibration_functional
        and calibration_movement["original_gap"] >= 0.03 - 1e-9
        and calibration_movement["belief_to_search_pass"]
        and calibration_movement["search_to_belief_pass"])

    source_differences = []
    for operation in ("belief", "search"):
        unsteered = holdout_raw[operation]["source_states"]
        steered = holdout_raw[
            f"{operation}_steered"]["source_states"]
        for history in ("clean", "natural"):
            source_differences.append(float(
                (unsteered[history] - steered[history]).abs().max()))
    maximum_source_difference = max(source_differences)
    source_invariant = bool(
        maximum_source_difference <= SOURCE_INVARIANCE_TOLERANCE)

    def evaluate_head_set(heads, label):
        blocked_belief = holdout["evaluate"](
            "belief", steered=True, gate_heads=heads,
            gate_bundle=holdout_raw["belief"]["gate_bundle"],
            label=f"{label}/belief_blockade")
        blocked_search = holdout["evaluate"](
            "search", steered=True, gate_heads=heads,
            gate_bundle=holdout_raw["search"]["gate_bundle"],
            label=f"{label}/search_blockade")
        rescued_belief = holdout["evaluate"](
            "belief", gate_heads=heads,
            gate_bundle=holdout_raw[
                "belief_steered"]["gate_bundle"],
            label=f"{label}/belief_rescue")
        rescued_search = holdout["evaluate"](
            "search", gate_heads=heads,
            gate_bundle=holdout_raw[
                "search_steered"]["gate_bundle"],
            label=f"{label}/search_rescue")
        metrics = _epistasis_metrics(
            holdout_raw["belief"], holdout_raw["search"],
            holdout_raw["belief_steered"],
            holdout_raw["search_steered"],
            blocked_belief, blocked_search,
            rescued_belief, rescued_search)
        return {
            "heads": [
                {"layer": GATE_LAYER, "head": int(head)}
                for head in heads
            ],
            "routes": {
                "belief_blockade": blocked_belief["public"],
                "search_blockade": blocked_search["public"],
                "belief_rescue": rescued_belief["public"],
                "search_rescue": rescued_search["public"],
            },
            "route_scores": {
                "belief_blockade": blocked_belief["route_score"],
                "search_blockade": blocked_search["route_score"],
                "belief_rescue": rescued_belief["route_score"],
                "search_rescue": rescued_search["route_score"],
            },
            "metrics": metrics,
        }

    selected_result = None
    random_results = []
    if discovery_result["stable"]:
        selected_result = evaluate_head_set(
            tuple(selected_heads), "selected")
        random_sets = _random_head_sets(
            candidate_heads, selected_heads)
        for random_index, heads in enumerate(random_sets):
            cell = evaluate_head_set(
                heads, f"random_{random_index}")
            random_results.append({
                "random_index": random_index,
                "heads": cell["heads"],
                "route_scores": cell["route_scores"],
                "metrics": cell["metrics"],
            })
    hb.done()

    specificity = {
        "blockade_empirical_p": None,
        "blockade_exceed_count": None,
        "rescue_empirical_p": None,
        "rescue_exceed_count": None,
        "all_random_contexts_functional": False,
        "blockade_specific": False,
        "rescue_specific": False,
    }
    if selected_result is not None:
        selected_blockade = selected_result[
            "metrics"]["blockade"]["bidirectional_fraction_score"]
        selected_rescue = selected_result[
            "metrics"]["rescue"]["bidirectional_fraction_score"]
        random_blockade = [
            cell["metrics"]["blockade"][
                "bidirectional_fraction_score"]
            for cell in random_results
        ]
        random_rescue = [
            cell["metrics"]["rescue"][
                "bidirectional_fraction_score"]
            for cell in random_results
        ]
        blockade_p, blockade_exceed = _tail_probability(
            selected_blockade, random_blockade)
        rescue_p, rescue_exceed = _tail_probability(
            selected_rescue, random_rescue)
        random_functional = all(
            cell["metrics"]["all_contexts_functional"]
            for cell in random_results)
        specificity = {
            "selected_blockade_score": selected_blockade,
            "selected_rescue_score": selected_rescue,
            "blockade_empirical_p": blockade_p,
            "blockade_exceed_count": blockade_exceed,
            "rescue_empirical_p": rescue_p,
            "rescue_exceed_count": rescue_exceed,
            "all_random_contexts_functional": random_functional,
            "blockade_specific": bool(
                random_functional
                and blockade_p <= MAXIMUM_NULL_P + 1e-12),
            "rescue_specific": bool(
                random_functional
                and rescue_p <= MAXIMUM_NULL_P + 1e-12),
        }

    selected_blockade_pass = bool(
        selected_result is not None
        and selected_result["metrics"]["blockade"]["pass"])
    selected_rescue_pass = bool(
        selected_result is not None
        and selected_result["metrics"]["rescue"]["pass"])
    blockade_specific = bool(specificity["blockade_specific"])
    rescue_specific = bool(specificity["rescue_specific"])
    verdict = _adjudicate(
        discovery_result["stable"], calibration_pass, source_invariant,
        selected_blockade_pass, selected_rescue_pass,
        blockade_specific, rescue_specific)

    result = {
        "stage": "delta_controller_circuit_epistasis",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "Activation-only discovery followed by same-prompt causal "
            "blockade and rescue tests whether the low-energy controller "
            "selects the frozen downstream transport circuit through "
            "immediate layer-22 gate heads."),
        "splits": {
            "location_donor_indices": location_indices[:DONOR_N],
            "ownership_donor_rows": ownership_rows,
            "color_donor_rows": color_donor_rows,
            "discovery_rows": discovery_rows,
            "causal_holdout_rows": holdout_rows,
            "holdout_excluded_from_head_selection": True,
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
        "frozen_transport_heads": [
            {"layer": int(layer), "head": int(head)}
            for layer, head in FROZEN_TOP8
        ],
        "discovery": discovery_result,
        "holdout_calibration": {
            "tasks": {
                name: route["public"]
                for name, route in holdout_raw.items()
            },
            "route_scores": {
                name: route["route_score"]
                for name, route in holdout_raw.items()
            },
            "movement": calibration_movement,
            "functional": calibration_functional,
            "pass": calibration_pass,
            "source_anchor_invariance": {
                "maximum_absolute_difference":
                    maximum_source_difference,
                "tolerance": SOURCE_INVARIANCE_TOLERANCE,
                "pass": source_invariant,
            },
        },
        "selected_head_set": selected_result,
        "random_head_null": {
            "n_random": len(random_results),
            "seed": RANDOM_SEED,
            "cells": random_results,
        },
        "specificity": specificity,
        "prospective_gate": {
            "discovery_n": DISCOVERY_N,
            "holdout_n": HOLDOUT_N,
            "selected_heads": SELECT_K,
            "minimum_effect_fraction": MINIMUM_EFFECT_FRACTION,
            "minimum_sign_successes": MINIMUM_SIGN_SUCCESSES,
            "maximum_sign_p": MAXIMUM_SIGN_P,
            "random_head_sets": N_RANDOM,
            "maximum_null_p": MAXIMUM_NULL_P,
        },
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_controller_circuit_epistasis_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"CONTROLLER-CIRCUIT verdict={verdict} "
        f"discovery={discovery_result['stable']} "
        f"calibration={calibration_pass} "
        f"source_invariant={source_invariant} "
        f"blockade={selected_blockade_pass}/{blockade_specific} "
        f"rescue={selected_rescue_pass}/{rescue_specific} "
        f"artifact={path}")
    return result
