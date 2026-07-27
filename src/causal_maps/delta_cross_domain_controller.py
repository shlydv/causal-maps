"""Cross-domain transfer of the frozen content-cancelled route controller."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _anchor_position, _resolve
from .delta_content_cancelled_controller import (
    DIRECTION_SEED,
    EPS,
    N_RANDOM,
    PATCH_WIDTH,
    POSITION_SEED,
    _donor_alignment,
    _fixed_patch,
    _functional_score,
    _movement,
    _norm_matched_directions,
    _random_position_sets,
    _summary,
    _world_mediation,
    _world_movements,
)
from .delta_distributed_label_transplant import (
    _aligned_batches,
    _capture_full_l21,
    _contiguous_groups,
    _tail_probability,
)
from .delta_operation_handoff_depth import (
    CAPTURE_LAYERS,
    LAYERS,
    _full_sites,
)
from .delta_preprint_battery import _compatible_world_rows
from .delta_source_head_mediation import (
    _capture_source_heads,
    _mediation_pass,
    _run_intervention,
)
from .delta_sparse_transport import (
    G_ACC,
    RATIO_GATE,
    _attention_geometry,
    _site_value,
)
from .delta_sparse_transport_confirmation import FROZEN_TOP8
from .delta_structured_workspace import (
    MARKER,
    _common_prefix,
    _marker_position,
)
from .delta_trajectory import _ld
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-25-p2-cross-domain-controller-v1"
PROTOCOL_SHA256 = (
    "D72BE5C785C0E8169B98B5ACCB2CD155621EA3E45F82F18592CFB83B2F34112D")
PRIOR_CONTROLLER_SHA256 = (
    "C59EA1539F4F3E63B953470A0EE94CEDCDC84D288C8C1DBB49EB2C19B5C91B71")
DONOR_N = 15
TEST_N = 15

DOMAIN_SPECS = {
    "ownership": {
        "values": (
            "Alice", "Bob", "Carol", "David",
            "Emma", "Frank", "Grace", "Henry",
        ),
        "rule": (
            "A person's private ownership belief is the last assignment "
            "that person personally witnessed. Reassignments while that "
            "person is absent do not update the private belief."
        ),
        "story": (
            "Rina personally watched the sapphire badge assigned to {state}, "
            "then left. While Rina was absent, the sapphire badge was "
            "reassigned to {d1}. Separately, the bronze badge was assigned "
            "to {d2}."
        ),
        "question": (
            "According to Rina's own information, who does she believe "
            "owns the sapphire badge?"
        ),
        "answer_description": "owner name",
    },
    "color_state": {
        "values": (
            "red", "blue", "green", "black",
            "white", "brown", "pink", "gray",
        ),
        "rule": (
            "A person's private color belief is the last color that person "
            "personally observed. Changes while that person is absent do "
            "not update the private belief."
        ),
        "story": (
            "Rina personally observed the primary status lamp glowing "
            "{state}, then left. While Rina was absent, the primary lamp "
            "changed to {d1}. Separately, the backup lamp glowed {d2}."
        ),
        "question": (
            "According to Rina's own information, what color does she "
            "believe the primary status lamp is?"
        ),
        "answer_description": "color word",
    },
    "key_value": {
        "values": (
            "one", "two", "three", "four",
            "five", "six", "seven", "eight",
        ),
        "rule": (
            "A person's cached key-value memory is the last value that "
            "person personally read. Server updates after disconnection do "
            "not update that private memory."
        ),
        "story": (
            "Rina personally read key ALPHA with value {state}, then "
            "disconnected. While Rina was offline, ALPHA changed to {d1}. "
            "Separately, key BETA had value {d2}."
        ),
        "question": (
            "According to Rina's own memory, what value does she remember "
            "for key ALPHA?"
        ),
        "answer_description": "number word",
    },
}


def _domain_rows(values, n_rows=TEST_N):
    if len(values) != 8 or int(n_rows) != TEST_N:
        raise ValueError("v1 requires eight values and exactly 15 rows")
    rows = []
    for index in range(int(n_rows)):
        source_index = index % len(values)
        shift = 1 if index < len(values) else 2
        target_index = (source_index + shift) % len(values)
        source = values[source_index]
        target = values[target_index]
        remaining = [
            values[(source_index + offset) % len(values)]
            for offset in range(1, len(values) + 1)
            if values[(source_index + offset) % len(values)]
            not in (source, target)
        ]
        rows.append({
            "row_index": index,
            "source": source,
            "target": target,
            "state": source,
            "d1": remaining[0],
            "d2": remaining[1],
        })
    pairs = {(row["source"], row["target"]) for row in rows}
    if len(pairs) != TEST_N:
        raise AssertionError("cross-domain source-target pairs are not unique")
    return rows


def _natural_rows(rows):
    return [
        {**row, "state": row["target"]}
        for row in rows
    ]


def _domain_user(row, spec, command):
    story = spec["story"].format(
        state=row["state"], d1=row["d1"], d2=row["d2"])
    return (
        f"Maintain the true state and Rina's private record. {spec['rule']} "
        f"{story} Silently compute all relevant state variables now. "
        f"{MARKER}.\n"
        f"Question: {spec['question']} Reply with exactly {command}, one "
        f"space, and the {spec['answer_description']}. Do not add anything "
        "else."
    )


def _domain_render(tok, row, spec, command):
    return tok.apply_chat_template(
        [{"role": "user",
          "content": _domain_user(row, spec, command)}],
        tokenize=False, add_generation_prompt=True)


def _domain_batch(tok, rows, spec, command, dev):
    values = spec["values"]
    texts = [
        _domain_render(tok, row, spec, command)
        for row in rows
    ]
    prefixes = []
    maps = []
    markers = []
    for text in texts:
        base = tok.encode(text, add_special_tokens=False)
        continuations = {}
        for value in values:
            full = tok.encode(
                text + f"{command} {value}",
                add_special_tokens=False)
            if full[:len(base)] != base:
                raise ValueError("answer contract resegmented")
            continuations[value] = full[len(base):]
        common = _common_prefix(list(continuations.values()))
        amap = {}
        for value, continuation in continuations.items():
            if len(continuation) <= len(common):
                raise ValueError(f"answer did not diverge for {value}")
            amap[value] = continuation[len(common)]
        if len(set(amap.values())) != len(values):
            raise ValueError("answer-token ids collide")
        prefixes.append(base + common)
        maps.append(amap)
        markers.append(_marker_position(tok, text))
    if len({len(prefix) for prefix in prefixes}) != 1:
        raise ValueError("domain batch is not length-aligned")
    if len(set(markers)) != 1:
        raise ValueError("domain marker position varies by row")
    if any(amap != maps[0] for amap in maps[1:]):
        raise ValueError("domain answer map varies by row")
    ids = torch.tensor(prefixes, dtype=torch.long, device=dev)
    return {
        "texts": texts,
        "ids": ids,
        "am": torch.ones_like(ids),
        "marker": int(markers[0]),
        "amap": maps[0],
        "values": list(values),
    }


def _domain_alignment(tok, dev, rows, spec):
    natural = _natural_rows(rows)
    batches = []
    reference_mask = None
    marker = None
    for history_rows in (rows, natural):
        belief = _domain_batch(
            tok, history_rows, spec, "BELIEF", dev)
        search = _domain_batch(
            tok, history_rows, spec, "X X SEARCH", dev)
        if belief["ids"].shape != search["ids"].shape:
            raise ValueError("BELIEF/SEARCH domain shapes differ")
        if belief["marker"] != search["marker"]:
            raise ValueError("BELIEF/SEARCH marker positions differ")
        diff = belief["ids"] != search["ids"]
        if not bool((diff == diff[0:1]).all()):
            raise ValueError("command difference mask varies by row")
        mask = diff[0].detach().cpu()
        if reference_mask is not None and not torch.equal(
                mask, reference_mask):
            raise ValueError("clean/natural command masks differ")
        reference_mask = mask
        if marker is not None and marker != belief["marker"]:
            raise ValueError("clean/natural marker positions differ")
        marker = int(belief["marker"])
        batches.append((belief, search))

    positions = torch.nonzero(
        reference_mask, as_tuple=False).flatten().tolist()
    groups = _contiguous_groups(positions)
    if len(groups) != 2 or any(
            len(group) != PATCH_WIDTH for group in groups):
        raise ValueError(
            f"expected two three-token command groups, found {groups}")
    readout = int(batches[0][0]["ids"].shape[1] - 1)
    if groups[1] != list(range(
            readout - PATCH_WIDTH + 1, readout + 1)):
        raise ValueError("answer-prefix command is not the final three tokens")

    clean_belief, clean_search = batches[0]
    natural_belief, natural_search = batches[1]
    _anchor_position(clean_belief, natural_belief)
    _anchor_position(clean_search, natural_search)
    excluded = set(positions)
    candidates = []
    all_batches = (
        clean_belief, clean_search,
        natural_belief, natural_search,
    )
    for position in range(marker + 1, groups[1][0]):
        if position in excluded:
            continue
        token_ids = {
            int(batch["ids"][row_index, position])
            for batch in all_batches
            for row_index in range(batch["ids"].shape[0])
        }
        if len(token_ids) == 1:
            candidates.append(position)
    if len(candidates) < PATCH_WIDTH:
        raise ValueError("too few token-identical random-position candidates")
    return {
        "batches": batches,
        "marker": marker,
        "readout": readout,
        "differing_positions": positions,
        "instruction_positions": list(groups[0]),
        "answer_positions": list(groups[1]),
        "random_candidates": candidates,
    }


def _generic_accuracy(logits, batch, expected, values):
    pool = torch.tensor([batch["amap"][value] for value in values])
    chosen = logits[:, pool].argmax(-1)
    gold = torch.tensor([values.index(value) for value in expected])
    return float((chosen == gold).float().mean())


def _generic_cell(clean_logits, natural_logits,
                  forward_logits, reverse_logits,
                  batch, source, target, values):
    source_ids = torch.tensor([batch["amap"][value] for value in source])
    target_ids = torch.tensor([batch["amap"][value] for value in target])
    clean_ld = _ld(clean_logits, target_ids, source_ids)
    natural_ld = _ld(natural_logits, target_ids, source_ids)
    natural_rows = natural_ld - clean_ld
    natural_effect = float(natural_rows.mean())
    forward_rows = (
        _ld(forward_logits, target_ids, source_ids) - clean_ld)
    reverse_rows = (
        natural_ld - _ld(reverse_logits, target_ids, source_ids))
    forward_ratio = (
        float(forward_rows.mean()) / natural_effect
        if abs(natural_effect) > EPS else None)
    reverse_ratio = (
        float(reverse_rows.mean()) / natural_effect
        if abs(natural_effect) > EPS else None)
    lo, hi = RATIO_GATE
    sufficient = bool(
        forward_ratio is not None and reverse_ratio is not None
        and lo <= forward_ratio <= hi
        and lo <= reverse_ratio <= hi
        and _generic_accuracy(
            forward_logits, batch, target, values) >= G_ACC
        and _generic_accuracy(
            reverse_logits, batch, source, values) >= G_ACC)
    return {
        "natural_effect": natural_effect,
        "forward_ratio": forward_ratio,
        "reverse_ratio": reverse_ratio,
        "forward_target_acc": float(_generic_accuracy(
            forward_logits, batch, target, values)),
        "reverse_clean_acc": float(_generic_accuracy(
            reverse_logits, batch, source, values)),
        "forward_effect_rows": forward_rows.tolist(),
        "reverse_effect_rows": reverse_rows.tolist(),
        "natural_effect_rows": natural_rows.tolist(),
        "sufficient": sufficient,
    }


def _generic_task_context(model, clean, natural, values,
                          source, target, layers, head_dim,
                          sequence_patch=None, source_layer=21,
                          blocked_sites=FROZEN_TOP8):
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
        sequence_patch=clean_patch, source_layer=source_layer)
    natural_logits, natural_source, natural_heads = _capture_source_heads(
        model, natural["ids"], natural["am"],
        source_position, readout_position, layers,
        sequence_patch=natural_patch, source_layer=source_layer)
    eligible = bool(min(
        _generic_accuracy(clean_logits, clean, source, values),
        _generic_accuracy(natural_logits, natural, target, values),
    ) >= G_ACC)

    forward = _run_intervention(
        model, clean["ids"], clean["am"],
        source_position, natural_source, readout_position,
        (), (), head_dim, sequence_patch=clean_patch,
        source_layer=source_layer)
    reverse = _run_intervention(
        model, natural["ids"], natural["am"],
        source_position, clean_source, readout_position,
        (), (), head_dim, sequence_patch=natural_patch,
        source_layer=source_layer)
    source_cell = _generic_cell(
        clean_logits, natural_logits, forward, reverse,
        clean, source, target, values)

    forward_blocked = _run_intervention(
        model, clean["ids"], clean["am"],
        source_position, natural_source, readout_position,
        blocked_sites,
        [_site_value(clean_heads, site, head_dim)
         for site in blocked_sites],
        head_dim, sequence_patch=clean_patch,
        source_layer=source_layer)
    reverse_blocked = _run_intervention(
        model, natural["ids"], natural["am"],
        source_position, clean_source, readout_position,
        blocked_sites,
        [_site_value(natural_heads, site, head_dim)
         for site in blocked_sites],
        head_dim, sequence_patch=natural_patch,
        source_layer=source_layer)
    blocked_cell = _generic_cell(
        clean_logits, natural_logits,
        forward_blocked, reverse_blocked,
        clean, source, target, values)
    mediation = _mediation_pass(
        source_cell, blocked_cell,
        _generic_accuracy(forward_blocked, clean, source, values),
        _generic_accuracy(reverse_blocked, natural, target, values))
    return {
        "clean": clean,
        "natural": natural,
        "values": list(values),
        "source": list(source),
        "target": list(target),
        "source_position": source_position,
        "source_layer": int(source_layer),
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
        "blocked_intervention": blocked_cell,
        "mediation": mediation,
    }


def _public_generic_task(context):
    omitted = {
        "clean", "natural", "values", "source", "target",
        "clean_sequence_patch", "natural_sequence_patch",
        "clean_logits", "natural_logits",
        "clean_source", "natural_source",
        "clean_heads", "natural_heads",
    }
    return {
        key: value for key, value in context.items()
        if key not in omitted
    }


def _generic_evaluate_sites(model, context, sites, head_dim):
    values = context["values"]
    forward = _run_intervention(
        model, context["clean"]["ids"], context["clean"]["am"],
        context["source_position"], context["natural_source"],
        context["readout_position"], sites,
        [_site_value(context["clean_heads"], site, head_dim)
         for site in sites],
        head_dim,
        sequence_patch=context["clean_sequence_patch"],
        source_layer=context["source_layer"])
    reverse = _run_intervention(
        model, context["natural"]["ids"], context["natural"]["am"],
        context["source_position"], context["clean_source"],
        context["readout_position"], sites,
        [_site_value(context["natural_heads"], site, head_dim)
         for site in sites],
        head_dim,
        sequence_patch=context["natural_sequence_patch"],
        source_layer=context["source_layer"])
    blocked = _generic_cell(
        context["clean_logits"], context["natural_logits"],
        forward, reverse, context["clean"],
        context["source"], context["target"], values)
    mediation = _mediation_pass(
        context["source_intervention"], blocked,
        _generic_accuracy(
            forward, context["clean"], context["source"], values),
        _generic_accuracy(
            reverse, context["natural"], context["target"], values))
    return {
        "layers": sorted({layer for layer, _head in sites}),
        "n_sites": len(sites),
        "blocked_intervention": blocked,
        "mediation": mediation,
    }


def _generic_curve(model, context, n_heads, head_dim):
    return {
        str(layer): _generic_evaluate_sites(
            model, context,
            _full_sites(
                tuple(value for value in LAYERS if value <= layer),
                n_heads),
            head_dim)
        for layer in LAYERS
    }


def _domain_verdict(tasks, summaries, movement, world_movement):
    names = (
        "belief_original", "search_original",
        "belief_to_search", "search_to_belief",
    )
    if not all(tasks[name]["eligible"] for name in names):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in names):
        return "SOURCE_SITE_INELIGIBLE"
    if any(summaries[name]["first_passing_prefix"] is None
           for name in names):
        return "DEPTH_UNRESOLVED"
    if movement["original_gap"] < 0.03 - 1e-9:
        return "ORIGINAL_GAP_ABSENT"
    down = movement["belief_to_search_pass"]
    up = movement["search_to_belief_pass"]
    if down != up:
        return "ASYMMETRIC_CROSS_DOMAIN_EFFECT"
    if not (down and up):
        return "NO_CROSS_DOMAIN_TRANSFER"
    if not world_movement["all_predicted_sign"]:
        return "NONUNIFORM_CROSS_DOMAIN_EFFECT"
    categorical = bool(
        summaries["belief_to_search"]["first_passing_prefix"]
        > summaries["belief_original"]["first_passing_prefix"]
        or summaries["search_to_belief"]["first_passing_prefix"]
        < summaries["search_original"]["first_passing_prefix"])
    return (
        "CROSS_DOMAIN_ROUTE_SWITCH"
        if categorical else "CONTINUOUS_CROSS_DOMAIN_TRANSFER")


def _original_domain_eligible(tasks, summaries, movement):
    names = ("belief_original", "search_original")
    return bool(
        all(tasks[name]["eligible"]
            and tasks[name]["source_intervention"]["sufficient"]
            for name in names)
        and all(summaries[name]["first_passing_prefix"] is not None
                for name in names)
        and movement["original_gap"] >= 0.03 - 1e-9)


def _second_largest(cells):
    scores = sorted(
        (float(cell["functional_bidirectional_score"])
         for cell in cells.values()),
        reverse=True)
    return scores[1] if len(scores) >= 2 else -1e9


def _overall_verdict(domains, instruction, direction_null, position_null):
    evaluation_eligible = [
        name for name, result in domains.items()
        if result["original_evaluation_eligible"]
    ]
    if len(evaluation_eligible) < 2:
        return "CROSS_DOMAIN_BEHAVIORALLY_INELIGIBLE"
    passing = [
        name for name, result in domains.items()
        if result["verdict"] in (
            "CROSS_DOMAIN_ROUTE_SWITCH",
            "CONTINUOUS_CROSS_DOMAIN_TRANSFER",
        )
    ]
    if len(passing) == 0:
        return "NO_CROSS_DOMAIN_TRANSFER"
    if len(passing) == 1:
        return "SINGLE_DOMAIN_TRANSFER"
    if direction_null["empirical_p"] > 0.05 + 1e-12:
        return "NONSPECIFIC_RANDOM_DIRECTION"
    if position_null["empirical_p"] > 0.05 + 1e-12:
        return "NONSPECIFIC_POSITION_EFFECT"
    selected = sorted(
        (float(domains[name]["functional_bidirectional_score"])
         for name in domains),
        reverse=True)[1]
    if (instruction["generalization_score"]
            >= 0.5 * selected - 1e-12):
        return "INSTRUCTION_LOCUS_EFFECT"
    return (
        "UNIVERSAL_CROSS_DOMAIN_ROUTE_CONTROLLER"
        if len(passing) == 3
        else "MULTI_DOMAIN_ROUTE_CONTROLLER")


@torch.no_grad()
def run_delta_cross_domain_controller(
        model_path, out_dir,
        model_key="qwen7b_cross_domain_controller",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != DONOR_N + TEST_N:
        raise ValueError("v1 is frozen to exactly 30 location worlds")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    location_rows, location_indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(location_rows) != DONOR_N + TEST_N:
        raise ValueError("v1 requires all 30 compatible location worlds")
    if max(LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("cross-domain route layers are absent")
    n_heads, head_dim = _attention_geometry(model)

    donor_pairs, _diff, donor_groups, _candidates, _marker, _readout = (
        _aligned_batches(tok, dev, location_rows[:DONOR_N]))
    answer_positions = list(donor_groups[1])
    if len(answer_positions) != PATCH_WIDTH:
        raise ValueError("location donor answer-prefix width changed")
    donor_samples = []
    for belief_batch, search_batch in donor_pairs:
        belief = _capture_full_l21(model, belief_batch)
        search = _capture_full_l21(model, search_batch)
        donor_samples.extend(
            belief[:, answer_positions, :]
            - search[:, answer_positions, :])
    donor_sample_tensor = torch.stack(donor_samples)
    displacement = donor_sample_tensor.mean(dim=0)
    random_directions = _norm_matched_directions(displacement)

    controller_path = os.path.join(
        out_dir, f"cross_domain_source_controller_{model_key}.npy")
    np.save(controller_path, displacement.numpy())
    with open(controller_path, "rb") as handle:
        controller_sha = hashlib.sha256(
            handle.read()).hexdigest().upper()

    prepared = {}
    alignment_errors = {}
    for domain_index, (name, spec) in enumerate(DOMAIN_SPECS.items()):
        rows = _domain_rows(spec["values"])
        try:
            alignment = _domain_alignment(tok, dev, rows, spec)
        except (AssertionError, ValueError) as exc:
            alignment_errors[name] = str(exc)
            continue
        states = {"belief": [], "search": []}
        for belief_batch, search_batch in alignment["batches"]:
            states["belief"].append(
                _capture_full_l21(model, belief_batch))
            states["search"].append(
                _capture_full_l21(model, search_batch))
        prepared[name] = {
            "domain_index": domain_index,
            "spec": spec,
            "rows": rows,
            "natural_rows": _natural_rows(rows),
            "alignment": alignment,
            "states": states,
            "random_positions": _random_position_sets(
                alignment["random_candidates"],
                seed=POSITION_SEED + 1009 * domain_index),
        }

    total_steps = (
        len(prepared) * 4 * (3 + len(LAYERS))
        + len(prepared) * 2 * 4
        + N_RANDOM * len(prepared) * 2 * 4
        + N_RANDOM * len(prepared) * 2 * 4)
    hb = Heartbeat(
        total_steps, "cross_domain_controller",
        every_sec=30, out_dir=out_dir)

    domain_results = {}
    retained = {}
    for name, data in prepared.items():
        spec = data["spec"]
        alignment = data["alignment"]
        source = [row["source"] for row in data["rows"]]
        target = [row["target"] for row in data["rows"]]
        clean_belief, clean_search = alignment["batches"][0]
        natural_belief, natural_search = alignment["batches"][1]
        selected_specs = {
            "belief_original": (
                clean_belief, natural_belief, None),
            "search_original": (
                clean_search, natural_search, None),
            "belief_to_search": (
                clean_belief, natural_belief,
                _fixed_patch(
                    data["states"]["belief"],
                    alignment["answer_positions"],
                    displacement, -1.0)),
            "search_to_belief": (
                clean_search, natural_search,
                _fixed_patch(
                    data["states"]["search"],
                    alignment["answer_positions"],
                    displacement, +1.0)),
        }
        tasks = {}
        curves = {}
        summaries = {}
        for task_name, (clean, natural, patch) in selected_specs.items():
            context = _generic_task_context(
                model, clean, natural, list(spec["values"]),
                source, target, CAPTURE_LAYERS, head_dim,
                sequence_patch=patch)
            tasks[task_name] = _public_generic_task(context)
            hb.step(extra=f"{name}/{task_name}/baseline")
            hb.step(extra=f"{name}/{task_name}/source")
            hb.step(extra=f"{name}/{task_name}/base_path")
            curves[task_name] = {}
            for layer, cell in _generic_curve(
                    model, context, n_heads, head_dim).items():
                curves[task_name][layer] = cell
                hb.step(extra=f"{name}/{task_name}/prefixL{layer}")

        for task_name, curve in curves.items():
            summaries[task_name] = _summary(curve)
        original = {
            "belief": summaries["belief_original"],
            "search": summaries["search_original"],
        }
        movement = _movement(
            original,
            summaries["belief_to_search"]["l24_minimum_mediation"],
            summaries["search_to_belief"]["l24_minimum_mediation"])
        world_movement = _world_movements(
            _world_mediation(
                tasks["belief_original"],
                curves["belief_original"]["24"]),
            _world_mediation(
                tasks["search_original"],
                curves["search_original"]["24"]),
            _world_mediation(
                tasks["belief_to_search"],
                curves["belief_to_search"]["24"]),
            _world_mediation(
                tasks["search_to_belief"],
                curves["search_to_belief"]["24"]))
        domain_verdict = _domain_verdict(
            tasks, summaries, movement, world_movement)
        functional = _functional_score(
            movement, [tasks[key] for key in selected_specs])
        domain_results[name] = {
            "values": list(spec["values"]),
            "rows": data["rows"],
            "alignment": {
                key: value for key, value in alignment.items()
                if key != "batches"
            },
            "tasks": tasks,
            "cumulative_prefix": curves,
            "selected_summaries": summaries,
            "primary_movement": movement,
            "per_world_movement": world_movement,
            "original_evaluation_eligible": _original_domain_eligible(
                tasks, summaries, movement),
            "functional": functional["functional"],
            "functional_bidirectional_score": (
                functional["functional_bidirectional_score"]),
            "verdict": domain_verdict,
        }
        retained[name] = {
            "data": data,
            "source": source,
            "target": target,
            "original": original,
        }
        log(
            f"CROSS DOMAIN selected {name} verdict={domain_verdict} "
            f"movement={movement} "
            f"worlds={world_movement['all_predicted_sign']}")

    control_layers = (21, 22, 23, 24)
    l24_sites = _full_sites((22, 23, 24), n_heads)

    def evaluate_control(domain_name, label, control_displacement, positions):
        record = retained[domain_name]
        data = record["data"]
        spec = data["spec"]
        alignment = data["alignment"]
        clean_belief, clean_search = alignment["batches"][0]
        natural_belief, natural_search = alignment["batches"][1]
        values = {}
        public = []
        for direction, clean, natural, base_states, sign in (
                ("belief_to_search", clean_belief, natural_belief,
                 data["states"]["belief"], -1.0),
                ("search_to_belief", clean_search, natural_search,
                 data["states"]["search"], +1.0)):
            patch = _fixed_patch(
                base_states, positions, control_displacement, sign)
            context = _generic_task_context(
                model, clean, natural, list(spec["values"]),
                record["source"], record["target"],
                control_layers, head_dim, sequence_patch=patch)
            task = _public_generic_task(context)
            public.append(task)
            hb.step(extra=(
                f"{label}/{domain_name}/{direction}/baseline"))
            hb.step(extra=f"{label}/{domain_name}/{direction}/source")
            hb.step(extra=f"{label}/{domain_name}/{direction}/base_path")
            cell = _generic_evaluate_sites(
                model, context, l24_sites, head_dim)
            values[direction] = float(
                cell["mediation"]["minimum_fraction"])
            hb.step(extra=f"{label}/{domain_name}/{direction}/L24")
        return _functional_score(
            _movement(
                record["original"],
                values["belief_to_search"],
                values["search_to_belief"]),
            public)

    instruction_cells = {}
    for name, record in retained.items():
        instruction_cells[name] = evaluate_control(
            name, "instruction", displacement,
            record["data"]["alignment"]["instruction_positions"])
    instruction_control = {
        "cells": instruction_cells,
        "generalization_score": _second_largest(instruction_cells),
    }

    direction_cells = []
    for random_index, random_direction in enumerate(random_directions):
        cells = {
            name: evaluate_control(
                name, f"random_direction{random_index}",
                random_direction,
                record["data"]["alignment"]["answer_positions"])
            for name, record in retained.items()
        }
        direction_cells.append({
            "random_index": random_index,
            "per_position_norms": random_direction.norm(
                dim=-1).tolist(),
            "domain_cells": cells,
            "generalization_score": _second_largest(cells),
        })

    position_cells = []
    for random_index in range(N_RANDOM):
        cells = {}
        positions = {}
        for name, record in retained.items():
            selected_positions = record["data"][
                "random_positions"][random_index]
            positions[name] = selected_positions
            cells[name] = evaluate_control(
                name, f"random_position{random_index}",
                displacement, selected_positions)
        position_cells.append({
            "random_index": random_index,
            "positions": positions,
            "domain_cells": cells,
            "generalization_score": _second_largest(cells),
        })
    hb.done()

    selected_score = _second_largest({
        name: {
            "functional_bidirectional_score":
                result["functional_bidirectional_score"]
        }
        for name, result in domain_results.items()
    })
    direction_scores = [
        cell["generalization_score"] for cell in direction_cells
    ]
    direction_p, direction_exceed = _tail_probability(
        selected_score, direction_scores)
    position_scores = [
        cell["generalization_score"] for cell in position_cells
    ]
    position_p, position_exceed = _tail_probability(
        selected_score, position_scores)
    direction_null = {
        "n_random": N_RANDOM,
        "seed": DIRECTION_SEED,
        "selected_generalization_score": selected_score,
        "empirical_p": direction_p,
        "exceed_count": direction_exceed,
        "cells": direction_cells,
    }
    position_null = {
        "n_random": N_RANDOM,
        "domain_seeds": {
            name: POSITION_SEED + 1009 * data["domain_index"]
            for name, data in prepared.items()
        },
        "selected_generalization_score": selected_score,
        "empirical_p": position_p,
        "exceed_count": position_exceed,
        "cells": position_cells,
    }
    overall = _overall_verdict(
        domain_results, instruction_control,
        direction_null, position_null)
    result = {
        "stage": "delta_cross_domain_controller",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "A location-derived, content-cancelled answer-prefix "
            "controller causally reconfigures access routes in new state "
            "domains without target-domain fitting."),
        "location_donor": {
            "indices_from_30": location_indices[:DONOR_N],
            "answer_prefix_positions": answer_positions,
        },
        "controller": {
            "construction": (
                "mean(BELIEF-SEARCH) over location donor worlds 0-14 "
                "and clean/natural arms"),
            "shape": list(displacement.shape),
            "frobenius_norm": float(displacement.norm()),
            "per_position_norms": displacement.norm(dim=-1).tolist(),
            "artifact": os.path.basename(controller_path),
            "artifact_sha256": controller_sha,
            "prior_artifact_sha256": PRIOR_CONTROLLER_SHA256,
            "bitwise_matches_prior": bool(
                controller_sha == PRIOR_CONTROLLER_SHA256),
            "donor_alignment": _donor_alignment(
                donor_sample_tensor, displacement),
        },
        "alignment_errors": alignment_errors,
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "domains": domain_results,
        "selected_generalization_score": selected_score,
        "instruction_control": instruction_control,
        "random_direction_control": direction_null,
        "random_position_control": position_null,
        "verdict": overall,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_cross_domain_controller_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    domain_verdicts = {
        name: value["verdict"]
        for name, value in domain_results.items()
    }
    log(
        f"CROSS DOMAIN CONTROLLER verdict={overall} "
        f"domains={domain_verdicts} "
        f"score={selected_score} p_direction={direction_p} "
        f"p_position={position_p} artifact={path}")
    return result
