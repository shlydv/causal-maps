"""Frozen eligibility screen for structurally heterogeneous computations."""
from __future__ import annotations

import hashlib
import json
import os

import torch

from .delta_anchor_write import _anchor_position, _resolve
from .delta_content_cancelled_controller import PATCH_WIDTH
from .delta_cross_domain_controller import (
    _domain_rows,
    _generic_evaluate_sites,
    _public_generic_task,
)
from .delta_distributed_label_transplant import (
    _contiguous_groups,
)
from .delta_operation_handoff_depth import _full_sites
from .delta_prospective_causal_sensitivity import _lean_generic_context
from .delta_sparse_transport import _attention_geometry
from .delta_structured_workspace import (
    MARKER,
    _common_prefix,
    _marker_position,
)
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)


PROTOCOL_VERSION = "2026-07-26-p2-heterogeneous-family-screen-v1.1"
FAMILY_ORDER = (
    "private_belief",
    "latest_update",
    "key_value_lookup",
    "two_hop_pointer",
    "conditional_selection",
    "maximum_score",
    "constraint_elimination",
    "temporal_slot",
)
VALUES = (
    "red", "blue", "green", "black",
    "white", "brown", "pink", "gray",
)
N_WORLD = 15
CONTROL_LAYERS = (21, 22, 23, 24)
MINIMUM_ROUTE_GAP = 0.03
MINIMUM_READY_FAMILIES = 4


FAMILY_SPECS = {
    "private_belief": {
        "preamble": (
            "Track a person's private information separately from external "
            "changes."
        ),
        "rule": (
            "Rina's private record is the last label she personally saw. "
            "A later change while she is absent does not update her record."
        ),
        "story": (
            "Rina personally saw the primary card labelled {state}, then "
            "left. While she was absent, it changed to {d1}. A separate "
            "backup card was labelled {d2}."
        ),
        "question": (
            "What label does Rina believe the primary card has?"
        ),
        "answer_description": "color word",
    },
    "latest_update": {
        "preamble": (
            "Execute the stated updates strictly in chronological order."
        ),
        "rule": (
            "The current register value is the value written by the final "
            "update to that register."
        ),
        "story": (
            "Register ALPHA began as {d1}. It was next updated to {d2}. "
            "The final update set ALPHA to {state}."
        ),
        "question": "What is the current value of register ALPHA?",
        "answer_description": "color word",
    },
    "key_value_lookup": {
        "preamble": (
            "Read the requested key exactly and ignore values under other "
            "keys."
        ),
        "rule": "Each table key has one current value.",
        "story": (
            "The table contains ALPHA = {state}, BETA = {d1}, and "
            "GAMMA = {d2}."
        ),
        "question": "What value is stored under key ALPHA?",
        "answer_description": "color word",
    },
    "two_hop_pointer": {
        "preamble": "Follow the pointer chain exactly two steps.",
        "rule": (
            "First follow START to a node, then read the value stored at "
            "that node."
        ),
        "story": (
            "START points to NODE_K. NODE_K stores {state}. NODE_M stores "
            "{d1}. NODE_N stores {d2}."
        ),
        "question": "What value is reached by following START?",
        "answer_description": "color word",
    },
    "conditional_selection": {
        "preamble": "Apply the conditional rule before selecting a value.",
        "rule": (
            "When SWITCH is ON, report PRIMARY; when it is OFF, report "
            "BACKUP."
        ),
        "story": (
            "SWITCH is ON. PRIMARY is {state}. BACKUP is {d1}. The unused "
            "audit value is {d2}."
        ),
        "question": "Which value does the rule select?",
        "answer_description": "color word",
    },
    "maximum_score": {
        "preamble": "Compare the numerical scores and return the winner.",
        "rule": "The winning label is the one with the greatest score.",
        "story": (
            "{state} scored  nine points. {d1} scored five points. {d2} "
            "scored two points."
        ),
        "question": "Which label won?",
        "answer_description": "color word",
    },
    "constraint_elimination": {
        "preamble": (
            "Eliminate every candidate that violates either constraint."
        ),
        "rule": (
            "Exactly one label satisfies both the shape and size "
            "constraints."
        ),
        "story": (
            "{d1} fails the shape constraint. {d2} fails the size "
            "constraint. The label satisfying both constraints is {state}."
        ),
        "question": "Which label satisfies both constraints?",
        "answer_description": "color word",
    },
    "temporal_slot": {
        "preamble": (
            "Retrieve the value from the requested time slot, not the most "
            "recent value."
        ),
        "rule": "Each time slot has its own recorded label.",
        "story": (
            "At dawn the code was {state}. At noon it was {d1}. At dusk it "
            "was {d2}."
        ),
        "question": "What code was recorded at dawn?",
        "answer_description": "color word",
    },
}


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "purpose": (
        "Select behaviorally and mechanistically eligible heterogeneous "
        "families before a prospective causal-geometry experiment."),
    "family_order": list(FAMILY_ORDER),
    "family_specs": FAMILY_SPECS,
    "values": list(VALUES),
    "n_world_per_family": N_WORLD,
    "commands": {
        "belief": "BELIEF",
        "search": "X X SEARCH",
    },
    "layers": list(CONTROL_LAYERS),
    "eligibility": {
        "clean_and_counterfactual_accuracy": 0.80,
        "source_intervention_sufficient": True,
        "minimum_absolute_l24_route_gap": MINIMUM_ROUTE_GAP,
    },
    "selection": (
        "First four passing families in frozen FAMILY_ORDER. All families "
        "and all failures are reported; prompts are not revised."),
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _family_user(row, spec, command):
    story = spec["story"].format(
        state=row["state"], d1=row["d1"], d2=row["d2"])
    return (
        f"{spec['preamble']} {spec['rule']} {story} "
        f"Silently compute the requested value now. {MARKER}.\n"
        f"Question: {spec['question']} Reply with exactly {command}, one "
        f"space, and the {spec['answer_description']}. Do not add anything "
        "else."
    )


def _family_render(tok, row, spec, command):
    return tok.apply_chat_template(
        [{"role": "user",
          "content": _family_user(row, spec, command)}],
        tokenize=False, add_generation_prompt=True)


def _family_batch(tok, rows, spec, command, dev):
    texts = [
        _family_render(tok, row, spec, command)
        for row in rows
    ]
    prefixes = []
    maps = []
    markers = []
    for text in texts:
        base = tok.encode(text, add_special_tokens=False)
        continuations = {}
        for value in VALUES:
            full = tok.encode(
                text + f"{command} {value}",
                add_special_tokens=False)
            if full[:len(base)] != base:
                raise ValueError("answer contract resegmented")
            continuations[value] = full[len(base):]
        common = _common_prefix(list(continuations.values()))
        answer_map = {}
        for value, continuation in continuations.items():
            if len(continuation) <= len(common):
                raise ValueError(f"answer did not diverge for {value}")
            answer_map[value] = continuation[len(common)]
        if len(set(answer_map.values())) != len(VALUES):
            raise ValueError("answer-token ids collide")
        prefixes.append(base + common)
        maps.append(answer_map)
        markers.append(_marker_position(tok, text))
    if len({len(prefix) for prefix in prefixes}) != 1:
        raise ValueError("family batch is not length-aligned")
    if len(set(markers)) != 1:
        raise ValueError("family marker position varies")
    if any(value != maps[0] for value in maps[1:]):
        raise ValueError("family answer map varies")
    ids = torch.tensor(prefixes, dtype=torch.long, device=dev)
    return {
        "texts": texts,
        "ids": ids,
        "am": torch.ones_like(ids),
        "marker": int(markers[0]),
        "amap": maps[0],
        "values": list(VALUES),
    }


def _family_alignment(tok, dev, rows, spec):
    natural_rows = [
        {**row, "state": row["target"]}
        for row in rows
    ]
    batches = []
    reference_mask = None
    marker = None
    for history_rows in (rows, natural_rows):
        belief = _family_batch(
            tok, history_rows, spec, "BELIEF", dev)
        search = _family_batch(
            tok, history_rows, spec, "X X SEARCH", dev)
        if belief["ids"].shape != search["ids"].shape:
            raise ValueError("BELIEF/SEARCH family shapes differ")
        if belief["marker"] != search["marker"]:
            raise ValueError("BELIEF/SEARCH marker positions differ")
        difference = belief["ids"] != search["ids"]
        if not bool((difference == difference[0:1]).all()):
            raise ValueError("command difference mask varies")
        mask = difference[0].detach().cpu()
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
        raise ValueError("answer-prefix command is not final")
    _anchor_position(batches[0][0], batches[1][0])
    _anchor_position(batches[0][1], batches[1][1])
    return {
        "batches": batches,
        "marker": marker,
        "readout": readout,
        "instruction_positions": list(groups[0]),
        "answer_positions": list(groups[1]),
    }


def _failure_reasons(tasks, values):
    reasons = []
    for operation in ("belief", "search"):
        task = tasks[operation]
        if not task["eligible"]:
            reasons.append(f"{operation.upper()}_ANSWER_INELIGIBLE")
        if not task["source_intervention"]["sufficient"]:
            reasons.append(f"{operation.upper()}_SOURCE_INELIGIBLE")
    gap_magnitude = abs(float(values["belief"] - values["search"]))
    if gap_magnitude < MINIMUM_ROUTE_GAP - 1e-9:
        reasons.append("ORIGINAL_ROUTE_GAP_BELOW_0.03")
    return reasons


@torch.no_grad()
def run_delta_heterogeneous_family_screen(
        model_path, out_dir,
        model_key="qwen7b_heterogeneous_family_screen",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=15):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != N_WORLD:
        raise ValueError("v1 is frozen to exactly 15 rows per family")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(CONTROL_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("frozen screen layers are absent")
    n_heads, head_dim = _attention_geometry(model)
    l24_sites = _full_sites((22, 23, 24), n_heads)

    heartbeat = Heartbeat(
        len(FAMILY_ORDER) * 2 * 4,
        "heterogeneous_family_screen",
        every_sec=30, out_dir=out_dir)
    results = {}
    alignment_failures = {}
    rows = _domain_rows(VALUES)
    for family in FAMILY_ORDER:
        spec = FAMILY_SPECS[family]
        try:
            alignment = _family_alignment(
                tok, dev, rows, spec)
        except (AssertionError, ValueError) as exc:
            alignment_failures[family] = str(exc)
            for _operation in ("belief", "search"):
                for _phase in range(4):
                    heartbeat.step(
                        extra=f"{family}/alignment_failure")
            continue
        source = [row["source"] for row in rows]
        target = [row["target"] for row in rows]
        tasks = {}
        cells = {}
        route_values = {}
        for operation in ("belief", "search"):
            batch_index = 0 if operation == "belief" else 1
            clean = alignment["batches"][0][batch_index]
            natural = alignment["batches"][1][batch_index]
            context = _lean_generic_context(
                model, clean, natural, list(VALUES),
                source, target, CONTROL_LAYERS, head_dim)
            tasks[operation] = _public_generic_task(context)
            for phase in (
                    "behavior", "source", "route_setup"):
                heartbeat.step(
                    extra=f"{family}/{operation}/{phase}")
            cells[operation] = _generic_evaluate_sites(
                model, context, l24_sites, head_dim)
            route_values[operation] = float(
                cells[operation]["mediation"]["minimum_fraction"])
            heartbeat.step(
                extra=f"{family}/{operation}/L24")
        reasons = _failure_reasons(tasks, route_values)
        results[family] = {
            "rows": rows,
            "alignment": {
                "marker": alignment["marker"],
                "readout": alignment["readout"],
                "instruction_positions":
                    alignment["instruction_positions"],
                "answer_positions": alignment["answer_positions"],
            },
            "tasks": tasks,
            "l24_cells": cells,
            "route_values": route_values,
            "original_route_gap_signed": float(
                route_values["belief"] - route_values["search"]),
            "original_route_gap_magnitude": abs(float(
                route_values["belief"] - route_values["search"])),
            "eligible": not reasons,
            "failure_reasons": reasons,
        }
        log(
            f"FAMILY-SCREEN {family} eligible={not reasons} "
            f"signed_gap="
            f"{results[family]['original_route_gap_signed']:+.5f} "
            f"|gap|="
            f"{results[family]['original_route_gap_magnitude']:.5f} "
            f"reasons={reasons}")
    heartbeat.done()

    passing = [
        family for family in FAMILY_ORDER
        if family in results and results[family]["eligible"]
    ]
    selected = passing[:MINIMUM_READY_FAMILIES]
    if alignment_failures:
        verdict = "TOKENIZATION_OR_ALIGNMENT_FAILURE"
    elif len(passing) >= MINIMUM_READY_FAMILIES:
        verdict = "HETEROGENEOUS_FAMILY_SET_READY"
    else:
        verdict = "INSUFFICIENT_HETEROGENEOUS_FAMILIES"
    result = {
        "stage": "delta_heterogeneous_family_screen",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "A frozen one-shot screen identifies at least four structurally "
            "heterogeneous computations suitable for prospective causal "
            "geometry without prompt tuning."),
        "alignment_failures": alignment_failures,
        "results": results,
        "passing_families": passing,
        "selected_first_four": selected,
        "selection_frozen_before_screen": True,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_heterogeneous_family_screen_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"HETEROGENEOUS-FAMILY-SCREEN verdict={verdict} "
        f"passing={passing} selected={selected} artifact={path}")
    return result
