"""Held-out screen for sparse backup heads conditional on the frozen path."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_preprint_battery import _compatible_world_rows
from .delta_source_head_mediation import (
    TASKS,
    _mediation_pass,
    _public_task,
    _run_intervention,
    _task_context,
)
from .delta_sparse_transport import (
    _attention_geometry,
    _cell,
    _fixed_split,
    _site_value,
)
from .delta_sparse_transport_confirmation import FROZEN_TOP8
from .delta_structured_workspace import _accuracy
from .delta_trajectory import _ld
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-conditional-backup-v1"
PROTOCOL_SHA256 = (
    "8647A535E8E9597C8EA68A6E52FF47369414795CA3C8B1A70F8CA80191C022C6")
BACKUP_LAYERS = (22, 23, 24)
CAPTURE_LAYERS = (21, 22, 23, 24)
TOP_K = (1, 2, 4, 8)
N_RANDOM = 9
RANDOM_SEED = 7759
EPS = 1e-8


def _candidate_sites(n_heads):
    frozen = set(FROZEN_TOP8)
    return [
        (layer, head)
        for layer in BACKUP_LAYERS
        for head in range(int(n_heads))
        if (layer, head) not in frozen
    ]


def _forward_ratio(context, logits):
    source_ids = torch.tensor([
        context["clean"]["amap"][x] for x in context["source"]])
    target_ids = torch.tensor([
        context["clean"]["amap"][x] for x in context["target"]])
    clean_ld = _ld(context["clean_logits"], target_ids, source_ids)
    patched_ld = _ld(logits, target_ids, source_ids)
    natural_effect = context["source_intervention"]["natural_effect"]
    if abs(float(natural_effect)) <= EPS:
        return None
    return float((patched_ld - clean_ld).mean()) / float(natural_effect)


def _evaluate_complement(model, context, extra_sites, head_dim):
    sites = tuple(FROZEN_TOP8) + tuple(extra_sites)
    forward = _run_intervention(
        model, context["clean"]["ids"], context["clean"]["am"],
        context["source_position"], context["natural_source"],
        context["readout_position"], sites,
        [_site_value(context["clean_heads"], site, head_dim)
         for site in sites], head_dim)
    reverse = _run_intervention(
        model, context["natural"]["ids"], context["natural"]["am"],
        context["source_position"], context["clean_source"],
        context["readout_position"], sites,
        [_site_value(context["natural_heads"], site, head_dim)
         for site in sites], head_dim)
    blocked = _cell(
        context["clean_logits"], context["natural_logits"],
        forward, reverse, context["clean"],
        context["source"], context["target"])
    mediation = _mediation_pass(
        context["source_intervention"], blocked,
        _accuracy(forward, context["clean"], context["source"]),
        _accuracy(reverse, context["natural"], context["target"]))
    return {
        "extra_sites": [
            {"layer": layer, "head": head}
            for layer, head in extra_sites],
        "blocked_intervention": blocked,
        "mediation": mediation,
    }


def _verdict(eval_tasks, random_cells, transfer):
    if not all(
            eval_tasks[name]["eligible"]
            and eval_tasks[name]["source_intervention"]["sufficient"]
            for name in TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    if any(
            cell["mediation"]["pass"]
            for cells in random_cells.values()
            for cell in cells):
        return "NONSPECIFIC_BACKUP"
    diagonal = [
        bool(transfer[name][name]["mediation"]["pass"])
        for name in TASKS
    ]
    shared = any(
        all(transfer[donor][target]["mediation"]["pass"]
            for target in TASKS)
        for donor in TASKS
    )
    if shared:
        return "SHARED_SPARSE_COMPLEMENT"
    if all(diagonal):
        return "SPARSE_QUERY_COMPLEMENTS"
    if any(diagonal):
        return "PARTIAL_BACKUP_LOCALIZATION"
    return "RESIDUAL_ROUTE_DISTRIBUTED_OR_OUTSIDE_HEAD_OUTPUTS"


@torch.no_grad()
def run_delta_conditional_backup(
        model_path, out_dir,
        model_key="qwen7b_conditional_backup",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30, n_donor=15, n_random=N_RANDOM,
        random_seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_random) != N_RANDOM:
        raise ValueError("v1 is frozen to nine random complements")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != 30:
        raise ValueError("v1 requires all 30 compatible worlds")
    donor_rows, eval_rows = _fixed_split(rows, int(n_donor))
    if max(BACKUP_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("backup layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)
    candidates = _candidate_sites(n_heads)
    expected = len(BACKUP_LAYERS) * n_heads - len(FROZEN_TOP8)
    if len(candidates) != expected:
        raise RuntimeError("candidate count does not match frozen geometry")

    total = len(TASKS) * (3 + len(candidates) + 3 + len(TOP_K))
    total += len(TASKS) * int(n_random)
    total += len(TASKS) * (len(TASKS) - 1)
    hb = Heartbeat(
        total, "conditional_backup", every_sec=30, out_dir=out_dir)

    discovery_contexts = {}
    rankings = {}
    discovery = {}
    for query in TASKS:
        context = _task_context(
            model, tok, dev, donor_rows, query,
            CAPTURE_LAYERS, head_dim)
        discovery_contexts[query] = context
        hb.step(extra=f"{query}/discovery_baseline")
        hb.step(extra=f"{query}/discovery_source")
        hb.step(extra=f"{query}/discovery_base_block")
        base_ratio = context["blocked_intervention"]["forward_ratio"]
        scores = []
        for site in candidates:
            sites = tuple(FROZEN_TOP8) + (site,)
            forward = _run_intervention(
                model, context["clean"]["ids"], context["clean"]["am"],
                context["source_position"], context["natural_source"],
                context["readout_position"], sites,
                [_site_value(context["clean_heads"], s, head_dim)
                 for s in sites], head_dim)
            ratio = _forward_ratio(context, forward)
            reduction = (
                abs(float(base_ratio)) - abs(float(ratio))
                if base_ratio is not None and ratio is not None
                else float("-inf"))
            score = {
                "layer": site[0],
                "head": site[1],
                "forward_residual_ratio": ratio,
                "absolute_residual_reduction": reduction,
                "origin_accuracy": float(_accuracy(
                    forward, context["clean"], context["source"])),
            }
            scores.append(score)
            hb.step(extra=(
                f"{query}/L{site[0]}H{site[1]} "
                f"reduce={reduction:.3f}"))
        scores.sort(
            key=lambda item: item["absolute_residual_reduction"],
            reverse=True)
        discovery[query] = scores
        rankings[query] = [
            (item["layer"], item["head"]) for item in scores]

    eval_contexts = {}
    eval_tasks = {}
    nested = {}
    for query in TASKS:
        context = _task_context(
            model, tok, dev, eval_rows, query,
            CAPTURE_LAYERS, head_dim)
        eval_contexts[query] = context
        eval_tasks[query] = _public_task(context)
        hb.step(extra=f"{query}/eval_baseline")
        hb.step(extra=f"{query}/eval_source")
        hb.step(extra=f"{query}/eval_base_block")
        nested[query] = {}
        for k in TOP_K:
            cell = _evaluate_complement(
                model, context, rankings[query][:k], head_dim)
            nested[query][str(k)] = cell
            hb.step(extra=(
                f"{query}/top{k}={cell['mediation']['pass']}"))

    rng = np.random.default_rng(int(random_seed))
    random_cells = {}
    for query in TASKS:
        selected = set(rankings[query][:max(TOP_K)])
        pool = [site for site in candidates if site not in selected]
        random_cells[query] = []
        for random_index in range(int(n_random)):
            choice = rng.choice(len(pool), size=max(TOP_K), replace=False)
            sites = [pool[int(i)] for i in choice]
            cell = _evaluate_complement(
                model, eval_contexts[query], sites, head_dim)
            random_cells[query].append({
                "random_index": random_index,
                **cell,
            })
            hb.step(extra=f"{query}/random{random_index}")

    transfer = {donor: {} for donor in TASKS}
    for donor in TASKS:
        extra = rankings[donor][:max(TOP_K)]
        for target in TASKS:
            if donor == target:
                cell = nested[target][str(max(TOP_K))]
            else:
                cell = _evaluate_complement(
                    model, eval_contexts[target], extra, head_dim)
                hb.step(extra=f"{donor}->{target}")
            transfer[donor][target] = cell
    hb.done()

    verdict = _verdict(eval_tasks, random_cells, transfer)
    result = {
        "stage": "delta_conditional_backup",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "n_discovery": len(donor_rows),
            "n_evaluation": len(eval_rows),
        },
        "backup_layers": list(BACKUP_LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_candidates": len(candidates),
        "frozen_top8": [
            {"layer": layer, "head": head}
            for layer, head in FROZEN_TOP8],
        "discovery": discovery,
        "evaluation_tasks": eval_tasks,
        "nested_complements": nested,
        "random_complements": random_cells,
        "cross_query_transfer": transfer,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_conditional_backup_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"CONDITIONAL BACKUP verdict={verdict} artifact={path}")
    return result
