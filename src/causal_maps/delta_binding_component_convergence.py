"""Held-out L8 component-convergence test for the affine binding operator.

Frozen design: BINDING_COMPONENT_CONVERGENCE_PROTOCOL.md.

The question is deliberately local: after a natural textual rewrite and an
affine L2 ADD have reached the previously validated L8 mediator, do the same
small set of L8 attention-head / MLP outputs at the final query token matter
for both effects?  This module does not claim to recover the full circuit.
"""
import json
import math
import os

import numpy as np
import torch

from .delta_operator import (
    DONOR_NAMES, INJECT_LAYER, MEDIATION_LAYER, _build_multi_group,
    _directions, _encode_uniform, _single_text, _trials, _values)
from .delta_trajectory import EPS, _cos_rows, _forward, _ld
from .logutil import log
from .model_utils import (
    get_decoder_layers, input_device, load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import _split_output

N_NULL = 100
TOP_K = 4
HEAD_KIND = "head"
MLP_KIND = "mlp"
PROTOCOL_VERSION = "2026-07-13-v1"


def _component_name(component):
    kind, index = component
    return "L8_MLP" if kind == MLP_KIND else f"L8_H{index}"


def _component_json(components):
    return [
        {"kind": kind, "head": index, "name": _component_name((kind, index))}
        for kind, index in components
    ]


def _split_trials(rows):
    """Split offsets 1/3 vs 5/7 within every source value.

    `_trials` emits four rows per source, alternating queried X and Y.  This
    split leaves ten discovery and ten held-out rows for each query role.
    """
    if len(rows) != 40 or len(rows) % 4:
        raise ValueError(f"expected 40 deterministic trials, got {len(rows)}")
    discovery, test = [], []
    for start in range(0, len(rows), 4):
        discovery.extend(rows[start:start + 2])
        test.extend(rows[start + 2:start + 4])
    for split_name, split in (("discovery", discovery), ("test", test)):
        counts = {query: sum(row["query"] == query for row in split)
                  for query in ("X", "Y")}
        if counts != {"X": 10, "Y": 10}:
            raise ValueError(f"{split_name} query split invalid: {counts}")
    return discovery, test


def _candidate_components(num_heads):
    return [(HEAD_KIND, head) for head in range(num_heads)] + [(MLP_KIND, None)]


def _top_components(scores, candidates, k=TOP_K):
    if len(scores) != len(candidates):
        raise ValueError("score/candidate length mismatch")
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return tuple(candidates[i] for i in order[:k])


def _jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 1.0


def _matched_random_sets(reference, num_heads, seed, n_null=N_NULL):
    """Sample size/type-matched head sets, preserving MLP inclusion."""
    n_head = sum(kind == HEAD_KIND for kind, _ in reference)
    has_mlp = any(kind == MLP_KIND for kind, _ in reference)
    if n_head + int(has_mlp) != len(reference):
        raise ValueError("unknown component kind")
    available = math.comb(num_heads, n_head) - 1
    if n_null > available:
        raise ValueError(
            f"need {n_null} matched random sets, but only {available} exist")
    rng = np.random.default_rng(seed)
    reference = tuple(sorted(reference))
    found = set()
    while len(found) < n_null:
        heads = tuple(sorted(int(item) for item in rng.choice(
            num_heads, size=n_head, replace=False)))
        candidate = tuple((HEAD_KIND, head) for head in heads)
        if has_mlp:
            candidate += ((MLP_KIND, None),)
        candidate = tuple(sorted(candidate))
        if candidate != reference:
            found.add(candidate)
    return sorted(found)


@torch.no_grad()
def _forward_components(model, input_ids, attention_mask, inject_position,
                        direction=None, components=(), component_position=None,
                        capture_positions=()):
    """Run one condition with an optional L2 ADD and L8 component ablation.

    Attention-head ablation zeros its pre-`o_proj` channel slice at one token.
    The MLP candidate zeros its output at that same token.  These are output
    component interventions: they do not alter attention patterns, Q/K, or
    upstream component computation.
    """
    layers = get_decoder_layers(model)
    if component_position is None:
        component_position = int(input_ids.shape[1] - 1)
    num_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    if hidden_size % num_heads:
        raise ValueError("hidden size is not divisible by number of heads")
    head_dim = hidden_size // num_heads
    heads = tuple(sorted(index for kind, index in components if kind == HEAD_KIND))
    use_mlp = any(kind == MLP_KIND for kind, _ in components)
    if any(index is None or not 0 <= index < num_heads for index in heads):
        raise ValueError(f"invalid attention head set: {heads}")
    if any(kind not in (HEAD_KIND, MLP_KIND) for kind, _ in components):
        raise ValueError(f"unknown component set: {components}")

    handles, cache = [], {}
    if direction is not None:
        def add_hook(_module, _inputs, output):
            hidden, rebuild = _split_output(output)
            edited = hidden.clone()
            delta = direction.to(device=edited.device, dtype=edited.dtype)
            if delta.ndim == 1:
                delta = delta.unsqueeze(0).expand(edited.shape[0], -1)
            if delta.shape != (edited.shape[0], edited.shape[-1]):
                raise ValueError(
                    f"direction shape {tuple(delta.shape)} incompatible with "
                    f"hidden state {tuple(edited.shape)}")
            edited[:, inject_position, :] += delta
            return rebuild(edited)
        handles.append(layers[INJECT_LAYER].register_forward_hook(add_hook))

    if heads:
        def head_pre_hook(_module, args):
            if not args:
                raise ValueError("o_proj did not receive hidden-state positional arg")
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[-1] != hidden_size:
                raise ValueError(f"unexpected o_proj input {tuple(hidden.shape)}")
            edited = hidden.clone()
            for head in heads:
                start, end = head * head_dim, (head + 1) * head_dim
                edited[:, component_position, start:end] = 0
            return (edited,) + tuple(args[1:])
        handles.append(
            layers[MEDIATION_LAYER].self_attn.o_proj.register_forward_pre_hook(
                head_pre_hook))

    if use_mlp:
        def mlp_hook(_module, _inputs, output):
            hidden, rebuild = _split_output(output)
            edited = hidden.clone()
            edited[:, component_position, :] = 0
            return rebuild(edited)
        handles.append(layers[MEDIATION_LAYER].mlp.register_forward_hook(mlp_hook))

    if capture_positions:
        def capture_hook(_module, _inputs, output):
            hidden, _ = _split_output(output)
            cache[MEDIATION_LAYER] = (
                hidden[:, capture_positions, :].detach().float().cpu())
        handles.append(layers[MEDIATION_LAYER].register_forward_hook(capture_hook))

    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False)
        logits = output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits, cache


def _group_data(model, tok, rows, prototypes, dev):
    """Prepare the two queried-slot groups and their affine operators."""
    groups = []
    for query in ("X", "Y"):
        selected, ci, cam, fi, fam, own_pos, _ = _build_multi_group(
            tok, rows, query)
        ci, cam = ci.to(dev), cam.to(dev)
        fi, fam = fi.to(dev), fam.to(dev)
        pos_ids = torch.tensor(
            [single_token_id(tok, row["target"]) for row in selected])
        neg_ids = torch.tensor(
            [single_token_id(tok, row["source"]) for row in selected])
        groups.append({
            "query": query,
            "rows": selected,
            "clean_ids": ci,
            "clean_am": cam,
            "natural_ids": fi,
            "natural_am": fam,
            "slot": int(own_pos),
            "last": int(ci.shape[1] - 1),
            "direction": _directions(prototypes, selected),
            "pos_ids": pos_ids,
            "neg_ids": neg_ids,
        })
    return groups


def _condition_effect(clean_logits, changed_logits, pos_ids, neg_ids):
    return _ld(changed_logits, pos_ids, neg_ids) - _ld(
        clean_logits, pos_ids, neg_ids)


@torch.no_grad()
def _measure(model, groups, components=(), component_position_kind="last",
             capture=False):
    """Measure natural and ADD effects with CLEAN re-run under same ablation."""
    natural_rows, add_rows = [], []
    clean_hits, natural_hits = [], []
    add_positive = []
    own_cos, last_cos = [], []
    group_records = []
    for group in groups:
        component_position = (
            group["last"] if component_position_kind == "last"
            else group["last"] - 1)
        capture_positions = (group["slot"], group["last"]) if capture else ()
        clean_logits, clean_cache = _forward_components(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            components=components, component_position=component_position,
            capture_positions=capture_positions)
        natural_logits, natural_cache = _forward_components(
            model, group["natural_ids"], group["natural_am"], group["slot"],
            components=components, component_position=component_position,
            capture_positions=capture_positions)
        add_logits, add_cache = _forward_components(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            direction=group["direction"], components=components,
            component_position=component_position,
            capture_positions=capture_positions)
        natural_effect = _condition_effect(
            clean_logits, natural_logits, group["pos_ids"], group["neg_ids"])
        add_effect = _condition_effect(
            clean_logits, add_logits, group["pos_ids"], group["neg_ids"])
        natural_rows.extend(natural_effect.tolist())
        add_rows.extend(add_effect.tolist())
        clean_hits.extend((clean_logits.argmax(-1) == group["neg_ids"]).tolist())
        natural_hits.extend(
            (natural_logits.argmax(-1) == group["pos_ids"]).tolist())
        add_positive.extend((add_effect > 0).tolist())
        record = {
            "query": group["query"],
            "n": len(group["rows"]),
            "natural_effect": float(natural_effect.mean()),
            "add_effect": float(add_effect.mean()),
        }
        if capture:
            clean_state = clean_cache[MEDIATION_LAYER]
            natural_state = natural_cache[MEDIATION_LAYER]
            add_state = add_cache[MEDIATION_LAYER]
            for index, name in enumerate(("slot", "last")):
                native = natural_state[:, index] - clean_state[:, index]
                induced = add_state[:, index] - clean_state[:, index]
                cosine = _cos_rows(induced, native)
                (own_cos if name == "slot" else last_cos).extend(cosine.tolist())
                record[f"l8_{name}_cos"] = float(cosine.mean())
        group_records.append(record)
    natural_mean = float(np.mean(natural_rows))
    add_mean = float(np.mean(add_rows))
    return {
        "natural_rows": natural_rows,
        "add_rows": add_rows,
        "natural_effect": natural_mean,
        "add_effect": add_mean,
        "clean_acc": float(np.mean(clean_hits)),
        "natural_acc": float(np.mean(natural_hits)),
        "add_positive_fraction": float(np.mean(add_positive)),
        "l8_slot_cos": float(np.mean(own_cos)) if own_cos else None,
        "l8_last_cos": float(np.mean(last_cos)) if last_cos else None,
        "groups": group_records,
    }


def _loss(base, ablated, key):
    effect = base[key]
    if effect <= EPS:
        return float("-inf")
    return float((effect - ablated[key]) / effect)


def _finite(value):
    return value is not None and bool(np.isfinite(value))


def _write_inelicitable(out_dir, model_path, quantization, seed, reason,
                        discovery_base=None, test_base=None):
    """Persist a null baseline as an interpretable experimental outcome."""
    result = {
        "stage": "delta_binding_component_convergence",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "inject_layer": INJECT_LAYER,
        "mediation_layer": MEDIATION_LAYER,
        "failure_reason": reason,
        "discovery_base": discovery_base,
        "heldout_base": test_base,
        "gates": {"G0": False, "G1": False},
        "verdict": "COMPONENT_CONVERGENCE_INELICITABLE",
    }
    path = os.path.join(out_dir, "results_delta_binding_component_convergence.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: COMPONENT_CONVERGENCE_INELICITABLE ({reason})")
    return result


@torch.no_grad()
def run_delta_binding_component_convergence(
        model_path, out_dir, quantization="8bit", device_map=None,
        seed=0, n_null=N_NULL):
    if n_null < 100:
        raise ValueError("component-convergence p<.01 gate requires >=100 nulls")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    layers = get_decoder_layers(model)
    if len(layers) <= MEDIATION_LAYER:
        raise ValueError("model lacks frozen L8 mediator")
    num_heads = int(model.config.num_attention_heads)
    values = _values(tok)
    rows = _trials(values)
    discovery_rows, test_rows = _split_trials(rows)

    donor_rows, donor_values = [], []
    for name in DONOR_NAMES:
        for value in values:
            donor_rows.append(_single_text(tok, name, value))
            donor_values.append(value)
    donor_ids, donor_am, donor_pos = _encode_uniform(tok, donor_rows)
    _, donor_cache = _forward(
        model, donor_ids.to(dev), donor_am.to(dev), (donor_pos,),
        (INJECT_LAYER,))
    donor_h = donor_cache[INJECT_LAYER][:, 0]
    prototypes = {
        value: donor_h[[i for i, observed in enumerate(donor_values)
                        if observed == value]].mean(0)
        for value in values
    }
    discovery = _group_data(model, tok, discovery_rows, prototypes, dev)
    test = _group_data(model, tok, test_rows, prototypes, dev)
    log("delta_binding_component_convergence: "
        f"discovery={len(discovery_rows)} test={len(test_rows)} "
        f"candidates={num_heads + 1} n_null={n_null}")

    discovery_base = _measure(model, discovery)
    if (discovery_base["natural_effect"] <= EPS
            or discovery_base["add_effect"] <= EPS):
        return _write_inelicitable(
            out_dir, model_path, quantization, seed,
            "discovery natural or ADD logit effect was non-positive",
            discovery_base=discovery_base)
    candidates = _candidate_components(num_heads)
    discovery_metrics = []
    for index, component in enumerate(candidates):
        ablated = _measure(model, discovery, (component,))
        natural_loss = _loss(discovery_base, ablated, "natural_effect")
        add_loss = _loss(discovery_base, ablated, "add_effect")
        discovery_metrics.append({
            "component": _component_json((component,))[0],
            "natural_loss": natural_loss,
            "add_loss": add_loss,
            "shared_score": min(natural_loss, add_loss),
        })
        log(f"discovery {index + 1}/{len(candidates)} "
            f"{_component_name(component)}: natural_loss={natural_loss:+.3f} "
            f"add_loss={add_loss:+.3f}")

    natural_set = _top_components(
        [item["natural_loss"] for item in discovery_metrics], candidates)
    add_set = _top_components(
        [item["add_loss"] for item in discovery_metrics], candidates)
    shared_set = _top_components(
        [item["shared_score"] for item in discovery_metrics], candidates)
    jaccard = _jaccard(natural_set, add_set)
    log(f"frozen discovery sets natural={_component_json(natural_set)} "
        f"add={_component_json(add_set)} shared={_component_json(shared_set)} "
        f"jaccard={jaccard:.3f}")

    test_base = _measure(model, test, capture=True)
    if (test_base["natural_effect"] <= EPS
            or test_base["add_effect"] <= EPS):
        return _write_inelicitable(
            out_dir, model_path, quantization, seed,
            "held-out natural or ADD logit effect was non-positive",
            discovery_base=discovery_base, test_base=test_base)
    test_natural = _measure(model, test, natural_set)
    test_add = _measure(model, test, add_set)
    test_shared = _measure(model, test, shared_set)
    test_control = _measure(
        model, test, shared_set, component_position_kind="previous")

    def set_summary(measurement):
        return {
            "natural_effect": measurement["natural_effect"],
            "add_effect": measurement["add_effect"],
            "natural_loss": _loss(test_base, measurement, "natural_effect"),
            "add_loss": _loss(test_base, measurement, "add_effect"),
        }

    natural_summary = set_summary(test_natural)
    add_summary = set_summary(test_add)
    shared_summary = set_summary(test_shared)
    control_summary = set_summary(test_control)

    random_sets = _matched_random_sets(shared_set, num_heads, seed + 7189, n_null)
    random_natural, random_add = [], []
    for index, component_set in enumerate(random_sets):
        measurement = _measure(model, test, component_set)
        random_natural.append(_loss(test_base, measurement, "natural_effect"))
        random_add.append(_loss(test_base, measurement, "add_effect"))
        if (index + 1) % 10 == 0 or index + 1 == len(random_sets):
            log(f"random component sets {index + 1}/{len(random_sets)}")
    natural_p = permutation_pvalue(
        shared_summary["natural_loss"], np.asarray(random_natural), "greater")
    add_p = permutation_pvalue(
        shared_summary["add_loss"], np.asarray(random_add), "greater")

    effect_ratio = (
        test_base["add_effect"] / test_base["natural_effect"]
        if test_base["natural_effect"] > EPS else float("-inf"))
    gates = {
        "G0": bool(
            test_base["clean_acc"] >= 0.80
            and test_base["natural_acc"] >= 0.80
            and test_base["add_positive_fraction"] >= 0.80
            and 0.70 <= effect_ratio <= 1.30),
        "G1": bool(
            test_base["l8_slot_cos"] >= 0.80
            and test_base["l8_last_cos"] >= 0.50),
        "C1": bool(jaccard >= 0.50),
        "C2": bool(
            shared_summary["natural_loss"] >= 0.50
            and shared_summary["add_loss"] >= 0.50),
        "C3": bool(natural_p < 0.01 and add_p < 0.01),
        "C4": bool(
            abs(control_summary["natural_loss"]) <= 0.20
            and abs(control_summary["add_loss"]) <= 0.20),
    }
    own_natural = natural_summary["natural_loss"] >= 0.50
    own_add = add_summary["add_loss"] >= 0.50
    divergent = bool(
        own_natural and own_add and not gates["C1"]
        and natural_summary["natural_loss"] - add_summary["natural_loss"] >= 0.20
        and add_summary["add_loss"] - natural_summary["add_loss"] >= 0.20)
    if not (gates["G0"] and gates["G1"]):
        verdict = "COMPONENT_CONVERGENCE_INELICITABLE"
    elif all(gates.values()):
        verdict = "SHARED_L8_COMPONENT_PATH"
    elif divergent:
        verdict = "DIVERGENT_L8_COMPONENT_PATHS"
    elif gates["C1"]:
        verdict = "OVERLAPPING_COMPONENTS_NOT_LOCALIZED"
    else:
        verdict = "DISTRIBUTED_OR_REDUNDANT_L8"

    result = {
        "stage": "delta_binding_component_convergence",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "inject_layer": INJECT_LAYER,
        "mediation_layer": MEDIATION_LAYER,
        "component_position": "final query/readout token",
        "n_trials": len(rows),
        "n_discovery": len(discovery_rows),
        "n_test": len(test_rows),
        "n_null": int(n_null),
        "num_attention_heads": num_heads,
        "donor_names": list(DONOR_NAMES),
        "discovery": {
            "base": {
                key: discovery_base[key] for key in (
                    "natural_effect", "add_effect", "clean_acc", "natural_acc",
                    "add_positive_fraction")},
            "candidate_metrics": discovery_metrics,
            "natural_set": _component_json(natural_set),
            "add_set": _component_json(add_set),
            "shared_set": _component_json(shared_set),
            "natural_add_jaccard": jaccard,
        },
        "heldout": {
            "base": {
                key: test_base[key] for key in (
                    "natural_effect", "add_effect", "clean_acc", "natural_acc",
                    "add_positive_fraction", "l8_slot_cos", "l8_last_cos")},
            "add_to_natural_effect_ratio": effect_ratio,
            "natural_set": natural_summary,
            "add_set": add_summary,
            "shared_set": shared_summary,
            "previous_position_control": control_summary,
            "random_shared_set_natural_losses": random_natural,
            "random_shared_set_add_losses": random_add,
            "shared_natural_p": float(natural_p),
            "shared_add_p": float(add_p),
            "groups": test_base["groups"],
        },
        "gates": gates,
        "verdict": verdict,
    }
    if not all(_finite(value) for value in (
            effect_ratio, shared_summary["natural_loss"], shared_summary["add_loss"],
            natural_p, add_p)):
        raise ValueError("non-finite summary metric")
    path = os.path.join(out_dir, "results_delta_binding_component_convergence.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
