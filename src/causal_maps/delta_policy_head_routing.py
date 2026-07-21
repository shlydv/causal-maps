"""Head-level causal routing from an upstream orchestration policy slot."""
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import os

import numpy as np
import torch

from .delta_agent_policy_broadcast import _continuation_id
from .delta_continuous_orchestration import _answer_ids, _followups
from .delta_orchestration_controller import (
    _assert_runtime, _encode_uniform)
from .delta_orchestration_label_transfer import (
    TEMPLATE_B_POSITION, _template_b_texts)
from .delta_orchestration_screen import DATABASE, MODEL_REVISION, _rows
from .delta_policy_slot_readers import (
    CONTROL_POSITION, INJECT_LAYER, _effect, _loss)
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import (
    get_decoder_layers, input_device, load_model_and_tokenizer)
from .patching import _split_output

CANDIDATE_LAYERS = (4, 5, 7, 10, 11, 15, 17, 18)
NUM_HEADS = 28
TOP_K = 8
N_RANDOM = 100
PROTOCOL_VERSION = "2026-07-13-v1"
PROTOCOL_DOCUMENT_SHA256 = (
    "ec7f931f0d3ba48ec103cd91a2dd9f05b7743546f056bb2e14c5e933cda9d554")
PROTOCOL_SPEC = {
    "inject_layer": INJECT_LAYER,
    "mode_position": TEMPLATE_B_POSITION,
    "control_position": CONTROL_POSITION,
    "candidate_layers": CANDIDATE_LAYERS,
    "num_query_heads": NUM_HEADS,
    "top_k": TOP_K,
    "n_random": N_RANDOM,
    "discovery": "original even-index donor rows only",
    "confirmation": "ten frozen unseen reversed-number rows",
    "native_call_accuracy": 0.90,
    "native_answer_accuracy": 0.80,
    "controller_call_accuracy": 0.90,
    "controller_answer_accuracy": 0.80,
    "minimum_effect_ratio": 0.70,
    "maximum_effect_ratio": 1.30,
    "minimum_positive_fraction": 0.80,
    "own_loss": 0.50,
    "random_exceedances": 0,
    "maximum_jaccard": 0.25,
    "own_cross_gap": 0.20,
    "maximum_control_loss": 0.20,
}


def _fresh_rows():
    specs = (
        (1, 2, "A"), (1, 3, "C"), (2, 3, "D"), (1, 4, "E"),
        (2, 4, "G"), (3, 4, "H"), (1, 5, "I"), (2, 5, "J"),
        (3, 5, "B"), (1, 6, "F"),
    )
    return [
        {"a": a, "b": b, "key": key, "database_value": DATABASE[key]}
        for a, b, key in specs
    ]


def _group_heads(blocked_heads):
    grouped = defaultdict(list)
    for layer_idx, head_idx in blocked_heads:
        if layer_idx not in CANDIDATE_LAYERS:
            raise ValueError(f"noncandidate layer: {layer_idx}")
        if not 0 <= head_idx < NUM_HEADS:
            raise ValueError(f"invalid head: {head_idx}")
        grouped[layer_idx].append(head_idx)
    return {
        layer_idx: tuple(sorted(set(heads)))
        for layer_idx, heads in grouped.items()}


@torch.no_grad()
def _forward_heads(
        model, input_ids, attention_mask, direction,
        blocked_heads=(), key_position=TEMPLATE_B_POSITION):
    layers = get_decoder_layers(model)
    grouped = _group_heads(blocked_heads)
    handles = []

    def add_hook(_module, _inputs, output):
        hidden, rebuild = _split_output(output)
        edited = hidden.clone()
        addition = direction
        if addition.ndim == 1:
            addition = addition.unsqueeze(0).expand(
                edited.shape[0], -1)
        if addition.shape != (
                edited.shape[0], edited.shape[-1]):
            raise ValueError(
                f"direction shape mismatch: {tuple(addition.shape)} "
                f"for hidden {tuple(edited.shape)}")
        edited[:, TEMPLATE_B_POSITION] += addition.to(
            edited.device, edited.dtype)
        return rebuild(edited)

    def make_edge_hook(head_indices):
        def edge_hook(_module, args, kwargs):
            mask = kwargs.get("attention_mask")
            if mask is None or mask.ndim != 4:
                raise ValueError(
                    f"expected 4D additive attention mask, got "
                    f"{None if mask is None else tuple(mask.shape)}")
            if (mask.shape[-2] <= TEMPLATE_B_POSITION
                    or mask.shape[-1] <= key_position):
                raise ValueError(
                    f"attention mask too short for key {key_position}: "
                    f"{tuple(mask.shape)}")
            if mask.shape[1] not in (1, NUM_HEADS):
                raise ValueError(
                    f"unexpected mask head dimension: {mask.shape[1]}")
            edited = mask.expand(
                mask.shape[0], NUM_HEADS,
                mask.shape[-2], mask.shape[-1]).clone()
            edited[
                :, list(head_indices),
                TEMPLATE_B_POSITION + 1:, key_position
            ] = torch.finfo(edited.dtype).min
            kwargs["attention_mask"] = edited
            return args, kwargs
        return edge_hook

    if direction is not None:
        handles.append(
            layers[INJECT_LAYER].register_forward_hook(add_hook))
    try:
        for layer_idx, head_indices in grouped.items():
            handles.append(
                layers[layer_idx].self_attn.register_forward_pre_hook(
                    make_edge_hook(head_indices), with_kwargs=True))
        batch, seq_len = input_ids.shape
        dtype = model.get_input_embeddings().weight.dtype
        additive_mask = torch.zeros(
            (batch, 1, seq_len, seq_len), dtype=dtype,
            device=input_ids.device)
        future = torch.triu(torch.ones(
            (seq_len, seq_len), dtype=torch.bool,
            device=input_ids.device), diagonal=1)
        additive_mask.masked_fill_(
            future.unsqueeze(0).unsqueeze(0),
            torch.finfo(dtype).min)
        additive_mask.masked_fill_(
            ~attention_mask.to(torch.bool)[:, None, None, :],
            torch.finfo(dtype).min)
        output = model(
            input_ids=input_ids,
            attention_mask={
                "full_attention": additive_mask,
                "sliding_attention": additive_mask,
            },
            use_cache=False,
            logits_to_keep=1)
        return output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def _paired_heads(
        model, input_ids, attention_mask, direction,
        blocked_heads=(), key_position=TEMPLATE_B_POSITION):
    batch = input_ids.shape[0]
    paired_ids = torch.cat([input_ids, input_ids], dim=0)
    paired_am = torch.cat([attention_mask, attention_mask], dim=0)
    paired_directions = torch.cat([
        torch.zeros((batch, direction.numel()), dtype=direction.dtype),
        direction.unsqueeze(0).expand(batch, -1),
    ], dim=0)
    logits = _forward_heads(
        model, paired_ids, paired_am, paired_directions,
        blocked_heads, key_position)
    return logits[:batch], logits[batch:]


def _matched_random_sets(reference, seed):
    counts = Counter(layer_idx for layer_idx, _ in reference)
    rng = np.random.default_rng(seed)
    found = set()
    reference = tuple(sorted(reference))
    while len(found) < N_RANDOM:
        sampled = []
        for layer_idx in sorted(counts):
            heads = rng.choice(
                NUM_HEADS, size=counts[layer_idx], replace=False)
            sampled.extend(
                (layer_idx, int(head_idx)) for head_idx in heads)
        candidate = tuple(sorted(sampled))
        if candidate != reference:
            found.add(candidate)
    return sorted(found)


def _head_json(heads):
    return [
        {"layer": layer_idx, "head": head_idx}
        for layer_idx, head_idx in heads]


def _accuracy(logits, ids, mask):
    return float((
        logits.argmax(-1)[mask] == ids[mask]).float().mean())


def _write_progress(out_dir, payload):
    path = os.path.join(
        out_dir, "progress_delta_policy_head_routing.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


@torch.no_grad()
def run_delta_policy_head_routing(
        model_path, out_dir, quantization="8bit", device_map=None,
        seed=0):
    if model_path != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(f"frozen model mismatch: {model_path}")
    if quantization != "8bit" or seed != 0:
        raise ValueError(
            f"frozen config mismatch: quant={quantization} seed={seed}")
    os.makedirs(out_dir, exist_ok=True)
    runtime = _assert_runtime()
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization,
        revision=MODEL_REVISION)
    dev = input_device(model)
    if len(get_decoder_layers(model)) != 28:
        raise ValueError("frozen model must have exactly 28 layers")
    if model.config.num_attention_heads != NUM_HEADS:
        raise ValueError(
            f"expected {NUM_HEADS} query heads, got "
            f"{model.config.num_attention_heads}")
    if model.config.num_key_value_heads != 4:
        raise ValueError(
            f"expected 4 KV heads, got "
            f"{model.config.num_key_value_heads}")
    if model.config._attn_implementation != "sdpa":
        raise ValueError(
            f"expected sdpa attention, got "
            f"{model.config._attn_implementation}")

    original_rows = _rows()
    donor_rows = original_rows[::2]
    fresh_rows = _fresh_rows()
    original_pairs = {
        (row["a"], row["b"]) for row in original_rows}
    if any(
            (row["a"], row["b"]) in original_pairs
            or row["a"] >= row["b"]
            or row["database_value"] == row["a"] + row["b"]
            for row in fresh_rows):
        raise ValueError("fresh confirmation row invariant failed")

    donor_source_texts = _template_b_texts(
        tok, donor_rows, "calculate")
    donor_target_texts = _template_b_texts(tok, donor_rows, "lookup")
    donor_source, donor_source_am = _encode_uniform(
        tok, donor_source_texts, dev)
    donor_target, donor_target_am = _encode_uniform(
        tok, donor_target_texts, dev)
    changed = [[
        idx for idx, (left, right) in enumerate(zip(source, target))
        if left != right
    ] for source, target in zip(
        donor_source.tolist(), donor_target.tolist())]
    if any(item != [TEMPLATE_B_POSITION] for item in changed):
        raise ValueError(f"template-B donor alignment failed: {changed}")
    _, source_cache = _forward(
        model, donor_source, donor_source_am, (TEMPLATE_B_POSITION,),
        (INJECT_LAYER,))
    _, target_cache = _forward(
        model, donor_target, donor_target_am, (TEMPLATE_B_POSITION,),
        (INJECT_LAYER,))
    direction = (
        target_cache[INJECT_LAYER][:, 0]
        - source_cache[INJECT_LAYER][:, 0]).mean(0)

    fresh_source_texts = _template_b_texts(
        tok, fresh_rows, "calculate")
    fresh_target_texts = _template_b_texts(
        tok, fresh_rows, "lookup")
    fresh_source, fresh_source_am = _encode_uniform(
        tok, fresh_source_texts, dev)
    fresh_target, fresh_target_am = _encode_uniform(
        tok, fresh_target_texts, dev)
    fresh_changed = [[
        idx for idx, (left, right) in enumerate(zip(source, target))
        if left != right
    ] for source, target in zip(
        fresh_source.tolist(), fresh_target.tolist())]
    if any(item != [TEMPLATE_B_POSITION] for item in fresh_changed):
        raise ValueError(
            f"template-B fresh alignment failed: {fresh_changed}")

    call_donor_texts = [
        text + "CALL" for text in donor_source_texts]
    call_fresh_source_texts = [
        text + "CALL" for text in fresh_source_texts]
    call_fresh_target_texts = [
        text + "CALL" for text in fresh_target_texts]
    call_donor, call_donor_am = _encode_uniform(
        tok, call_donor_texts, dev)
    call_fresh_source, call_fresh_source_am = _encode_uniform(
        tok, call_fresh_source_texts, dev)
    call_fresh_target_input, call_fresh_target_am = _encode_uniform(
        tok, call_fresh_target_texts, dev)
    call_contexts = (
        call_donor_texts + call_fresh_source_texts
        + call_fresh_target_texts)
    call_target_id = _continuation_id(tok, call_contexts, " database")
    call_source_id = _continuation_id(tok, call_contexts, " calculator")
    call_donor_target = torch.full(
        (len(donor_rows),), call_target_id, dtype=torch.long)
    call_donor_source = torch.full(
        (len(donor_rows),), call_source_id, dtype=torch.long)
    call_fresh_target_ids = torch.full(
        (len(fresh_rows),), call_target_id, dtype=torch.long)
    call_fresh_source_ids = torch.full(
        (len(fresh_rows),), call_source_id, dtype=torch.long)

    answer_donor_texts, _, _ = _followups(
        tok, donor_rows, "calculate", "lookup")
    answer_fresh_source_texts, _, _ = _followups(
        tok, fresh_rows, "calculate", "lookup")
    answer_fresh_native_source_texts, _, _ = _followups(
        tok, fresh_rows, "calculate", "calculate")
    answer_fresh_target_texts, _, _ = _followups(
        tok, fresh_rows, "lookup", "lookup")
    answer_donor, answer_donor_am = _encode_uniform(
        tok, answer_donor_texts, dev)
    answer_fresh_source, answer_fresh_source_am = _encode_uniform(
        tok, answer_fresh_source_texts, dev)
    answer_fresh_native_source, answer_fresh_native_source_am = (
        _encode_uniform(
            tok, answer_fresh_native_source_texts, dev))
    answer_fresh_target_input, answer_fresh_target_am = _encode_uniform(
        tok, answer_fresh_target_texts, dev)
    donor_target_answers = [
        str(row["database_value"]) for row in donor_rows]
    donor_source_answers = [
        str(row["a"] + row["b"]) for row in donor_rows]
    fresh_target_answers = [
        str(row["database_value"]) for row in fresh_rows]
    fresh_source_answers = [
        str(row["a"] + row["b"]) for row in fresh_rows]
    answer_donor_mask = torch.tensor([
        target != source for target, source in zip(
            donor_target_answers, donor_source_answers)])
    answer_fresh_mask = torch.ones(
        len(fresh_rows), dtype=torch.bool)
    answer_donor_target = _answer_ids(
        tok, answer_donor_texts, donor_target_answers)
    answer_donor_source = _answer_ids(
        tok, answer_donor_texts, donor_source_answers)
    answer_fresh_target_ids = _answer_ids(
        tok, answer_fresh_source_texts, fresh_target_answers)
    answer_fresh_source_ids = _answer_ids(
        tok, answer_fresh_source_texts, fresh_source_answers)
    answer_native_target = _answer_ids(
        tok, answer_fresh_target_texts, fresh_target_answers)
    answer_native_source = _answer_ids(
        tok, answer_fresh_native_source_texts, fresh_source_answers)

    all_mask_donor = torch.ones(
        len(donor_rows), dtype=torch.bool)
    all_mask_fresh = torch.ones(
        len(fresh_rows), dtype=torch.bool)
    stages = {
        "call": {
            "donor_ids": call_donor,
            "donor_am": call_donor_am,
            "donor_target": call_donor_target,
            "donor_source": call_donor_source,
            "donor_mask": all_mask_donor,
            "fresh_ids": call_fresh_source,
            "fresh_am": call_fresh_source_am,
            "fresh_target": call_fresh_target_ids,
            "fresh_source": call_fresh_source_ids,
            "fresh_mask": all_mask_fresh,
        },
        "answer": {
            "donor_ids": answer_donor,
            "donor_am": answer_donor_am,
            "donor_target": answer_donor_target,
            "donor_source": answer_donor_source,
            "donor_mask": answer_donor_mask,
            "fresh_ids": answer_fresh_source,
            "fresh_am": answer_fresh_source_am,
            "fresh_target": answer_fresh_target_ids,
            "fresh_source": answer_fresh_source_ids,
            "fresh_mask": answer_fresh_mask,
        },
    }

    candidate_heads = tuple(
        (layer_idx, head_idx)
        for layer_idx in CANDIDATE_LAYERS
        for head_idx in range(NUM_HEADS))
    discovery = {}
    raw_tensors = {}
    for name, stage in stages.items():
        clean, base = _paired_heads(
            model, stage["donor_ids"], stage["donor_am"], direction)
        base_effect = _effect(
            clean, base, stage["donor_target"],
            stage["donor_source"], stage["donor_mask"])
        head_losses = {}
        head_effect_rows = {}
        for head in candidate_heads:
            blocked_clean, blocked = _paired_heads(
                model, stage["donor_ids"], stage["donor_am"],
                direction, (head,))
            blocked_effect = _effect(
                blocked_clean, blocked, stage["donor_target"],
                stage["donor_source"], stage["donor_mask"])
            head_losses[head] = _loss(
                base_effect["mean"], blocked_effect["mean"])
            head_effect_rows[head] = blocked_effect["rows"]
            if head[1] == NUM_HEADS - 1:
                _write_progress(out_dir, {
                    "phase": "discovery",
                    "stage": name,
                    "completed_layer": head[0],
                    "completed_heads": len(head_losses),
                    "base_effect": base_effect["mean"],
                    "head_losses": [
                        {
                            "layer": layer_idx,
                            "head": head_idx,
                            "loss": loss,
                        }
                        for (layer_idx, head_idx), loss
                        in sorted(head_losses.items())
                    ],
                })
        selected = tuple(sorted(
            sorted(
                candidate_heads,
                key=lambda item: (-head_losses[item], item)
            )[:TOP_K]))
        stage["selected"] = selected
        stage["donor_base_effect"] = base_effect
        discovery[name] = {
            "selected_heads": _head_json(selected),
            "donor_base_effect": base_effect["mean"],
            "head_losses": [
                {
                    "layer": layer_idx, "head": head_idx,
                    "loss": head_losses[(layer_idx, head_idx)],
                }
                for layer_idx, head_idx in candidate_heads
            ],
        }
        raw_tensors[f"{name}_donor_effects"] = {
            "base": base_effect["rows"],
            **{
                f"L{layer_idx}H{head_idx}": rows
                for (layer_idx, head_idx), rows
                in head_effect_rows.items()
            },
        }

    call_heads = stages["call"]["selected"]
    answer_heads = stages["answer"]["selected"]
    call_set, answer_set = set(call_heads), set(answer_heads)
    intersection = tuple(sorted(call_set & answer_set))
    call_unique = tuple(sorted(call_set - answer_set))
    answer_unique = tuple(sorted(answer_set - call_set))
    union = call_set | answer_set
    jaccard = len(intersection) / len(union)
    named_sets = {
        "own": {"call": call_heads, "answer": answer_heads},
        "cross": {"call": answer_heads, "answer": call_heads},
        "intersection": {
            "call": intersection, "answer": intersection},
        "call_unique": {
            "call": call_unique, "answer": call_unique},
        "answer_unique": {
            "call": answer_unique, "answer": answer_unique},
    }

    native_call_source = _forward_heads(
        model, call_fresh_source, call_fresh_source_am, None)
    native_call_target = _forward_heads(
        model, call_fresh_target_input, call_fresh_target_am, None)
    native_answer_source = _forward_heads(
        model, answer_fresh_native_source,
        answer_fresh_native_source_am, None)
    native_answer_target = _forward_heads(
        model, answer_fresh_target_input, answer_fresh_target_am, None)
    native = {
        "call_source_accuracy": _accuracy(
            native_call_source, call_fresh_source_ids, all_mask_fresh),
        "call_target_accuracy": _accuracy(
            native_call_target, call_fresh_target_ids, all_mask_fresh),
        "answer_source_accuracy": _accuracy(
            native_answer_source, answer_native_source, answer_fresh_mask),
        "answer_target_accuracy": _accuracy(
            native_answer_target, answer_native_target, answer_fresh_mask),
    }
    raw_tensors["native_logits"] = {
        "call_source": native_call_source,
        "call_target": native_call_target,
        "answer_source": native_answer_source,
        "answer_target": native_answer_target,
    }

    confirmation = {}
    random_sets_by_stage = {}
    natural_target_logits = {
        "call": native_call_target,
        "answer": native_answer_target,
    }
    for stage_idx, (name, stage) in enumerate(stages.items()):
        clean, base = _paired_heads(
            model, stage["fresh_ids"], stage["fresh_am"], direction)
        base_effect = _effect(
            clean, base, stage["fresh_target"],
            stage["fresh_source"], stage["fresh_mask"])
        natural_effect = _effect(
            clean, natural_target_logits[name], stage["fresh_target"],
            stage["fresh_source"], stage["fresh_mask"])
        effect_ratio = (
            base_effect["mean"] / natural_effect["mean"]
            if abs(natural_effect["mean"]) > 1e-8 else None)
        condition_results = {}
        condition_logits = {"clean": clean, "base": base}
        condition_effects = {"base": base_effect["rows"]}
        for condition, per_stage in named_sets.items():
            heads = per_stage[name]
            blocked_clean, blocked = _paired_heads(
                model, stage["fresh_ids"], stage["fresh_am"],
                direction, heads)
            blocked_effect = _effect(
                blocked_clean, blocked, stage["fresh_target"],
                stage["fresh_source"], stage["fresh_mask"])
            condition_results[condition] = {
                "heads": _head_json(heads),
                "effect": blocked_effect["mean"],
                "loss": _loss(
                    base_effect["mean"], blocked_effect["mean"]),
                "target_accuracy": _accuracy(
                    blocked, stage["fresh_target"],
                    stage["fresh_mask"]),
            }
            condition_logits[f"{condition}_clean"] = blocked_clean
            condition_logits[condition] = blocked
            condition_effects[condition] = blocked_effect["rows"]
        own_heads = named_sets["own"][name]
        control_clean, control = _paired_heads(
            model, stage["fresh_ids"], stage["fresh_am"], direction,
            own_heads, CONTROL_POSITION)
        control_effect = _effect(
            control_clean, control, stage["fresh_target"],
            stage["fresh_source"], stage["fresh_mask"])
        control_loss = _loss(
            base_effect["mean"], control_effect["mean"])
        random_sets = _matched_random_sets(
            own_heads, seed + 3101 + stage_idx)
        random_losses = []
        for random_idx, random_heads in enumerate(random_sets):
            random_clean, random_blocked = _paired_heads(
                model, stage["fresh_ids"], stage["fresh_am"],
                direction, random_heads)
            random_effect = _effect(
                random_clean, random_blocked, stage["fresh_target"],
                stage["fresh_source"], stage["fresh_mask"])
            random_losses.append(_loss(
                base_effect["mean"], random_effect["mean"]))
            if (random_idx + 1) % 10 == 0:
                _write_progress(out_dir, {
                    "phase": "confirmation_random",
                    "stage": name,
                    "completed_random_sets": random_idx + 1,
                    "selected_heads": _head_json(own_heads),
                    "base_effect": base_effect["mean"],
                    "conditions": condition_results,
                    "control_loss": control_loss,
                    "random_losses": random_losses,
                })
        own_loss = condition_results["own"]["loss"]
        confirmation[name] = {
            "base_effect": base_effect["mean"],
            "base_positive_fraction": base_effect["positive_fraction"],
            "natural_effect": natural_effect["mean"],
            "effect_ratio": effect_ratio,
            "controller_target_accuracy": _accuracy(
                base, stage["fresh_target"], stage["fresh_mask"]),
            "conditions": condition_results,
            "control": {
                "effect": control_effect["mean"],
                "loss": control_loss,
                "target_accuracy": _accuracy(
                    control, stage["fresh_target"],
                    stage["fresh_mask"]),
            },
            "random_losses": random_losses,
            "random_exceedances": sum(
                loss >= own_loss for loss in random_losses),
        }
        random_sets_by_stage[name] = random_sets
        raw_tensors[f"{name}_confirmation_logits"] = {
            **condition_logits,
            "control_clean": control_clean,
            "control": control,
        }
        raw_tensors[f"{name}_confirmation_effects"] = {
            **condition_effects,
            "natural": natural_effect["rows"],
            "control": control_effect["rows"],
        }

    g0 = bool(
        native["call_source_accuracy"] >= 0.90
        and native["call_target_accuracy"] >= 0.90
        and native["answer_source_accuracy"] >= 0.80
        and native["answer_target_accuracy"] >= 0.80
        and confirmation["call"]["controller_target_accuracy"] >= 0.90
        and confirmation["answer"]["controller_target_accuracy"] >= 0.80
        and all(
            discovery[name]["donor_base_effect"] > 0
            and confirmation[name]["base_effect"] > 0
            and confirmation[name]["natural_effect"] > 0
            and confirmation[name]["effect_ratio"] is not None
            and 0.70 <= confirmation[name]["effect_ratio"] <= 1.30
            and confirmation[name]["base_positive_fraction"] >= 0.80
            for name in stages))
    h0 = all(
        confirmation[name]["conditions"]["own"]["loss"] >= 0.50
        for name in stages)
    h1 = all(
        confirmation[name]["random_exceedances"] == 0
        for name in stages)
    h2 = bool(
        jaccard <= 0.25
        and all(
            confirmation[name]["conditions"]["own"]["loss"]
            - confirmation[name]["conditions"]["cross"]["loss"] >= 0.20
            for name in stages))
    c0 = all(
        abs(confirmation[name]["control"]["loss"]) <= 0.20
        for name in stages)
    gates = {"G0": g0, "H0": h0, "H1": h1, "H2": h2, "C0": c0}
    if not g0:
        verdict = "POLICY_HEAD_ROUTING_DIAGNOSTIC_INVALID"
    elif h0 and h1 and c0 and h2:
        verdict = "STAGE_SPECIFIC_POLICY_ROUTING_HEADS"
    elif h0 and h1 and c0:
        verdict = "SHARED_OR_OVERLAPPING_POLICY_ROUTING_HEADS"
    else:
        verdict = "NO_LOCALIZED_POLICY_ROUTING_HEADS"

    tensor_artifact = "raw_delta_policy_head_routing.pt"
    torch.save({
        "direction": direction,
        "donor_caches": {
            "source": source_cache, "target": target_cache},
        "input_ids": {
            "call_donor": call_donor.detach().cpu(),
            "call_fresh_source": call_fresh_source.detach().cpu(),
            "call_fresh_target": call_fresh_target_input.detach().cpu(),
            "answer_donor": answer_donor.detach().cpu(),
            "answer_fresh_source": answer_fresh_source.detach().cpu(),
            "answer_fresh_native_source": (
                answer_fresh_native_source.detach().cpu()),
            "answer_fresh_target": answer_fresh_target_input.detach().cpu(),
        },
        "attention_masks": {
            "call_donor": call_donor_am.detach().cpu(),
            "call_fresh_source": call_fresh_source_am.detach().cpu(),
            "call_fresh_target": call_fresh_target_am.detach().cpu(),
            "answer_donor": answer_donor_am.detach().cpu(),
            "answer_fresh_source": answer_fresh_source_am.detach().cpu(),
            "answer_fresh_native_source": (
                answer_fresh_native_source_am.detach().cpu()),
            "answer_fresh_target": answer_fresh_target_am.detach().cpu(),
        },
        "output_ids": {
            "call_target": call_fresh_target_ids,
            "call_source": call_fresh_source_ids,
            "answer_target": answer_fresh_target_ids,
            "answer_source": answer_fresh_source_ids,
            "answer_native_target": answer_native_target,
            "answer_native_source": answer_native_source,
        },
        **raw_tensors,
    }, os.path.join(out_dir, tensor_artifact))
    result = {
        "stage": "delta_policy_head_routing",
        "model_path": model_path,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "quantization": quantization,
        "seed": seed,
        "protocol": PROTOCOL_SPEC,
        "provenance": {
            "protocol_version": PROTOCOL_VERSION,
            "source_sha256": hashlib.sha256(
                open(__file__, "rb").read()).hexdigest(),
            "protocol_sha256": hashlib.sha256(json.dumps(
                PROTOCOL_SPEC, sort_keys=True, default=list
            ).encode()).hexdigest(),
            "protocol_document_sha256": PROTOCOL_DOCUMENT_SHA256,
            "helpers_sha256": hashlib.sha256("".join(
                inspect.getsource(helper) for helper in (
                    _forward, _ld, _answer_ids, _followups,
                    _continuation_id, _template_b_texts, _rows,
                    _effect, _loss, _split_output, get_decoder_layers,
                    _assert_runtime, _encode_uniform,
                    load_model_and_tokenizer)).encode()).hexdigest(),
            "original_rows_sha256": hashlib.sha256(json.dumps(
                original_rows, sort_keys=True).encode()).hexdigest(),
            "fresh_rows_sha256": hashlib.sha256(json.dumps(
                fresh_rows, sort_keys=True).encode()).hexdigest(),
        },
        "preflight": {
            "direction_norm": float(direction.norm()),
            "attention_implementation": (
                model.config._attn_implementation),
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "donor_changed_positions": changed,
            "fresh_changed_positions": fresh_changed,
            "call_target_id": call_target_id,
            "call_source_id": call_source_id,
            "answer_donor_diagnostic": answer_donor_mask.tolist(),
            "prompts": {
                "donor_source": donor_source_texts,
                "donor_target": donor_target_texts,
                "fresh_source": fresh_source_texts,
                "fresh_target": fresh_target_texts,
                "call_donor": call_donor_texts,
                "call_fresh_source": call_fresh_source_texts,
                "call_fresh_target": call_fresh_target_texts,
                "answer_donor": answer_donor_texts,
                "answer_fresh_source": answer_fresh_source_texts,
                "answer_fresh_native_source": (
                    answer_fresh_native_source_texts),
                "answer_fresh_target": answer_fresh_target_texts,
            },
        },
        "donor_rows": donor_rows,
        "fresh_rows": fresh_rows,
        "native": native,
        "discovery": discovery,
        "head_sets": {
            "call": _head_json(call_heads),
            "answer": _head_json(answer_heads),
            "intersection": _head_json(intersection),
            "call_unique": _head_json(call_unique),
            "answer_unique": _head_json(answer_unique),
            "jaccard": jaccard,
        },
        "random_head_sets": {
            name: [_head_json(heads) for heads in sets]
            for name, sets in random_sets_by_stage.items()
        },
        "confirmation": confirmation,
        "gates": gates,
        "raw_tensor_artifact": tensor_artifact,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_policy_head_routing.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    _write_progress(out_dir, {
        "phase": "complete",
        "gates": gates,
        "verdict": verdict,
    })
    log(f"head_sets call={call_heads} answer={answer_heads} "
        f"jaccard={jaccard:.3f}")
    log(f"native={native}")
    log(f"confirmation={confirmation}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
