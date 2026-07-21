"""Causal read-layer pathways from an upstream orchestration policy slot."""
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
from .delta_orchestration_screen import MODEL_REVISION, _rows
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import (
    get_decoder_layers, input_device, load_model_and_tokenizer)
from .patching import _split_output

INJECT_LAYER = 2
CANDIDATE_LAYERS = tuple(range(3, 28))
TOP_K = 6
N_RANDOM = 100
CONTROL_POSITION = TEMPLATE_B_POSITION - 1
N_DIAGNOSTIC = 8
PROTOCOL_VERSION = "2026-07-13-v1"
PROTOCOL_DOCUMENT_SHA256 = (
    "7bb900dc8f39e25a9d3e7d85cdd737a9e1ed62f34a157db2d54225a6d1a0ff61")
PROTOCOL_SPEC = {
    "inject_layer": INJECT_LAYER,
    "mode_position": TEMPLATE_B_POSITION,
    "control_position": CONTROL_POSITION,
    "candidate_layers": CANDIDATE_LAYERS,
    "top_k": TOP_K,
    "n_random": N_RANDOM,
    "split": "even donor / odd test",
    "base_call_accuracy": 0.90,
    "base_answer_accuracy": 0.875,
    "own_loss": 0.50,
    "random_exceedances": 0,
    "maximum_jaccard": 0.33,
    "own_cross_gap": 0.20,
    "maximum_control_loss": 0.20,
}


@torch.no_grad()
def _forward_reader(
        model, input_ids, attention_mask, direction,
        blocked_layers=(), key_position=TEMPLATE_B_POSITION):
    layers = get_decoder_layers(model)
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

    def edge_hook(_module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.ndim != 4:
            raise ValueError(
                f"expected 4D additive attention mask, got "
                f"{None if mask is None else tuple(mask.shape)}")
        if (mask.shape[-2] <= key_position
                or mask.shape[-1] <= key_position):
            raise ValueError(
                f"attention mask too short for key {key_position}: "
                f"{tuple(mask.shape)}")
        edited = mask.clone()
        fill = (
            torch.finfo(edited.dtype).min
            if edited.dtype.is_floating_point else False)
        edited[
            ..., TEMPLATE_B_POSITION + 1:, key_position] = fill
        kwargs["attention_mask"] = edited
        return args, kwargs

    if direction is not None:
        handles.append(
            layers[INJECT_LAYER].register_forward_hook(add_hook))
    try:
        for layer_idx in blocked_layers:
            handles.append(
                layers[layer_idx].self_attn.register_forward_pre_hook(
                    edge_hook, with_kwargs=True))
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
            use_cache=False)
        return output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def _paired_reader(
        model, input_ids, attention_mask, direction,
        blocked_layers=(), key_position=TEMPLATE_B_POSITION):
    batch = input_ids.shape[0]
    paired_ids = torch.cat([input_ids, input_ids], dim=0)
    paired_am = torch.cat([attention_mask, attention_mask], dim=0)
    paired_directions = torch.cat([
        torch.zeros((batch, direction.numel()), dtype=direction.dtype),
        direction.unsqueeze(0).expand(batch, -1),
    ], dim=0)
    logits = _forward_reader(
        model, paired_ids, paired_am, paired_directions,
        blocked_layers, key_position)
    return logits[:batch], logits[batch:]


def _effect(clean, intervention, target_ids, source_ids, mask):
    values = (
        _ld(intervention, target_ids, source_ids)
        - _ld(clean, target_ids, source_ids))
    selected = values[mask]
    return {
        "mean": float(selected.mean()),
        "positive_fraction": float((selected > 0).float().mean()),
        "rows": values,
    }


def _loss(base_mean, blocked_mean):
    if abs(base_mean) < 1e-8:
        raise ValueError(f"zero baseline effect: {base_mean}")
    return float((base_mean - blocked_mean) / abs(base_mean))


def _random_sets(seed):
    rng = np.random.default_rng(seed)
    found = set()
    while len(found) < N_RANDOM:
        found.add(tuple(sorted(
            int(item) for item in rng.choice(
                CANDIDATE_LAYERS, size=TOP_K, replace=False))))
    return sorted(found)


@torch.no_grad()
def run_delta_policy_slot_readers(
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
    rows = _rows()
    donor_rows, test_rows = rows[::2], rows[1::2]
    if len(get_decoder_layers(model)) != 28:
        raise ValueError("frozen model must have exactly 28 layers")

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
        raise ValueError(f"template-B alignment failed: {changed}")
    if not torch.equal(
            donor_source[:, CONTROL_POSITION],
            donor_target[:, CONTROL_POSITION]):
        raise ValueError("control position differs across policies")
    _, source_cache = _forward(
        model, donor_source, donor_source_am, (TEMPLATE_B_POSITION,),
        (INJECT_LAYER,))
    _, target_cache = _forward(
        model, donor_target, donor_target_am, (TEMPLATE_B_POSITION,),
        (INJECT_LAYER,))
    direction = (
        target_cache[INJECT_LAYER][:, 0]
        - source_cache[INJECT_LAYER][:, 0]).mean(0)

    call_donor_texts = [
        text + "CALL" for text in donor_source_texts]
    call_test_texts = [
        text + "CALL" for text in _template_b_texts(
            tok, test_rows, "calculate")]
    call_target_contexts = [
        text + "CALL" for text in (
            donor_target_texts
            + _template_b_texts(tok, test_rows, "lookup"))]
    call_donor, call_donor_am = _encode_uniform(
        tok, call_donor_texts, dev)
    call_test, call_test_am = _encode_uniform(
        tok, call_test_texts, dev)
    call_target_id = _continuation_id(
        tok, call_donor_texts + call_test_texts + call_target_contexts,
        " database")
    call_source_id = _continuation_id(
        tok, call_donor_texts + call_test_texts + call_target_contexts,
        " calculator")
    call_donor_target = torch.full(
        (len(donor_rows),), call_target_id, dtype=torch.long)
    call_donor_source = torch.full(
        (len(donor_rows),), call_source_id, dtype=torch.long)
    call_test_target = torch.full(
        (len(test_rows),), call_target_id, dtype=torch.long)
    call_test_source = torch.full(
        (len(test_rows),), call_source_id, dtype=torch.long)
    call_donor_mask = torch.ones(len(donor_rows), dtype=torch.bool)
    call_test_mask = torch.ones(len(test_rows), dtype=torch.bool)

    answer_donor_texts, _, _ = _followups(
        tok, donor_rows, "calculate", "lookup")
    answer_test_texts, _, _ = _followups(
        tok, test_rows, "calculate", "lookup")
    answer_donor, answer_donor_am = _encode_uniform(
        tok, answer_donor_texts, dev)
    answer_test, answer_test_am = _encode_uniform(
        tok, answer_test_texts, dev)
    donor_target_answers = [
        str(row["database_value"]) for row in donor_rows]
    donor_source_answers = [
        str(row["a"] + row["b"]) for row in donor_rows]
    test_target_answers = [
        str(row["database_value"]) for row in test_rows]
    test_source_answers = [
        str(row["a"] + row["b"]) for row in test_rows]
    answer_donor_mask = torch.tensor([
        target != source for target, source in zip(
            donor_target_answers, donor_source_answers)])
    answer_test_mask = torch.tensor([
        target != source for target, source in zip(
            test_target_answers, test_source_answers)])
    if int(answer_test_mask.sum()) != N_DIAGNOSTIC:
        raise ValueError(
            f"test diagnostic rows changed: {answer_test_mask.tolist()}")
    if int(answer_donor_mask.sum()) < TOP_K:
        raise ValueError("too few donor diagnostic answer rows")
    answer_donor_target = _answer_ids(
        tok, answer_donor_texts, donor_target_answers)
    answer_donor_source = _answer_ids(
        tok, answer_donor_texts, donor_source_answers)
    answer_test_target = _answer_ids(
        tok, answer_test_texts, test_target_answers)
    answer_test_source = _answer_ids(
        tok, answer_test_texts, test_source_answers)

    stages = {
        "call": {
            "donor_ids": call_donor,
            "donor_am": call_donor_am,
            "donor_target": call_donor_target,
            "donor_source": call_donor_source,
            "donor_mask": call_donor_mask,
            "test_ids": call_test,
            "test_am": call_test_am,
            "test_target": call_test_target,
            "test_source": call_test_source,
            "test_mask": call_test_mask,
        },
        "answer": {
            "donor_ids": answer_donor,
            "donor_am": answer_donor_am,
            "donor_target": answer_donor_target,
            "donor_source": answer_donor_source,
            "donor_mask": answer_donor_mask,
            "test_ids": answer_test,
            "test_am": answer_test_am,
            "test_target": answer_test_target,
            "test_source": answer_test_source,
            "test_mask": answer_test_mask,
        },
    }
    tensors = {}
    summaries = {}
    for name, stage in stages.items():
        donor_clean, donor_base = _paired_reader(
            model, stage["donor_ids"], stage["donor_am"], direction)
        donor_base_effect = _effect(
            donor_clean, donor_base, stage["donor_target"],
            stage["donor_source"], stage["donor_mask"])
        layer_losses = {}
        layer_effect_rows = {}
        for layer_idx in CANDIDATE_LAYERS:
            blocked_clean, blocked = _paired_reader(
                model, stage["donor_ids"], stage["donor_am"], direction,
                (layer_idx,))
            blocked_effect = _effect(
                blocked_clean, blocked, stage["donor_target"],
                stage["donor_source"], stage["donor_mask"])
            layer_effect_rows[layer_idx] = blocked_effect["rows"]
            layer_losses[layer_idx] = _loss(
                donor_base_effect["mean"], blocked_effect["mean"])
        selected = tuple(sorted(
            sorted(
                CANDIDATE_LAYERS,
                key=lambda layer: (-layer_losses[layer], layer)
            )[:TOP_K]))
        stage["selected"] = selected
        stage["donor_clean"] = donor_clean
        stage["donor_base"] = donor_base
        stage["donor_base_effect"] = donor_base_effect
        stage["layer_losses"] = layer_losses
        tensors[f"{name}_donor_layer_effects"] = layer_effect_rows
        summaries[name] = {
            "selected_layers": list(selected),
            "donor_base_effect": donor_base_effect["mean"],
            "donor_layer_losses": {
                str(key): value for key, value in layer_losses.items()},
        }

    call_set = stages["call"]["selected"]
    answer_set = stages["answer"]["selected"]
    random_layer_sets = _random_sets(seed + 2301)
    test_details = {}
    for name, stage in stages.items():
        clean, base = _paired_reader(
            model, stage["test_ids"], stage["test_am"], direction)
        own_set = call_set if name == "call" else answer_set
        cross_set = answer_set if name == "call" else call_set
        own_clean, own = _paired_reader(
            model, stage["test_ids"], stage["test_am"], direction,
            own_set)
        cross_clean, cross = _paired_reader(
            model, stage["test_ids"], stage["test_am"], direction,
            cross_set)
        control_clean, control = _paired_reader(
            model, stage["test_ids"], stage["test_am"], direction,
            own_set, CONTROL_POSITION)
        base_effect = _effect(
            clean, base, stage["test_target"],
            stage["test_source"], stage["test_mask"])
        own_effect = _effect(
            own_clean, own, stage["test_target"],
            stage["test_source"], stage["test_mask"])
        cross_effect = _effect(
            cross_clean, cross, stage["test_target"],
            stage["test_source"], stage["test_mask"])
        control_effect = _effect(
            control_clean, control, stage["test_target"],
            stage["test_source"], stage["test_mask"])
        own_loss = _loss(base_effect["mean"], own_effect["mean"])
        cross_loss = _loss(base_effect["mean"], cross_effect["mean"])
        control_loss = _loss(base_effect["mean"], control_effect["mean"])
        random_losses = []
        for layer_set in random_layer_sets:
            blocked_clean, blocked = _paired_reader(
                model, stage["test_ids"], stage["test_am"], direction,
                layer_set)
            blocked_effect = _effect(
                blocked_clean, blocked, stage["test_target"],
                stage["test_source"], stage["test_mask"])
            random_losses.append(
                _loss(base_effect["mean"], blocked_effect["mean"]))
        exceedances = sum(
            loss >= own_loss for loss in random_losses)
        target_predictions = base.argmax(-1)
        target_accuracy = float((
            target_predictions[stage["test_mask"]]
            == stage["test_target"][stage["test_mask"]]).float().mean())
        test_details[name] = {
            "base_effect": base_effect["mean"],
            "base_positive_fraction": base_effect["positive_fraction"],
            "target_accuracy": target_accuracy,
            "own_loss": own_loss,
            "cross_loss": cross_loss,
            "own_cross_gap": own_loss - cross_loss,
            "control_loss": control_loss,
            "random_losses": random_losses,
            "random_exceedances": exceedances,
        }
        tensors[f"{name}_test_logits"] = {
            "clean": clean, "base": base, "own": own,
            "cross": cross, "control": control,
            "own_clean": own_clean, "cross_clean": cross_clean,
            "control_clean": control_clean,
        }
        tensors[f"{name}_effects"] = {
            "base": base_effect["rows"],
            "own": own_effect["rows"],
            "cross": cross_effect["rows"],
            "control": control_effect["rows"],
        }

    intersection = len(set(call_set) & set(answer_set))
    union = len(set(call_set) | set(answer_set))
    jaccard = intersection / union
    g0 = bool(
        test_details["call"]["target_accuracy"] >= 0.90
        and test_details["answer"]["target_accuracy"] >= 0.875
        and all(
            summaries[name]["donor_base_effect"] > 0
            for name in stages)
        and all(
            test_details[name]["base_effect"] > 0
            for name in stages))
    r0 = all(
        test_details[name]["own_loss"] >= 0.50
        for name in stages)
    r1 = all(
        test_details[name]["random_exceedances"] == 0
        for name in stages)
    r2 = bool(
        jaccard <= 0.33
        and all(
            test_details[name]["own_cross_gap"] >= 0.20
            for name in stages))
    c0 = all(
        abs(test_details[name]["control_loss"]) <= 0.20
        for name in stages)
    gates = {"G0": g0, "R0": r0, "R1": r1, "R2": r2, "C0": c0}
    if not g0:
        verdict = "POLICY_SLOT_READER_DIAGNOSTIC_INVALID"
    elif r0 and r1 and c0 and r2:
        verdict = "STAGE_SPECIFIC_POLICY_SLOT_READERS"
    elif r0 and r1 and c0:
        verdict = "POLICY_SLOT_READERS_NOT_STAGE_SPECIFIC"
    else:
        verdict = "NO_LOCALIZED_POLICY_SLOT_READERS"

    tensor_artifact = "raw_delta_policy_slot_readers.pt"
    torch.save({
        "direction": direction,
        "donor_caches": {
            "source": source_cache, "target": target_cache},
        "input_ids": {
            f"{name}_{split}_ids": stage[f"{split}_ids"].detach().cpu()
            for name, stage in stages.items()
            for split in ("donor", "test")
        },
        "attention_masks": {
            f"{name}_{split}_am": stage[f"{split}_am"].detach().cpu()
            for name, stage in stages.items()
            for split in ("donor", "test")
        },
        "output_ids": {
            f"{name}_{split}_{side}": stage[
                f"{split}_{side}"].detach().cpu()
            for name, stage in stages.items()
            for split in ("donor", "test")
            for side in ("target", "source")
        },
        "diagnostic_masks": {
            f"{name}_{split}": stage[
                f"{split}_mask"].detach().cpu()
            for name, stage in stages.items()
            for split in ("donor", "test")
        },
        **tensors,
    }, os.path.join(out_dir, tensor_artifact))
    result = {
        "stage": "delta_policy_slot_readers",
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
                    _split_output, get_decoder_layers, _assert_runtime,
                    _encode_uniform,
                    load_model_and_tokenizer)).encode()).hexdigest(),
            "rows_sha256": hashlib.sha256(json.dumps(
                rows, sort_keys=True).encode()).hexdigest(),
        },
        "preflight": {
            "direction_norm": float(direction.norm()),
            "donor_changed_positions": changed,
            "control_position": CONTROL_POSITION,
            "call_target_id": call_target_id,
            "call_source_id": call_source_id,
            "answer_donor_diagnostic": answer_donor_mask.tolist(),
            "answer_test_diagnostic": answer_test_mask.tolist(),
            "prompts": {
                "donor_source": donor_source_texts,
                "donor_target": donor_target_texts,
                "call_donor": call_donor_texts,
                "call_test": call_test_texts,
                "answer_donor": answer_donor_texts,
                "answer_test": answer_test_texts,
            },
        },
        "donor_rows": donor_rows,
        "test_rows": test_rows,
        "discovery": summaries,
        "reader_sets": {
            "call": list(call_set),
            "answer": list(answer_set),
            "intersection": intersection,
            "union": union,
            "jaccard": jaccard,
        },
        "random_layer_sets": [list(item) for item in random_layer_sets],
        "test": test_details,
        "gates": gates,
        "raw_tensor_artifact": tensor_artifact,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_policy_slot_readers.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    log(f"reader_sets call={call_set} answer={answer_set} "
        f"jaccard={jaccard:.3f}")
    log(f"test={test_details}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
