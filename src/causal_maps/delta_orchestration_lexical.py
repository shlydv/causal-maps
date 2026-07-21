"""Scale-matched lexical replay controls for orchestration."""
import json
import os

import numpy as np
import torch

from .delta_orchestration_controller import (
    INJECT_LAYER, LAYERS, MEDIATION_LAYER, MODE_POSITION, N_NULL,
    _assert_runtime, _encode_uniform, _evaluate_workflow, _generate_with_add,
    _safe_ratio, _task_texts)
from .delta_orchestration_screen import MODEL_REVISION, _generate, _rows
from .delta_reasoning_controller import _candidate_metrics
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

EPS = 1e-8


@torch.no_grad()
def _evaluate_variant(model, tok, dev, test_rows, calc_texts, lookup_texts,
                      calc_call_ids, calc_call_am, clean_logits, clean_cache,
                      natural_logits, natural_cache, clean_ld, calc_tool_tensor,
                      lookup_tool_tensor, direction, generator, n_null,
                      mode_position=MODE_POSITION, content_fn=None):
    call_last = int(calc_call_ids.shape[1] - 1)
    batch = direction.unsqueeze(0).expand(len(test_rows), -1)
    logits, cache = _forward(
        model, calc_call_ids, calc_call_am, (mode_position, call_last), LAYERS,
        add=(INJECT_LAYER, mode_position, batch))
    natural_rows = (
        _ld(natural_logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld)
    effect_rows = (
        _ld(logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld)
    natural_effect = float(natural_rows.mean())
    effect = float(effect_rows.mean())
    ratio = _safe_ratio(effect, natural_effect)
    native_l20 = (
        natural_cache[MEDIATION_LAYER][:, 1]
        - clean_cache[MEDIATION_LAYER][:, 1])
    variant_l20 = (
        cache[MEDIATION_LAYER][:, 1]
        - clean_cache[MEDIATION_LAYER][:, 1])
    l20 = _candidate_metrics(variant_l20, native_l20)

    raw_forward = _generate_with_add(
        model, tok, calc_texts, dev, mode_position, direction)
    raw_reverse = _generate_with_add(
        model, tok, lookup_texts, dev, mode_position, -direction)
    forward_workflow = _evaluate_workflow(
        model, tok, dev, test_rows, "calculate", "lookup", raw_forward,
        **({"content_fn": content_fn} if content_fn is not None else {}))
    reverse_workflow = _evaluate_workflow(
        model, tok, dev, test_rows, "lookup", "calculate", raw_reverse,
        **({"content_fn": content_fn} if content_fn is not None else {}))

    null_means = np.zeros(n_null)
    null_rows = np.zeros((n_null, len(test_rows)))
    null_directions = torch.zeros((n_null, direction.numel()))
    norm = direction.norm().clamp(min=EPS)
    for idx in range(n_null):
        random = torch.randn(
            direction.shape, generator=generator, dtype=torch.float32)
        random = random / random.norm() * norm
        null_directions[idx] = random
        random_batch = random.unsqueeze(0).expand(len(test_rows), -1)
        random_logits, _ = _forward(
            model, calc_call_ids, calc_call_am, (mode_position, call_last),
            add=(INJECT_LAYER, mode_position, random_batch))
        rows = (
            _ld(random_logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld)
        null_rows[idx] = rows.numpy()
        null_means[idx] = float(rows.mean())
    null_exceedances = int((null_means >= effect).sum())
    positive_fraction = float((effect_rows > 0).float().mean())
    forward_metrics = forward_workflow["metrics"]
    reverse_metrics = reverse_workflow["metrics"]
    output_pass = bool(
        ratio is not None and 0.70 <= ratio <= 1.30
        and positive_fraction >= 0.80 and null_exceedances <= 1)
    workflow_pass = bool(
        forward_metrics["exact_call_acc"] >= 0.80
        and forward_metrics["end_to_end_acc"] >= 0.80)
    reverse_pass = bool(
        reverse_metrics["exact_call_acc"] >= 0.80
        and reverse_metrics["end_to_end_acc"] >= 0.80)
    trajectory_pass = bool(
        l20["cosine"] >= 0.80 and l20["error"] <= 0.60)
    return {
        "summary": {
            "norm": float(direction.norm()),
            "effect": effect,
            "natural_effect": natural_effect,
            "ratio": ratio,
            "positive_fraction": positive_fraction,
            "null_exceedances": null_exceedances,
            "l20_cosine": l20["cosine"],
            "l20_error": l20["error"],
            "forward_metrics": forward_metrics,
            "reverse_metrics": reverse_metrics,
            "gates": {
                "output": output_pass,
                "workflow": workflow_pass,
                "reverse": reverse_pass,
                "trajectory": trajectory_pass,
            },
            "replay_pass": bool(
                output_pass and workflow_pass and reverse_pass
                and trajectory_pass),
            "workflow_output_pass": bool(
                output_pass and workflow_pass and reverse_pass),
        },
        "forward_workflow": forward_workflow,
        "reverse_workflow": reverse_workflow,
        "raw": {
            "direction": direction,
            "logits": logits,
            "cache": cache,
            "effect_rows": effect_rows,
            "l20_cosine_rows": l20["cosine_rows"],
            "l20_error_rows": l20["error_rows"],
            "null_means": torch.from_numpy(null_means),
            "null_rows": torch.from_numpy(null_rows),
            "null_directions": null_directions,
        },
    }


@torch.no_grad()
def run_delta_orchestration_lexical(
        model_path, out_dir, quantization="8bit", device_map=None,
        seed=0, n_null=N_NULL):
    if model_path != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(f"frozen model mismatch: {model_path}")
    if quantization != "8bit" or seed != 0 or n_null != N_NULL:
        raise ValueError(
            f"frozen config mismatch: quant={quantization} "
            f"seed={seed} null={n_null}")
    os.makedirs(out_dir, exist_ok=True)
    runtime = _assert_runtime()
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization,
        revision=MODEL_REVISION)
    dev = input_device(model)
    rows = _rows()
    train_rows, test_rows = rows[::2], rows[1::2]
    train_calc_texts = _task_texts(tok, train_rows, "calculate")
    train_lookup_texts = _task_texts(tok, train_rows, "lookup")
    test_calc_texts = _task_texts(tok, test_rows, "calculate")
    test_lookup_texts = _task_texts(tok, test_rows, "lookup")
    tr_calc, tr_calc_am = _encode_uniform(tok, train_calc_texts, dev)
    tr_lookup, tr_lookup_am = _encode_uniform(tok, train_lookup_texts, dev)
    te_calc, _ = _encode_uniform(tok, test_calc_texts, dev)
    te_lookup, _ = _encode_uniform(tok, test_lookup_texts, dev)
    if len({
            tr_calc.shape[1], tr_lookup.shape[1],
            te_calc.shape[1], te_lookup.shape[1]}) != 1:
        raise ValueError("train/test prompt lengths differ")
    changed = [[
        idx for idx, (left, right) in enumerate(zip(calc, lookup))
        if left != right
    ] for calc, lookup in zip(
        tr_calc.tolist() + te_calc.tolist(),
        tr_lookup.tolist() + te_lookup.tolist())]
    if any(row != [MODE_POSITION] for row in changed):
        raise ValueError(f"mode alignment failed: {changed}")

    raw_native_calc = _generate(model, tok, test_calc_texts, dev, 8)
    raw_native_lookup = _generate(model, tok, test_lookup_texts, dev, 8)
    native_calc = _evaluate_workflow(
        model, tok, dev, test_rows, "calculate", "calculate", raw_native_calc)
    native_lookup = _evaluate_workflow(
        model, tok, dev, test_rows, "lookup", "lookup", raw_native_lookup)
    native_valid = all(
        value >= 0.90
        for workflow in (native_calc, native_lookup)
        for value in workflow["metrics"].values())

    _, train_calc_cache = _forward(
        model, tr_calc, tr_calc_am, (MODE_POSITION,), (INJECT_LAYER,))
    _, train_lookup_cache = _forward(
        model, tr_lookup, tr_lookup_am, (MODE_POSITION,), (INJECT_LAYER,))
    learned = (
        train_lookup_cache[INJECT_LAYER][:, 0]
        - train_calc_cache[INJECT_LAYER][:, 0]).mean(0)
    embedding = model.get_input_embeddings().weight
    calc_token = int(tr_calc[0, MODE_POSITION])
    lookup_token = int(tr_lookup[0, MODE_POSITION])
    lexical = (
        embedding[lookup_token].detach().float().cpu()
        - embedding[calc_token].detach().float().cpu())
    norm_scale = float(learned.norm() / lexical.norm().clamp(min=EPS))
    optimal_scale = float(
        torch.dot(lexical, learned)
        / torch.dot(lexical, lexical).clamp(min=EPS))
    norm_matched = lexical * norm_scale
    donor_optimal = lexical * optimal_scale

    calc_call_texts = [text + "CALL" for text in test_calc_texts]
    lookup_call_texts = [text + "CALL" for text in test_lookup_texts]
    calc_call_ids, calc_call_am = _encode_uniform(tok, calc_call_texts, dev)
    lookup_call_ids, lookup_call_am = _encode_uniform(
        tok, lookup_call_texts, dev)
    prompt_rows = te_calc.detach().cpu().tolist() + te_lookup.detach().cpu().tolist()
    call_rows = (
        calc_call_ids.detach().cpu().tolist()
        + lookup_call_ids.detach().cpu().tolist())
    if any(
            call[:len(prompt)] != prompt or len(call) != len(prompt) + 1
            for prompt, call in zip(prompt_rows, call_rows)):
        raise ValueError("CALL is not one stable continuation token")
    call_token_id = call_rows[0][-1]
    if any(call[-1] != call_token_id for call in call_rows):
        raise ValueError("CALL token varies across prompts")
    call_last = int(calc_call_ids.shape[1] - 1)
    if call_last != int(lookup_call_ids.shape[1] - 1):
        raise ValueError("CALL-prefix lengths differ")
    clean_logits, clean_cache = _forward(
        model, calc_call_ids, calc_call_am, (MODE_POSITION, call_last), LAYERS)
    natural_logits, natural_cache = _forward(
        model, lookup_call_ids, lookup_call_am,
        (MODE_POSITION, call_last), LAYERS)
    calc_tool_rows = [
        tok.encode(text + " calculator", add_special_tokens=False)
        for text in calc_call_texts]
    lookup_tool_rows = [
        tok.encode(text + " database", add_special_tokens=False)
        for text in lookup_call_texts]
    if any(
            full[:len(prefix)] != prefix or len(full) != len(prefix) + 1
            for prefix, full in zip(
                calc_call_ids.detach().cpu().tolist(), calc_tool_rows)):
        raise ValueError("calculator is not one stable continuation token")
    if any(
            full[:len(prefix)] != prefix or len(full) != len(prefix) + 1
            for prefix, full in zip(
                lookup_call_ids.detach().cpu().tolist(), lookup_tool_rows)):
        raise ValueError("database is not one stable continuation token")
    calc_tool_id = calc_tool_rows[0][-1]
    lookup_tool_id = lookup_tool_rows[0][-1]
    if any(row[-1] != calc_tool_id for row in calc_tool_rows):
        raise ValueError("calculator token varies across prompts")
    if any(row[-1] != lookup_tool_id for row in lookup_tool_rows):
        raise ValueError("database token varies across prompts")
    calc_tool_tensor = torch.full(
        (len(test_rows),), calc_tool_id, dtype=torch.long)
    lookup_tool_tensor = torch.full(
        (len(test_rows),), lookup_tool_id, dtype=torch.long)
    clean_ld = _ld(clean_logits, lookup_tool_tensor, calc_tool_tensor)

    variants = {}
    for idx, (name, direction) in enumerate((
            ("learned", learned),
            ("norm_matched_embedding", norm_matched),
            ("donor_optimal_embedding", donor_optimal))):
        variants[name] = _evaluate_variant(
            model, tok, dev, test_rows, test_calc_texts, test_lookup_texts,
            calc_call_ids, calc_call_am, clean_logits, clean_cache,
            natural_logits, natural_cache, clean_ld,
            calc_tool_tensor, lookup_tool_tensor, direction,
            torch.Generator(device="cpu").manual_seed(
                seed + (919 if idx == 0 else 1201 + idx)),
            n_null)

    positive = variants["learned"]["summary"]
    positive_valid = bool(
        positive["ratio"] is not None
        and 0.70 <= positive["ratio"] <= 1.30
        and positive["positive_fraction"] >= 0.80
        and positive["gates"]["workflow"]
        and positive["gates"]["reverse"]
        and positive["gates"]["trajectory"])
    lexical_variants = [
        variants["norm_matched_embedding"]["summary"],
        variants["donor_optimal_embedding"]["summary"],
    ]
    if not native_valid or not positive_valid:
        verdict = "LEXICAL_CONTROL_INVALID"
    elif any(row["replay_pass"] for row in lexical_variants):
        verdict = "SCALE_MATCHED_LEXICAL_REPLAY"
    elif any(row["workflow_output_pass"] for row in lexical_variants):
        verdict = "LEXICAL_WORKFLOW_REPLAY_WITHOUT_STATE_EQUIVALENCE"
    else:
        verdict = "BEYOND_SCALE_MATCHED_EMBEDDING"

    raw_artifact = "raw_delta_orchestration_lexical.pt"
    preflight = {
        "quantization": quantization,
        "seed": seed,
        "n_null": n_null,
        "inject_layer": INJECT_LAYER,
        "mediation_layer": MEDIATION_LAYER,
        "layers": list(LAYERS),
        "mode_position": MODE_POSITION,
        "call_position": call_last,
        "call_token_id": int(call_token_id),
        "calculate_mode_token_id": int(tr_calc[0, MODE_POSITION]),
        "lookup_mode_token_id": int(tr_lookup[0, MODE_POSITION]),
        "calculate_tool_token_id": int(calc_tool_id),
        "database_tool_token_id": int(lookup_tool_id),
        "changed_positions": changed,
        "prompts": {
            "train_calculate": train_calc_texts,
            "train_lookup": train_lookup_texts,
            "test_calculate": test_calc_texts,
            "test_lookup": test_lookup_texts,
        },
    }
    torch.save({
        "learned_direction": learned,
        "raw_embedding_direction": lexical,
        "norm_matched_direction": norm_matched,
        "donor_optimal_direction": donor_optimal,
        "train_calculate_states": train_calc_cache,
        "train_lookup_states": train_lookup_cache,
        "clean_logits": clean_logits,
        "natural_logits": natural_logits,
        "clean_cache": clean_cache,
        "natural_cache": natural_cache,
        "input_ids": {
            "train_calculate": tr_calc.detach().cpu(),
            "train_lookup": tr_lookup.detach().cpu(),
            "test_calculate": te_calc.detach().cpu(),
            "test_lookup": te_lookup.detach().cpu(),
            "call_calculate": calc_call_ids.detach().cpu(),
            "call_lookup": lookup_call_ids.detach().cpu(),
        },
        "variants": {
            name: value["raw"] for name, value in variants.items()},
    }, os.path.join(out_dir, raw_artifact))
    result = {
        "stage": "delta_orchestration_lexical",
        "model_path": model_path,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "seed": seed,
        "n_null": n_null,
        "quantization": quantization,
        "preflight": preflight,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "native_calculate": native_calc,
        "native_lookup": native_lookup,
        "scales": {
            "learned_norm": float(learned.norm()),
            "embedding_norm": float(lexical.norm()),
            "norm_match_scale": norm_scale,
            "donor_optimal_scale": optimal_scale,
            "learned_embedding_cosine": float(torch.nn.functional.cosine_similarity(
                learned.unsqueeze(0), lexical.unsqueeze(0))),
        },
        "variants": {
            name: {
                "summary": value["summary"],
                "forward_workflow": value["forward_workflow"],
                "reverse_workflow": value["reverse_workflow"],
            } for name, value in variants.items()},
        "raw_tensor_artifact": raw_artifact,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_orchestration_lexical.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    log(f"scales={result['scales']}")
    log(f"learned={variants['learned']['summary']}")
    log(f"norm_matched={variants['norm_matched_embedding']['summary']}")
    log(f"donor_optimal={variants['donor_optimal_embedding']['summary']}")
    log(f"VERDICT: {verdict}")
    return result
