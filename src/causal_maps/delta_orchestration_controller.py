"""Causal workflow switch from calculator to database tool use."""
import importlib.metadata
import json
import os

import numpy as np
import torch

from .delta_orchestration_screen import (
    MODEL_REVISION, _correct_action, _execute, _expected_answer,
    _expected_call, _generate, _normalize, _parse_call, _render, _rows,
    _task_content)
from .delta_reasoning_controller import _candidate_metrics
from .delta_trajectory import _cos_rows, _forward, _ld
from .logutil import log
from .model_utils import (
    _ensure_bitsandbytes, input_device, load_model_and_tokenizer)
from .nulls import permutation_pvalue

INJECT_LAYER = 2
MEDIATION_LAYER = 20
LAYERS = (2, 8, 14, 20, 26)
MODE_POSITION = 97
N_NULL = 100
EPS = 1e-8
EXPECTED_RUNTIME = {
    "torch": "2.10.0+cu128",
    "transformers": "5.0.0",
    "bitsandbytes": "0.49.2",
}


def _assert_runtime():
    base = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }
    expected_base = {
        key: EXPECTED_RUNTIME[key] for key in ("torch", "transformers")}
    if base != expected_base:
        raise RuntimeError(
            f"frozen runtime mismatch: expected={expected_base}, got={base}")
    _ensure_bitsandbytes()
    runtime = {
        **base,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
    }
    if runtime != EXPECTED_RUNTIME:
        raise RuntimeError(
            f"frozen runtime mismatch: expected={EXPECTED_RUNTIME}, got={runtime}")
    return runtime


def _safe_ratio(numerator, denominator):
    if denominator <= EPS:
        return None
    return numerator / denominator


def _between(value, low, high):
    return value is not None and low <= value <= high


def _verdict(gates):
    core = ("G0", "A1", "O1", "W1", "R1", "Q1", "M1", "M2")
    if all(gates[key] for key in core):
        return (
            "LATENT_ORCHESTRATION_CONTROLLER"
            if gates["B1"] else "LEXICAL_ORCHESTRATION_REPLAY")
    if gates["O1"] and gates["W1"] and not all(
            gates[key] for key in ("Q1", "M1", "M2")):
        return "ORCHESTRATION_ALTERNATE_PATH"
    if gates["O1"] and gates["W1"]:
        return "ORCHESTRATION_OPERATOR_AMBIGUOUS"
    return "ORCHESTRATION_CONTROL_NULL"


def _task_texts(tok, rows, mode):
    return [
        _render(tok, [{"role": "user", "content": _task_content(row, mode)}])
        for row in rows]


def _encode_uniform(tok, texts, dev):
    encoded = [tok.encode(text, add_special_tokens=False) for text in texts]
    lengths = {len(ids) for ids in encoded}
    if len(lengths) != 1:
        raise ValueError(f"nonuniform causal prompts: {sorted(lengths)}")
    ids = torch.tensor(encoded, dtype=torch.long, device=dev)
    return ids, torch.ones_like(ids)


def _tool_decision_batch(tok, texts, dev):
    """Teacher-force the token prefix shared by the two possible tool calls.

    This is equivalent to appending `CALL` for Qwen, but remains valid for
    tokenizers that resegment `CALL` when the following tool name is added.
    """
    prefixes, calc_next, lookup_next, audits = [], [], [], []
    for text in texts:
        calc = tok.encode(
            text + "CALL calculator", add_special_tokens=False)
        lookup = tok.encode(
            text + "CALL database", add_special_tokens=False)
        common = 0
        for left, right in zip(calc, lookup):
            if left != right:
                break
            common += 1
        if common == 0 or common >= min(len(calc), len(lookup)):
            raise ValueError(
                f"tool alternatives have invalid shared prefix: {common}")
        prefix = calc[:common]
        prefixes.append(prefix)
        calc_next.append(calc[common])
        lookup_next.append(lookup[common])
        audits.append({
            "shared_prefix_length": common,
            "shared_suffix_text": tok.decode(prefix[-4:]),
            "calculate_next_id": int(calc[common]),
            "calculate_next_text": tok.decode([calc[common]]),
            "lookup_next_id": int(lookup[common]),
            "lookup_next_text": tok.decode([lookup[common]]),
        })
    if len({len(row) for row in prefixes}) != 1:
        raise ValueError("tool-decision prefix lengths vary by row")
    if len(set(calc_next)) != 1 or len(set(lookup_next)) != 1:
        raise ValueError("first differentiating tool tokens vary by row")
    ids = torch.tensor(prefixes, dtype=torch.long, device=dev)
    return (ids, torch.ones_like(ids), int(calc_next[0]),
            int(lookup_next[0]), audits)


@torch.no_grad()
def _generate_with_add(model, tok, texts, dev, mode_pos, direction,
                       inject_layer=INJECT_LAYER,
                       max_new_tokens=8):
    ids, am = _encode_uniform(tok, texts, dev)
    prompt_length = ids.shape[1]
    batch_direction = direction.unsqueeze(0).expand(ids.shape[0], -1)
    eos_ids = {int(tok.eos_token_id)}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if (im_end is not None
            and tok.convert_ids_to_tokens(int(im_end)) == "<|im_end|>"):
        eos_ids.add(int(im_end))
    finished = torch.zeros(ids.shape[0], dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits, _ = _forward(
            model, ids, am, (mode_pos,),
            add=(inject_layer, mode_pos, batch_direction))
        next_ids = logits.argmax(-1).long()
        next_ids[finished] = int(tok.eos_token_id)
        finished |= torch.tensor(
            [int(token) in eos_ids for token in next_ids], dtype=torch.bool)
        ids = torch.cat([ids, next_ids.to(dev).unsqueeze(1)], dim=1)
        am = torch.cat([
            am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=dev)
        ], dim=1)
        if bool(finished.all()):
            break
    continuation = ids[:, prompt_length:].detach().cpu()
    return [
        tok.decode(row.tolist(), skip_special_tokens=False)
        for row in continuation]


@torch.no_grad()
def _evaluate_workflow(model, tok, dev, rows, prompt_mode, target_mode,
                       raw_calls, content_fn=_task_content):
    calls = [_normalize(text) for text in raw_calls]
    parsed = [_parse_call(call) for call in calls]
    correct_actions = [
        _correct_action(row, target_mode, parsed_call)
        for row, parsed_call in zip(rows, parsed)]
    tool_results = [_execute(parsed_call) for parsed_call in parsed]
    followups = []
    for row, call, result in zip(rows, calls, tool_results):
        followups.append(_render(tok, [
            {"role": "user", "content": content_fn(row, prompt_mode)},
            {"role": "assistant", "content": call},
            {"role": "user", "content": (
                f"Tool result: {result if result is not None else 'ERROR'}. "
                "Return only the final answer.")},
        ]))
    raw_answers = _generate(model, tok, followups, dev, 4)
    answers = [_normalize(text) for text in raw_answers]
    expected_calls = [_expected_call(row, target_mode) for row in rows]
    expected_answers = [_expected_answer(row, target_mode) for row in rows]
    exact_calls = [
        call == expected for call, expected in zip(calls, expected_calls)]
    correct_answers = [
        answer == expected
        for answer, expected in zip(answers, expected_answers)]
    answer_matches_result = [
        result is not None and answer == result
        for answer, result in zip(answers, tool_results)]
    end_to_end = [
        action and task_answer and result_answer
        for action, task_answer, result_answer in zip(
            correct_actions, correct_answers, answer_matches_result)]
    return {
        "metrics": {
            "exact_call_acc": sum(exact_calls) / len(rows),
            "correct_action_acc": sum(correct_actions) / len(rows),
            "final_answer_acc": sum(correct_answers) / len(rows),
            "answer_matches_tool_result_acc": (
                sum(answer_matches_result) / len(rows)),
            "end_to_end_acc": sum(end_to_end) / len(rows),
        },
        "rows": [{
            "raw_call": raw_calls[idx],
            "call": calls[idx],
            "parsed_call": parsed[idx],
            "tool_result": tool_results[idx],
            "expected_call": expected_calls[idx],
            "raw_answer": raw_answers[idx],
            "answer": answers[idx],
            "expected_answer": expected_answers[idx],
            "exact_call": exact_calls[idx],
            "correct_action": correct_actions[idx],
            "correct_answer": correct_answers[idx],
            "answer_matches_tool_result": answer_matches_result[idx],
            "end_to_end": end_to_end[idx],
        } for idx in range(len(rows))],
    }


@torch.no_grad()
def run_delta_orchestration_controller(
        model_path, out_dir, quantization="8bit", device_map=None,
        seed=0, n_null=N_NULL, confirmation=False):
    if not confirmation and model_path != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(f"frozen model mismatch: {model_path}")
    if quantization != "8bit" or seed != 0 or n_null != N_NULL:
        raise ValueError(
            f"frozen config mismatch: quant={quantization} "
            f"seed={seed} null={n_null}")
    os.makedirs(out_dir, exist_ok=True)
    runtime = _assert_runtime()
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization,
        revision=None if confirmation else MODEL_REVISION)
    if confirmation:
        if int(model.config.num_hidden_layers) != 32:
            raise ValueError(
                "frozen confirmation expects a 32-layer Mistral-family model")
        inject_layer = 2
        mediation_layer = 23
        layers = (2, 9, 16, 23, 30)
        model_revision = None
        stage_name = "delta_orchestration_cross_model_confirmation"
        result_name = "results_delta_orchestration_cross_model_confirmation.json"
    else:
        inject_layer = INJECT_LAYER
        mediation_layer = MEDIATION_LAYER
        layers = LAYERS
        model_revision = MODEL_REVISION
        stage_name = "delta_orchestration_controller"
        result_name = "results_delta_orchestration_controller.json"
    dev = input_device(model)
    all_rows = _rows()
    train_rows, test_rows = all_rows[::2], all_rows[1::2]
    train_calc_texts = _task_texts(tok, train_rows, "calculate")
    train_lookup_texts = _task_texts(tok, train_rows, "lookup")
    test_calc_texts = _task_texts(tok, test_rows, "calculate")
    test_lookup_texts = _task_texts(tok, test_rows, "lookup")
    tr_calc, tr_calc_am = _encode_uniform(tok, train_calc_texts, dev)
    tr_lookup, tr_lookup_am = _encode_uniform(tok, train_lookup_texts, dev)
    te_calc, te_calc_am = _encode_uniform(tok, test_calc_texts, dev)
    te_lookup, te_lookup_am = _encode_uniform(tok, test_lookup_texts, dev)
    all_pairs = [
        *(zip(tr_calc.tolist(), tr_lookup.tolist())),
        *(zip(te_calc.tolist(), te_lookup.tolist())),
    ]
    changed_positions = [[
        idx for idx, (left, right) in enumerate(zip(calc, lookup))
        if left != right
    ] for calc, lookup in all_pairs]
    if confirmation:
        unique_positions = {tuple(changed) for changed in changed_positions}
        if len(unique_positions) != 1 or len(next(iter(unique_positions))) != 1:
            raise ValueError(f"confirmation mode position mismatch: {changed_positions}")
        mode_position = next(iter(unique_positions))[0]
    else:
        mode_position = MODE_POSITION
        if any(changed != [mode_position] for changed in changed_positions):
            raise ValueError(f"frozen mode position mismatch: {changed_positions}")
    (calc_call_ids, calc_call_am, calc_tool_id, lookup_tool_id,
     calc_decision_audit) = _tool_decision_batch(tok, test_calc_texts, dev)
    (lookup_call_ids, lookup_call_am, lookup_calc_tool_id,
     lookup_lookup_tool_id,
     lookup_decision_audit) = _tool_decision_batch(
         tok, test_lookup_texts, dev)
    if (calc_tool_id != lookup_calc_tool_id
            or lookup_tool_id != lookup_lookup_tool_id):
        raise ValueError("tool decision tokens differ across prompt modes")
    call_token_id = int(calc_call_ids[0, -1])
    call_last = int(calc_call_ids.shape[1] - 1)
    if call_last != int(lookup_call_ids.shape[1] - 1):
        raise ValueError("CALL-prefix lengths differ")
    calc_tool_tensor = torch.full(
        (len(test_rows),), int(calc_tool_id), dtype=torch.long)
    lookup_tool_tensor = torch.full(
        (len(test_rows),), int(lookup_tool_id), dtype=torch.long)
    preflight = {
        "model_revision": model_revision,
        "runtime": runtime,
        "quantization": quantization,
        "seed": seed,
        "n_null": n_null,
        "layers": list(layers),
        "inject_layer": inject_layer,
        "mediation_layer": mediation_layer,
        "mode_position": mode_position,
        "call_position": call_last,
        "call_token_id": int(call_token_id),
        "prompt_length": int(te_calc.shape[1]),
        "changed_positions": changed_positions,
        "calculate_mode_token_id": int(te_calc[0, mode_position]),
        "lookup_mode_token_id": int(te_lookup[0, mode_position]),
        "calculate_tool_token_id": int(calc_tool_id),
        "database_tool_token_id": int(lookup_tool_id),
        "tool_decision_definition": "first token after longest shared call prefix",
        "calculate_decision_audit": calc_decision_audit,
        "lookup_decision_audit": lookup_decision_audit,
        "train_input_ids": {
            "calculate": tr_calc.detach().cpu().tolist(),
            "lookup": tr_lookup.detach().cpu().tolist(),
        },
        "test_input_ids": {
            "calculate": te_calc.detach().cpu().tolist(),
            "lookup": te_lookup.detach().cpu().tolist(),
        },
        "train_prompts": {
            "calculate": train_calc_texts,
            "lookup": train_lookup_texts,
        },
        "test_prompts": {
            "calculate": test_calc_texts,
            "lookup": test_lookup_texts,
        },
    }

    raw_native_calc = _generate(model, tok, test_calc_texts, dev, 8)
    raw_native_lookup = _generate(model, tok, test_lookup_texts, dev, 8)
    native_calc = _evaluate_workflow(
        model, tok, dev, test_rows, "calculate", "calculate",
        raw_native_calc)
    native_lookup = _evaluate_workflow(
        model, tok, dev, test_rows, "lookup", "lookup",
        raw_native_lookup)
    g0 = all(
        value >= 0.90
        for workflow in (native_calc, native_lookup)
        for value in workflow["metrics"].values())
    if not g0:
        result = {
            "stage": stage_name,
            "model_path": model_path,
            "preflight": preflight,
            "native_calculate": native_calc,
            "native_lookup": native_lookup,
            "gates": {"G0": False},
            "verdict": "ORCHESTRATION_INELICITABLE",
        }
        with open(os.path.join(
                out_dir, result_name), "w") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        log(f"VERDICT: ORCHESTRATION_INELICITABLE "
            f"{native_calc['metrics']} {native_lookup['metrics']}")
        return result

    _, tr_calc_cache = _forward(
        model, tr_calc, tr_calc_am, (mode_position,), (inject_layer,))
    _, tr_lookup_cache = _forward(
        model, tr_lookup, tr_lookup_am, (mode_position,), (inject_layer,))
    direction = (
        tr_lookup_cache[inject_layer][:, 0]
        - tr_calc_cache[inject_layer][:, 0]).mean(0)
    batch_direction = direction.unsqueeze(0).expand(len(test_rows), -1)
    calc_token_id = int(te_calc[0, mode_position])
    lookup_token_id = int(te_lookup[0, mode_position])
    embedding = model.get_input_embeddings().weight
    embed_direction = (
        embedding[lookup_token_id].detach().float().cpu()
        - embedding[calc_token_id].detach().float().cpu())
    batch_embed = embed_direction.unsqueeze(0).expand(len(test_rows), -1)

    clean_logits, clean_cache = _forward(
        model, calc_call_ids, calc_call_am, (mode_position, call_last), layers)
    natural_logits, natural_cache = _forward(
        model, lookup_call_ids, lookup_call_am,
        (mode_position, call_last), layers)
    add_logits, add_cache = _forward(
        model, calc_call_ids, calc_call_am, (mode_position, call_last), layers,
        add=(inject_layer, mode_position, batch_direction))
    embed_logits, embed_cache = _forward(
        model, calc_call_ids, calc_call_am, (mode_position, call_last), layers,
        add=(inject_layer, mode_position, batch_embed))
    clean_ld = _ld(clean_logits, lookup_tool_tensor, calc_tool_tensor)
    natural_rows = _ld(
        natural_logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld
    add_rows = _ld(
        add_logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld
    embed_rows = _ld(
        embed_logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld
    natural_effect = float(natural_rows.mean())
    add_effect = float(add_rows.mean())
    embed_effect = float(embed_rows.mean())
    output_ratio = _safe_ratio(add_effect, natural_effect)
    embed_ratio = _safe_ratio(embed_effect, natural_effect)

    trajectory, embed_trajectory = {}, {}
    trajectory_rows, embed_trajectory_rows = {}, {}
    for layer in layers:
        native = natural_cache[layer][:, 1] - clean_cache[layer][:, 1]
        active = add_cache[layer][:, 1] - clean_cache[layer][:, 1]
        baseline = embed_cache[layer][:, 1] - clean_cache[layer][:, 1]
        active_metric = _candidate_metrics(active, native)
        baseline_metric = _candidate_metrics(baseline, native)
        trajectory[str(layer)] = {
            "cosine": active_metric["cosine"],
            "error": active_metric["error"],
        }
        embed_trajectory[str(layer)] = {
            "cosine": baseline_metric["cosine"],
            "error": baseline_metric["error"],
        }
        trajectory_rows[str(layer)] = {
            "cosine": active_metric["cosine_rows"].tolist(),
            "error": active_metric["error_rows"].tolist(),
        }
        embed_trajectory_rows[str(layer)] = {
            "cosine": baseline_metric["cosine_rows"].tolist(),
            "error": baseline_metric["error_rows"].tolist(),
        }
    native_local = (
        natural_cache[inject_layer][:, 0]
        - clean_cache[inject_layer][:, 0])
    active_local = (
        add_cache[inject_layer][:, 0]
        - clean_cache[inject_layer][:, 0])
    local_metric = _candidate_metrics(active_local, native_local)

    raw_steered = _generate_with_add(
        model, tok, test_calc_texts, dev, mode_position, direction,
        inject_layer=inject_layer)
    raw_embed = _generate_with_add(
        model, tok, test_calc_texts, dev, mode_position, embed_direction,
        inject_layer=inject_layer)
    raw_reverse = _generate_with_add(
        model, tok, test_lookup_texts, dev, mode_position, -direction,
        inject_layer=inject_layer)
    steered_workflow = _evaluate_workflow(
        model, tok, dev, test_rows, "calculate", "lookup", raw_steered)
    embed_workflow = _evaluate_workflow(
        model, tok, dev, test_rows, "calculate", "lookup", raw_embed)
    reverse_workflow = _evaluate_workflow(
        model, tok, dev, test_rows, "lookup", "calculate", raw_reverse)

    patch_add_logits, _ = _forward(
        model, calc_call_ids, calc_call_am, (mode_position, call_last),
        patch=(mediation_layer, call_last,
               add_cache[mediation_layer][:, 1]))
    patch_natural_logits, _ = _forward(
        model, calc_call_ids, calc_call_am, (mode_position, call_last),
        patch=(mediation_layer, call_last,
               natural_cache[mediation_layer][:, 1]))
    block_add_logits, _ = _forward(
        model, calc_call_ids, calc_call_am, (mode_position, call_last),
        add=(inject_layer, mode_position, batch_direction),
        patch=(mediation_layer, call_last,
               clean_cache[mediation_layer][:, 1]))
    block_natural_logits, _ = _forward(
        model, lookup_call_ids, lookup_call_am, (mode_position, call_last),
        patch=(mediation_layer, call_last,
               clean_cache[mediation_layer][:, 1]))
    patch_add_rows = (
        _ld(patch_add_logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld)
    patch_natural_rows = (
        _ld(
            patch_natural_logits, lookup_tool_tensor, calc_tool_tensor)
        - clean_ld)
    blocked_add_rows = (
        _ld(block_add_logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld)
    blocked_natural_rows = (
        _ld(
            block_natural_logits, lookup_tool_tensor, calc_tool_tensor)
        - clean_ld)
    patch_add = float(patch_add_rows.mean())
    patch_natural = float(patch_natural_rows.mean())
    patch_ratio = _safe_ratio(patch_add, patch_natural)
    blocked_add = float(blocked_add_rows.mean())
    blocked_natural = float(blocked_natural_rows.mean())
    add_block_fraction = _safe_ratio(add_effect - blocked_add, add_effect)
    natural_block_fraction = _safe_ratio(
        natural_effect - blocked_natural, natural_effect)
    block_gap = (
        abs(add_block_fraction - natural_block_fraction)
        if add_block_fraction is not None
        and natural_block_fraction is not None else None)

    generator = torch.Generator(device="cpu").manual_seed(seed + 919)
    null_output = np.zeros(n_null)
    null_embed_output = np.zeros(n_null)
    null_local_cos = np.zeros(n_null)
    null_output_rows = np.zeros((n_null, len(test_rows)))
    null_embed_output_rows = np.zeros((n_null, len(test_rows)))
    null_local_cos_rows = np.zeros((n_null, len(test_rows)))
    null_directions = torch.zeros((n_null, direction.numel()))
    null_embed_directions = torch.zeros((n_null, embed_direction.numel()))
    for idx in range(n_null):
        random = torch.randn(
            direction.shape, generator=generator, dtype=torch.float32)
        random = random / random.norm() * direction.norm()
        null_directions[idx] = random
        random_batch = random.unsqueeze(0).expand(len(test_rows), -1)
        random_logits, random_cache = _forward(
            model, calc_call_ids, calc_call_am, (mode_position, call_last),
            (inject_layer,),
            add=(inject_layer, mode_position, random_batch))
        random_effect_rows = (
            _ld(random_logits, lookup_tool_tensor, calc_tool_tensor) - clean_ld)
        null_output_rows[idx] = random_effect_rows.numpy()
        null_output[idx] = float(random_effect_rows.mean())
        random_local = (
            random_cache[inject_layer][:, 0]
            - clean_cache[inject_layer][:, 0])
        random_local_cos_rows = _cos_rows(random_local, native_local)
        null_local_cos_rows[idx] = random_local_cos_rows.numpy()
        null_local_cos[idx] = float(random_local_cos_rows.mean())
        random_embed = torch.randn(
            embed_direction.shape, generator=generator, dtype=torch.float32)
        random_embed = random_embed / random_embed.norm() * embed_direction.norm()
        null_embed_directions[idx] = random_embed
        random_embed_batch = random_embed.unsqueeze(0).expand(
            len(test_rows), -1)
        random_embed_logits, _ = _forward(
            model, calc_call_ids, calc_call_am, (mode_position, call_last),
            add=(inject_layer, mode_position, random_embed_batch))
        random_embed_effect_rows = (
            _ld(
                random_embed_logits, lookup_tool_tensor, calc_tool_tensor)
            - clean_ld)
        null_embed_output_rows[idx] = random_embed_effect_rows.numpy()
        null_embed_output[idx] = float(random_embed_effect_rows.mean())

    output_exceed = int((null_output >= add_effect).sum())
    local_exceed = int(
        (null_local_cos >= local_metric["cosine"]).sum())
    add_positive = float((add_rows > 0).float().mean())
    q20 = trajectory[str(mediation_layer)]
    steered_metrics = steered_workflow["metrics"]
    reverse_metrics = reverse_workflow["metrics"]
    gates = {
        "G0": True,
        "A1": bool(local_metric["cosine"] >= 0.80
                   and local_metric["error"] <= 0.60
                   and local_exceed <= 1),
        "O1": bool(_between(output_ratio, 0.70, 1.30)
                   and add_positive >= 0.80 and output_exceed <= 1
                   and steered_metrics["exact_call_acc"] >= 0.80),
        "W1": bool(steered_metrics["correct_action_acc"] >= 0.80
                   and steered_metrics["final_answer_acc"] >= 0.80
                   and steered_metrics[
                       "answer_matches_tool_result_acc"] >= 0.80
                   and steered_metrics["end_to_end_acc"] >= 0.80),
        "R1": bool(reverse_metrics["exact_call_acc"] >= 0.80
                   and reverse_metrics["end_to_end_acc"] >= 0.80),
        "Q1": bool(q20["cosine"] >= 0.80 and q20["error"] <= 0.60),
        "M1": bool(patch_add > 0 and patch_natural > 0
                   and _between(patch_ratio, 0.70, 1.30)),
        "M2": bool(add_block_fraction is not None
                   and natural_block_fraction is not None
                   and block_gap is not None
                   and add_block_fraction >= 0.70
                   and natural_block_fraction >= 0.70 and block_gap <= 0.20),
    }
    embed_metrics = embed_workflow["metrics"]
    embed_o = bool(
        _between(embed_ratio, 0.70, 1.30)
        and float((embed_rows > 0).float().mean()) >= 0.80
        and int((null_embed_output >= embed_effect).sum()) <= 1
        and embed_metrics["exact_call_acc"] >= 0.80)
    embed_w = bool(
        embed_metrics["correct_action_acc"] >= 0.80
        and embed_metrics["final_answer_acc"] >= 0.80
        and embed_metrics["answer_matches_tool_result_acc"] >= 0.80
        and embed_metrics["end_to_end_acc"] >= 0.80)
    embed_q = bool(
        embed_trajectory[str(mediation_layer)]["cosine"] >= 0.80
        and embed_trajectory[str(mediation_layer)]["error"] <= 0.60)
    gates["B1"] = not (embed_o and embed_w and embed_q)
    verdict = _verdict(gates)

    row_results = []
    for idx, row in enumerate(test_rows):
        row_results.append({
            **row,
            "native_tool_effect": float(natural_rows[idx]),
            "add_tool_effect": float(add_rows[idx]),
            "embed_tool_effect": float(embed_rows[idx]),
            "local_cosine": float(local_metric["cosine_rows"][idx]),
            "local_error": float(local_metric["error_rows"][idx]),
            "patch_add_effect": float(patch_add_rows[idx]),
            "patch_natural_effect": float(patch_natural_rows[idx]),
            "blocked_add_effect": float(blocked_add_rows[idx]),
            "blocked_natural_effect": float(blocked_natural_rows[idx]),
            "trajectory": {
                layer: {
                    "cosine": trajectory_rows[layer]["cosine"][idx],
                    "error": trajectory_rows[layer]["error"][idx],
                    "embed_cosine": embed_trajectory_rows[layer]["cosine"][idx],
                    "embed_error": embed_trajectory_rows[layer]["error"][idx],
                } for layer in trajectory_rows},
            "steered_workflow": steered_workflow["rows"][idx],
            "embedding_workflow": embed_workflow["rows"][idx],
            "reverse_workflow": reverse_workflow["rows"][idx],
        })
    raw_artifact = "raw_delta_orchestration_controller.pt"
    torch.save({
        "direction": direction,
        "embedding_direction": embed_direction,
        "input_ids": {
            "train_calculate": tr_calc.detach().cpu(),
            "train_lookup": tr_lookup.detach().cpu(),
            "test_calculate": te_calc.detach().cpu(),
            "test_lookup": te_lookup.detach().cpu(),
            "call_calculate": calc_call_ids.detach().cpu(),
            "call_lookup": lookup_call_ids.detach().cpu(),
        },
        "logits": {
            "clean": clean_logits,
            "natural": natural_logits,
            "add": add_logits,
            "embedding": embed_logits,
            "patch_add": patch_add_logits,
            "patch_natural": patch_natural_logits,
            "block_add": block_add_logits,
            "block_natural": block_natural_logits,
        },
        "states": {
            "train_calculate": tr_calc_cache,
            "train_lookup": tr_lookup_cache,
            "clean": clean_cache,
            "natural": natural_cache,
            "add": add_cache,
            "embedding": embed_cache,
        },
        "row_effects": {
            "natural": natural_rows,
            "add": add_rows,
            "embedding": embed_rows,
            "patch_add": patch_add_rows,
            "patch_natural": patch_natural_rows,
            "block_add": blocked_add_rows,
            "block_natural": blocked_natural_rows,
        },
        "null": {
            "directions": null_directions,
            "embedding_directions": null_embed_directions,
            "output_rows": torch.from_numpy(null_output_rows),
            "embedding_output_rows": torch.from_numpy(null_embed_output_rows),
            "local_cosine_rows": torch.from_numpy(null_local_cos_rows),
        },
    }, os.path.join(out_dir, raw_artifact))
    result = {
        "stage": stage_name,
        "model_path": model_path,
        "model_revision": model_revision,
        "runtime": runtime,
        "preflight": preflight,
        "raw_tensor_artifact": raw_artifact,
        "train_rows": train_rows,
        "test_rows": row_results,
        "native_calculate": native_calc,
        "native_lookup": native_lookup,
        "direction_norm": float(direction.norm()),
        "embedding_direction_norm": float(embed_direction.norm()),
        "local": {
            "cosine": local_metric["cosine"],
            "error": local_metric["error"],
            "null_exceedances": local_exceed,
        },
        "tool_output": {
            "natural_effect": natural_effect,
            "add_effect": add_effect,
            "ratio": output_ratio,
            "positive_fraction": add_positive,
            "null_exceedances": output_exceed,
            "p": float(permutation_pvalue(
                add_effect, null_output, "greater")),
            "embed_effect": embed_effect,
            "embed_ratio": embed_ratio,
        },
        "steered_workflow_metrics": steered_metrics,
        "embedding_workflow_metrics": embed_metrics,
        "reverse_workflow_metrics": reverse_metrics,
        "trajectory": trajectory,
        "embedding_trajectory": embed_trajectory,
        "mediation": {
            "patch_add_effect": patch_add,
            "patch_natural_effect": patch_natural,
            "patch_ratio": patch_ratio,
            "blocked_add_effect": blocked_add,
            "blocked_natural_effect": blocked_natural,
            "add_block_fraction": add_block_fraction,
            "natural_block_fraction": natural_block_fraction,
            "block_gap": block_gap,
        },
        "null": {
            "output": null_output.tolist(),
            "embedding_output": null_embed_output.tolist(),
            "local_cosine": null_local_cos.tolist(),
        },
        "embedding_baseline_gates": {
            "O1_like": embed_o, "W1": embed_w, "Q1": embed_q},
        "gates": gates,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, result_name), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    log(f"steered={steered_metrics} reverse={reverse_metrics}")
    log(f"tool ratio={output_ratio} L{mediation_layer}={q20} gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
