"""Cross-stage interchange test for an agent policy representation."""
import hashlib
import inspect
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .delta_continuous_orchestration import (
    _answer_ids, _followups, _output_metrics, _score)
from .delta_orchestration_controller import (
    EPS, N_NULL, _assert_runtime, _encode_uniform, _safe_ratio)
from .delta_orchestration_label_transfer import (
    _template_b_content, _template_b_texts)
from .delta_orchestration_screen import (
    MODEL_REVISION, _execute, _expected_call, _normalize, _parse_call,
    _render, _rows)
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

POLICY_LAYER = 20
N_DIAGNOSTIC = 8
PROTOCOL_VERSION = "2026-07-13-v1"
PROTOCOL_SPEC = {
    "layer": POLICY_LAYER,
    "split": "even donor / odd test",
    "diagnostic_rows": N_DIAGNOSTIC,
    "geometry_cosine": 0.80,
    "behavior_accuracy": 0.875,
    "output_ratio": [0.70, 1.30],
    "positive_fraction": 0.75,
    "null_exceedances": 1,
    "cross_scale": "donor direction to recipient donor-reference norm",
    "call_application": "first tool-token prediction only",
}


def _mean_direction(target_cache, source_cache):
    return (target_cache[POLICY_LAYER][:, 0]
            - source_cache[POLICY_LAYER][:, 0]).mean(0)


def _to_norm(direction, reference):
    return (
        direction / direction.norm().clamp(min=EPS)
        * reference.norm())


def _continuation_id(tok, texts, continuation):
    found = set()
    for text in texts:
        prefix = tok.encode(text, add_special_tokens=False)
        full = tok.encode(text + continuation, add_special_tokens=False)
        if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(
                f"unstable continuation {continuation!r}: "
                f"{full[len(prefix):]}")
        found.add(full[-1])
    if len(found) != 1:
        raise ValueError(
            f"continuation token varies for {continuation!r}: {found}")
    return next(iter(found))


def _pass_output(metrics, null_exceedances=None):
    ratio = metrics["ratio"]
    passed = bool(
        ratio is not None and 0.70 <= ratio <= 1.30
        and metrics["positive_fraction"] >= 0.75)
    if null_exceedances is not None:
        passed &= null_exceedances <= 1
    return passed


@torch.no_grad()
def _generate_at_decision(
        model, tok, texts, dev, position, direction,
        max_new_tokens=4, first_step_only=False):
    ids, am = _encode_uniform(tok, texts, dev)
    prompt_length = ids.shape[1]
    batch = direction.unsqueeze(0).expand(ids.shape[0], -1)
    eos_ids = {int(tok.eos_token_id)}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        eos_ids.add(int(im_end))
    finished = torch.zeros(ids.shape[0], dtype=torch.bool)
    for step in range(max_new_tokens):
        add = (
            (POLICY_LAYER, position, batch)
            if not first_step_only or step == 0 else None)
        logits, _ = _forward(model, ids, am, (position,), add=add)
        next_ids = logits.argmax(-1).long()
        next_ids[finished] = int(tok.eos_token_id)
        finished |= torch.tensor(
            [int(token) in eos_ids for token in next_ids], dtype=torch.bool)
        ids = torch.cat([ids, next_ids.to(dev).unsqueeze(1)], dim=1)
        am = torch.cat([
            am, torch.ones(
                (am.shape[0], 1), dtype=am.dtype, device=dev)
        ], dim=1)
        if bool(finished.all()):
            break
    continuation = ids[:, prompt_length:].detach().cpu()
    return (
        [tok.decode(row.tolist(), skip_special_tokens=False)
         for row in continuation],
        continuation,
    )


def _score_calls(raw_continuations, rows, diagnostic):
    calls = [_normalize("CALL" + text) for text in raw_continuations]
    expected = [_expected_call(row, "lookup") for row in rows]
    correct = [call == target for call, target in zip(calls, expected)]
    diag_idx = [idx for idx, keep in enumerate(diagnostic) if keep]
    return {
        "exact_call_acc": sum(correct) / len(correct),
        "diagnostic_exact_call_acc": (
            sum(correct[idx] for idx in diag_idx) / len(diag_idx)),
        "calls": calls,
        "expected": expected,
        "parsed": [_parse_call(call) for call in calls],
        "correct_rows": correct,
    }


def _null_exceedances(
        model, ids, am, position, direction, clean_logits,
        target_ids, source_ids, diagnostic, seed, n_null):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mask = torch.tensor(diagnostic, dtype=torch.bool)
    clean_ld = _ld(clean_logits, target_ids, source_ids)[mask]
    means = np.zeros(n_null)
    rows = np.zeros((n_null, int(mask.sum())))
    directions = torch.zeros((n_null, direction.numel()))
    for idx in range(n_null):
        random = torch.randn(
            direction.shape, generator=generator, dtype=torch.float32)
        random = random / random.norm().clamp(min=EPS) * direction.norm()
        directions[idx] = random
        logits, _ = _forward(
            model, ids, am, (position,),
            add=(POLICY_LAYER, position, random.unsqueeze(0).expand(
                ids.shape[0], -1)))
        effect = _ld(logits, target_ids, source_ids)[mask] - clean_ld
        rows[idx] = effect.numpy()
        means[idx] = float(effect.mean())
    return means, rows, directions


@torch.no_grad()
def run_delta_agent_policy_broadcast(
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
    target_answers = [str(row["database_value"]) for row in test_rows]
    source_answers = [str(row["a"] + row["b"]) for row in test_rows]
    diagnostic = [
        target != source
        for target, source in zip(target_answers, source_answers)]
    if sum(diagnostic) != N_DIAGNOSTIC:
        raise ValueError(f"diagnostic row mismatch: {diagnostic}")

    call_train_source_texts = [
        text + "CALL"
        for text in _template_b_texts(tok, train_rows, "calculate")]
    call_train_target_texts = [
        text + "CALL"
        for text in _template_b_texts(tok, train_rows, "lookup")]
    call_test_source_texts = [
        text + "CALL"
        for text in _template_b_texts(tok, test_rows, "calculate")]
    call_test_target_texts = [
        text + "CALL"
        for text in _template_b_texts(tok, test_rows, "lookup")]
    call_train_source, call_train_source_am = _encode_uniform(
        tok, call_train_source_texts, dev)
    call_train_target, call_train_target_am = _encode_uniform(
        tok, call_train_target_texts, dev)
    call_test_source, call_test_source_am = _encode_uniform(
        tok, call_test_source_texts, dev)
    call_test_target, call_test_target_am = _encode_uniform(
        tok, call_test_target_texts, dev)
    call_position = int(call_train_source.shape[1] - 1)
    if any(int(ids.shape[1] - 1) != call_position for ids in (
            call_train_target, call_test_source, call_test_target)):
        raise ValueError("CALL positions differ")

    answer_train_source_texts, _, _ = _followups(
        tok, train_rows, "calculate", "lookup")
    answer_train_target_texts, _, _ = _followups(
        tok, train_rows, "lookup", "lookup")
    answer_test_source_texts, _, _ = _followups(
        tok, test_rows, "calculate", "lookup")
    answer_test_target_texts, _, _ = _followups(
        tok, test_rows, "lookup", "lookup")
    answer_train_source, answer_train_source_am = _encode_uniform(
        tok, answer_train_source_texts, dev)
    answer_train_target, answer_train_target_am = _encode_uniform(
        tok, answer_train_target_texts, dev)
    answer_test_source, answer_test_source_am = _encode_uniform(
        tok, answer_test_source_texts, dev)
    answer_test_target, answer_test_target_am = _encode_uniform(
        tok, answer_test_target_texts, dev)
    answer_position = int(answer_train_source.shape[1] - 1)
    if any(int(ids.shape[1] - 1) != answer_position for ids in (
            answer_train_target, answer_test_source, answer_test_target)):
        raise ValueError("answer positions differ")

    _, call_train_source_cache = _forward(
        model, call_train_source, call_train_source_am,
        (call_position,), (POLICY_LAYER,))
    _, call_train_target_cache = _forward(
        model, call_train_target, call_train_target_am,
        (call_position,), (POLICY_LAYER,))
    _, answer_train_source_cache = _forward(
        model, answer_train_source, answer_train_source_am,
        (answer_position,), (POLICY_LAYER,))
    _, answer_train_target_cache = _forward(
        model, answer_train_target, answer_train_target_am,
        (answer_position,), (POLICY_LAYER,))
    d_call = _mean_direction(
        call_train_target_cache, call_train_source_cache)
    d_answer = _mean_direction(
        answer_train_target_cache, answer_train_source_cache)
    d_answer_to_call = _to_norm(d_answer, d_call)
    d_call_to_answer = _to_norm(d_call, d_answer)
    direction_cosine = float(F.cosine_similarity(
        d_call.unsqueeze(0), d_answer.unsqueeze(0)))

    call_clean_logits, _ = _forward(
        model, call_test_source, call_test_source_am, (call_position,))
    call_natural_logits, _ = _forward(
        model, call_test_target, call_test_target_am, (call_position,))
    call_same_logits, _ = _forward(
        model, call_test_source, call_test_source_am, (call_position,),
        add=(POLICY_LAYER, call_position, d_call.unsqueeze(0).expand(
            len(test_rows), -1)))
    call_cross_logits, _ = _forward(
        model, call_test_source, call_test_source_am, (call_position,),
        add=(POLICY_LAYER, call_position,
             d_answer_to_call.unsqueeze(0).expand(len(test_rows), -1)))
    all_call_texts = (
        call_train_source_texts + call_train_target_texts
        + call_test_source_texts + call_test_target_texts)
    call_target_id = _continuation_id(tok, all_call_texts, " database")
    call_source_id = _continuation_id(tok, all_call_texts, " calculator")
    call_target_ids = torch.full(
        (len(test_rows),), call_target_id, dtype=torch.long)
    call_source_ids = torch.full(
        (len(test_rows),), call_source_id, dtype=torch.long)
    call_same_output = _output_metrics(
        call_clean_logits, call_natural_logits, call_same_logits,
        call_target_ids, call_source_ids, diagnostic)
    call_cross_output = _output_metrics(
        call_clean_logits, call_natural_logits, call_cross_logits,
        call_target_ids, call_source_ids, diagnostic)

    answer_target_ids = _answer_ids(
        tok, answer_test_source_texts, target_answers)
    answer_source_ids = _answer_ids(
        tok, answer_test_source_texts, source_answers)
    answer_clean_logits, _ = _forward(
        model, answer_test_source, answer_test_source_am, (answer_position,))
    answer_natural_logits, _ = _forward(
        model, answer_test_target, answer_test_target_am, (answer_position,))
    answer_same_logits, _ = _forward(
        model, answer_test_source, answer_test_source_am, (answer_position,),
        add=(POLICY_LAYER, answer_position, d_answer.unsqueeze(0).expand(
            len(test_rows), -1)))
    answer_cross_logits, _ = _forward(
        model, answer_test_source, answer_test_source_am, (answer_position,),
        add=(POLICY_LAYER, answer_position,
             d_call_to_answer.unsqueeze(0).expand(len(test_rows), -1)))
    answer_same_output = _output_metrics(
        answer_clean_logits, answer_natural_logits, answer_same_logits,
        answer_target_ids, answer_source_ids, diagnostic)
    answer_cross_output = _output_metrics(
        answer_clean_logits, answer_natural_logits, answer_cross_logits,
        answer_target_ids, answer_source_ids, diagnostic)

    call_same_raw, call_same_tokens = _generate_at_decision(
        model, tok, call_test_source_texts, dev, call_position, d_call,
        max_new_tokens=7, first_step_only=True)
    call_cross_raw, call_cross_tokens = _generate_at_decision(
        model, tok, call_test_source_texts, dev, call_position,
        d_answer_to_call, max_new_tokens=7, first_step_only=True)
    call_same_score = _score_calls(call_same_raw, test_rows, diagnostic)
    call_cross_score = _score_calls(call_cross_raw, test_rows, diagnostic)
    answer_same_raw, answer_same_tokens = _generate_at_decision(
        model, tok, answer_test_source_texts, dev, answer_position,
        d_answer)
    answer_cross_raw, answer_cross_tokens = _generate_at_decision(
        model, tok, answer_test_source_texts, dev, answer_position,
        d_call_to_answer)
    answer_same_score = _score(
        answer_same_raw, target_answers, source_answers, diagnostic)
    answer_cross_score = _score(
        answer_cross_raw, target_answers, source_answers, diagnostic)

    call_null_means, call_null_rows, call_null_directions = _null_exceedances(
        model, call_test_source, call_test_source_am, call_position,
        d_answer_to_call, call_clean_logits, call_target_ids, call_source_ids,
        diagnostic, seed + 1801, n_null)
    answer_null_means, answer_null_rows, answer_null_directions = (
        _null_exceedances(
            model, answer_test_source, answer_test_source_am,
            answer_position, d_call_to_answer, answer_clean_logits,
            answer_target_ids, answer_source_ids, diagnostic,
            seed + 1802, n_null))
    call_exceedances = int(
        (call_null_means >= call_cross_output["effect"]).sum())
    answer_exceedances = int(
        (answer_null_means >= answer_cross_output["effect"]).sum())

    call_same_pass = bool(
        call_same_score["diagnostic_exact_call_acc"] >= 0.875)
    answer_same_pass = bool(
        answer_same_score["diagnostic_target_acc"] >= 0.875)
    geometry_pass = direction_cosine >= 0.80
    call_cross_behavior = bool(
        call_cross_score["diagnostic_exact_call_acc"] >= 0.875)
    call_cross_output_pass = _pass_output(
        call_cross_output, call_exceedances)
    answer_cross_behavior = bool(
        answer_cross_score["diagnostic_target_acc"] >= 0.875)
    answer_cross_output_pass = _pass_output(
        answer_cross_output, answer_exceedances)
    gates = {
        "same_call_behavior": call_same_pass,
        "same_answer_behavior": answer_same_pass,
        "geometry": geometry_pass,
        "answer_to_call_behavior": call_cross_behavior,
        "answer_to_call_output": call_cross_output_pass,
        "call_to_answer_behavior": answer_cross_behavior,
        "call_to_answer_output": answer_cross_output_pass,
    }
    if not call_same_pass or not answer_same_pass:
        verdict = "POLICY_BROADCAST_DIAGNOSTIC_INVALID"
    elif all(gates.values()):
        verdict = "SHARED_AGENT_POLICY_STATE"
    else:
        verdict = "NO_SHARED_POLICY_INTERCHANGE"

    def serializable(metrics):
        return {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()}

    raw_artifact = "raw_delta_agent_policy_broadcast.pt"
    torch.save({
        "directions": {
            "call": d_call,
            "answer": d_answer,
            "answer_to_call": d_answer_to_call,
            "call_to_answer": d_call_to_answer,
        },
        "train_caches": {
            "call_source": call_train_source_cache,
            "call_target": call_train_target_cache,
            "answer_source": answer_train_source_cache,
            "answer_target": answer_train_target_cache,
        },
        "test_logits": {
            "call_clean": call_clean_logits,
            "call_natural": call_natural_logits,
            "call_same": call_same_logits,
            "call_cross": call_cross_logits,
            "answer_clean": answer_clean_logits,
            "answer_natural": answer_natural_logits,
            "answer_same": answer_same_logits,
            "answer_cross": answer_cross_logits,
        },
        "generated_tokens": {
            "call_same": call_same_tokens,
            "call_cross": call_cross_tokens,
            "answer_same": answer_same_tokens,
            "answer_cross": answer_cross_tokens,
        },
        "input_ids": {
            "call_train_source": call_train_source.detach().cpu(),
            "call_train_target": call_train_target.detach().cpu(),
            "call_test_source": call_test_source.detach().cpu(),
            "call_test_target": call_test_target.detach().cpu(),
            "answer_train_source": answer_train_source.detach().cpu(),
            "answer_train_target": answer_train_target.detach().cpu(),
            "answer_test_source": answer_test_source.detach().cpu(),
            "answer_test_target": answer_test_target.detach().cpu(),
        },
        "attention_masks": {
            "call_train_source": call_train_source_am.detach().cpu(),
            "call_train_target": call_train_target_am.detach().cpu(),
            "call_test_source": call_test_source_am.detach().cpu(),
            "call_test_target": call_test_target_am.detach().cpu(),
            "answer_train_source": answer_train_source_am.detach().cpu(),
            "answer_train_target": answer_train_target_am.detach().cpu(),
            "answer_test_source": answer_test_source_am.detach().cpu(),
            "answer_test_target": answer_test_target_am.detach().cpu(),
        },
        "output_ids": {
            "call_target": call_target_ids,
            "call_source": call_source_ids,
            "answer_target": answer_target_ids,
            "answer_source": answer_source_ids,
            "diagnostic": torch.tensor(diagnostic, dtype=torch.bool),
        },
        "nulls": {
            "call_means": torch.from_numpy(call_null_means),
            "call_rows": torch.from_numpy(call_null_rows),
            "call_directions": call_null_directions,
            "answer_means": torch.from_numpy(answer_null_means),
            "answer_rows": torch.from_numpy(answer_null_rows),
            "answer_directions": answer_null_directions,
        },
    }, os.path.join(out_dir, raw_artifact))
    result = {
        "stage": "delta_agent_policy_broadcast",
        "model_path": model_path,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "quantization": quantization,
        "seed": seed,
        "n_null": n_null,
        "provenance": {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_spec_sha256": hashlib.sha256(json.dumps(
                PROTOCOL_SPEC, sort_keys=True).encode()).hexdigest(),
            "source_sha256": hashlib.sha256(
                open(__file__, "rb").read()).hexdigest(),
            "helpers_sha256": hashlib.sha256("".join(
                inspect.getsource(helper) for helper in (
                    _forward, _output_metrics, _answer_ids, _followups,
                    _score, _template_b_content, _template_b_texts,
                    _encode_uniform, _safe_ratio, _ld,
                    _expected_call, _normalize, _parse_call,
                    _execute, _render, _rows)).encode()).hexdigest(),
            "rows_sha256": hashlib.sha256(json.dumps(
                rows, sort_keys=True).encode()).hexdigest(),
        },
        "preflight": {
            "policy_layer": POLICY_LAYER,
            "call_position": call_position,
            "answer_position": answer_position,
            "diagnostic_rows": diagnostic,
            "call_direction_norm": float(d_call.norm()),
            "answer_direction_norm": float(d_answer.norm()),
            "answer_to_call_norm": float(d_answer_to_call.norm()),
            "call_to_answer_norm": float(d_call_to_answer.norm()),
            "direction_cosine": direction_cosine,
            "call_target_token_id": int(call_target_id),
            "call_source_token_id": int(call_source_id),
            "prompts": {
                "call_train_source": call_train_source_texts,
                "call_train_target": call_train_target_texts,
                "call_test_source": call_test_source_texts,
                "call_test_target": call_test_target_texts,
                "answer_train_source": answer_train_source_texts,
                "answer_train_target": answer_train_target_texts,
                "answer_test_source": answer_test_source_texts,
                "answer_test_target": answer_test_target_texts,
            },
        },
        "train_rows": train_rows,
        "test_rows": test_rows,
        "same_stage": {
            "call": {
                "behavior": call_same_score,
                "output": serializable(call_same_output),
                "raw_continuations": call_same_raw,
            },
            "answer": {
                "behavior": answer_same_score,
                "output": serializable(answer_same_output),
                "raw_answers": answer_same_raw,
            },
        },
        "cross_stage": {
            "answer_to_call": {
                "behavior": call_cross_score,
                "output": serializable(call_cross_output),
                "null_exceedances": call_exceedances,
                "raw_continuations": call_cross_raw,
            },
            "call_to_answer": {
                "behavior": answer_cross_score,
                "output": serializable(answer_cross_output),
                "null_exceedances": answer_exceedances,
                "raw_answers": answer_cross_raw,
            },
        },
        "gates": gates,
        "raw_tensor_artifact": raw_artifact,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_agent_policy_broadcast.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    log(f"direction_cosine={direction_cosine:.3f}")
    log(f"same_call={call_same_score} same_answer={answer_same_score}")
    log(f"cross_call={call_cross_score} cross_answer={answer_cross_score}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
