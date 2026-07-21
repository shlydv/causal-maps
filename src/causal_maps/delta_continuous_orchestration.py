"""Answer-stage test of reapplying a template-specific controller."""
import hashlib
import inspect
import json
import os

import numpy as np
import torch

from .delta_orchestration_controller import (
    EPS, INJECT_LAYER, N_NULL, _assert_runtime, _encode_uniform,
    _safe_ratio)
from .delta_orchestration_label_transfer import (
    TEMPLATE_B_POSITION, _template_b_content, _template_b_texts)
from .delta_orchestration_screen import (
    MODEL_REVISION, _execute, _expected_call, _normalize,
    _parse_call, _render, _rows)
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

PROTOCOL_VERSION = "2026-07-13-v1"
PROTOCOL_SPEC = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "model_revision": MODEL_REVISION,
    "runtime": {
        "torch": "2.10.0+cu128",
        "transformers": "5.0.0",
        "bitsandbytes": "0.49.2",
    },
    "quantization": "8bit",
    "seed": 0,
    "n_null": 100,
    "inject_layer": INJECT_LAYER,
    "mode_position": TEMPLATE_B_POSITION,
    "alpha": 1.0,
    "split": "even-index donor / odd-index test",
    "conditions": (
        "natural_target", "natural_source", "source_target_unsteered",
        "source_target_reapplied", "source_target_lexical"),
    "diagnostic_rows": 8,
    "gates": {
        "G0": 0.875,
        "R0_max": 0.625,
        "C1_accuracy": 0.875,
        "C1_gain": 0.25,
        "C2_ratio": [0.70, 1.30],
        "C2_positive": 0.75,
        "C2_null_exceedances": 1,
        "B1": "lexical fails C1 or C2",
    },
    "verdicts": (
        "ANSWER_TURN_LATENT_CONTROL", "ANSWER_TURN_CONTROL_LEXICAL",
        "ANSWER_TURN_DIAGNOSTIC_INVALID", "ANSWER_TURN_CONTROL_NULL"),
}


def _followups(tok, rows, prompt_mode, call_mode):
    texts, calls, results = [], [], []
    for row in rows:
        call = _expected_call(row, call_mode)
        result = _execute(_parse_call(call))
        if result is None:
            raise ValueError(f"frozen call did not execute: {call}")
        texts.append(_render(tok, [
            {"role": "user",
             "content": _template_b_content(row, prompt_mode)},
            {"role": "assistant", "content": call},
            {"role": "user", "content": (
                f"Tool result: {result}. Return only the final answer.")},
        ]))
        calls.append(call)
        results.append(result)
    return texts, calls, results


def _answer_ids(tok, texts, answers):
    ids = []
    for text, answer in zip(texts, answers):
        prefix = tok.encode(text, add_special_tokens=False)
        full = tok.encode(text + answer, add_special_tokens=False)
        continuation = full[len(prefix):]
        if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(
                f"answer is not one contextual token: {answer} "
                f"{continuation}")
        ids.append(continuation[0])
    return torch.tensor(ids, dtype=torch.long)


def _score(raw_answers, target_answers, source_answers, diagnostic):
    answers = [_normalize(answer) for answer in raw_answers]
    target_rows = [
        answer == target for answer, target in zip(answers, target_answers)]
    source_rows = [
        answer == source for answer, source in zip(answers, source_answers)]
    diag_idx = [idx for idx, value in enumerate(diagnostic) if value]
    collision_idx = [idx for idx, value in enumerate(diagnostic) if not value]

    def mean(rows, indices):
        return sum(rows[idx] for idx in indices) / len(indices)

    return {
        "target_acc": sum(target_rows) / len(target_rows),
        "source_acc": sum(source_rows) / len(source_rows),
        "diagnostic_target_acc": mean(target_rows, diag_idx),
        "diagnostic_source_acc": mean(source_rows, diag_idx),
        "collision_exact_acc": mean(target_rows, collision_idx),
        "collision_answers": [answers[idx] for idx in collision_idx],
        "answers": answers,
        "target_rows": target_rows,
        "source_rows": source_rows,
    }


@torch.no_grad()
def _generate_answers(
        model, tok, texts, dev, direction=None, max_new_tokens=4,
        inject_layer=INJECT_LAYER, mode_position=TEMPLATE_B_POSITION):
    ids, am = _encode_uniform(tok, texts, dev)
    prompt_length = ids.shape[1]
    batch_direction = (
        None if direction is None
        else direction.unsqueeze(0).expand(ids.shape[0], -1))
    eos_ids = {int(tok.eos_token_id)}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if (im_end is not None
            and tok.convert_ids_to_tokens(int(im_end)) == "<|im_end|>"):
        eos_ids.add(int(im_end))
    finished = torch.zeros(ids.shape[0], dtype=torch.bool)
    for _ in range(max_new_tokens):
        add = (
            None if batch_direction is None
            else (inject_layer, mode_position, batch_direction))
        logits, _ = _forward(
            model, ids, am, (mode_position,), add=add)
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
    token_ids = continuation.tolist()
    return (
        [tok.decode(row, skip_special_tokens=False) for row in token_ids],
        token_ids,
    )


def _output_metrics(
        clean_logits, natural_logits, candidate_logits,
        target_ids, source_ids, diagnostic):
    mask = torch.tensor(diagnostic, dtype=torch.bool)
    clean_ld = _ld(clean_logits, target_ids, source_ids)[mask]
    natural_rows = (
        _ld(natural_logits, target_ids, source_ids)[mask] - clean_ld)
    effect_rows = (
        _ld(candidate_logits, target_ids, source_ids)[mask] - clean_ld)
    natural_effect = float(natural_rows.mean())
    effect = float(effect_rows.mean())
    return {
        "effect": effect,
        "natural_effect": natural_effect,
        "ratio": _safe_ratio(effect, natural_effect),
        "positive_fraction": float((effect_rows > 0).float().mean()),
        "effect_rows": effect_rows,
        "natural_rows": natural_rows,
        "clean_ld": clean_ld,
    }


def _output_pass(metrics, null_exceedances):
    ratio = metrics["ratio"]
    return bool(
        ratio is not None and 0.70 <= ratio <= 1.30
        and metrics["positive_fraction"] >= 0.75
        and null_exceedances <= 1)


@torch.no_grad()
def run_delta_answer_turn_control(
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
    all_rows = _rows()
    train_rows, test_rows = all_rows[::2], all_rows[1::2]

    train_source_texts = _template_b_texts(tok, train_rows, "calculate")
    train_target_texts = _template_b_texts(tok, train_rows, "lookup")
    train_source, train_source_am = _encode_uniform(
        tok, train_source_texts, dev)
    train_target, train_target_am = _encode_uniform(
        tok, train_target_texts, dev)
    changed = [[
        idx for idx, (left, right) in enumerate(zip(source, target))
        if left != right
    ] for source, target in zip(
        train_source.tolist(), train_target.tolist())]
    if any(row != [TEMPLATE_B_POSITION] for row in changed):
        raise ValueError(f"template-B alignment failed: {changed}")
    _, source_cache = _forward(
        model, train_source, train_source_am, (TEMPLATE_B_POSITION,),
        (INJECT_LAYER,))
    _, target_cache = _forward(
        model, train_target, train_target_am, (TEMPLATE_B_POSITION,),
        (INJECT_LAYER,))
    direction = (
        target_cache[INJECT_LAYER][:, 0]
        - source_cache[INJECT_LAYER][:, 0]).mean(0)

    source_label_id = int(train_source[0, TEMPLATE_B_POSITION])
    target_label_id = int(train_target[0, TEMPLATE_B_POSITION])
    embedding = model.get_input_embeddings().weight
    lexical_raw = (
        embedding[target_label_id].detach().float().cpu()
        - embedding[source_label_id].detach().float().cpu())
    lexical_direction = (
        lexical_raw / lexical_raw.norm().clamp(min=EPS)
        * direction.norm())

    natural_target_texts, target_calls, target_results = _followups(
        tok, test_rows, "lookup", "lookup")
    source_target_texts, _, source_target_results = _followups(
        tok, test_rows, "calculate", "lookup")
    natural_source_texts, source_calls, source_results = _followups(
        tok, test_rows, "calculate", "calculate")
    if target_results != source_target_results:
        raise ValueError("target execution differs across prompt modes")
    target_answers = [str(row["database_value"]) for row in test_rows]
    source_answers = [str(row["a"] + row["b"]) for row in test_rows]
    diagnostic = [
        target != source
        for target, source in zip(target_answers, source_answers)]
    if sum(diagnostic) != 8:
        raise ValueError(f"expected eight diagnostic rows: {diagnostic}")

    source_target_ids, source_target_am = _encode_uniform(
        tok, source_target_texts, dev)
    natural_target_ids, natural_target_am = _encode_uniform(
        tok, natural_target_texts, dev)
    if int(source_target_ids[0, TEMPLATE_B_POSITION]) != source_label_id:
        raise ValueError("source follow-up mode token moved")
    if int(natural_target_ids[0, TEMPLATE_B_POSITION]) != target_label_id:
        raise ValueError("target follow-up mode token moved")
    target_ids = _answer_ids(tok, source_target_texts, target_answers)
    source_ids = _answer_ids(tok, source_target_texts, source_answers)

    clean_logits, _ = _forward(
        model, source_target_ids, source_target_am, (TEMPLATE_B_POSITION,))
    natural_logits, _ = _forward(
        model, natural_target_ids, natural_target_am, (TEMPLATE_B_POSITION,))
    batch_direction = direction.unsqueeze(0).expand(len(test_rows), -1)
    continuous_logits, _ = _forward(
        model, source_target_ids, source_target_am, (TEMPLATE_B_POSITION,),
        add=(INJECT_LAYER, TEMPLATE_B_POSITION, batch_direction))
    lexical_batch = lexical_direction.unsqueeze(0).expand(len(test_rows), -1)
    lexical_logits, _ = _forward(
        model, source_target_ids, source_target_am, (TEMPLATE_B_POSITION,),
        add=(INJECT_LAYER, TEMPLATE_B_POSITION, lexical_batch))

    continuous_output = _output_metrics(
        clean_logits, natural_logits, continuous_logits,
        target_ids, source_ids, diagnostic)
    lexical_output = _output_metrics(
        clean_logits, natural_logits, lexical_logits,
        target_ids, source_ids, diagnostic)

    generator = torch.Generator(device="cpu").manual_seed(seed + 1701)
    null_means = np.zeros(n_null)
    null_rows_all = np.zeros((n_null, sum(diagnostic)))
    null_directions = torch.zeros((n_null, direction.numel()))
    mask = torch.tensor(diagnostic, dtype=torch.bool)
    clean_diag_ld = _ld(clean_logits, target_ids, source_ids)[mask]
    for idx in range(n_null):
        random = torch.randn(
            direction.shape, generator=generator, dtype=torch.float32)
        random = random / random.norm().clamp(min=EPS) * direction.norm()
        null_directions[idx] = random
        random_logits, _ = _forward(
            model, source_target_ids, source_target_am,
            (TEMPLATE_B_POSITION,), add=(
                INJECT_LAYER, TEMPLATE_B_POSITION,
                random.unsqueeze(0).expand(len(test_rows), -1)))
        null_rows = (
            _ld(random_logits, target_ids, source_ids)[mask] - clean_diag_ld)
        null_rows_all[idx] = null_rows.numpy()
        null_means[idx] = float(null_rows.mean())
    continuous_exceedances = int(
        (null_means >= continuous_output["effect"]).sum())
    lexical_exceedances = int(
        (null_means >= lexical_output["effect"]).sum())

    generation_specs = {
        "natural_target": (natural_target_texts, None),
        "natural_source": (natural_source_texts, None),
        "source_target_unsteered": (source_target_texts, None),
        "source_target_reapplied": (source_target_texts, direction),
        "source_target_lexical": (
            source_target_texts, lexical_direction),
    }
    raw, generated_token_ids = {}, {}
    for name, (texts, intervention) in generation_specs.items():
        raw[name], generated_token_ids[name] = _generate_answers(
            model, tok, texts, dev, direction=intervention)
    scores = {
        name: _score(values, target_answers, source_answers, diagnostic)
        for name, values in raw.items()}
    g0 = bool(
        scores["natural_target"]["diagnostic_target_acc"] >= 0.875
        and scores["natural_source"]["diagnostic_source_acc"] >= 0.875)
    r0 = bool(
        scores["source_target_unsteered"]["diagnostic_target_acc"] <= 0.625)
    continuous_behavior = bool(
        scores["source_target_reapplied"]["diagnostic_target_acc"] >= 0.875
        and scores["source_target_reapplied"]["diagnostic_target_acc"]
        - scores["source_target_unsteered"]["diagnostic_target_acc"] >= 0.25)
    continuous_causal = _output_pass(
        continuous_output, continuous_exceedances)
    lexical_behavior = bool(
        scores["source_target_lexical"]["diagnostic_target_acc"] >= 0.875
        and scores["source_target_lexical"]["diagnostic_target_acc"]
        - scores["source_target_unsteered"]["diagnostic_target_acc"] >= 0.25)
    lexical_causal = _output_pass(lexical_output, lexical_exceedances)
    b1 = not (lexical_behavior and lexical_causal)
    gates = {
        "G0": g0,
        "R0": r0,
        "C1": continuous_behavior,
        "C2": continuous_causal,
        "B1": b1,
    }
    if not g0 or not r0:
        verdict = "ANSWER_TURN_DIAGNOSTIC_INVALID"
    elif continuous_behavior and continuous_causal:
        verdict = (
            "ANSWER_TURN_LATENT_CONTROL"
            if b1 else "ANSWER_TURN_CONTROL_LEXICAL")
    else:
        verdict = "ANSWER_TURN_CONTROL_NULL"

    raw_artifact = "raw_delta_answer_turn_control.pt"
    torch.save({
        "direction": direction,
        "lexical_raw": lexical_raw,
        "lexical_direction": lexical_direction,
        "source_cache": source_cache,
        "target_cache": target_cache,
        "input_ids": {
            "source_target": source_target_ids.detach().cpu(),
            "natural_target": natural_target_ids.detach().cpu(),
        },
        "logits": {
            "clean": clean_logits,
            "natural": natural_logits,
            "reapplied": continuous_logits,
            "lexical": lexical_logits,
        },
        "answer_token_ids": {
            "target": target_ids,
            "source": source_ids,
        },
        "output_rows": {
            "natural": continuous_output["natural_rows"],
            "reapplied": continuous_output["effect_rows"],
            "lexical": lexical_output["effect_rows"],
        },
        "null_directions": null_directions,
        "null_means": torch.from_numpy(null_means),
        "null_rows": torch.from_numpy(null_rows_all),
        "generated_token_ids": generated_token_ids,
    }, os.path.join(out_dir, raw_artifact))
    result = {
        "stage": "delta_answer_turn_control",
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
            "module_source_sha256": hashlib.sha256(
                open(__file__, "rb").read()).hexdigest(),
            "imported_helpers_sha256": hashlib.sha256("".join(
                inspect.getsource(helper) for helper in (
                    _forward, _template_b_content, _template_b_texts,
                    _render, _expected_call, _execute, _parse_call,
                    _normalize, _rows, _ld, _safe_ratio, _encode_uniform,
                    _assert_runtime)).encode()).hexdigest(),
            "rows_sha256": hashlib.sha256(json.dumps(
                all_rows, sort_keys=True).encode()).hexdigest(),
        },
        "preflight": {
            "mode_position": TEMPLATE_B_POSITION,
            "source_label_id": source_label_id,
            "target_label_id": target_label_id,
            "direction_norm": float(direction.norm()),
            "lexical_raw_norm": float(lexical_raw.norm()),
            "diagnostic_rows": diagnostic,
            "train_prompts": {
                "source": train_source_texts,
                "target": train_target_texts,
            },
            "test_prompts": {
                "natural_target": natural_target_texts,
                "source_target": source_target_texts,
                "natural_source": natural_source_texts,
            },
        },
        "train_rows": train_rows,
        "test_rows": test_rows,
        "calls": {
            "target": target_calls,
            "source": source_calls,
            "target_results": target_results,
            "source_results": source_results,
        },
        "target_answers": target_answers,
        "source_answers": source_answers,
        "raw_answers": raw,
        "generated_token_ids": generated_token_ids,
        "scores": scores,
        "output": {
            "reapplied": {
                key: (
                    value.tolist() if isinstance(value, torch.Tensor)
                    else value)
                for key, value in continuous_output.items()},
            "lexical": {
                key: (
                    value.tolist() if isinstance(value, torch.Tensor)
                    else value)
                for key, value in lexical_output.items()},
            "reapplied_null_exceedances": continuous_exceedances,
            "lexical_null_exceedances": lexical_exceedances,
            "null_means": null_means.tolist(),
        },
        "gates": gates,
        "raw_tensor_artifact": raw_artifact,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_answer_turn_control.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    log(f"scores={scores}")
    log(f"reapplied_output={result['output']['reapplied']}")
    log(f"lexical_output={result['output']['lexical']}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
