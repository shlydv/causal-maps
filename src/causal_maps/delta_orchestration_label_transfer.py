"""Cross-label/template transfer of the orchestration controller."""
import json
import os

import torch

from .delta_orchestration_controller import (
    INJECT_LAYER, LAYERS, MEDIATION_LAYER, MODE_POSITION, N_NULL,
    _assert_runtime, _encode_uniform, _evaluate_workflow, _task_texts)
from .delta_orchestration_lexical import _evaluate_variant
from .delta_orchestration_screen import (
    MODEL_REVISION, _generate, _render, _rows)
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

TEMPLATE_B_POSITION = 103


def _template_b_content(row, mode):
    label = "red" if mode == "calculate" else "blue"
    return (
        "Two service endpoints are available. "
        "The calculator endpoint adds packet numbers and uses: "
        "CALL calculator <first> <second>. "
        "The database endpoint retrieves the packet key and uses: "
        "CALL database <key> 0. "
        "Policy red dispatches to the calculator endpoint. "
        "Policy blue dispatches to the database endpoint. "
        f"Packet first: {row['a']}. Packet second: {row['b']}. "
        f"Packet key: {row['key']}. Active policy: {label}. "
        "Emit exactly one endpoint call and nothing else."
    )


def _template_b_texts(tok, rows, mode):
    return [
        _render(tok, [{"role": "user",
                       "content": _template_b_content(row, mode)}])
        for row in rows]


def _passes(summary):
    return all(summary["gates"].values())


@torch.no_grad()
def run_delta_orchestration_label_transfer(
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

    a_train_calc_texts = _task_texts(tok, train_rows, "calculate")
    a_train_lookup_texts = _task_texts(tok, train_rows, "lookup")
    a_train_calc, a_train_calc_am = _encode_uniform(
        tok, a_train_calc_texts, dev)
    a_train_lookup, a_train_lookup_am = _encode_uniform(
        tok, a_train_lookup_texts, dev)
    a_changed_positions = [[
        idx for idx, (left, right) in enumerate(zip(calc, lookup))
        if left != right
    ] for calc, lookup in zip(
        a_train_calc.tolist(), a_train_lookup.tolist())]
    if any(changed != [MODE_POSITION] for changed in a_changed_positions):
        raise ValueError(f"template-A alignment failed: {a_changed_positions}")
    _, a_calc_cache = _forward(
        model, a_train_calc, a_train_calc_am,
        (MODE_POSITION,), (INJECT_LAYER,))
    _, a_lookup_cache = _forward(
        model, a_train_lookup, a_train_lookup_am,
        (MODE_POSITION,), (INJECT_LAYER,))
    a_direction = (
        a_lookup_cache[INJECT_LAYER][:, 0]
        - a_calc_cache[INJECT_LAYER][:, 0]).mean(0)

    b_train_calc_texts = _template_b_texts(tok, train_rows, "calculate")
    b_train_lookup_texts = _template_b_texts(tok, train_rows, "lookup")
    b_test_calc_texts = _template_b_texts(tok, test_rows, "calculate")
    b_test_lookup_texts = _template_b_texts(tok, test_rows, "lookup")
    b_train_calc, b_train_calc_am = _encode_uniform(
        tok, b_train_calc_texts, dev)
    b_train_lookup, b_train_lookup_am = _encode_uniform(
        tok, b_train_lookup_texts, dev)
    b_test_calc, _ = _encode_uniform(tok, b_test_calc_texts, dev)
    b_test_lookup, _ = _encode_uniform(tok, b_test_lookup_texts, dev)
    changed_positions = [[
        idx for idx, (left, right) in enumerate(zip(calc, lookup))
        if left != right
    ] for calc, lookup in zip(
        b_train_calc.tolist() + b_test_calc.tolist(),
        b_train_lookup.tolist() + b_test_lookup.tolist())]
    if any(changed != [TEMPLATE_B_POSITION] for changed in changed_positions):
        raise ValueError(f"template-B alignment failed: {changed_positions}")
    if len({
            b_train_calc.shape[1], b_train_lookup.shape[1],
            b_test_calc.shape[1], b_test_lookup.shape[1]}) != 1:
        raise ValueError("template-B prompt lengths differ")
    if int(b_test_calc.shape[1]) != 119:
        raise ValueError(
            f"template-B frozen length mismatch: {b_test_calc.shape[1]}")

    raw_native_calc = _generate(model, tok, b_test_calc_texts, dev, 8)
    raw_native_lookup = _generate(model, tok, b_test_lookup_texts, dev, 8)
    native_calc = _evaluate_workflow(
        model, tok, dev, test_rows, "calculate", "calculate",
        raw_native_calc, content_fn=_template_b_content)
    native_lookup = _evaluate_workflow(
        model, tok, dev, test_rows, "lookup", "lookup",
        raw_native_lookup, content_fn=_template_b_content)
    native_valid = all(
        value >= 0.90
        for workflow in (native_calc, native_lookup)
        for value in workflow["metrics"].values())
    _, b_calc_cache = _forward(
        model, b_train_calc, b_train_calc_am,
        (TEMPLATE_B_POSITION,), (INJECT_LAYER,))
    _, b_lookup_cache = _forward(
        model, b_train_lookup, b_train_lookup_am,
        (TEMPLATE_B_POSITION,), (INJECT_LAYER,))
    b_direction = (
        b_lookup_cache[INJECT_LAYER][:, 0]
        - b_calc_cache[INJECT_LAYER][:, 0]).mean(0)

    b_calc_call_texts = [text + "CALL" for text in b_test_calc_texts]
    b_lookup_call_texts = [text + "CALL" for text in b_test_lookup_texts]
    b_calc_call, b_calc_call_am = _encode_uniform(
        tok, b_calc_call_texts, dev)
    b_lookup_call, b_lookup_call_am = _encode_uniform(
        tok, b_lookup_call_texts, dev)
    call_last = int(b_calc_call.shape[1] - 1)
    if call_last != int(b_lookup_call.shape[1] - 1):
        raise ValueError("template-B CALL-prefix lengths differ")
    call_ids = {
        tuple(tok.encode(call_text, add_special_tokens=False)[
            len(tok.encode(prompt, add_special_tokens=False)):])
        for prompt, call_text in zip(
            b_test_calc_texts + b_test_lookup_texts,
            b_calc_call_texts + b_lookup_call_texts)}
    if len(call_ids) != 1 or len(next(iter(call_ids))) != 1:
        raise ValueError(f"CALL continuation is unstable: {call_ids}")
    tool_extensions = {"calculator": set(), "database": set()}
    for prompt in b_calc_call_texts + b_lookup_call_texts:
        prefix_ids = tok.encode(prompt, add_special_tokens=False)
        for tool in tool_extensions:
            full_ids = tok.encode(
                prompt + f" {tool}", add_special_tokens=False)
            tool_extensions[tool].add(tuple(full_ids[len(prefix_ids):]))
    if any(len(values) != 1 or len(next(iter(values))) != 1
           for values in tool_extensions.values()):
        raise ValueError(f"tool continuations are unstable: {tool_extensions}")
    clean_logits, clean_cache = _forward(
        model, b_calc_call, b_calc_call_am,
        (TEMPLATE_B_POSITION, call_last), LAYERS)
    natural_logits, natural_cache = _forward(
        model, b_lookup_call, b_lookup_call_am,
        (TEMPLATE_B_POSITION, call_last), LAYERS)
    call_token_id = next(iter(next(iter(call_ids))))
    calc_tool_id = next(iter(next(iter(tool_extensions["calculator"]))))
    lookup_tool_id = next(iter(next(iter(tool_extensions["database"]))))
    preflight = {
        "template_a_length": int(a_train_calc.shape[1]),
        "template_a_mode_position": MODE_POSITION,
        "template_a_changed_positions": a_changed_positions,
        "template_b_length": int(b_test_calc.shape[1]),
        "template_b_mode_position": TEMPLATE_B_POSITION,
        "template_b_changed_positions": changed_positions,
        "call_position": call_last,
        "call_token_id": int(call_token_id),
        "calculate_tool_token_id": int(calc_tool_id),
        "database_tool_token_id": int(lookup_tool_id),
        "prompts": {
            "a_train_calculate": a_train_calc_texts,
            "a_train_lookup": a_train_lookup_texts,
            "b_train_calculate": b_train_calc_texts,
            "b_train_lookup": b_train_lookup_texts,
            "b_test_calculate": b_test_calc_texts,
            "b_test_lookup": b_test_lookup_texts,
        },
    }
    if not native_valid:
        result = {
            "stage": "delta_orchestration_label_transfer",
            "model_path": model_path,
            "model_revision": MODEL_REVISION,
            "runtime": runtime,
            "quantization": quantization,
            "seed": seed,
            "n_null": n_null,
            "preflight": preflight,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "native_calculate": native_calc,
            "native_lookup": native_lookup,
            "verdict": "TEMPLATE_TRANSFER_INELICITABLE",
        }
        with open(os.path.join(
                out_dir, "results_delta_orchestration_label_transfer.json"), "w") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        log(f"VERDICT: TEMPLATE_TRANSFER_INELICITABLE "
            f"{native_calc['metrics']} {native_lookup['metrics']}")
        return result
    calc_tool_tensor = torch.full(
        (len(test_rows),), calc_tool_id, dtype=torch.long)
    lookup_tool_tensor = torch.full(
        (len(test_rows),), lookup_tool_id, dtype=torch.long)
    clean_ld = _ld(clean_logits, lookup_tool_tensor, calc_tool_tensor)

    variants = {}
    for idx, (name, direction) in enumerate((
            ("cross_template_a_direction", a_direction),
            ("template_b_specific_direction", b_direction))):
        variants[name] = _evaluate_variant(
            model, tok, dev, test_rows, b_test_calc_texts,
            b_test_lookup_texts, b_calc_call, b_calc_call_am,
            clean_logits, clean_cache, natural_logits, natural_cache,
            clean_ld, calc_tool_tensor, lookup_tool_tensor, direction,
            torch.Generator(device="cpu").manual_seed(seed + 1501 + idx),
            n_null, mode_position=TEMPLATE_B_POSITION,
            content_fn=_template_b_content)

    cross_pass = _passes(variants["cross_template_a_direction"]["summary"])
    specific_pass = _passes(
        variants["template_b_specific_direction"]["summary"])
    if cross_pass and specific_pass:
        verdict = "CROSS_LABEL_TEMPLATE_TRANSFER"
    elif cross_pass:
        verdict = "CROSS_TRANSFER_REFERENCE_INVALID"
    elif specific_pass:
        verdict = "TEMPLATE_SPECIFIC_CONTROLLER"
    else:
        verdict = "CONTROLLER_NOT_REPLICATED"

    raw_artifact = "raw_delta_orchestration_label_transfer.pt"
    torch.save({
        "template_a_direction": a_direction,
        "template_b_direction": b_direction,
        "template_a_train_states": {
            "calculate": a_calc_cache, "lookup": a_lookup_cache},
        "template_b_train_states": {
            "calculate": b_calc_cache, "lookup": b_lookup_cache},
        "template_b_clean_logits": clean_logits,
        "template_b_natural_logits": natural_logits,
        "template_b_clean_cache": clean_cache,
        "template_b_natural_cache": natural_cache,
        "variants": {
            name: value["raw"] for name, value in variants.items()},
        "input_ids": {
            "a_train_calculate": a_train_calc.detach().cpu(),
            "a_train_lookup": a_train_lookup.detach().cpu(),
            "b_train_calculate": b_train_calc.detach().cpu(),
            "b_train_lookup": b_train_lookup.detach().cpu(),
            "b_test_calculate": b_test_calc.detach().cpu(),
            "b_test_lookup": b_test_lookup.detach().cpu(),
            "b_call_calculate": b_calc_call.detach().cpu(),
            "b_call_lookup": b_lookup_call.detach().cpu(),
        },
    }, os.path.join(out_dir, raw_artifact))
    result = {
        "stage": "delta_orchestration_label_transfer",
        "model_path": model_path,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "quantization": quantization,
        "seed": seed,
        "n_null": n_null,
        "preflight": {
            **preflight,
            "template_a_direction_norm": float(a_direction.norm()),
            "template_b_direction_norm": float(b_direction.norm()),
            "direction_cosine": float(torch.nn.functional.cosine_similarity(
                a_direction.unsqueeze(0), b_direction.unsqueeze(0))),
        },
        "train_rows": train_rows,
        "test_rows": test_rows,
        "native_calculate": native_calc,
        "native_lookup": native_lookup,
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
            out_dir, "results_delta_orchestration_label_transfer.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    log(f"A/B direction cos={result['preflight']['direction_cosine']:.3f}")
    log(f"cross={variants['cross_template_a_direction']['summary']}")
    log(f"specific={variants['template_b_specific_direction']['summary']}")
    log(f"VERDICT: {verdict}")
    return result
