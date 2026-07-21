"""Behavior-only qualification of prompt contexts for agent workspace tests."""
import hashlib
import inspect
import json
import os

import torch

from .delta_orchestration_controller import (
    _assert_runtime, _evaluate_workflow)
from .delta_orchestration_screen import (
    MODEL_REVISION, _generate, _render, _rows)
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

CONTEXTS = {
    "A": ("calculate", "lookup"),
    "B": ("red", "blue"),
    "C": ("north", "south"),
    "D": ("left", "right"),
    "E": ("alpha", "beta"),
    "F": ("open", "close"),
}
PROTOCOL_VERSION = "2026-07-13-v1"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _content(row, context, mode):
    first, second = CONTEXTS[context]
    label = first if mode == "calculate" else second
    if context == "A":
        return (
            "You are an agent that must call exactly one tool. To add the "
            "payload numbers, write: CALL calculator <first> <second>. To "
            "retrieve the payload key, write: CALL database <key> 0. "
            f"Payload numbers: {row['a']} and {row['b']}. "
            f"Payload key: {row['key']}. Mode calculate means use the "
            "calculator. Mode lookup means use the database. "
            f"Mode: {label}. Output exactly one tool call and nothing else.")
    if context == "B":
        return (
            "Two service endpoints are available. The calculator endpoint "
            "adds packet numbers and uses: CALL calculator <first> <second>. "
            "The database endpoint retrieves the packet key and uses: "
            "CALL database <key> 0. Policy red dispatches to the calculator "
            "endpoint. Policy blue dispatches to the database endpoint. "
            f"Packet first: {row['a']}. Packet second: {row['b']}. "
            f"Packet key: {row['key']}. Active policy: {label}. "
            "Emit exactly one endpoint call and nothing else.")
    if context == "C":
        return (
            "Routing manual: route north sends the two numbers to calculator "
            "with CALL calculator <first> <second>; route south sends the key "
            "to storage with CALL database <key> 0. "
            f"Select route {label} for this request. The request contains "
            f"key {row['key']}, first number {row['a']}, and second number "
            f"{row['b']}. Return only the exact CALL line.")
    if context == "D":
        return (
            f"Process this packet using channel {label}. Channel left requires "
            "CALL calculator <first> <second> and adds the numeric fields. "
            "Channel right requires CALL database <key> 0 and retrieves the "
            f"keyed record. Numeric fields are {row['a']} and {row['b']}; "
            f"key field is {row['key']}. Produce one tool call without "
            "commentary.")
    if context == "E":
        return (
            "API dispatch rule: alpha chooses the calculator API; beta chooses "
            "the database API. Calculator syntax is CALL calculator <first> "
            "<second>. Database syntax is CALL database <key> 0. "
            f"Dispatch code: {label}. Arguments include key {row['key']}, "
            f"first {row['a']}, and second {row['b']}. Respond with exactly "
            "the selected API call.")
    return (
        f"Instruction flag {label} controls the operation. Flag open means "
        "add values through CALL calculator <first> <second>. Flag close means "
        "fetch a record through CALL database <key> 0. Current values: "
        f"first {row['a']}, second {row['b']}, key {row['key']}. "
        "Output the required call only.")


def _texts(tok, rows, context, mode):
    return [
        _render(tok, [{"role": "user",
                       "content": _content(row, context, mode)}])
        for row in rows]


@torch.no_grad()
def run_delta_agent_workspace_screen(
        model_path, out_dir, quantization="8bit", device_map=None, seed=0):
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
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "screen_source_sha256": _sha256(
            open(__file__, "rb").read()),
        "workflow_helper_sha256": _sha256(
            inspect.getsource(_evaluate_workflow).encode()),
        "generation_helper_sha256": _sha256(
            inspect.getsource(_generate).encode()),
        "contexts_sha256": _sha256(json.dumps(
            CONTEXTS, sort_keys=True).encode()),
        "rows_sha256": _sha256(json.dumps(
            rows, sort_keys=True).encode()),
    }
    contexts = {}
    all_eligible = True
    for context in CONTEXTS:
        mode_texts = {
            mode: _texts(tok, rows, context, mode)
            for mode in ("calculate", "lookup")}
        encoded = {
            mode: [
                tok.encode(text, add_special_tokens=False)
                for text in mode_texts[mode]]
            for mode in mode_texts}
        lengths = {
            len(ids) for values in encoded.values() for ids in values}
        changed_positions = []
        for calculate, lookup in zip(
                encoded["calculate"], encoded["lookup"]):
            if len(calculate) != len(lookup):
                raise ValueError(
                    f"context {context} mode lengths differ: "
                    f"{len(calculate)} != {len(lookup)}")
            changed_positions.append([
                idx for idx, (left, right) in enumerate(
                    zip(calculate, lookup)) if left != right])
        if len(lengths) != 1:
            raise ValueError(
                f"context {context} payload lengths vary: {sorted(lengths)}")
        if any(len(changed) != 1 for changed in changed_positions):
            raise ValueError(
                f"context {context} not one-token aligned: "
                f"{changed_positions}")
        position = changed_positions[0][0]
        if any(changed != [position] for changed in changed_positions):
            raise ValueError(
                f"context {context} positions vary: {changed_positions}")

        workflows = {}
        for mode in ("calculate", "lookup"):
            raw_calls = _generate(
                model, tok, mode_texts[mode], dev, max_new_tokens=8)
            workflows[mode] = _evaluate_workflow(
                model, tok, dev, rows, mode, mode, raw_calls,
                content_fn=lambda row, selected, context=context: _content(
                    row, context, selected))
        eligible = all(
            value >= 0.90
            for workflow in workflows.values()
            for value in workflow["metrics"].values())
        all_eligible &= eligible
        contexts[context] = {
            "labels": list(CONTEXTS[context]),
            "prompt_length": next(iter(lengths)),
            "mode_position": position,
            "changed_positions": changed_positions,
            "eligible": eligible,
            "calculate": workflows["calculate"],
            "lookup": workflows["lookup"],
            "prompts": mode_texts,
        }
        log(
            f"context={context} eligible={eligible} "
            f"calc={workflows['calculate']['metrics']} "
            f"lookup={workflows['lookup']['metrics']}")

    verdict = (
        "WORKSPACE_CONTEXTS_ELIGIBLE"
        if all_eligible else "WORKSPACE_CONTEXTS_INELICITABLE")
    result = {
        "stage": "delta_agent_workspace_screen",
        "model_path": model_path,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "quantization": quantization,
        "seed": seed,
        "provenance": provenance,
        "rows": rows,
        "contexts": contexts,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_agent_workspace_screen.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    log(f"VERDICT: {verdict}")
    return result
