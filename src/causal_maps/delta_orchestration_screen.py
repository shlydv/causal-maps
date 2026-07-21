"""Behavior-only screen for a structured two-tool agent workflow."""
import json
import os
import re
import importlib.metadata

import torch

from .delta_reasoning_screen import _arithmetic_rows
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

LETTERS = tuple("ABCDEFGHIJ")
MODES = ("calculate", "lookup")
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DATABASE = {
    letter: (3 * idx + 1) % 10 for idx, letter in enumerate(LETTERS)}
CALL_RE = re.compile(
    r"^CALL\s+(calculator|database)\s+([A-J0-9])\s+([0-9])$")


def _rows():
    return [
        {
            **pair,
            "key": LETTERS[idx % len(LETTERS)],
            "database_value": DATABASE[LETTERS[idx % len(LETTERS)]],
        }
        for idx, pair in enumerate(_arithmetic_rows())
    ]


def _task_content(row, mode):
    return (
        "You are an agent that must call exactly one tool. "
        "To add the payload numbers, write: "
        "CALL calculator <first> <second>. "
        "To retrieve the payload key, write: "
        "CALL database <key> 0. "
        f"Payload numbers: {row['a']} and {row['b']}. "
        f"Payload key: {row['key']}. "
        "Mode calculate means use the calculator. "
        "Mode lookup means use the database. "
        f"Mode: {mode}. "
        "Output exactly one tool call and nothing else."
    )


def _render(tok, messages):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def _normalize(text):
    for marker in ("<|im_end|>", "</s>", "<|eot_id|>"):
        text = text.split(marker, 1)[0]
    return text.strip()


def _expected_call(row, mode):
    if mode == "calculate":
        return f"CALL calculator {row['a']} {row['b']}"
    return f"CALL database {row['key']} 0"


def _expected_answer(row, mode):
    return str(
        row["a"] + row["b"]
        if mode == "calculate" else row["database_value"])


@torch.no_grad()
def _generate(model, tok, texts, dev, max_new_tokens):
    old_side = tok.padding_side
    old_pad = tok.pad_token
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        encoded = tok(
            texts, add_special_tokens=False, padding=True,
            return_tensors="pt")
    finally:
        tok.padding_side = old_side
        if old_pad is None:
            tok.pad_token = None
    ids = encoded["input_ids"].to(dev)
    am = encoded["attention_mask"].to(dev)
    generated = model.generate(
        input_ids=ids,
        attention_mask=am,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
    )
    continuation = generated[:, ids.shape[1]:].detach().cpu()
    return [
        tok.decode(row.tolist(), skip_special_tokens=False)
        for row in continuation]


def _parse_call(text):
    match = CALL_RE.fullmatch(text)
    if match is None:
        return None
    return {
        "tool": match.group(1),
        "arg1": match.group(2),
        "arg2": match.group(3),
    }


def _correct_action(row, mode, parsed):
    if parsed is None:
        return False
    if mode == "calculate":
        return (
            parsed["tool"] == "calculator"
            and parsed["arg1"] == str(row["a"])
            and parsed["arg2"] == str(row["b"]))
    return (
        parsed["tool"] == "database"
        and parsed["arg1"] == row["key"]
        and parsed["arg2"] == "0")


def _execute(parsed):
    if parsed is None:
        return None
    if parsed["tool"] == "calculator":
        if not parsed["arg1"].isdigit() or not parsed["arg2"].isdigit():
            return None
        return str(int(parsed["arg1"]) + int(parsed["arg2"]))
    if parsed["arg1"] not in DATABASE or parsed["arg2"] != "0":
        return None
    return str(DATABASE[parsed["arg1"]])


@torch.no_grad()
def run_delta_orchestration_screen(model_path, out_dir, quantization="8bit",
                                   device_map=None, seed=0):
    if model_path != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(f"frozen model mismatch: {model_path}")
    if quantization != "8bit" or seed != 0:
        raise ValueError(
            f"frozen config mismatch: quantization={quantization}, seed={seed}")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization,
        revision=MODEL_REVISION)
    dev = input_device(model)
    rows = _rows()

    prompt_ids = {
        mode: [
            tok.encode(
                _render(tok, [{"role": "user",
                               "content": _task_content(row, mode)}]),
                add_special_tokens=False)
            for row in rows]
        for mode in MODES}
    lengths = {
        len(ids) for mode in MODES for ids in prompt_ids[mode]}
    if len(lengths) != 1:
        raise ValueError(f"nonuniform orchestration prompts: {sorted(lengths)}")
    changed_positions = []
    for calculate, lookup in zip(
            prompt_ids["calculate"], prompt_ids["lookup"]):
        changed = [
            idx for idx, (left, right) in enumerate(zip(calculate, lookup))
            if left != right]
        if len(calculate) != len(lookup) or len(changed) != 1:
            raise ValueError(
                f"mode alignment failed: {len(calculate)}/{len(lookup)} "
                f"{changed}")
        changed_positions.append(changed[0])

    families = {}
    for mode in MODES:
        task_messages = [
            [{"role": "user", "content": _task_content(row, mode)}]
            for row in rows]
        task_texts = [_render(tok, messages) for messages in task_messages]
        raw_calls = _generate(model, tok, task_texts, dev, 8)
        calls = [_normalize(text) for text in raw_calls]
        parsed = [_parse_call(call) for call in calls]
        correct_actions = [
            _correct_action(row, mode, parsed_call)
            for row, parsed_call in zip(rows, parsed)]
        tool_results = [
            _execute(parsed_call) for parsed_call in parsed]
        followups = []
        for messages, call, result in zip(task_messages, calls, tool_results):
            followups.append(_render(tok, [
                *messages,
                {"role": "assistant", "content": call},
                {"role": "user", "content": (
                    f"Tool result: {result if result is not None else 'ERROR'}. "
                    "Return only the final answer.")},
            ]))
        raw_answers = _generate(model, tok, followups, dev, 4)
        answers = [_normalize(text) for text in raw_answers]

        expected_calls = [
            _expected_call(row, mode) for row in rows]
        expected_answers = [
            _expected_answer(row, mode) for row in rows]
        exact_call_rows = [
            call == expected
            for call, expected in zip(calls, expected_calls)]
        correct_action_rows = correct_actions
        answer_rows = [
            answer == expected
            for answer, expected in zip(answers, expected_answers)]
        end_to_end_rows = [
            action and answer
            for action, answer in zip(correct_action_rows, answer_rows)]
        metrics = {
            "exact_call_acc": sum(exact_call_rows) / len(rows),
            "correct_action_acc": sum(correct_action_rows) / len(rows),
            "final_answer_acc": sum(answer_rows) / len(rows),
            "end_to_end_acc": sum(end_to_end_rows) / len(rows),
        }
        eligible = all(value >= 0.90 for value in metrics.values())
        families[mode] = {
            "metrics": metrics,
            "eligible": bool(eligible),
            "rows": [{
                **row,
                "expected_call": expected_calls[idx],
                "generated_call": calls[idx],
                "raw_call": raw_calls[idx],
                "parsed_call": parsed[idx],
                "tool_result": tool_results[idx],
                "expected_answer": expected_answers[idx],
                "generated_answer": answers[idx],
                "raw_answer": raw_answers[idx],
                "exact_call": exact_call_rows[idx],
                "correct_action": correct_action_rows[idx],
                "correct_final_answer": answer_rows[idx],
                "correct_end_to_end": end_to_end_rows[idx],
            } for idx, row in enumerate(rows)],
        }

    eligible = all(families[mode]["eligible"] for mode in MODES)
    verdict = (
        "ORCHESTRATION_ELIGIBLE"
        if eligible else "ORCHESTRATION_INELIGIBLE")
    result = {
        "stage": "delta_orchestration_screen",
        "model_path": model_path,
        "model_revision": MODEL_REVISION,
        "quantization": quantization,
        "seed": seed,
        "runtime": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        },
        "n": len(rows),
        "prompt_length": next(iter(lengths)),
        "changed_positions": changed_positions,
        "families": families,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_orchestration_screen.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"calculate={families['calculate']['metrics']}")
    log(f"lookup={families['lookup']['metrics']}")
    log(f"VERDICT: {verdict}")
    return result
