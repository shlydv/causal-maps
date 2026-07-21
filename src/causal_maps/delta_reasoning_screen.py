"""Behavior-only eligibility screen for nontrivial reasoning controls."""
import json
import os
import random

import torch

from .delta_reasoning_controller import _continuation_token_id
from .logutil import log
from .model_utils import input_device, last_token_logits, load_model_and_tokenizer

LETTERS = tuple("ABCDEFGHIJ")


def _chat(tok, user):
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def _kinship_rows(seed=0):
    rng = random.Random(seed + 2001)
    rows = []
    for repeat in (0, 1):
        for grand_idx, grandparent in enumerate(LETTERS):
            parent = LETTERS[(grand_idx + 3 + 2 * repeat) % len(LETTERS)]
            remaining = [
                letter for letter in LETTERS
                if letter not in (grandparent, parent)]
            rng.shuffle(remaining)
            child, distractor_parent, distractor_child = remaining[:3]
            facts = [
                f"{parent} is the parent of {child}",
                f"{grandparent} is the parent of {parent}",
                f"{distractor_parent} is the parent of {distractor_child}",
            ]
            rng.shuffle(facts)
            rows.append({
                "child": child,
                "parent": parent,
                "grandparent": grandparent,
                "facts": facts,
            })
    rng.shuffle(rows)
    return rows


def _kinship_prompt(tok, row, mode):
    user = (
        f"Facts: {'; '.join(row['facts'])}. "
        f"Mode one means answer with {row['child']}'s parent. "
        f"Mode two means answer with {row['child']}'s grandparent. "
        f"Mode: {mode}. Who is requested? "
        "Answer with one capital letter only."
    )
    return _chat(tok, user)


def _arithmetic_rows():
    return [
        {"a": a, "b": b}
        for a in range(1, 9)
        for b in range(1, min(a, 9 - a) + 1)
    ]


def _arithmetic_prompt(tok, row, mode):
    user = (
        f"First number: {row['a']}. Second number: {row['b']}. "
        "Mode add means add the numbers. "
        "Mode subtract means subtract the second number from the first. "
        f"Mode: {mode}. Calculate the result. "
        "Return only the integer answer."
    )
    return _chat(tok, user)


def _encode_family(tok, rows, prompt_fn, modes, answer_fn, dev):
    prompts = {
        mode: [prompt_fn(tok, row, mode) for row in rows]
        for mode in modes}
    encoded = {
        mode: [tok.encode(text, add_special_tokens=False)
               for text in prompts[mode]]
        for mode in modes}
    lengths = {len(ids) for mode in modes for ids in encoded[mode]}
    if len(lengths) != 1:
        raise ValueError(f"nonuniform prompt lengths: {sorted(lengths)}")
    changed_positions = []
    for left, right in zip(encoded[modes[0]], encoded[modes[1]]):
        changed = [
            idx for idx, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(left) != len(right) or len(changed) != 1:
            raise ValueError(
                f"mode alignment failed: {len(left)}/{len(right)} {changed}")
        changed_positions.append(changed[0])
    ids = {
        mode: torch.tensor(encoded[mode], dtype=torch.long, device=dev)
        for mode in modes}
    masks = {mode: torch.ones_like(ids[mode]) for mode in modes}
    answers = {
        mode: [str(answer_fn(row, mode)) for row in rows]
        for mode in modes}
    answer_ids = {
        mode: torch.tensor([
            _continuation_token_id(tok, prompt, answer)
            for prompt, answer in zip(prompts[mode], answers[mode])])
        for mode in modes}
    return ids, masks, answer_ids, answers, changed_positions, next(iter(lengths))


@torch.no_grad()
def _evaluate(model, tok, name, rows, prompt_fn, modes, answer_fn, dev):
    ids, masks, answer_ids, answers, changed, length = _encode_family(
        tok, rows, prompt_fn, modes, answer_fn, dev)
    result = {
        "name": name,
        "n": len(rows),
        "prompt_length": length,
        "changed_positions": changed,
        "modes": {},
    }
    eligible = True
    for mode in modes:
        logits = last_token_logits(model, ids[mode], masks[mode]).float().cpu()
        prediction_ids = logits.argmax(-1)
        accuracy = float(
            (prediction_ids == answer_ids[mode]).float().mean())
        result["modes"][mode] = {
            "accuracy": accuracy,
            "answers": answers[mode],
            "answer_ids": answer_ids[mode].tolist(),
            "prediction_ids": prediction_ids.tolist(),
            "predictions": [
                tok.decode([int(token_id)]) for token_id in prediction_ids],
        }
        eligible = eligible and accuracy >= 0.90
    result["eligible"] = bool(eligible)
    return result


@torch.no_grad()
def run_delta_reasoning_screen(model_path, out_dir, quantization="8bit",
                               device_map=None, seed=0):
    if model_path != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(f"frozen model mismatch: {model_path}")
    if quantization != "8bit" or seed != 0:
        raise ValueError(
            f"frozen config mismatch: quantization={quantization}, seed={seed}")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    kinship = _evaluate(
        model, tok, "kinship_depth", _kinship_rows(seed),
        _kinship_prompt, ("one", "two"),
        lambda row, mode: (
            row["parent"] if mode == "one" else row["grandparent"]),
        dev)
    arithmetic = _evaluate(
        model, tok, "arithmetic_program", _arithmetic_rows(),
        _arithmetic_prompt, ("add", "subtract"),
        lambda row, mode: (
            row["a"] + row["b"] if mode == "add" else row["a"] - row["b"]),
        dev)
    selected = (
        "kinship_depth" if kinship["eligible"]
        else "arithmetic_program" if arithmetic["eligible"]
        else None)
    verdict = (
        f"ELIGIBLE_{selected.upper()}" if selected
        else "NO_REASONING_TASK_ELIGIBLE")
    result = {
        "stage": "delta_reasoning_screen",
        "model_path": model_path,
        "quantization": quantization,
        "seed": seed,
        "families": [kinship, arithmetic],
        "selected": selected,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_reasoning_screen.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"kinship eligible={kinship['eligible']} modes={kinship['modes']}")
    log(f"arithmetic eligible={arithmetic['eligible']} "
        f"modes={arithmetic['modes']}")
    log(f"VERDICT: {verdict}")
    return result
