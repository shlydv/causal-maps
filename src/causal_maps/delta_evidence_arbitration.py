"""Causal arbitration between an internal answer and executed tool evidence."""
import json
import os

import numpy as np
import torch

from .delta_continuous_orchestration import (
    _answer_ids, _generate_answers, _output_metrics, _score)
from .delta_orchestration_controller import (
    EPS, _assert_runtime, _encode_uniform, _safe_ratio)
from .delta_orchestration_screen import (
    _execute, _expected_call, _normalize, _parse_call, _render, _rows,
    _task_content)
from .delta_reasoning_controller import _candidate_metrics
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer


PROTOCOL_VERSION = "2026-07-14-v2"
INJECT_LAYER = 2
LAYERS = (2, 9, 16, 23, 30)
MEDIATION_LAYER = 23
N_NULL = 100


def _original_followups(tok, rows, prompt_mode):
    texts, results = [], []
    for row in rows:
        call = _expected_call(row, "lookup")
        result = _execute(_parse_call(call))
        suffix = f"Tool result: {result}. Return only the final answer."
        texts.append(_render(tok, [
            {"role": "user", "content": _task_content(row, prompt_mode)},
            {"role": "assistant", "content": call},
            {"role": "user", "content": suffix},
        ]))
        results.append(result)
    return texts, results


def _evidence_content(row, mode):
    return (
        f"Payload numbers: {row['a']} and {row['b']}. "
        f"Your internal calculation result is {row['a'] + row['b']}. "
        f"An external database tool executed successfully and returned "
        f"{row['database_value']}. "
        "Mode calculate means use the internal calculation result. "
        "Mode lookup means use the executed external tool result. "
        f"Mode: {mode}. Which result should be used? "
        "Return only the single integer answer and no explanation."
    )


def _mode_batches(tok, rows, dev):
    source_texts = [
        _render(tok, [{"role": "user", "content": _evidence_content(row, "calculate")}])
        for row in rows]
    target_texts = [
        _render(tok, [{"role": "user", "content": _evidence_content(row, "lookup")}])
        for row in rows]
    source, source_am = _encode_uniform(tok, source_texts, dev)
    target, target_am = _encode_uniform(tok, target_texts, dev)
    changes = []
    for left, right in zip(source.tolist(), target.tolist()):
        changes.append([i for i, (a, b) in enumerate(zip(left, right)) if a != b])
    unique = {tuple(x) for x in changes}
    if len(unique) != 1 or len(next(iter(unique))) != 1:
        raise ValueError(f"evidence mode alignment failed: {changes}")
    return (source_texts, target_texts, source, source_am, target, target_am,
            next(iter(unique))[0])


def _mean(rows, mask):
    values = [x for x, keep in zip(rows, mask) if keep]
    return sum(values) / len(values)


def _generation_pass(score):
    return score["diagnostic_target_acc"] >= .875


@torch.no_grad()
def run_delta_evidence_arbitration(
        model_path, out_dir, quantization="8bit", device_map=None,
        seed=0, n_null=N_NULL):
    if quantization != "8bit" or seed != 0 or n_null != 100:
        raise ValueError("frozen evidence-arbitration config mismatch")
    os.makedirs(out_dir, exist_ok=True)
    runtime = _assert_runtime()
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    if int(model.config.num_hidden_layers) != 32:
        raise ValueError("frozen evidence arbitration expects 32 layers")
    dev = input_device(model)
    all_rows = _rows()
    train_rows, test_rows = all_rows[::2], all_rows[1::2]

    (train_source_texts, train_target_texts, train_source, train_source_am,
     train_target, train_target_am, mode_position) = _mode_batches(
         tok, train_rows, dev)
    (test_source_texts, test_target_texts, _test_source_task, _test_source_am,
     _test_target_task, _test_target_am, test_mode_position) = _mode_batches(
         tok, test_rows, dev)
    if test_mode_position != mode_position:
        raise ValueError("mode position differs between donor and heldout rows")

    train_source_followups, train_target_followups = (
        train_source_texts, train_target_texts)
    test_source_followups, test_target_followups = (
        test_source_texts, test_target_texts)
    train_internal = [str(row["a"] + row["b"]) for row in train_rows]
    test_internal = [str(row["a"] + row["b"]) for row in test_rows]
    train_external = [str(row["database_value"]) for row in train_rows]
    test_external = [str(row["database_value"]) for row in test_rows]
    train_diag = [a != b for a, b in zip(train_external, train_internal)]
    test_diag = [a != b for a, b in zip(test_external, test_internal)]
    if sum(train_diag) < 8 or sum(test_diag) != 8:
        raise ValueError("unexpected evidence-conflict partition")

    # Behavior is screened before extracting or applying any direction.
    behavior_texts = {
        "train_internal": train_source_followups,
        "train_external": train_target_followups,
        "test_internal": test_source_followups,
        "test_external": test_target_followups,
    }
    behavior_expected = {
        "train_internal": (train_internal, train_external, train_diag, False),
        "train_external": (train_external, train_internal, train_diag, True),
        "test_internal": (test_internal, test_external, test_diag, False),
        "test_external": (test_external, test_internal, test_diag, True),
    }
    behavior = {}
    for name, texts in behavior_texts.items():
        raw, token_ids = _generate_answers(
            model, tok, texts, dev, inject_layer=INJECT_LAYER,
            mode_position=mode_position)
        expected, rival, mask, _target = behavior_expected[name]
        behavior[name] = {
            "score": _score(raw, expected, rival, mask),
            "raw": raw, "token_ids": token_ids,
        }
        log(f"G0 {name}: {behavior[name]['score']}")
    g0 = all(_generation_pass(behavior[name]["score"])
             for name in behavior_expected)
    base = {
        "stage": "delta_evidence_arbitration",
        "protocol_version": PROTOCOL_VERSION, "model_path": model_path,
        "runtime": runtime, "layers": list(LAYERS),
        "inject_layer": INJECT_LAYER, "mediation_layer": MEDIATION_LAYER,
        "mode_position": mode_position, "train_rows": train_rows,
        "test_rows": test_rows, "train_conflicts": train_diag,
        "test_conflicts": test_diag, "behavior": behavior,
    }
    if not g0:
        base["gates"] = {"G0": False}
        base["verdict"] = "EVIDENCE_ARBITRATION_BEHAVIORALLY_INELIGIBLE"
        return _write(out_dir, base)

    _, source_cache = _forward(
        model, train_source, train_source_am, (mode_position,), (INJECT_LAYER,))
    _, target_cache = _forward(
        model, train_target, train_target_am, (mode_position,), (INJECT_LAYER,))
    donor_rows = target_cache[INJECT_LAYER][:, 0] - source_cache[INJECT_LAYER][:, 0]
    direction = donor_rows.mean(0)
    source_label = int(train_source[0, mode_position])
    target_label = int(train_target[0, mode_position])
    embedding = model.get_input_embeddings().weight.detach().float().cpu()
    lexical_raw = embedding[target_label] - embedding[source_label]
    lexical = lexical_raw / lexical_raw.norm().clamp(min=EPS) * direction.norm()

    clean_ids, clean_am = _encode_uniform(tok, test_source_followups, dev)
    natural_ids, natural_am = _encode_uniform(tok, test_target_followups, dev)
    if (int(clean_ids[0, mode_position]) != source_label
            or int(natural_ids[0, mode_position]) != target_label):
        raise ValueError("answer-turn evidence address moved")
    last = clean_ids.shape[1] - 1
    target_ids = _answer_ids(tok, test_source_followups, test_external)
    source_ids = _answer_ids(tok, test_source_followups, test_internal)
    batch = direction.unsqueeze(0).expand(len(test_rows), -1)
    lexical_batch = lexical.unsqueeze(0).expand(len(test_rows), -1)

    clean_logits, clean_cache = _forward(
        model, clean_ids, clean_am, (mode_position, last), LAYERS)
    natural_logits, natural_cache = _forward(
        model, natural_ids, natural_am, (mode_position, last), LAYERS)
    add_logits, add_cache = _forward(
        model, clean_ids, clean_am, (mode_position, last), LAYERS,
        add=(INJECT_LAYER, mode_position, batch))
    lexical_logits, lexical_cache = _forward(
        model, clean_ids, clean_am, (mode_position, last), LAYERS,
        add=(INJECT_LAYER, mode_position, lexical_batch))
    wrong_address = max(0, mode_position - 8)
    wrong_logits, _ = _forward(
        model, clean_ids, clean_am, (mode_position, last),
        add=(INJECT_LAYER, wrong_address, batch))

    output = {
        "learned": _output_metrics(clean_logits, natural_logits, add_logits,
                                    target_ids, source_ids, test_diag),
        "lexical": _output_metrics(clean_logits, natural_logits, lexical_logits,
                                    target_ids, source_ids, test_diag),
        "wrong_address": _output_metrics(
            clean_logits, natural_logits, wrong_logits,
            target_ids, source_ids, test_diag),
    }
    trajectory = {}
    for layer in LAYERS:
        native = natural_cache[layer][:, 1] - clean_cache[layer][:, 1]
        active = add_cache[layer][:, 1] - clean_cache[layer][:, 1]
        lex = lexical_cache[layer][:, 1] - clean_cache[layer][:, 1]
        a = _candidate_metrics(active, native)
        e = _candidate_metrics(lex, native)
        trajectory[str(layer)] = {
            "learned": {"cosine": a["cosine"], "error": a["error"]},
            "lexical": {"cosine": e["cosine"], "error": e["error"]},
        }
    native_local = natural_cache[INJECT_LAYER][:, 0] - clean_cache[INJECT_LAYER][:, 0]
    active_local = add_cache[INJECT_LAYER][:, 0] - clean_cache[INJECT_LAYER][:, 0]
    local = _candidate_metrics(active_local, native_local)

    clean_ld = _ld(clean_logits, target_ids, source_ids)
    patch_add_logits, _ = _forward(
        model, clean_ids, clean_am, (last,),
        patch=(MEDIATION_LAYER, last, add_cache[MEDIATION_LAYER][:, 1]))
    patch_nat_logits, _ = _forward(
        model, clean_ids, clean_am, (last,),
        patch=(MEDIATION_LAYER, last, natural_cache[MEDIATION_LAYER][:, 1]))
    block_add_logits, _ = _forward(
        model, clean_ids, clean_am, (last,),
        add=(INJECT_LAYER, mode_position, batch),
        patch=(MEDIATION_LAYER, last, clean_cache[MEDIATION_LAYER][:, 1]))
    block_nat_logits, _ = _forward(
        model, natural_ids, natural_am, (last,),
        patch=(MEDIATION_LAYER, last, clean_cache[MEDIATION_LAYER][:, 1]))
    mask = torch.tensor(test_diag, dtype=torch.bool)
    def eff(logits):
        return float((_ld(logits, target_ids, source_ids) - clean_ld)[mask].mean())
    patch_add, patch_nat = eff(patch_add_logits), eff(patch_nat_logits)
    block_add, block_nat = eff(block_add_logits), eff(block_nat_logits)
    learned_effect = output["learned"]["effect"]
    natural_effect = output["learned"]["natural_effect"]
    add_block = _safe_ratio(learned_effect - block_add, learned_effect)
    nat_block = _safe_ratio(natural_effect - block_nat, natural_effect)
    mediation = {
        "patch_add": patch_add, "patch_natural": patch_nat,
        "patch_ratio": _safe_ratio(patch_add, patch_nat),
        "blocked_add": block_add, "blocked_natural": block_nat,
        "add_block_fraction": add_block, "natural_block_fraction": nat_block,
        "block_gap": (abs(add_block - nat_block)
                      if add_block is not None and nat_block is not None else None),
    }

    generator = torch.Generator().manual_seed(seed + 8113)
    null = []
    for i in range(n_null):
        random = torch.randn(direction.shape, generator=generator)
        random = random / random.norm().clamp(min=EPS) * direction.norm()
        logits, _ = _forward(
            model, clean_ids, clean_am, (last,),
            add=(INJECT_LAYER, mode_position,
                 random.unsqueeze(0).expand(len(test_rows), -1)))
        null.append(eff(logits))
        if (i + 1) % 10 == 0:
            log(f"evidence random {i + 1}/{n_null}")
    null_exceed = sum(x >= learned_effect for x in null)

    generation_specs = {
        "internal": (test_source_followups, None, mode_position),
        "external": (test_target_followups, None, mode_position),
        "learned": (test_source_followups, direction, mode_position),
        "reverse": (test_target_followups, -direction, mode_position),
        "lexical": (test_source_followups, lexical, mode_position),
        "wrong_address": (test_source_followups, direction, wrong_address),
    }
    generations, scores = {}, {}
    for name, (texts, delta, position) in generation_specs.items():
        raw, ids = _generate_answers(
            model, tok, texts, dev, direction=delta,
            inject_layer=INJECT_LAYER, mode_position=position)
        generations[name] = {"raw": raw, "ids": ids}
        scores[name] = _score(raw, test_external, test_internal, test_diag)

    # Supplemental bridge to the untouched native Mistral failure condition.
    # Locate the same literal mode-label address in the original agent prompt.
    original_source = [
        _render(tok, [{"role": "user", "content": _task_content(row, "calculate")}])
        for row in test_rows]
    original_target = [
        _render(tok, [{"role": "user", "content": _task_content(row, "lookup")}])
        for row in test_rows]
    original_changes = []
    for left, right in zip(original_source, original_target):
        li = tok.encode(left, add_special_tokens=False)
        ri = tok.encode(right, add_special_tokens=False)
        original_changes.append([
            i for i, (a, b) in enumerate(zip(li, ri)) if a != b])
    original_unique = {tuple(x) for x in original_changes}
    if len(original_unique) != 1 or len(next(iter(original_unique))) != 1:
        raise ValueError("original mode-token bridge is not uniquely aligned")
    original_mode_position = next(iter(original_unique))[0]
    spontaneous_texts, _ = _original_followups(tok, test_rows, "lookup")
    spontaneous_base, _ = _generate_answers(
        model, tok, spontaneous_texts, dev,
        mode_position=original_mode_position)
    spontaneous_add, _ = _generate_answers(
        model, tok, spontaneous_texts, dev, direction=direction,
        mode_position=original_mode_position)
    base_answers = [_normalize(x) for x in spontaneous_base]
    add_answers = [_normalize(x) for x in spontaneous_add]
    spontaneous_failures = [
        i for i, (answer, target) in enumerate(zip(base_answers, test_external))
        if answer != target]
    spontaneous = {
        "base_answers": base_answers, "amplified_answers": add_answers,
        "failure_indices": spontaneous_failures,
        "rescued": [add_answers[i] == test_external[i]
                    for i in spontaneous_failures],
        "targets": test_external, "internal_answers": test_internal,
        "original_mode_position": original_mode_position,
    }

    ratio = output["learned"]["ratio"]
    output_gate = bool(ratio is not None and .70 <= ratio <= 1.30
                       and output["learned"]["positive_fraction"] >= .75
                       and null_exceed <= 1)
    behavior_gate = bool(
        scores["learned"]["diagnostic_target_acc"] >= .875
        and scores["learned"]["diagnostic_target_acc"]
        - scores["internal"]["diagnostic_target_acc"] >= .25)
    reverse_gate = scores["reverse"]["diagnostic_source_acc"] >= .875
    q = trajectory[str(MEDIATION_LAYER)]["learned"]
    trajectory_gate = q["cosine"] >= .80 and q["error"] <= .60
    patch_ratio = mediation["patch_ratio"]
    mediation_gate = bool(
        patch_add > 0 and patch_nat > 0 and patch_ratio is not None
        and .70 <= patch_ratio <= 1.30 and add_block is not None
        and nat_block is not None and add_block >= .70 and nat_block >= .70
        and mediation["block_gap"] <= .20)
    controls_gate = bool(
        scores["lexical"]["diagnostic_target_acc"] < .875
        and scores["wrong_address"]["diagnostic_target_acc"] < .875)
    gates = {
        "G0": True,
        "A1": bool(local["cosine"] >= .80 and local["error"] <= .60),
        "O1": output_gate, "W1": behavior_gate, "R1": reverse_gate,
        "Q1": trajectory_gate, "M1_M2": mediation_gate,
        "B1": controls_gate,
    }
    raw_artifact = "raw_delta_evidence_arbitration.pt"
    torch.save({
        "direction": direction, "donor_direction_rows": donor_rows,
        "lexical_raw": lexical_raw, "lexical_direction": lexical,
        "input_ids": {
            "train_internal": train_source.detach().cpu(),
            "train_external": train_target.detach().cpu(),
            "test_internal_followup": clean_ids.detach().cpu(),
            "test_external_followup": natural_ids.detach().cpu(),
        },
        "states": {"clean": clean_cache, "natural": natural_cache,
                   "learned": add_cache, "lexical": lexical_cache},
        "logits": {"clean": clean_logits, "natural": natural_logits,
                   "learned": add_logits, "lexical": lexical_logits,
                   "wrong_address": wrong_logits,
                   "patch_learned": patch_add_logits,
                   "patch_natural": patch_nat_logits,
                   "block_learned": block_add_logits,
                   "block_natural": block_nat_logits},
        "null_effects": torch.tensor(null),
    }, os.path.join(out_dir, raw_artifact))
    base.update({
        "direction_norm": float(direction.norm()),
        "lexical_norm": float(lexical.norm()), "wrong_address": wrong_address,
        "local": {"cosine": local["cosine"], "error": local["error"]},
        "output": {k: {x: (v.tolist() if torch.is_tensor(v) else v)
                        for x, v in value.items()}
                   for k, value in output.items()},
        "trajectory": trajectory, "mediation": mediation,
        "null": {"effects": null, "exceedances": null_exceed},
        "generations": generations, "scores": scores,
        "spontaneous_bridge": spontaneous, "gates": gates,
        "raw_tensor_artifact": raw_artifact,
        "verdict": ("CAUSAL_EVIDENCE_ARBITRATION_STATE"
                    if all(gates.values()) else
                    "EVIDENCE_ARBITRATION_NOT_ESTABLISHED"),
    })
    return _write(out_dir, base)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_evidence_arbitration.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
