"""Does amplifying the native lookup-policy state rescue tool-result rejection?"""
import json
import os

import torch

from .delta_continuous_orchestration import (
    _answer_ids, _generate_answers, _output_metrics)
from .delta_evidence_arbitration import _mode_batches as _clean_mode_batches
from .delta_orchestration_controller import (
    EPS, _assert_runtime, _encode_uniform, _safe_ratio)
from .delta_orchestration_screen import (
    _expected_call, _normalize, _render, _rows, _task_content)
from .delta_reasoning_controller import _candidate_metrics
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer


PROTOCOL_VERSION = "2026-07-14-v1"
INJECT_LAYER = 2
LAYERS = (2, 9, 16, 23, 30)
MEDIATION_LAYER = 23
N_NULL = 100


def _zero_rows():
    return [{**row, "key": "D", "database_value": 0} for row in _rows()]


def _task_texts(tok, rows, mode):
    return [_render(tok, [{"role": "user", "content": _task_content(row, mode)}])
            for row in rows]


def _mode_batches(tok, rows, dev):
    source_texts = _task_texts(tok, rows, "calculate")
    target_texts = _task_texts(tok, rows, "lookup")
    source, source_am = _encode_uniform(tok, source_texts, dev)
    target, target_am = _encode_uniform(tok, target_texts, dev)
    changes = [[i for i, (a, b) in enumerate(zip(left, right)) if a != b]
               for left, right in zip(source.tolist(), target.tolist())]
    unique = {tuple(x) for x in changes}
    if len(unique) != 1 or len(next(iter(unique))) != 1:
        raise ValueError(f"native mode alignment failed: {changes}")
    return (source_texts, target_texts, source, source_am, target, target_am,
            next(iter(unique))[0])


def _database_followups(tok, rows, mode, authoritative=False):
    texts = []
    for row in rows:
        suffix = "Tool result: 0. "
        if authoritative:
            suffix += (
                "The executed database result is authoritative and must "
                "override any arithmetic calculation. "
            )
        suffix += "Return only the final answer."
        texts.append(_render(tok, [
            {"role": "user", "content": _task_content(row, mode)},
            {"role": "assistant", "content": _expected_call(row, "lookup")},
            {"role": "user", "content": suffix},
        ]))
    return texts


def _json_metrics(metrics):
    return {k: (v.tolist() if torch.is_tensor(v) else v)
            for k, v in metrics.items()}


def _score_conflicts(raw_answers, target_answers, source_answers):
    """Score an all-conflict batch without requiring collision rows."""
    answers = [_normalize(answer) for answer in raw_answers]
    target_rows = [
        answer == target for answer, target in zip(answers, target_answers)]
    source_rows = [
        answer == source for answer, source in zip(answers, source_answers)]
    return {
        "target_acc": sum(target_rows) / len(target_rows),
        "source_acc": sum(source_rows) / len(source_rows),
        "diagnostic_target_acc": sum(target_rows) / len(target_rows),
        "diagnostic_source_acc": sum(source_rows) / len(source_rows),
        "collision_exact_acc": None,
        "collision_answers": [],
        "answers": answers,
        "target_rows": target_rows,
        "source_rows": source_rows,
    }


@torch.no_grad()
def run_delta_multiturn_evidence_bridge(
        model_path, out_dir, quantization="8bit", device_map=None,
        seed=0, n_null=N_NULL):
    if quantization != "8bit" or seed != 0 or n_null != 100:
        raise ValueError("frozen multiturn bridge config mismatch")
    os.makedirs(out_dir, exist_ok=True)
    runtime = _assert_runtime()
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    if int(model.config.num_hidden_layers) != 32:
        raise ValueError("frozen bridge expects 32 layers")
    dev = input_device(model)
    rows = _zero_rows()
    train_rows, test_rows = rows[::2], rows[1::2]
    (train_source_texts, train_lookup_texts, train_source, train_source_am,
     train_lookup, train_lookup_am, mode_position) = _mode_batches(
         tok, train_rows, dev)
    (_, _, test_source_task, _test_source_am, test_lookup_task,
     _test_lookup_am, test_mode_position) = _mode_batches(tok, test_rows, dev)
    if test_mode_position != mode_position:
        raise ValueError("mode position differs across splits")

    source_texts = _database_followups(tok, test_rows, "calculate")
    lookup_texts = _database_followups(tok, test_rows, "lookup")
    trusted_texts = _database_followups(
        tok, test_rows, "lookup", authoritative=True)
    target_answers = ["0"] * len(test_rows)
    source_answers = [str(row["a"] + row["b"]) for row in test_rows]
    diagnostic = [True] * len(test_rows)

    behavior = {}
    for name, texts, target, source in (
            ("calculate_conflict", source_texts, source_answers, target_answers),
            ("lookup_conflict", lookup_texts, target_answers, source_answers),
            ("authoritative_upper_bound", trusted_texts,
             target_answers, source_answers)):
        raw, ids = _generate_answers(
            model, tok, texts, dev, mode_position=mode_position)
        behavior[name] = {
            "score": _score_conflicts(raw, target, source),
            "raw": raw, "token_ids": ids,
        }
        log(f"bridge behavior {name}: {behavior[name]['score']}")
    g0 = bool(
        behavior["calculate_conflict"]["score"]["diagnostic_target_acc"] >= .80
        and behavior["authoritative_upper_bound"]["score"][
            "diagnostic_target_acc"] >= .90
        and behavior["lookup_conflict"]["score"][
            "diagnostic_target_acc"] <= .50)
    result = {
        "stage": "delta_multiturn_evidence_bridge",
        "protocol_version": PROTOCOL_VERSION, "model_path": model_path,
        "runtime": runtime, "layers": list(LAYERS),
        "mode_position": mode_position, "train_rows": train_rows,
        "test_rows": test_rows, "behavior": behavior,
    }
    if not g0:
        result["gates"] = {"G0": False}
        result["verdict"] = "MULTITURN_BRIDGE_DIAGNOSTIC_INELIGIBLE"
        return _write(out_dir, result)

    # Learn the original agent-prompt policy displacement from donors only.
    # This early-token state cannot contain the later tool result; the test is
    # whether strengthening it is sufficient to change later evidence use.
    _, source_cache = _forward(
        model, train_source, train_source_am, (mode_position,), (INJECT_LAYER,))
    _, lookup_cache = _forward(
        model, train_lookup, train_lookup_am, (mode_position,), (INJECT_LAYER,))
    donor_rows = lookup_cache[INJECT_LAYER][:, 0] - source_cache[INJECT_LAYER][:, 0]
    native_direction = donor_rows.mean(0)

    # Recompute the successful clean-context direction on the same donors.
    (_cs_text, _ct_text, clean_source, clean_source_am, clean_target,
     clean_target_am, clean_mode_position) = _clean_mode_batches(
         tok, train_rows, dev)
    _, clean_source_cache = _forward(
        model, clean_source, clean_source_am, (clean_mode_position,),
        (INJECT_LAYER,))
    _, clean_target_cache = _forward(
        model, clean_target, clean_target_am, (clean_mode_position,),
        (INJECT_LAYER,))
    clean_direction = (
        clean_target_cache[INJECT_LAYER][:, 0]
        - clean_source_cache[INJECT_LAYER][:, 0]).mean(0)
    direction_cosine = float(torch.nn.functional.cosine_similarity(
        native_direction, clean_direction, dim=0))

    source_label = int(train_source[0, mode_position])
    target_label = int(train_lookup[0, mode_position])
    embedding = model.get_input_embeddings().weight.detach().float().cpu()
    lexical_raw = embedding[target_label] - embedding[source_label]
    lexical = lexical_raw / lexical_raw.norm().clamp(min=EPS) * native_direction.norm()

    source_ids, source_am = _encode_uniform(tok, source_texts, dev)
    lookup_ids, lookup_am = _encode_uniform(tok, lookup_texts, dev)
    trusted_ids, trusted_am = _encode_uniform(tok, trusted_texts, dev)
    if (int(source_ids[0, mode_position]) != source_label
            or int(lookup_ids[0, mode_position]) != target_label):
        raise ValueError("answer-turn mode-token address moved")
    last = lookup_ids.shape[1] - 1
    trusted_last = trusted_ids.shape[1] - 1
    if trusted_last != last:
        # Different suffix length is valid, but sparse patching compares each
        # condition at its own final decision token.
        pass
    target_token_ids = _answer_ids(tok, lookup_texts, target_answers)
    source_token_ids = _answer_ids(tok, lookup_texts, source_answers)
    batch_native = native_direction.unsqueeze(0).expand(len(test_rows), -1)
    batch_clean = clean_direction.unsqueeze(0).expand(len(test_rows), -1)
    batch_lexical = lexical.unsqueeze(0).expand(len(test_rows), -1)
    wrong_address = max(0, mode_position - 8)

    lookup_logits, lookup_state = _forward(
        model, lookup_ids, lookup_am, (mode_position, last), LAYERS)
    trusted_logits, trusted_state = _forward(
        model, trusted_ids, trusted_am, (mode_position, trusted_last), LAYERS)
    amplified_logits, amplified_state = _forward(
        model, lookup_ids, lookup_am, (mode_position, last), LAYERS,
        add=(INJECT_LAYER, mode_position, batch_native))
    clean_transfer_logits, clean_transfer_state = _forward(
        model, lookup_ids, lookup_am, (mode_position, last), LAYERS,
        add=(INJECT_LAYER, mode_position, batch_clean))
    lexical_logits, lexical_state = _forward(
        model, lookup_ids, lookup_am, (mode_position, last), LAYERS,
        add=(INJECT_LAYER, mode_position, batch_lexical))
    wrong_logits, _ = _forward(
        model, lookup_ids, lookup_am, (last,),
        add=(INJECT_LAYER, wrong_address, batch_native))

    output = {
        "native_amplification": _output_metrics(
            lookup_logits, trusted_logits, amplified_logits,
            target_token_ids, source_token_ids, diagnostic),
        "clean_transfer": _output_metrics(
            lookup_logits, trusted_logits, clean_transfer_logits,
            target_token_ids, source_token_ids, diagnostic),
        "lexical": _output_metrics(
            lookup_logits, trusted_logits, lexical_logits,
            target_token_ids, source_token_ids, diagnostic),
        "wrong_address": _output_metrics(
            lookup_logits, trusted_logits, wrong_logits,
            target_token_ids, source_token_ids, diagnostic),
    }
    # The L2 local reference is computed on held-out initial task prompts.
    _, test_source_cache = _forward(
        model, test_source_task, _test_source_am, (mode_position,), (INJECT_LAYER,))
    _, test_lookup_cache = _forward(
        model, test_lookup_task, _test_lookup_am, (mode_position,), (INJECT_LAYER,))
    _, add_task_cache = _forward(
        model, test_source_task, _test_source_am, (mode_position,),
        (INJECT_LAYER,), add=(INJECT_LAYER, mode_position, batch_native))
    local = _candidate_metrics(
        add_task_cache[INJECT_LAYER][:, 0] - test_source_cache[INJECT_LAYER][:, 0],
        test_lookup_cache[INJECT_LAYER][:, 0] - test_source_cache[INJECT_LAYER][:, 0])

    trajectory = {}
    for layer in LAYERS:
        native = (trusted_state[layer][:, 1]
                  - lookup_state[layer][:, 1])
        candidates = {
            "native_amplification": amplified_state[layer][:, 1]
            - lookup_state[layer][:, 1],
            "clean_transfer": clean_transfer_state[layer][:, 1]
            - lookup_state[layer][:, 1],
            "lexical": lexical_state[layer][:, 1]
            - lookup_state[layer][:, 1],
        }
        trajectory[str(layer)] = {}
        for name, candidate in candidates.items():
            metric = _candidate_metrics(candidate, native)
            trajectory[str(layer)][name] = {
                "cosine": metric["cosine"], "error": metric["error"]}

    lookup_ld = _ld(lookup_logits, target_token_ids, source_token_ids)
    patch_amp_logits, _ = _forward(
        model, lookup_ids, lookup_am, (last,),
        patch=(MEDIATION_LAYER, last,
               amplified_state[MEDIATION_LAYER][:, 1]))
    # Row-matched trusted states live at the trusted prompt's own last token.
    patch_trusted_logits, _ = _forward(
        model, lookup_ids, lookup_am, (last,),
        patch=(MEDIATION_LAYER, last,
               trusted_state[MEDIATION_LAYER][:, 1]))
    block_amp_logits, _ = _forward(
        model, lookup_ids, lookup_am, (last,),
        add=(INJECT_LAYER, mode_position, batch_native),
        patch=(MEDIATION_LAYER, last, lookup_state[MEDIATION_LAYER][:, 1]))
    def effect(logits):
        return float((_ld(logits, target_token_ids, source_token_ids) - lookup_ld).mean())
    amp_effect = output["native_amplification"]["effect"]
    mediation = {
        "patch_amplified": effect(patch_amp_logits),
        "patch_trusted": effect(patch_trusted_logits),
        "blocked_amplified": effect(block_amp_logits),
    }
    mediation["patch_ratio"] = _safe_ratio(
        mediation["patch_amplified"], mediation["patch_trusted"])
    mediation["block_fraction"] = _safe_ratio(
        amp_effect - mediation["blocked_amplified"], amp_effect)

    generator = torch.Generator().manual_seed(seed + 9917)
    null_effects = []
    for i in range(n_null):
        random = torch.randn(native_direction.shape, generator=generator)
        random = random / random.norm().clamp(min=EPS) * native_direction.norm()
        logits, _ = _forward(
            model, lookup_ids, lookup_am, (last,),
            add=(INJECT_LAYER, mode_position,
                 random.unsqueeze(0).expand(len(test_rows), -1)))
        null_effects.append(effect(logits))
        if (i + 1) % 10 == 0:
            log(f"bridge random {i + 1}/{n_null}")
    null_exceed = sum(x >= amp_effect for x in null_effects)

    generation_specs = {
        "lookup_base": (lookup_texts, None, mode_position),
        "trusted": (trusted_texts, None, mode_position),
        "native_amplification": (lookup_texts, native_direction, mode_position),
        "clean_transfer": (lookup_texts, clean_direction, mode_position),
        "lexical": (lookup_texts, lexical, mode_position),
        "wrong_address": (lookup_texts, native_direction, wrong_address),
    }
    generations, scores = {}, {}
    for name, (texts, direction, position) in generation_specs.items():
        raw, ids = _generate_answers(
            model, tok, texts, dev, direction=direction,
            mode_position=position)
        generations[name] = {"raw": raw, "token_ids": ids}
        scores[name] = _score_conflicts(
            raw, target_answers, source_answers)

    ratio = output["native_amplification"]["ratio"]
    output_gate = bool(
        ratio is not None and .70 <= ratio <= 1.30
        and output["native_amplification"]["positive_fraction"] >= .80
        and null_exceed <= 1)
    behavior_gate = bool(
        scores["native_amplification"]["diagnostic_target_acc"] >= .80
        and scores["native_amplification"]["diagnostic_target_acc"]
        - scores["lookup_base"]["diagnostic_target_acc"] >= .50)
    q = trajectory[str(MEDIATION_LAYER)]["native_amplification"]
    trajectory_gate = q["cosine"] >= .80 and q["error"] <= .60
    med_ratio = mediation["patch_ratio"]
    mediation_gate = bool(
        med_ratio is not None and .70 <= med_ratio <= 1.30
        and mediation["block_fraction"] is not None
        and mediation["block_fraction"] >= .70)
    controls_gate = all(
        scores[name]["diagnostic_target_acc"] < .80
        for name in ("clean_transfer", "lexical", "wrong_address"))
    gates = {
        "G0": True,
        "A1": bool(local["cosine"] >= .80 and local["error"] <= .60),
        "O1": output_gate, "W1": behavior_gate,
        "Q1": trajectory_gate, "M1_M2": mediation_gate,
        "B1": controls_gate,
    }
    verdict = ("CONTEXT_ALIGNED_MULTITURN_EVIDENCE_RESCUE"
               if all(gates.values()) else
               "MULTITURN_EVIDENCE_NOT_SIMPLE_AMPLIFICATION")
    raw_artifact = "raw_delta_multiturn_evidence_bridge.pt"
    torch.save({
        "native_direction": native_direction,
        "clean_direction": clean_direction,
        "donor_rows": donor_rows,
        "logits": {"lookup": lookup_logits, "trusted": trusted_logits,
                   "amplified": amplified_logits,
                   "clean_transfer": clean_transfer_logits,
                   "lexical": lexical_logits},
        "states": {"lookup": lookup_state, "trusted": trusted_state,
                   "amplified": amplified_state,
                   "clean_transfer": clean_transfer_state},
        "null_effects": torch.tensor(null_effects),
    }, os.path.join(out_dir, raw_artifact))
    result.update({
        "direction_norms": {"native": float(native_direction.norm()),
                            "clean": float(clean_direction.norm()),
                            "lexical": float(lexical.norm())},
        "native_clean_direction_cosine": direction_cosine,
        "local": {"cosine": local["cosine"], "error": local["error"]},
        "wrong_address": wrong_address,
        "output": {k: _json_metrics(v) for k, v in output.items()},
        "trajectory": trajectory, "mediation": mediation,
        "null": {"effects": null_effects, "exceedances": null_exceed},
        "generations": generations, "scores": scores,
        "gates": gates, "raw_tensor_artifact": raw_artifact,
        "verdict": verdict,
    })
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_multiturn_evidence_bridge.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
