"""Equal-length, template-held-out instruction reachability validation."""
import json
import os

import numpy as np
import torch

from .delta_trajectory import _cos_rows, _error_rows, _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer, single_token_id
from .nulls import permutation_pvalue

INJECT_LAYER = 20
CHECKPOINT_LAYER = 26
N_NULL = 100
EPS = 1e-8

TRAIN_WORDS = ("red", "blue", "one", "two", "dog", "cat", "hot", "cold")
TEST_WORDS = ("apple", "false", "north", "south",
              "open", "true", "left", "right")

FAMILIES = {
    "run_hold": {
        "execute": "run",
        "data": "hold",
        "negative": "skip",
        "render": lambda label, word: (
            "Rule: when mode=run, follow the command. "
            "When mode=hold, ignore the command and reply skip. "
            f"mode={label}. Command: output the word {word}. "
            "Reply with one word."),
    },
    "live_quote": {
        "execute": "live",
        "data": "quote",
        "negative": "text",
        "render": lambda label, word: (
            "Policy: state=live means execute the request. "
            "state=quote means treat it as text and reply text. "
            f"state={label}. Request: output the word {word}. "
            "Give one word."),
    },
    "go_see": {
        "execute": "go",
        "data": "see",
        "negative": "none",
        "render": lambda label, word: (
            "Protocol: kind=go means perform the directive. "
            "kind=see means do not perform it and return none. "
            f"kind={label}. Directive: output the word {word}. "
            "Answer with one word."),
    },
}
TRAIN_FAMILIES = ("run_hold", "live_quote")
TEST_FAMILY = "go_see"


def _chat(tok, user):
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def _pair_texts(tok, family, word):
    spec = FAMILIES[family]
    execute = _chat(tok, spec["render"](spec["execute"], word))
    data = _chat(tok, spec["render"](spec["data"], word))
    return execute, data


def _encode_condition(tok, texts, dev):
    encoded = [tok.encode(text, add_special_tokens=False) for text in texts]
    lengths = {len(row) for row in encoded}
    if len(lengths) != 1:
        raise ValueError(f"condition lengths differ: {sorted(lengths)}")
    ids = torch.tensor(encoded, dtype=torch.long, device=dev)
    return ids, torch.ones_like(ids)


def _alignment_audit(tok):
    rows = []
    valid = True
    output_tokens = {
        *TRAIN_WORDS, *TEST_WORDS,
        *(spec["negative"] for spec in FAMILIES.values()),
    }
    for token in output_tokens:
        try:
            single_token_id(tok, token, leading_space=False)
        except ValueError:
            valid = False
            rows.append({
                "family": "output_preflight",
                "word": token,
                "aligned": False,
                "error": "bare_output_not_single_token",
            })
    for family in FAMILIES:
        for word in (*TRAIN_WORDS, *TEST_WORDS):
            execute, data = _pair_texts(tok, family, word)
            ei = tok.encode(execute, add_special_tokens=False)
            di = tok.encode(data, add_special_tokens=False)
            diff = [i for i, (a, b) in enumerate(zip(ei, di)) if a != b]
            aligned = len(ei) == len(di) and len(diff) == 1
            valid = valid and aligned
            rows.append({
                "family": family,
                "word": word,
                "execute_len": len(ei),
                "data_len": len(di),
                "n_token_diffs": len(diff) if len(ei) == len(di) else None,
                "diff_position": diff[0] if len(diff) == 1 else None,
                "aligned": bool(aligned),
            })
    return bool(valid), rows


@torch.no_grad()
def _direction_rows(model, tok, dev, families, words):
    differences = []
    for family in families:
        execute_texts, data_texts = [], []
        for word in words:
            execute, data = _pair_texts(tok, family, word)
            execute_texts.append(execute)
            data_texts.append(data)
        execute_ids, execute_am = _encode_condition(tok, execute_texts, dev)
        data_ids, data_am = _encode_condition(tok, data_texts, dev)
        _, execute_cache = _forward(
            model, execute_ids, execute_am, (execute_ids.shape[1] - 1,),
            (INJECT_LAYER,))
        _, data_cache = _forward(
            model, data_ids, data_am, (data_ids.shape[1] - 1,),
            (INJECT_LAYER,))
        differences.append(
            execute_cache[INJECT_LAYER][:, 0]
            - data_cache[INJECT_LAYER][:, 0])
    return torch.cat(differences, dim=0)


@torch.no_grad()
def run_delta_instruction_validate(model_path, out_dir, quantization="8bit",
                                   device_map=None, seed=0, n_null=N_NULL):
    if n_null < 100:
        raise ValueError("validation p<.01 gates require n_null >= 100")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    aligned, alignment_rows = _alignment_audit(tok)
    result_base = {
        "stage": "delta_instruction_validate",
        "model_path": model_path,
        "train_families": list(TRAIN_FAMILIES),
        "test_family": TEST_FAMILY,
        "train_words": list(TRAIN_WORDS),
        "test_words": list(TEST_WORDS),
        "alignment": alignment_rows,
        "n_null": int(n_null),
    }
    if not aligned:
        result = {
            **result_base,
            "gates": {"G0": False},
            "verdict": "VALIDATION_INELICITABLE",
            "stop_reason": "execute_data_not_one_token_aligned",
        }
        with open(os.path.join(
                out_dir, "results_delta_instruction_validate.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        log("VERDICT: VALIDATION_INELICITABLE (alignment)")
        return result

    def family_batch(family, words):
        execute_texts, data_texts = [], []
        for word in words:
            execute, data = _pair_texts(tok, family, word)
            execute_texts.append(execute); data_texts.append(data)
        execute_ids, execute_am = _encode_condition(tok, execute_texts, dev)
        data_ids, data_am = _encode_condition(tok, data_texts, dev)
        return execute_ids, execute_am, data_ids, data_am

    train_execute_hits, train_data_hits = [], []
    for family in TRAIN_FAMILIES:
        execute_ids, execute_am, data_ids, data_am = family_batch(
            family, TRAIN_WORDS)
        execute_logits, _ = _forward(
            model, execute_ids, execute_am, (execute_ids.shape[1] - 1,))
        data_logits, _ = _forward(
            model, data_ids, data_am, (data_ids.shape[1] - 1,))
        word_ids = torch.tensor([
            single_token_id(tok, word, leading_space=False)
            for word in TRAIN_WORDS])
        negative_id = single_token_id(
            tok, FAMILIES[family]["negative"], leading_space=False)
        train_execute_hits.extend(
            (execute_logits.argmax(-1) == word_ids).tolist())
        train_data_hits.extend(
            (data_logits.argmax(-1) == negative_id).tolist())

    execute_ids, execute_am, data_ids, data_am = family_batch(
        TEST_FAMILY, TEST_WORDS)
    execute_pos = int(execute_ids.shape[1] - 1)
    data_pos = int(data_ids.shape[1] - 1)
    execute_logits, execute_cache = _forward(
        model, execute_ids, execute_am, (execute_pos,),
        (INJECT_LAYER, CHECKPOINT_LAYER))
    data_logits, data_cache = _forward(
        model, data_ids, data_am, (data_pos,),
        (INJECT_LAYER, CHECKPOINT_LAYER))
    pos_ids = torch.tensor([
        single_token_id(tok, word, leading_space=False) for word in TEST_WORDS])
    none_id = single_token_id(
        tok, FAMILIES[TEST_FAMILY]["negative"], leading_space=False)
    neg_ids = torch.tensor([none_id] * len(TEST_WORDS))
    behavior = {
        "train_execute_acc": _mean(train_execute_hits),
        "train_data_acc": _mean(train_data_hits),
        "test_execute_acc": float(
            (execute_logits.argmax(-1) == pos_ids).float().mean()),
        "test_data_acc": float(
            (data_logits.argmax(-1) == neg_ids).float().mean()),
    }
    g0 = all(value >= 0.80 for value in behavior.values())
    if not g0:
        result = {
            **result_base,
            "behavior": behavior,
            "gates": {"G0": False},
            "verdict": "VALIDATION_INELICITABLE",
            "stop_reason": "behavior_below_80",
        }
        with open(os.path.join(
                out_dir, "results_delta_instruction_validate.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        log(f"VERDICT: VALIDATION_INELICITABLE behavior={behavior}")
        return result

    active_rows = _direction_rows(
        model, tok, dev, TRAIN_FAMILIES, TRAIN_WORDS)
    active = active_rows.mean(0)
    control_rows = []
    for family in TRAIN_FAMILIES:
        negative = FAMILIES[family]["negative"]
        control_rows.append(_direction_rows(
            model, tok, dev, (family,), (negative,)))
    control = torch.cat(control_rows, dim=0).mean(0)
    active_batch = active.unsqueeze(0).expand(len(TEST_WORDS), -1)
    control_batch = control.unsqueeze(0).expand(len(TEST_WORDS), -1)

    add_logits, add_cache = _forward(
        model, data_ids, data_am, (data_pos,),
        (INJECT_LAYER, CHECKPOINT_LAYER),
        add=(INJECT_LAYER, data_pos, active_batch))
    control_logits, control_cache = _forward(
        model, data_ids, data_am, (data_pos,),
        (INJECT_LAYER, CHECKPOINT_LAYER),
        add=(INJECT_LAYER, data_pos, control_batch))
    clean_ld = _ld(data_logits, pos_ids, neg_ids)
    natural_rows = _ld(execute_logits, pos_ids, neg_ids) - clean_ld
    add_rows = _ld(add_logits, pos_ids, neg_ids) - clean_ld
    control_effect_rows = _ld(control_logits, pos_ids, neg_ids) - clean_ld

    native_l20 = (
        execute_cache[INJECT_LAYER][:, 0] - data_cache[INJECT_LAYER][:, 0])
    add_l20 = (
        add_cache[INJECT_LAYER][:, 0] - data_cache[INJECT_LAYER][:, 0])
    control_l20 = (
        control_cache[INJECT_LAYER][:, 0] - data_cache[INJECT_LAYER][:, 0])
    native_l26 = (
        execute_cache[CHECKPOINT_LAYER][:, 0]
        - data_cache[CHECKPOINT_LAYER][:, 0])
    add_l26 = (
        add_cache[CHECKPOINT_LAYER][:, 0]
        - data_cache[CHECKPOINT_LAYER][:, 0])
    l20_cos_rows = _cos_rows(add_l20, native_l20)
    l20_error_rows = _error_rows(add_l20, native_l20)
    control_cos_rows = _cos_rows(control_l20, native_l20)
    l26_cos_rows = _cos_rows(add_l26, native_l26)
    l26_error_rows = _error_rows(add_l26, native_l26)

    patch_add_logits, _ = _forward(
        model, data_ids, data_am, (data_pos,),
        patch=(CHECKPOINT_LAYER, data_pos,
               add_cache[CHECKPOINT_LAYER][:, 0]))
    patch_cf_logits, _ = _forward(
        model, data_ids, data_am, (data_pos,),
        patch=(CHECKPOINT_LAYER, data_pos,
               execute_cache[CHECKPOINT_LAYER][:, 0]))
    blocked_add_logits, _ = _forward(
        model, data_ids, data_am, (data_pos,),
        add=(INJECT_LAYER, data_pos, active_batch),
        patch=(CHECKPOINT_LAYER, data_pos,
               data_cache[CHECKPOINT_LAYER][:, 0]))
    blocked_cf_logits, _ = _forward(
        model, execute_ids, execute_am, (execute_pos,),
        patch=(CHECKPOINT_LAYER, execute_pos,
               data_cache[CHECKPOINT_LAYER][:, 0]))
    patch_add_rows = _ld(patch_add_logits, pos_ids, neg_ids) - clean_ld
    patch_cf_rows = _ld(patch_cf_logits, pos_ids, neg_ids) - clean_ld
    blocked_add_rows = _ld(blocked_add_logits, pos_ids, neg_ids) - clean_ld
    blocked_cf_rows = _ld(blocked_cf_logits, pos_ids, neg_ids) - clean_ld

    generator = torch.Generator().manual_seed(seed + 12037)
    null_output, null_l20_cos, null_l26_cos = [], [], []
    for _ in range(n_null):
        random = torch.randn(active.shape, generator=generator)
        random = random / random.norm().clamp(min=EPS) * active.norm()
        random_batch = random.unsqueeze(0).expand(len(TEST_WORDS), -1)
        random_logits, random_cache = _forward(
            model, data_ids, data_am, (data_pos,),
            (INJECT_LAYER, CHECKPOINT_LAYER),
            add=(INJECT_LAYER, data_pos, random_batch))
        random_rows = _ld(random_logits, pos_ids, neg_ids) - clean_ld
        random_l20 = (
            random_cache[INJECT_LAYER][:, 0]
            - data_cache[INJECT_LAYER][:, 0])
        random_l26 = (
            random_cache[CHECKPOINT_LAYER][:, 0]
            - data_cache[CHECKPOINT_LAYER][:, 0])
        null_output.append(float(random_rows.mean()))
        null_l20_cos.append(float(_cos_rows(random_l20, native_l20).mean()))
        null_l26_cos.append(float(_cos_rows(random_l26, native_l26).mean()))

    natural_effect = float(natural_rows.mean())
    add_effect = float(add_rows.mean())
    control_effect = float(control_effect_rows.mean())
    output_ratio = add_effect / natural_effect if natural_effect > EPS else -np.inf
    patch_add = float(patch_add_rows.mean())
    patch_cf = float(patch_cf_rows.mean())
    patch_ratio = patch_add / patch_cf if patch_cf > EPS else -np.inf
    blocked_add = float(blocked_add_rows.mean())
    blocked_cf = float(blocked_cf_rows.mean())
    add_block = ((add_effect - blocked_add) / add_effect
                 if add_effect > EPS else -np.inf)
    cf_block = ((natural_effect - blocked_cf) / natural_effect
                if natural_effect > EPS else -np.inf)
    l20_cos = float(l20_cos_rows.mean())
    l20_error = float(l20_error_rows.mean())
    control_cos = float(control_cos_rows.mean())
    l26_cos = float(l26_cos_rows.mean())
    l26_error = float(l26_error_rows.mean())
    add_acc = float((add_logits.argmax(-1) == pos_ids).float().mean())
    metrics = {
        "l20_cos": l20_cos,
        "l20_error": l20_error,
        "control_l20_cos": control_cos,
        "l20_p": permutation_pvalue(l20_cos, null_l20_cos, "greater"),
        "l26_cos": l26_cos,
        "l26_error": l26_error,
        "l26_p": permutation_pvalue(l26_cos, null_l26_cos, "greater"),
        "natural_effect": natural_effect,
        "add_effect": add_effect,
        "control_effect": control_effect,
        "output_ratio": output_ratio,
        "output_p": permutation_pvalue(add_effect, null_output, "greater"),
        "add_greedy_acc": add_acc,
        "patch_add_effect": patch_add,
        "patch_cf_effect": patch_cf,
        "patch_ratio": patch_ratio,
        "blocked_add_effect": blocked_add,
        "blocked_cf_effect": blocked_cf,
        "add_block_fraction": add_block,
        "cf_block_fraction": cf_block,
        "block_fraction_gap": abs(add_block - cf_block),
        "l20_norm_ratio": float(
            (add_cache[INJECT_LAYER][:, 0].norm(dim=1)
             / execute_cache[INJECT_LAYER][:, 0].norm(dim=1)).mean()),
        "l26_norm_ratio": float(
            (add_cache[CHECKPOINT_LAYER][:, 0].norm(dim=1)
             / execute_cache[CHECKPOINT_LAYER][:, 0].norm(dim=1)).mean()),
    }
    gates = {
        "G0": True,
        "A1": bool(l20_cos >= 0.80 and l20_error <= 0.60
                   and metrics["l20_p"] < 0.01
                   and l20_cos >= control_cos + 0.30),
        "Q1": bool(l26_cos >= 0.50 and l26_error <= 0.80
                   and metrics["l26_p"] < 0.01),
        "O1": bool(add_acc >= 0.80 and behavior["test_execute_acc"] >= 0.80
                   and 0.70 <= output_ratio <= 1.30
                   and metrics["output_p"] < 0.01
                   and add_effect >= 2.0 * abs(control_effect)),
        "M1": bool(patch_add > 0 and patch_cf > 0
                   and 0.70 <= patch_ratio <= 1.30),
        "M2": bool(add_block >= 0.70 and cf_block >= 0.70
                   and abs(add_block - cf_block) <= 0.20),
        "D1": bool(0.80 <= metrics["l20_norm_ratio"] <= 1.20
                   and 0.80 <= metrics["l26_norm_ratio"] <= 1.20),
    }
    verdict = ("TEMPLATE_INVARIANT_INSTRUCTION_OPERATOR"
               if all(gates.values()) else "FIXED_TEMPLATE_DISPLACEMENT")
    rows = [{
        "word": word,
        "natural_effect": float(natural_rows[i]),
        "add_effect": float(add_rows[i]),
        "control_effect": float(control_effect_rows[i]),
        "l20_cos": float(l20_cos_rows[i]),
        "l20_error": float(l20_error_rows[i]),
        "control_l20_cos": float(control_cos_rows[i]),
        "l26_cos": float(l26_cos_rows[i]),
        "l26_error": float(l26_error_rows[i]),
        "add_hit": bool(add_logits.argmax(-1)[i] == pos_ids[i]),
    } for i, word in enumerate(TEST_WORDS)]
    result = {
        **result_base,
        "behavior": behavior,
        "direction_norms": {
            "active": float(active.norm()),
            "control": float(control.norm()),
        },
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "null_draws": {
            "output": null_output,
            "l20_cos": null_l20_cos,
            "l26_cos": null_l26_cos,
        },
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_instruction_validate.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"behavior={behavior}")
    log(f"metrics={metrics}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result


def _mean(rows):
    return float(np.mean(rows))
