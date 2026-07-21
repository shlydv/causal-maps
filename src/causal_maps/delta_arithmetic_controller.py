"""Held-out operand add-to-subtract latent operator test."""
import json
import os

import numpy as np
import torch

from .delta_reasoning_controller import _candidate_metrics
from .delta_reasoning_screen import (
    _arithmetic_prompt, _arithmetic_rows, _encode_family)
from .delta_trajectory import _cos_rows, _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer
from .nulls import permutation_pvalue

INJECT_LAYER = 8
MEDIATION_LAYER = 20
LAYERS = (8, 12, 16, 20, 26)
N_NULL = 100
EPS = 1e-8


def _answer(row, mode):
    return row["a"] + row["b"] if mode == "add" else row["a"] - row["b"]


@torch.no_grad()
def run_delta_arithmetic_controller(model_path, out_dir, quantization="8bit",
                                    device_map=None, seed=0, n_null=N_NULL):
    if model_path != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(f"frozen model mismatch: {model_path}")
    if quantization != "8bit" or seed != 0 or n_null != N_NULL:
        raise ValueError(
            f"frozen config mismatch: quant={quantization} "
            f"seed={seed} null={n_null}")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    rows = _arithmetic_rows()
    train_rows, test_rows = rows[::2], rows[1::2]
    train = _encode_family(
        tok, train_rows, _arithmetic_prompt, ("add", "subtract"),
        _answer, dev)
    test = _encode_family(
        tok, test_rows, _arithmetic_prompt, ("add", "subtract"),
        _answer, dev)
    tr_ids, tr_am, _tr_answers, _tr_text, tr_changed, tr_length = train
    te_ids, te_am, te_answers, te_text, te_changed, te_length = test
    if tr_length != te_length or set(tr_changed + te_changed) != {56}:
        raise ValueError(
            f"frozen alignment failed: lengths={tr_length}/{te_length}, "
            f"positions={set(tr_changed + te_changed)}")
    source_ids = te_answers["add"]
    target_ids = te_answers["subtract"]
    last = int(te_ids["add"].shape[1] - 1)

    clean_logits, clean_cache = _forward(
        model, te_ids["add"], te_am["add"], (last,), LAYERS)
    natural_logits, natural_cache = _forward(
        model, te_ids["subtract"], te_am["subtract"], (last,), LAYERS)
    clean_acc = float(
        (clean_logits.argmax(-1) == source_ids).float().mean())
    natural_acc = float(
        (natural_logits.argmax(-1) == target_ids).float().mean())
    g0 = clean_acc >= 0.90 and natural_acc >= 0.90
    preflight = {
        "train_n": len(train_rows),
        "test_n": len(test_rows),
        "prompt_length": te_length,
        "changed_position": 56,
        "clean_acc": clean_acc,
        "natural_acc": natural_acc,
        "pass": bool(g0),
        "rows": [{
            **row,
            "add_answer": te_text["add"][idx],
            "subtract_answer": te_text["subtract"][idx],
            "add_answer_id": int(source_ids[idx]),
            "subtract_answer_id": int(target_ids[idx]),
            "clean_prediction_id": int(clean_logits.argmax(-1)[idx]),
            "natural_prediction_id": int(natural_logits.argmax(-1)[idx]),
        } for idx, row in enumerate(test_rows)],
    }
    if not g0:
        result = {
            "stage": "delta_arithmetic_controller",
            "model_path": model_path,
            "preflight": preflight,
            "train_rows": train_rows,
            "gates": {"G0": False},
            "verdict": "ARITHMETIC_INELICITABLE",
        }
        with open(os.path.join(
                out_dir, "results_delta_arithmetic_controller.json"), "w") as f:
            json.dump(result, f, indent=2)
        log(f"VERDICT: ARITHMETIC_INELICITABLE {preflight}")
        return result

    _, train_add_cache = _forward(
        model, tr_ids["add"], tr_am["add"], (last,), (INJECT_LAYER,))
    _, train_sub_cache = _forward(
        model, tr_ids["subtract"], tr_am["subtract"], (last,), (INJECT_LAYER,))
    direction = (
        train_sub_cache[INJECT_LAYER][:, 0]
        - train_add_cache[INJECT_LAYER][:, 0]).mean(0)
    batch_direction = direction.unsqueeze(0).expand(len(test_rows), -1)
    add_token_id = int(te_ids["add"][0, 56])
    sub_token_id = int(te_ids["subtract"][0, 56])
    embedding = model.get_input_embeddings().weight
    embed_direction = (
        embedding[sub_token_id].detach().float().cpu()
        - embedding[add_token_id].detach().float().cpu())
    batch_embed = embed_direction.unsqueeze(0).expand(len(test_rows), -1)

    steered_logits, steered_cache = _forward(
        model, te_ids["add"], te_am["add"], (last,), LAYERS,
        add=(INJECT_LAYER, last, batch_direction))
    embed_logits, embed_cache = _forward(
        model, te_ids["add"], te_am["add"], (last,), LAYERS,
        add=(INJECT_LAYER, last, batch_embed))
    reverse_logits, _ = _forward(
        model, te_ids["subtract"], te_am["subtract"], (last,), LAYERS,
        add=(INJECT_LAYER, last, -batch_direction))

    clean_ld = _ld(clean_logits, target_ids, source_ids)
    natural_rows = _ld(natural_logits, target_ids, source_ids) - clean_ld
    steered_rows = _ld(steered_logits, target_ids, source_ids) - clean_ld
    embed_rows = _ld(embed_logits, target_ids, source_ids) - clean_ld
    natural_effect = float(natural_rows.mean())
    steered_effect = float(steered_rows.mean())
    embed_effect = float(embed_rows.mean())
    output_ratio = (
        steered_effect / natural_effect if natural_effect > EPS else -np.inf)
    embed_ratio = (
        embed_effect / natural_effect if natural_effect > EPS else -np.inf)
    steered_acc = float(
        (steered_logits.argmax(-1) == target_ids).float().mean())
    embed_acc = float(
        (embed_logits.argmax(-1) == target_ids).float().mean())
    reverse_acc = float(
        (reverse_logits.argmax(-1) == source_ids).float().mean())
    positive_fraction = float((steered_rows > 0).float().mean())
    embed_positive_fraction = float((embed_rows > 0).float().mean())

    candidate_ids = torch.tensor([
        int(_encode_family(
            tok, [test_rows[0]], _arithmetic_prompt, ("add", "subtract"),
            lambda _row, _mode, digit=digit: digit, dev)[2]["add"][0])
        for digit in range(10)])
    clean_candidates = clean_logits[:, candidate_ids]
    native_candidates = natural_logits[:, candidate_ids] - clean_candidates
    steered_candidates = steered_logits[:, candidate_ids] - clean_candidates
    embed_candidates = embed_logits[:, candidate_ids] - clean_candidates
    content = _candidate_metrics(steered_candidates, native_candidates)
    embed_content = _candidate_metrics(embed_candidates, native_candidates)

    trajectory, embed_trajectory = {}, {}
    trajectory_rows, embed_trajectory_rows = {}, {}
    for layer in LAYERS:
        native = natural_cache[layer][:, 0] - clean_cache[layer][:, 0]
        steered = steered_cache[layer][:, 0] - clean_cache[layer][:, 0]
        embedded = embed_cache[layer][:, 0] - clean_cache[layer][:, 0]
        active_metric = _candidate_metrics(steered, native)
        embed_metric = _candidate_metrics(embedded, native)
        trajectory[str(layer)] = {
            "cosine": active_metric["cosine"],
            "error": active_metric["error"],
        }
        trajectory_rows[str(layer)] = {
            "cosine": active_metric["cosine_rows"].tolist(),
            "error": active_metric["error_rows"].tolist(),
        }
        embed_trajectory[str(layer)] = {
            "cosine": embed_metric["cosine"],
            "error": embed_metric["error"],
        }
        embed_trajectory_rows[str(layer)] = {
            "cosine": embed_metric["cosine_rows"].tolist(),
            "error": embed_metric["error_rows"].tolist(),
        }

    patch_steered_logits, _ = _forward(
        model, te_ids["add"], te_am["add"], (last,),
        patch=(MEDIATION_LAYER, last,
               steered_cache[MEDIATION_LAYER][:, 0]))
    patch_natural_logits, _ = _forward(
        model, te_ids["add"], te_am["add"], (last,),
        patch=(MEDIATION_LAYER, last,
               natural_cache[MEDIATION_LAYER][:, 0]))
    block_steered_logits, _ = _forward(
        model, te_ids["add"], te_am["add"], (last,),
        add=(INJECT_LAYER, last, batch_direction),
        patch=(MEDIATION_LAYER, last,
               clean_cache[MEDIATION_LAYER][:, 0]))
    block_natural_logits, _ = _forward(
        model, te_ids["subtract"], te_am["subtract"], (last,),
        patch=(MEDIATION_LAYER, last,
               clean_cache[MEDIATION_LAYER][:, 0]))
    patch_steered_rows = (
        _ld(patch_steered_logits, target_ids, source_ids) - clean_ld)
    patch_natural_rows = (
        _ld(patch_natural_logits, target_ids, source_ids) - clean_ld)
    blocked_steered_rows = (
        _ld(block_steered_logits, target_ids, source_ids) - clean_ld)
    blocked_natural_rows = (
        _ld(block_natural_logits, target_ids, source_ids) - clean_ld)
    patch_steered = float(patch_steered_rows.mean())
    patch_natural = float(patch_natural_rows.mean())
    patch_ratio = (
        patch_steered / patch_natural if patch_natural > EPS else -np.inf)
    blocked_steered = float(blocked_steered_rows.mean())
    blocked_natural = float(blocked_natural_rows.mean())
    steered_block_fraction = (
        (steered_effect - blocked_steered) / steered_effect
        if steered_effect > EPS else -np.inf)
    natural_block_fraction = (
        (natural_effect - blocked_natural) / natural_effect
        if natural_effect > EPS else -np.inf)
    block_gap = abs(steered_block_fraction - natural_block_fraction)

    generator = torch.Generator(device="cpu").manual_seed(seed + 811)
    null_output = np.zeros(n_null, dtype=np.float64)
    null_embed_output = np.zeros(n_null, dtype=np.float64)
    null_inject_cos = np.zeros(n_null, dtype=np.float64)
    native_inject = (
        natural_cache[INJECT_LAYER][:, 0]
        - clean_cache[INJECT_LAYER][:, 0])
    direction_norm = direction.norm().clamp(min=EPS)
    embed_norm = embed_direction.norm().clamp(min=EPS)
    for idx in range(n_null):
        random_direction = torch.randn(
            direction.shape, generator=generator, dtype=torch.float32)
        random_direction = (
            random_direction / random_direction.norm() * direction_norm)
        random_batch = random_direction.unsqueeze(0).expand(len(test_rows), -1)
        random_logits, random_cache = _forward(
            model, te_ids["add"], te_am["add"], (last,), (INJECT_LAYER,),
            add=(INJECT_LAYER, last, random_batch))
        null_output[idx] = float(
            (_ld(random_logits, target_ids, source_ids) - clean_ld).mean())
        random_disp = (
            random_cache[INJECT_LAYER][:, 0]
            - clean_cache[INJECT_LAYER][:, 0])
        null_inject_cos[idx] = float(
            _cos_rows(random_disp, native_inject).mean())
        random_embed = torch.randn(
            embed_direction.shape, generator=generator, dtype=torch.float32)
        random_embed = random_embed / random_embed.norm() * embed_norm
        random_embed_batch = random_embed.unsqueeze(0).expand(
            len(test_rows), -1)
        random_embed_logits, _ = _forward(
            model, te_ids["add"], te_am["add"], (last,),
            add=(INJECT_LAYER, last, random_embed_batch))
        null_embed_output[idx] = float(
            (_ld(random_embed_logits, target_ids, source_ids) - clean_ld).mean())

    inject = trajectory[str(INJECT_LAYER)]
    l20 = trajectory[str(MEDIATION_LAYER)]
    output_exceedances = int((null_output >= steered_effect).sum())
    inject_exceedances = int(
        (null_inject_cos >= inject["cosine"]).sum())
    output_p = permutation_pvalue(steered_effect, null_output, "greater")
    inject_p = permutation_pvalue(
        inject["cosine"], null_inject_cos, "greater")
    gates = {
        "G0": True,
        "A1": bool(inject["cosine"] >= 0.50
                   and inject["error"] <= 0.80
                   and inject_exceedances <= 1),
        "O1": bool(steered_acc >= 0.80 and positive_fraction >= 0.80
                   and 0.70 <= output_ratio <= 1.30
                   and output_exceedances <= 1),
        "C1": bool(content["cosine"] >= 0.80 and content["error"] <= 0.60),
        "R1": bool(reverse_acc >= 0.80),
        "Q1": bool(l20["cosine"] >= 0.80 and l20["error"] <= 0.60),
        "M1": bool(patch_steered > 0 and patch_natural > 0
                   and 0.70 <= patch_ratio <= 1.30),
        "M2": bool(steered_block_fraction >= 0.70
                   and natural_block_fraction >= 0.70 and block_gap <= 0.20),
    }
    embed_o = bool(
        embed_acc >= 0.80 and embed_positive_fraction >= 0.80
        and 0.70 <= embed_ratio <= 1.30
        and int((null_embed_output >= embed_effect).sum()) <= 1)
    embed_c = bool(
        embed_content["cosine"] >= 0.80 and embed_content["error"] <= 0.60)
    embed_q = bool(
        embed_trajectory[str(MEDIATION_LAYER)]["cosine"] >= 0.80
        and embed_trajectory[str(MEDIATION_LAYER)]["error"] <= 0.60)
    gates["B1"] = not (embed_o and embed_c and embed_q)
    core = ("G0", "A1", "O1", "C1", "R1", "Q1", "M1", "M2")
    if all(gates[key] for key in core):
        verdict = (
            "LATENT_ARITHMETIC_CONTROLLER"
            if gates["B1"] else "LEXICAL_ARITHMETIC_REPLAY")
    elif gates["O1"] and not all(
            gates[key] for key in ("C1", "Q1", "M1", "M2")):
        verdict = "ARITHMETIC_OUTPUT_ONLY"
    elif gates["O1"]:
        verdict = "ARITHMETIC_OPERATOR_AMBIGUOUS"
    else:
        verdict = "ARITHMETIC_CONTROL_NULL"

    row_results = []
    for idx, row in enumerate(test_rows):
        row_results.append({
            **row,
            "add_answer": te_text["add"][idx],
            "subtract_answer": te_text["subtract"][idx],
            "natural_effect": float(natural_rows[idx]),
            "steered_effect": float(steered_rows[idx]),
            "embed_effect": float(embed_rows[idx]),
            "content_cosine": float(content["cosine_rows"][idx]),
            "content_error": float(content["error_rows"][idx]),
            "embed_content_cosine": float(
                embed_content["cosine_rows"][idx]),
            "embed_content_error": float(
                embed_content["error_rows"][idx]),
            "patch_steered_effect": float(patch_steered_rows[idx]),
            "patch_natural_effect": float(patch_natural_rows[idx]),
            "blocked_steered_effect": float(blocked_steered_rows[idx]),
            "blocked_natural_effect": float(blocked_natural_rows[idx]),
            "trajectory": {
                layer: {
                    "cosine": trajectory_rows[layer]["cosine"][idx],
                    "error": trajectory_rows[layer]["error"][idx],
                    "embed_cosine": embed_trajectory_rows[layer]["cosine"][idx],
                    "embed_error": embed_trajectory_rows[layer]["error"][idx],
                }
                for layer in trajectory_rows
            },
        })
    result = {
        "stage": "delta_arithmetic_controller",
        "model_path": model_path,
        "quantization": quantization,
        "seed": seed,
        "preflight": preflight,
        "train_rows": train_rows,
        "test_rows": row_results,
        "direction_norm": float(direction.norm()),
        "embedding_direction_norm": float(embed_direction.norm()),
        "behavior": {
            "clean_acc": clean_acc,
            "natural_acc": natural_acc,
            "steered_acc": steered_acc,
            "embed_acc": embed_acc,
            "reverse_acc": reverse_acc,
        },
        "predictions": {
            "source_answer_ids": source_ids.tolist(),
            "target_answer_ids": target_ids.tolist(),
            "clean_ids": clean_logits.argmax(-1).tolist(),
            "natural_ids": natural_logits.argmax(-1).tolist(),
            "steered_ids": steered_logits.argmax(-1).tolist(),
            "embedding_ids": embed_logits.argmax(-1).tolist(),
            "reverse_ids": reverse_logits.argmax(-1).tolist(),
        },
        "output": {
            "natural_effect": natural_effect,
            "steered_effect": steered_effect,
            "ratio": float(output_ratio),
            "positive_fraction": positive_fraction,
            "p": float(output_p),
            "null_exceedances": output_exceedances,
            "embed_effect": embed_effect,
            "embed_ratio": float(embed_ratio),
        },
        "content": {
            "cosine": content["cosine"],
            "error": content["error"],
            "embed_cosine": embed_content["cosine"],
            "embed_error": embed_content["error"],
        },
        "trajectory": trajectory,
        "embedding_trajectory": embed_trajectory,
        "mediation": {
            "patch_steered_effect": patch_steered,
            "patch_natural_effect": patch_natural,
            "patch_ratio": float(patch_ratio),
            "blocked_steered_effect": blocked_steered,
            "blocked_natural_effect": blocked_natural,
            "steered_block_fraction": float(steered_block_fraction),
            "natural_block_fraction": float(natural_block_fraction),
            "block_gap": float(block_gap),
        },
        "null": {
            "n": n_null,
            "output": null_output.tolist(),
            "embedding_output": null_embed_output.tolist(),
            "inject_cosine": null_inject_cos.tolist(),
            "inject_exceedances": inject_exceedances,
            "embedding_output_exceedances": int(
                (null_embed_output >= embed_effect).sum()),
            "inject_p": float(inject_p),
        },
        "embedding_baseline_gates": {
            "O1_like": embed_o, "C1": embed_c, "Q1": embed_q},
        "gates": gates,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_arithmetic_controller.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"behavior clean/natural/steered/embed/reverse={clean_acc:.0%}/"
        f"{natural_acc:.0%}/{steered_acc:.0%}/{embed_acc:.0%}/{reverse_acc:.0%}")
    log(f"output ratio={output_ratio:.3f} p={output_p:.4f}; "
        f"L20 cos={l20['cosine']:.3f} error={l20['error']:.3f}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
