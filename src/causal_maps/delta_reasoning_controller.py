"""Held-out-answer latent controller for two-step graph reasoning."""
import json
import os
import random

import numpy as np
import torch

from .delta_trajectory import _cos_rows, _error_rows, _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer, single_token_id
from .nulls import permutation_pvalue

INJECT_LAYER = 8
MEDIATION_LAYER = 20
LAYERS = (8, 12, 16, 20, 26)
LETTERS = tuple("ABCDEFGHIJ")
N_NULL = 100
EPS = 1e-8


def _continuation_token_id(tok, prompt, answer):
    base = tok.encode(prompt, add_special_tokens=False)
    full = tok.encode(prompt + answer, add_special_tokens=False)
    if full[:len(base)] != base or len(full) != len(base) + 1:
        raise ValueError(
            f"{answer!r} is not one stable continuation token after prompt")
    return full[-1]


def _graph_rows(seed, split):
    """Balanced endpoints with disjoint deterministic graph assignments."""
    rng = random.Random(seed + (0 if split == "train" else 10000))
    rows = []
    repeats = (0, 1) if split == "train" else (2, 3)
    for repeat in repeats:
        for target_idx, blue_end in enumerate(LETTERS):
            red_end = LETTERS[(target_idx + 3 + 2 * repeat) % len(LETTERS)]
            forbidden = {blue_end, red_end}
            available = [letter for letter in LETTERS if letter not in forbidden]
            rng.shuffle(available)
            start, red_mid, blue_mid = available[:3]
            rows.append({
                "start": start,
                "red_mid": red_mid,
                "blue_mid": blue_mid,
                "red_end": red_end,
                "blue_end": blue_end,
            })
    rng.shuffle(rows)
    return rows


def _prompt(tok, row, control):
    user = (
        "A colored edge points from one node to another. "
        f"Edges: {row['start']} -red-> {row['red_mid']}; "
        f"{row['start']} -blue-> {row['blue_mid']}; "
        f"{row['red_mid']} -green-> {row['red_end']}; "
        f"{row['blue_mid']} -green-> {row['blue_end']}. "
        f"Begin at {row['start']}. Follow {control}, then green. "
        "Which node is reached? Answer with one capital letter only."
    )
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def _encode_split(tok, rows, dev):
    red_texts = [_prompt(tok, row, "red") for row in rows]
    blue_texts = [_prompt(tok, row, "blue") for row in rows]
    red_encoded = [
        tok.encode(text, add_special_tokens=False) for text in red_texts]
    blue_encoded = [
        tok.encode(text, add_special_tokens=False) for text in blue_texts]
    lengths = {len(ids) for ids in red_encoded + blue_encoded}
    if len(lengths) != 1:
        raise ValueError(f"graph prompts have nonuniform lengths: {sorted(lengths)}")
    audits = []
    for red_ids, blue_ids in zip(red_encoded, blue_encoded):
        changed = [
            idx for idx, (red, blue) in enumerate(zip(red_ids, blue_ids))
            if red != blue]
        audits.append(changed)
        if len(red_ids) != len(blue_ids) or len(changed) != 1:
            raise ValueError(
                f"red/blue alignment failed: lengths "
                f"{len(red_ids)}/{len(blue_ids)}, changed={changed}")
    red_ids = torch.tensor(red_encoded, dtype=torch.long, device=dev)
    blue_ids = torch.tensor(blue_encoded, dtype=torch.long, device=dev)
    red_am = torch.ones_like(red_ids)
    blue_am = torch.ones_like(blue_ids)
    source_ids = torch.tensor([
        _continuation_token_id(tok, prompt, row["red_end"])
        for prompt, row in zip(red_texts, rows)])
    target_ids = torch.tensor([
        _continuation_token_id(tok, prompt, row["blue_end"])
        for prompt, row in zip(blue_texts, rows)])
    return red_ids, red_am, blue_ids, blue_am, source_ids, target_ids, audits


def _candidate_metrics(candidate, native):
    cos_rows = _cos_rows(candidate, native)
    error_rows = _error_rows(candidate, native)
    return {
        "cosine": float(cos_rows.mean()),
        "error": float(error_rows.mean()),
        "cosine_rows": cos_rows,
        "error_rows": error_rows,
    }


@torch.no_grad()
def run_delta_reasoning_controller(model_path, out_dir, quantization="8bit",
                                   device_map=None, seed=0, n_null=N_NULL):
    if model_path != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(f"frozen model mismatch: {model_path}")
    if quantization != "8bit":
        raise ValueError(f"frozen quantization mismatch: {quantization}")
    if seed != 0 or n_null != N_NULL:
        raise ValueError(
            f"frozen seed/null mismatch: seed={seed}, n_null={n_null}")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    train_rows = _graph_rows(seed, "train")
    test_rows = _graph_rows(seed, "test")
    train_data = _encode_split(tok, train_rows, dev)
    test_data = _encode_split(tok, test_rows, dev)
    (tr_red, tr_red_am, tr_blue, tr_blue_am,
     _tr_source, _tr_target, train_audits) = train_data
    (te_red, te_red_am, te_blue, te_blue_am,
     source_ids, target_ids, test_audits) = test_data
    last = int(te_red.shape[1] - 1)
    candidate_ids = torch.tensor([
        _continuation_token_id(
            tok, _prompt(tok, test_rows[0], "red"), letter)
        for letter in LETTERS])

    clean_logits, clean_cache = _forward(
        model, te_red, te_red_am, (last,), LAYERS)
    natural_logits, natural_cache = _forward(
        model, te_blue, te_blue_am, (last,), LAYERS)
    clean_acc = float(
        (clean_logits.argmax(-1) == source_ids).float().mean())
    natural_acc = float(
        (natural_logits.argmax(-1) == target_ids).float().mean())
    g0 = clean_acc >= 0.80 and natural_acc >= 0.80
    preflight = {
        "train_n": len(train_rows),
        "test_n": len(test_rows),
        "uniform_length": int(te_red.shape[1]),
        "train_changed_positions": train_audits,
        "test_changed_positions": test_audits,
        "clean_acc": clean_acc,
        "natural_acc": natural_acc,
        "pass": bool(g0),
    }
    if not g0:
        result = {
            "stage": "delta_reasoning_controller",
            "model_path": model_path,
            "preflight": preflight,
            "gates": {"G0": False},
            "verdict": "REASONING_INELICITABLE",
        }
        with open(os.path.join(
                out_dir, "results_delta_reasoning_controller.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        log(f"VERDICT: REASONING_INELICITABLE clean={clean_acc:.0%} "
            f"natural={natural_acc:.0%}")
        return result

    _, train_red_cache = _forward(
        model, tr_red, tr_red_am, (last,), (INJECT_LAYER,))
    _, train_blue_cache = _forward(
        model, tr_blue, tr_blue_am, (last,), (INJECT_LAYER,))
    direction = (
        train_blue_cache[INJECT_LAYER][:, 0]
        - train_red_cache[INJECT_LAYER][:, 0]).mean(0)
    batch_direction = direction.unsqueeze(0).expand(te_red.shape[0], -1)
    embedding = model.get_input_embeddings().weight
    red_id = single_token_id(tok, "red")
    blue_id = single_token_id(tok, "blue")
    embed_direction = (
        embedding[blue_id].detach().float().cpu()
        - embedding[red_id].detach().float().cpu())
    batch_embed = embed_direction.unsqueeze(0).expand(te_red.shape[0], -1)

    add_logits, add_cache = _forward(
        model, te_red, te_red_am, (last,), LAYERS,
        add=(INJECT_LAYER, last, batch_direction))
    embed_logits, embed_cache = _forward(
        model, te_red, te_red_am, (last,), LAYERS,
        add=(INJECT_LAYER, last, batch_embed))
    reverse_logits, _ = _forward(
        model, te_blue, te_blue_am, (last,), LAYERS,
        add=(INJECT_LAYER, last, -batch_direction))

    clean_ld = _ld(clean_logits, target_ids, source_ids)
    natural_rows = _ld(natural_logits, target_ids, source_ids) - clean_ld
    add_rows = _ld(add_logits, target_ids, source_ids) - clean_ld
    embed_rows = _ld(embed_logits, target_ids, source_ids) - clean_ld
    natural_effect = float(natural_rows.mean())
    add_effect = float(add_rows.mean())
    embed_effect = float(embed_rows.mean())
    output_ratio = add_effect / natural_effect if natural_effect > EPS else -np.inf
    embed_ratio = (
        embed_effect / natural_effect if natural_effect > EPS else -np.inf)
    add_acc = float((add_logits.argmax(-1) == target_ids).float().mean())
    embed_acc = float((embed_logits.argmax(-1) == target_ids).float().mean())
    reverse_acc = float(
        (reverse_logits.argmax(-1) == source_ids).float().mean())
    positive_fraction = float((add_rows > 0).float().mean())
    embed_positive_fraction = float((embed_rows > 0).float().mean())

    clean_candidates = clean_logits[:, candidate_ids]
    native_candidates = natural_logits[:, candidate_ids] - clean_candidates
    add_candidates = add_logits[:, candidate_ids] - clean_candidates
    embed_candidates = embed_logits[:, candidate_ids] - clean_candidates
    content = _candidate_metrics(add_candidates, native_candidates)
    embed_content = _candidate_metrics(embed_candidates, native_candidates)

    trajectory = {}
    embed_trajectory = {}
    for layer in LAYERS:
        native_disp = (
            natural_cache[layer][:, 0] - clean_cache[layer][:, 0])
        add_disp = add_cache[layer][:, 0] - clean_cache[layer][:, 0]
        embed_disp = embed_cache[layer][:, 0] - clean_cache[layer][:, 0]
        active_metric = _candidate_metrics(add_disp, native_disp)
        baseline_metric = _candidate_metrics(embed_disp, native_disp)
        trajectory[str(layer)] = {
            "cosine": active_metric["cosine"],
            "error": active_metric["error"],
        }
        embed_trajectory[str(layer)] = {
            "cosine": baseline_metric["cosine"],
            "error": baseline_metric["error"],
        }

    patch_add_logits, _ = _forward(
        model, te_red, te_red_am, (last,),
        patch=(MEDIATION_LAYER, last,
               add_cache[MEDIATION_LAYER][:, 0]))
    patch_natural_logits, _ = _forward(
        model, te_red, te_red_am, (last,),
        patch=(MEDIATION_LAYER, last,
               natural_cache[MEDIATION_LAYER][:, 0]))
    block_add_logits, _ = _forward(
        model, te_red, te_red_am, (last,),
        add=(INJECT_LAYER, last, batch_direction),
        patch=(MEDIATION_LAYER, last,
               clean_cache[MEDIATION_LAYER][:, 0]))
    block_natural_logits, _ = _forward(
        model, te_blue, te_blue_am, (last,),
        patch=(MEDIATION_LAYER, last,
               clean_cache[MEDIATION_LAYER][:, 0]))
    patch_add_rows = (
        _ld(patch_add_logits, target_ids, source_ids) - clean_ld)
    patch_natural_rows = (
        _ld(patch_natural_logits, target_ids, source_ids) - clean_ld)
    blocked_add_rows = (
        _ld(block_add_logits, target_ids, source_ids) - clean_ld)
    blocked_natural_rows = (
        _ld(block_natural_logits, target_ids, source_ids) - clean_ld)
    patch_add = float(patch_add_rows.mean())
    patch_natural = float(patch_natural_rows.mean())
    patch_ratio = (
        patch_add / patch_natural if patch_natural > EPS else -np.inf)
    blocked_add = float(blocked_add_rows.mean())
    blocked_natural = float(blocked_natural_rows.mean())
    add_block_fraction = (
        (add_effect - blocked_add) / add_effect if add_effect > EPS else -np.inf)
    natural_block_fraction = (
        (natural_effect - blocked_natural) / natural_effect
        if natural_effect > EPS else -np.inf)
    block_gap = abs(add_block_fraction - natural_block_fraction)

    generator = torch.Generator(device="cpu").manual_seed(seed + 701)
    null_output = np.zeros(n_null, dtype=np.float64)
    null_inject_cos = np.zeros(n_null, dtype=np.float64)
    native_inject = (
        natural_cache[INJECT_LAYER][:, 0]
        - clean_cache[INJECT_LAYER][:, 0])
    norm = direction.norm().clamp(min=EPS)
    for idx in range(n_null):
        random_direction = torch.randn(
            direction.shape, generator=generator, dtype=torch.float32)
        random_direction = random_direction / random_direction.norm() * norm
        random_batch = random_direction.unsqueeze(0).expand(te_red.shape[0], -1)
        random_logits, random_cache = _forward(
            model, te_red, te_red_am, (last,), (INJECT_LAYER,),
            add=(INJECT_LAYER, last, random_batch))
        random_rows = (
            _ld(random_logits, target_ids, source_ids) - clean_ld)
        random_disp = (
            random_cache[INJECT_LAYER][:, 0]
            - clean_cache[INJECT_LAYER][:, 0])
        null_output[idx] = float(random_rows.mean())
        null_inject_cos[idx] = float(
            _cos_rows(random_disp, native_inject).mean())

    inject = trajectory[str(INJECT_LAYER)]
    l20 = trajectory[str(MEDIATION_LAYER)]
    output_p = permutation_pvalue(add_effect, null_output, "greater")
    inject_p = permutation_pvalue(
        inject["cosine"], null_inject_cos, "greater")
    output_null_exceedances = int((null_output >= add_effect).sum())
    inject_null_exceedances = int(
        (null_inject_cos >= inject["cosine"]).sum())
    gates = {
        "G0": True,
        "A1": bool(inject["cosine"] >= 0.50
                   and inject["error"] <= 0.80
                   and inject_null_exceedances <= 1),
        "O1": bool(add_acc >= 0.80 and positive_fraction >= 0.80
                   and 0.70 <= output_ratio <= 1.30
                   and output_null_exceedances <= 1),
        "C1": bool(content["cosine"] >= 0.80 and content["error"] <= 0.60),
        "R1": bool(reverse_acc >= 0.80),
        "Q1": bool(l20["cosine"] >= 0.80 and l20["error"] <= 0.60),
        "M1": bool(patch_add > 0 and patch_natural > 0
                   and 0.70 <= patch_ratio <= 1.30),
        "M2": bool(add_block_fraction >= 0.70
                   and natural_block_fraction >= 0.70 and block_gap <= 0.20),
    }
    embed_o = bool(
        embed_acc >= 0.80 and embed_positive_fraction >= 0.80
        and 0.70 <= embed_ratio <= 1.30
        and int((null_output >= embed_effect).sum()) <= 1)
    embed_c = bool(
        embed_content["cosine"] >= 0.80 and embed_content["error"] <= 0.60)
    embed_q = bool(
        embed_trajectory[str(MEDIATION_LAYER)]["cosine"] >= 0.80
        and embed_trajectory[str(MEDIATION_LAYER)]["error"] <= 0.60)
    gates["B1"] = not (embed_o and embed_c and embed_q)
    core = ("G0", "A1", "O1", "C1", "R1", "Q1", "M1", "M2")
    if all(gates[key] for key in core):
        verdict = (
            "LATENT_REASONING_CONTROLLER"
            if gates["B1"] else "LEXICAL_REPLAY_EQUIVALENT")
    elif gates["O1"] and not all(
            gates[key] for key in ("C1", "Q1", "M1", "M2")):
        verdict = "REASONING_OUTPUT_ONLY"
    elif gates["O1"]:
        verdict = "REASONING_OPERATOR_AMBIGUOUS"
    else:
        verdict = "REASONING_CONTROL_NULL"

    rows = []
    for idx, row in enumerate(test_rows):
        rows.append({
            **row,
            "natural_effect": float(natural_rows[idx]),
            "add_effect": float(add_rows[idx]),
            "embed_effect": float(embed_rows[idx]),
            "content_cosine": float(content["cosine_rows"][idx]),
            "content_error": float(content["error_rows"][idx]),
            "patch_add_effect": float(patch_add_rows[idx]),
            "patch_natural_effect": float(patch_natural_rows[idx]),
            "blocked_add_effect": float(blocked_add_rows[idx]),
            "blocked_natural_effect": float(blocked_natural_rows[idx]),
        })
    result = {
        "stage": "delta_reasoning_controller",
        "model_path": model_path,
        "quantization": quantization,
        "seed": seed,
        "preflight": preflight,
        "train_rows": train_rows,
        "test_rows": rows,
        "direction_norm": float(direction.norm()),
        "embedding_direction_norm": float(embed_direction.norm()),
        "behavior": {
            "clean_acc": clean_acc,
            "natural_acc": natural_acc,
            "add_acc": add_acc,
            "embed_acc": embed_acc,
            "reverse_acc": reverse_acc,
        },
        "output": {
            "natural_effect": natural_effect,
            "add_effect": add_effect,
            "output_ratio": float(output_ratio),
            "positive_fraction": positive_fraction,
            "p": float(output_p),
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
            "patch_add_effect": patch_add,
            "patch_natural_effect": patch_natural,
            "patch_ratio": float(patch_ratio),
            "blocked_add_effect": blocked_add,
            "blocked_natural_effect": blocked_natural,
            "add_block_fraction": float(add_block_fraction),
            "natural_block_fraction": float(natural_block_fraction),
            "block_gap": float(block_gap),
        },
        "null": {
            "n": n_null,
            "output": null_output.tolist(),
            "inject_cosine": null_inject_cos.tolist(),
            "output_exceedances": output_null_exceedances,
            "inject_exceedances": inject_null_exceedances,
            "inject_p": float(inject_p),
        },
        "embedding_baseline_gates": {
            "O1_like": embed_o, "C1": embed_c, "Q1": embed_q},
        "gates": gates,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_reasoning_controller.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"behavior clean/natural/add/embed/reverse={clean_acc:.0%}/"
        f"{natural_acc:.0%}/{add_acc:.0%}/{embed_acc:.0%}/{reverse_acc:.0%}")
    log(f"output ratio={output_ratio:.3f} p={output_p:.4f}; "
        f"L20 cos={l20['cosine']:.3f} error={l20['error']:.3f}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
