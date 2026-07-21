"""Target-specific multiclass readout audit for delta_operator."""
import json
import os

import numpy as np
import torch

from .delta_operator import (_build_multi_group, _directions, _encode_uniform,
                             _single_text, _trials, _values, DONOR_NAMES,
                             INJECT_LAYER)
from .delta_trajectory import _forward
from .logutil import log
from .model_utils import (input_device, load_model_and_tokenizer,
                          single_token_id)

EPS = 1e-8


def _margins(pool_logits, intended_idx):
    rows = torch.arange(pool_logits.shape[0])
    intended = pool_logits[rows, intended_idx]
    masked = pool_logits.clone()
    masked[rows, intended_idx] = float("-inf")
    return intended - masked.max(dim=1).values


@torch.no_grad()
def run_delta_operator_readout(model_path, out_dir, quantization="8bit",
                               device_map=None, seed=0):
    del seed
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    values = _values(tok)
    value_idx = {value: i for i, value in enumerate(values)}
    value_ids = torch.tensor([single_token_id(tok, value) for value in values])
    rows = _trials(values)

    donor_rows, donor_values = [], []
    for name in DONOR_NAMES:
        for value in values:
            donor_rows.append(_single_text(tok, name, value))
            donor_values.append(value)
    donor_ids, donor_am, donor_pos = _encode_uniform(tok, donor_rows)
    _, donor_cache = _forward(
        model, donor_ids.to(dev), donor_am.to(dev), (donor_pos,),
        (INJECT_LAYER,))
    donor_h = donor_cache[INJECT_LAYER][:, 0]
    prototypes = {
        value: donor_h[[i for i, observed in enumerate(donor_values)
                        if observed == value]].mean(0)
        for value in values
    }

    records = []
    correct_target_hits, wrong_intended_hits, wrong_original_hits = [], [], []
    natural_margins, correct_margins = [], []
    wrong_intended_margins, wrong_original_margins = [], []

    for query in ("X", "Y"):
        group, ci, cam, fi, fam, own_pos, _ = _build_multi_group(
            tok, rows, query)
        ci, cam, fi, fam = ci.to(dev), cam.to(dev), fi.to(dev), fam.to(dev)
        last = int(ci.shape[1] - 1)
        correct = _directions(prototypes, group)
        wrong = _directions(prototypes, group, key="wrong")
        target_ids = torch.tensor(
            [single_token_id(tok, row["target"]) for row in group])
        wrong_ids = torch.tensor(
            [single_token_id(tok, row["wrong"]) for row in group])
        target_idx = torch.tensor([value_idx[row["target"]] for row in group])
        wrong_idx = torch.tensor([value_idx[row["wrong"]] for row in group])

        clean_logits, _ = _forward(model, ci, cam, (own_pos, last))
        natural_logits, _ = _forward(model, fi, fam, (own_pos, last))
        correct_logits, _ = _forward(
            model, ci, cam, (own_pos, last),
            add=(INJECT_LAYER, own_pos, correct))
        wrong_logits, _ = _forward(
            model, ci, cam, (own_pos, last),
            add=(INJECT_LAYER, own_pos, wrong))

        natural_pool = natural_logits[:, value_ids]
        correct_pool = correct_logits[:, value_ids]
        wrong_pool = wrong_logits[:, value_ids]
        natural_margin = _margins(natural_pool, target_idx)
        correct_margin = _margins(correct_pool, target_idx)
        wrong_intended_margin = _margins(wrong_pool, wrong_idx)
        wrong_original_margin = _margins(wrong_pool, target_idx)

        correct_hit = correct_logits.argmax(-1) == target_ids
        wrong_intended_hit = wrong_logits.argmax(-1) == wrong_ids
        wrong_original_hit = wrong_logits.argmax(-1) == target_ids
        correct_target_hits.extend(correct_hit.tolist())
        wrong_intended_hits.extend(wrong_intended_hit.tolist())
        wrong_original_hits.extend(wrong_original_hit.tolist())
        natural_margins.extend(natural_margin.tolist())
        correct_margins.extend(correct_margin.tolist())
        wrong_intended_margins.extend(wrong_intended_margin.tolist())
        wrong_original_margins.extend(wrong_original_margin.tolist())

        for i, row in enumerate(group):
            records.append({
                **row,
                "clean_greedy_id": int(clean_logits.argmax(-1)[i]),
                "natural_greedy_id": int(natural_logits.argmax(-1)[i]),
                "correct_greedy_id": int(correct_logits.argmax(-1)[i]),
                "wrong_greedy_id": int(wrong_logits.argmax(-1)[i]),
                "correct_target_hit": bool(correct_hit[i]),
                "wrong_intended_hit": bool(wrong_intended_hit[i]),
                "wrong_original_target_hit": bool(wrong_original_hit[i]),
                "natural_target_margin": float(natural_margin[i]),
                "correct_target_margin": float(correct_margin[i]),
                "wrong_intended_margin": float(wrong_intended_margin[i]),
                "wrong_original_target_margin": float(
                    wrong_original_margin[i]),
            })

    correct_acc = float(np.mean(correct_target_hits))
    wrong_acc = float(np.mean(wrong_intended_hits))
    wrong_original_acc = float(np.mean(wrong_original_hits))
    natural_margin = float(np.mean(natural_margins))
    correct_margin = float(np.mean(correct_margins))
    wrong_intended_margin = float(np.mean(wrong_intended_margins))
    wrong_original_margin = float(np.mean(wrong_original_margins))
    margin_ratio = correct_margin / max(natural_margin, EPS)
    gates = {
        "T1": bool(correct_acc >= 0.80 and margin_ratio >= 0.70),
        "T2": bool(wrong_acc >= 0.80 and wrong_original_acc <= 0.20
                   and wrong_intended_margin > 0),
        "T3": bool(correct_margin > wrong_original_margin),
    }
    verdict = ("TARGET_SPECIFIC_OPERATOR" if all(gates.values())
               else "NONSPECIFIC_REPLACEMENT")
    result = {
        "stage": "delta_operator_readout",
        "model_path": model_path,
        "n_trials": len(rows),
        "values": values,
        "metrics": {
            "correct_target_greedy_acc": correct_acc,
            "wrong_intended_greedy_acc": wrong_acc,
            "wrong_original_target_greedy_acc": wrong_original_acc,
            "natural_target_margin": natural_margin,
            "correct_target_margin": correct_margin,
            "correct_over_natural_margin": float(margin_ratio),
            "wrong_intended_margin": wrong_intended_margin,
            "wrong_original_target_margin": wrong_original_margin,
        },
        "rows": records,
        "gates": gates,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_operator_readout.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"metrics={result['metrics']}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
