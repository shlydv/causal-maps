"""Multiclass content-specificity control for affine counterfactual operators."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .delta_operator import (_build_multi_group, _directions, _encode_uniform,
                             _single_text, _trials, _values, DONOR_NAMES,
                             INJECT_LAYER, EPS)
from .delta_trajectory import _forward
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer, single_token_id


def _centered_metrics(induced, natural):
    induced = induced.float() - induced.float().mean(dim=1, keepdim=True)
    natural = natural.float() - natural.float().mean(dim=1, keepdim=True)
    cosine = F.cosine_similarity(induced, natural, dim=1, eps=EPS)
    error = ((induced - natural).norm(dim=1)
             / natural.norm(dim=1).clamp(min=EPS))
    return cosine, error


def _js_rows(logits_a, logits_b):
    pa = F.softmax(logits_a.float(), dim=1).clamp(min=EPS)
    pb = F.softmax(logits_b.float(), dim=1).clamp(min=EPS)
    mean = 0.5 * (pa + pb)
    return 0.5 * (
        (pa * (pa.log() - mean.log())).sum(dim=1)
        + (pb * (pb.log() - mean.log())).sum(dim=1))


@torch.no_grad()
def run_delta_operator_content(model_path, out_dir, quantization="8bit",
                               device_map=None, seed=0):
    del seed
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    values = _values(tok)
    rows = _trials(values)
    value_index = {value: i for i, value in enumerate(values)}
    candidate_ids = torch.tensor([single_token_id(tok, value)
                                  for value in values])

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
    for query in ("X", "Y"):
        group, ci, cam, fi, fam, own_pos, _ = _build_multi_group(
            tok, rows, query)
        wrong_design = [{**row, "target": row["wrong"]} for row in rows]
        wrong_group, wci, wcam, wfi, wfam, wrong_pos, _ = _build_multi_group(
            tok, wrong_design, query)
        if own_pos != wrong_pos or not torch.equal(ci, wci):
            raise ValueError("wrong natural-control clean prompts misaligned")
        ci, cam = ci.to(dev), cam.to(dev)
        fi, fam = fi.to(dev), fam.to(dev)
        wfi, wfam = wfi.to(dev), wfam.to(dev)
        last = int(ci.shape[1] - 1)
        target_delta = _directions(prototypes, group)
        wrong_delta = _directions(prototypes, group, key="wrong")

        clean_logits, _ = _forward(model, ci, cam, (own_pos, last))
        target_nat_logits, _ = _forward(model, fi, fam, (own_pos, last))
        wrong_nat_logits, _ = _forward(model, wfi, wfam, (own_pos, last))
        target_add_logits, _ = _forward(
            model, ci, cam, (own_pos, last),
            add=(INJECT_LAYER, own_pos, target_delta))
        wrong_add_logits, _ = _forward(
            model, ci, cam, (own_pos, last),
            add=(INJECT_LAYER, own_pos, wrong_delta))

        clean_c = clean_logits[:, candidate_ids]
        target_nat_c = target_nat_logits[:, candidate_ids]
        wrong_nat_c = wrong_nat_logits[:, candidate_ids]
        target_add_c = target_add_logits[:, candidate_ids]
        wrong_add_c = wrong_add_logits[:, candidate_ids]
        target_cos, target_error = _centered_metrics(
            target_add_c - clean_c, target_nat_c - clean_c)
        wrong_cos, wrong_error = _centered_metrics(
            wrong_add_c - clean_c, wrong_nat_c - clean_c)
        target_js = _js_rows(target_add_c, target_nat_c)
        wrong_js = _js_rows(wrong_add_c, wrong_nat_c)

        for i, row in enumerate(group):
            source_idx = value_index[row["source"]]
            target_idx = value_index[row["target"]]
            wrong_idx = value_index[row["wrong"]]
            target_id = single_token_id(tok, row["target"])
            wrong_id = single_token_id(tok, row["wrong"])
            rec = {
                **row,
                "natural_target_hit": bool(
                    target_nat_logits.argmax(-1)[i] == target_id),
                "add_target_hit": bool(
                    target_add_logits.argmax(-1)[i] == target_id),
                "natural_wrong_hit": bool(
                    wrong_nat_logits.argmax(-1)[i] == wrong_id),
                "add_wrong_hit": bool(
                    wrong_add_logits.argmax(-1)[i] == wrong_id),
                "target_logit_cos": float(target_cos[i]),
                "target_logit_error": float(target_error[i]),
                "target_js": float(target_js[i]),
                "wrong_logit_cos": float(wrong_cos[i]),
                "wrong_logit_error": float(wrong_error[i]),
                "wrong_js": float(wrong_js[i]),
                "target_intended_gain": float(
                    target_add_c[i, target_idx] - clean_c[i, target_idx]),
                "target_source_suppression": float(
                    clean_c[i, source_idx] - target_add_c[i, source_idx]),
                "wrong_effect_on_target": float(
                    wrong_add_c[i, target_idx] - clean_c[i, target_idx]),
                "wrong_intended_gain": float(
                    wrong_add_c[i, wrong_idx] - clean_c[i, wrong_idx]),
                "wrong_source_suppression": float(
                    clean_c[i, source_idx] - wrong_add_c[i, source_idx]),
                "target_discriminates": bool(
                    target_add_c[i, target_idx]
                    > torch.maximum(target_add_c[i, source_idx],
                                    target_add_c[i, wrong_idx])),
                "wrong_discriminates": bool(
                    wrong_add_c[i, wrong_idx]
                    > torch.maximum(wrong_add_c[i, source_idx],
                                    wrong_add_c[i, target_idx])),
            }
            records.append(rec)

    def mean(key):
        return float(np.mean([row[key] for row in records]))

    summary = {
        "natural_target_acc": mean("natural_target_hit"),
        "natural_wrong_acc": mean("natural_wrong_hit"),
        "add_target_acc": mean("add_target_hit"),
        "add_wrong_acc": mean("add_wrong_hit"),
        "target_logit_cos": mean("target_logit_cos"),
        "target_logit_error": mean("target_logit_error"),
        "target_js": mean("target_js"),
        "wrong_logit_cos": mean("wrong_logit_cos"),
        "wrong_logit_error": mean("wrong_logit_error"),
        "wrong_js": mean("wrong_js"),
        "target_discrimination": mean("target_discriminates"),
        "wrong_discrimination": mean("wrong_discriminates"),
        "target_intended_gain": mean("target_intended_gain"),
        "target_source_suppression": mean("target_source_suppression"),
        "wrong_effect_on_target": mean("wrong_effect_on_target"),
        "wrong_intended_gain": mean("wrong_intended_gain"),
        "wrong_source_suppression": mean("wrong_source_suppression"),
    }
    gates = {
        "G0": bool(summary["natural_target_acc"] >= 0.90
                   and summary["natural_wrong_acc"] >= 0.90),
        "C1": bool(summary["add_target_acc"] >= 0.90
                   and summary["add_wrong_acc"] >= 0.90),
        "C2": bool(summary["target_logit_cos"] >= 0.95
                   and summary["target_logit_error"] <= 0.25
                   and summary["wrong_logit_cos"] >= 0.95
                   and summary["wrong_logit_error"] <= 0.25),
        "C3": bool(summary["target_discrimination"] >= 0.90
                   and summary["wrong_discrimination"] >= 0.90),
    }
    verdict = ("CONTENT_SPECIFIC_COUNTERFACTUAL_OPERATOR"
               if all(gates.values()) else "OUTPUT_EQUIVALENCE_FAILED")
    result = {
        "stage": "delta_operator_content",
        "model_path": model_path,
        "values": values,
        "n_trials": len(records),
        "summary": summary,
        "gates": gates,
        "rows": records,
        "verdict": verdict,
    }
    with open(os.path.join(
            out_dir, "results_delta_operator_content.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"summary={summary}")
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
