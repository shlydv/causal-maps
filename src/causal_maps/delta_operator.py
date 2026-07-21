"""Affine counterfactual-operator validity gate.

Frozen design: COUNTERFACTUAL_OPERATOR_PROTOCOL.md.
"""
import json
import os

import numpy as np
import torch

from . import variable_pairs
from .delta_multislot import _two_var_text
from .delta_trajectory import (_cos_rows, _error_rows, _forward, _ld,
                               _random_like_rows)
from .logutil import log
from .model_utils import (input_device, load_model_and_tokenizer,
                          single_token_id)
from .nulls import permutation_pvalue
from .tensorize import _anchor_token_index

INJECT_LAYER = 2
MEDIATION_LAYER = 8
N_NULL = 100
EPS = 1e-8
DONOR_NAMES = ("X", "Y", "Z", "W")
HELDOUT_NAME = "K"
OFFSETS = (1, 3, 5, 7)


def _single_text(tok, name, value):
    user = f"Let {name} = {value}. What is the value of {name}?"
    text = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True) + f"{name} ="
    marker = f"Let {name} = "
    offset = text.find(marker) + len(marker)
    return text, offset


def _encode_uniform(tok, rows):
    """rows are (text, char_offset); return ids, mask, uniform token position."""
    encoded, positions = [], []
    for text, offset in rows:
        encoded.append(tok.encode(text, add_special_tokens=False))
        positions.append(_anchor_token_index(tok, text, offset))
    if None in positions:
        raise ValueError(f"unstable anchor positions: {positions}")
    lengths = {len(row) for row in encoded}
    if len(lengths) != 1:
        raise ValueError(f"non-uniform token lengths: {sorted(lengths)}")
    if len(set(positions)) != 1:
        raise ValueError(f"non-uniform token positions: {sorted(set(positions))}")
    ids = torch.tensor(encoded, dtype=torch.long)
    return ids, torch.ones_like(ids), int(positions[0])


def _values(tok):
    values = []
    for _, value in variable_pairs._VALUE_PAIRS:
        single_token_id(tok, value)
        values.append(value)
    return values


def _trials(values):
    rows = []
    n = len(values)
    for source_idx, source in enumerate(values):
        for j, offset in enumerate(OFFSETS):
            target_idx = (source_idx + offset) % n
            target = values[target_idx]
            wrong_idx = (target_idx + 2) % n
            while wrong_idx in (source_idx, target_idx):
                wrong_idx = (wrong_idx + 1) % n
            distractor_idx = (source_idx + offset + 4) % n
            while distractor_idx in (source_idx, target_idx, wrong_idx):
                distractor_idx = (distractor_idx + 1) % n
            rows.append({
                "source": source,
                "target": target,
                "wrong": values[wrong_idx],
                "distractor": values[distractor_idx],
                "query": "X" if j % 2 == 0 else "Y",
            })
    return rows


def _directions(prototypes, rows, key="target"):
    return torch.stack([
        prototypes[row[key]] - prototypes[row["source"]] for row in rows
    ])


def _metrics(clean_cache, cf_cache, add_cache):
    result = {}
    for layer in (INJECT_LAYER, MEDIATION_LAYER):
        for site_idx, site in enumerate(("slot", "last")):
            native = cf_cache[layer][:, site_idx] - clean_cache[layer][:, site_idx]
            induced = add_cache[layer][:, site_idx] - clean_cache[layer][:, site_idx]
            result[f"L{layer}_{site}"] = {
                "cos_rows": _cos_rows(induced, native),
                "error_rows": _error_rows(induced, native),
                "norm_ratio_rows": (
                    add_cache[layer][:, site_idx].norm(dim=1)
                    / cf_cache[layer][:, site_idx].norm(dim=1).clamp(min=EPS)
                ),
            }
    return result


def _extend_metric_store(store, metrics):
    for key, row in metrics.items():
        target = store.setdefault(
            key, {"cos": [], "error": [], "norm_ratio": []})
        target["cos"].extend(row["cos_rows"].tolist())
        target["error"].extend(row["error_rows"].tolist())
        target["norm_ratio"].extend(row["norm_ratio_rows"].tolist())


def _summarize_metrics(store):
    return {
        key: {metric: float(np.mean(values))
              for metric, values in row.items()}
        for key, row in store.items()
    }


def _build_multi_group(tok, rows, query):
    selected = [row for row in rows if row["query"] == query]
    clean_rows, cf_rows = [], []
    x_positions, y_positions = [], []
    for row in selected:
        if query == "X":
            clean_args = (row["source"], row["distractor"])
            cf_args = (row["target"], row["distractor"])
        else:
            clean_args = (row["distractor"], row["source"])
            cf_args = (row["distractor"], row["target"])
        clean, clean_x, clean_y = _two_var_text(
            tok, "X", "Y", *clean_args, query)
        cf, cf_x, cf_y = _two_var_text(tok, "X", "Y", *cf_args, query)
        clean_rows.append((clean, clean_x if query == "X" else clean_y))
        cf_rows.append((cf, cf_x if query == "X" else cf_y))
        x_positions.append(_anchor_token_index(tok, clean, clean_x))
        y_positions.append(_anchor_token_index(tok, clean, clean_y))
    ci, cam, own_pos = _encode_uniform(tok, clean_rows)
    fi, fam, cf_own_pos = _encode_uniform(tok, cf_rows)
    if ci.shape != fi.shape or own_pos != cf_own_pos:
        raise ValueError("clean/cf multi-binding alignment failed")
    if len(set(x_positions)) != 1 or len(set(y_positions)) != 1:
        raise ValueError("multi-binding slot positions are not uniform")
    other_pos = y_positions[0] if query == "X" else x_positions[0]
    return selected, ci, cam, fi, fam, own_pos, int(other_pos)


@torch.no_grad()
def _secondary_single(model, tok, prototypes, rows, dev):
    clean_rows, cf_rows = [], []
    for row in rows:
        clean_rows.append(_single_text(tok, HELDOUT_NAME, row["source"]))
        cf_rows.append(_single_text(tok, HELDOUT_NAME, row["target"]))
    ci, cam, pos = _encode_uniform(tok, clean_rows)
    fi, fam, cf_pos = _encode_uniform(tok, cf_rows)
    if ci.shape != fi.shape or pos != cf_pos:
        raise ValueError("single-variable clean/cf alignment failed")
    ci, cam, fi, fam = ci.to(dev), cam.to(dev), fi.to(dev), fam.to(dev)
    last = int(ci.shape[1] - 1)
    directions = _directions(prototypes, rows)
    pos_ids = torch.tensor(
        [single_token_id(tok, row["target"]) for row in rows])
    neg_ids = torch.tensor(
        [single_token_id(tok, row["source"]) for row in rows])
    clean_logits, clean_cache = _forward(
        model, ci, cam, (pos, last), (INJECT_LAYER, MEDIATION_LAYER))
    cf_logits, cf_cache = _forward(
        model, fi, fam, (pos, last), (INJECT_LAYER, MEDIATION_LAYER))
    add_logits, add_cache = _forward(
        model, ci, cam, (pos, last), (INJECT_LAYER, MEDIATION_LAYER),
        add=(INJECT_LAYER, pos, directions))
    clean_ld = _ld(clean_logits, pos_ids, neg_ids)
    natural_effect = _ld(cf_logits, pos_ids, neg_ids) - clean_ld
    add_effect = _ld(add_logits, pos_ids, neg_ids) - clean_ld
    metrics = _metrics(clean_cache, cf_cache, add_cache)
    ratio = float(add_effect.mean() / natural_effect.mean().clamp(min=EPS))
    summary = {
        "l2_cos": float(metrics["L2_slot"]["cos_rows"].mean()),
        "l2_error": float(metrics["L2_slot"]["error_rows"].mean()),
        "l8_last_cos": float(metrics["L8_last"]["cos_rows"].mean()),
        "output_effect": float(add_effect.mean()),
        "natural_effect": float(natural_effect.mean()),
        "effect_ratio": ratio,
    }
    summary["S1"] = bool(
        summary["l2_cos"] >= 0.80
        and summary["l2_error"] <= 0.60
        and summary["l8_last_cos"] >= 0.50
        and summary["effect_ratio"] >= 0.70)
    return summary


@torch.no_grad()
def run_delta_operator(model_path, out_dir, quantization="8bit",
                       device_map=None, seed=0, n_null=N_NULL):
    if n_null < 100:
        raise ValueError("counterfactual-operator p<.01 gate requires >=100 nulls")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    values = _values(tok)
    rows = _trials(values)
    generator = torch.Generator().manual_seed(seed + 3407)

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
    log(f"delta_operator: donors={len(donor_rows)} trials={len(rows)} "
        f"n_null={n_null} donor_pos={donor_pos}")

    # Absolute behavioral gate before any operator intervention.
    pre_clean_hits, pre_cf_hits = [], []
    for query in ("X", "Y"):
        group, ci, cam, fi, fam, own_pos, _ = _build_multi_group(
            tok, rows, query)
        ci, cam, fi, fam = ci.to(dev), cam.to(dev), fi.to(dev), fam.to(dev)
        last = int(ci.shape[1] - 1)
        pos_ids = torch.tensor(
            [single_token_id(tok, row["target"]) for row in group])
        neg_ids = torch.tensor(
            [single_token_id(tok, row["source"]) for row in group])
        clean_logits, _ = _forward(model, ci, cam, (own_pos, last))
        cf_logits, _ = _forward(model, fi, fam, (own_pos, last))
        pre_clean_hits.extend((clean_logits.argmax(-1) == neg_ids).tolist())
        pre_cf_hits.extend((cf_logits.argmax(-1) == pos_ids).tolist())
    pre_clean_acc = float(np.mean(pre_clean_hits))
    pre_cf_acc = float(np.mean(pre_cf_hits))
    if pre_clean_acc < 0.80 or pre_cf_acc < 0.80:
        result = {
            "stage": "delta_operator",
            "model_path": model_path,
            "values": values,
            "n_trials": len(rows),
            "behavior": {
                "clean_acc": pre_clean_acc, "cf_acc": pre_cf_acc},
            "gates": {"G0": False},
            "verdict": "OPERATOR_INELICITABLE",
        }
        with open(os.path.join(out_dir, "results_delta_operator.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        log(f"G0 failed clean={pre_clean_acc:.0%} cf={pre_cf_acc:.0%}")
        log("VERDICT: OPERATOR_INELICITABLE")
        return result

    secondary = _secondary_single(model, tok, prototypes, rows, dev)
    log(f"secondary K: {secondary}")

    store = {}
    add_effect_rows, natural_effect_rows = [], []
    wrong_effect_rows, other_effect_rows = [], []
    wrong_cos_rows = []
    patch_add_rows, patch_cf_rows = [], []
    blocked_rows, natural_blocked_rows = [], []
    clean_hits, cf_hits = [], []
    row_records = []
    null_output = np.zeros(n_null, dtype=np.float64)
    null_last_cos = np.zeros(n_null, dtype=np.float64)

    for query in ("X", "Y"):
        group, ci, cam, fi, fam, own_pos, other_pos = _build_multi_group(
            tok, rows, query)
        ci, cam, fi, fam = ci.to(dev), cam.to(dev), fi.to(dev), fam.to(dev)
        last = int(ci.shape[1] - 1)
        directions = _directions(prototypes, group)
        wrong = _directions(prototypes, group, key="wrong")
        pos_ids = torch.tensor(
            [single_token_id(tok, row["target"]) for row in group])
        neg_ids = torch.tensor(
            [single_token_id(tok, row["source"]) for row in group])

        clean_logits, clean_cache = _forward(
            model, ci, cam, (own_pos, last), (INJECT_LAYER, MEDIATION_LAYER))
        cf_logits, cf_cache = _forward(
            model, fi, fam, (own_pos, last), (INJECT_LAYER, MEDIATION_LAYER))
        clean_ld = _ld(clean_logits, pos_ids, neg_ids)
        natural_effect = _ld(cf_logits, pos_ids, neg_ids) - clean_ld
        clean_hits.extend((clean_logits.argmax(-1) == neg_ids).tolist())
        cf_hits.extend((cf_logits.argmax(-1) == pos_ids).tolist())

        add_logits, add_cache = _forward(
            model, ci, cam, (own_pos, last), (INJECT_LAYER, MEDIATION_LAYER),
            add=(INJECT_LAYER, own_pos, directions))
        add_effect = _ld(add_logits, pos_ids, neg_ids) - clean_ld
        add_effect_rows.extend(add_effect.tolist())
        natural_effect_rows.extend(natural_effect.tolist())
        group_metrics = _metrics(clean_cache, cf_cache, add_cache)
        _extend_metric_store(store, group_metrics)

        wrong_logits, wrong_cache = _forward(
            model, ci, cam, (own_pos, last), (INJECT_LAYER,),
            add=(INJECT_LAYER, own_pos, wrong))
        wrong_effect = _ld(wrong_logits, pos_ids, neg_ids) - clean_ld
        wrong_effect_rows.extend(wrong_effect.tolist())
        native_l2 = (cf_cache[INJECT_LAYER][:, 0]
                     - clean_cache[INJECT_LAYER][:, 0])
        wrong_l2 = (wrong_cache[INJECT_LAYER][:, 0]
                    - clean_cache[INJECT_LAYER][:, 0])
        wrong_cos_group = _cos_rows(wrong_l2, native_l2)
        wrong_cos_rows.extend(wrong_cos_group.tolist())

        other_logits, _ = _forward(
            model, ci, cam, (own_pos, last),
            add=(INJECT_LAYER, other_pos, directions))
        other_effect_group = _ld(other_logits, pos_ids, neg_ids) - clean_ld
        other_effect_rows.extend(other_effect_group.tolist())

        patch_add_logits, _ = _forward(
            model, ci, cam, (own_pos, last),
            patch=(MEDIATION_LAYER, own_pos,
                   add_cache[MEDIATION_LAYER][:, 0]))
        patch_cf_logits, _ = _forward(
            model, ci, cam, (own_pos, last),
            patch=(MEDIATION_LAYER, own_pos,
                   cf_cache[MEDIATION_LAYER][:, 0]))
        blocked_logits, _ = _forward(
            model, ci, cam, (own_pos, last),
            add=(INJECT_LAYER, own_pos, directions),
            patch=(MEDIATION_LAYER, own_pos,
                   clean_cache[MEDIATION_LAYER][:, 0]))
        natural_blocked_logits, _ = _forward(
            model, fi, fam, (own_pos, last),
            patch=(MEDIATION_LAYER, own_pos,
                   clean_cache[MEDIATION_LAYER][:, 0]))
        patch_add_group = _ld(patch_add_logits, pos_ids, neg_ids) - clean_ld
        patch_cf_group = _ld(patch_cf_logits, pos_ids, neg_ids) - clean_ld
        blocked_group = _ld(blocked_logits, pos_ids, neg_ids) - clean_ld
        natural_blocked_group = (
            _ld(natural_blocked_logits, pos_ids, neg_ids) - clean_ld)
        patch_add_rows.extend(patch_add_group.tolist())
        patch_cf_rows.extend(patch_cf_group.tolist())
        blocked_rows.extend(blocked_group.tolist())
        natural_blocked_rows.extend(natural_blocked_group.tolist())

        for i, row in enumerate(group):
            row_records.append({
                **row,
                "clean_hit": bool(clean_logits.argmax(-1)[i] == neg_ids[i]),
                "cf_hit": bool(cf_logits.argmax(-1)[i] == pos_ids[i]),
                "natural_effect": float(natural_effect[i]),
                "add_effect": float(add_effect[i]),
                "wrong_effect": float(wrong_effect[i]),
                "other_slot_effect": float(other_effect_group[i]),
                "l2_slot_cos": float(
                    group_metrics["L2_slot"]["cos_rows"][i]),
                "l2_slot_error": float(
                    group_metrics["L2_slot"]["error_rows"][i]),
                "l8_slot_cos": float(
                    group_metrics["L8_slot"]["cos_rows"][i]),
                "l8_last_cos": float(
                    group_metrics["L8_last"]["cos_rows"][i]),
                "patch_add_effect": float(patch_add_group[i]),
                "patch_cf_effect": float(patch_cf_group[i]),
                "add_blocked_effect": float(blocked_group[i]),
                "natural_blocked_effect": float(natural_blocked_group[i]),
                "wrong_l2_cos": float(wrong_cos_group[i]),
                "trajectory": {
                    key: {
                        metric.removesuffix("_rows"): float(values[i])
                        for metric, values in metrics.items()
                    }
                    for key, metrics in group_metrics.items()
                },
            })

        native_last = (cf_cache[MEDIATION_LAYER][:, 1]
                       - clean_cache[MEDIATION_LAYER][:, 1])
        for null_idx in range(n_null):
            random = _random_like_rows(directions, generator)
            random_logits, random_cache = _forward(
                model, ci, cam, (own_pos, last), (MEDIATION_LAYER,),
                add=(INJECT_LAYER, own_pos, random))
            random_effect = _ld(random_logits, pos_ids, neg_ids) - clean_ld
            random_last = (random_cache[MEDIATION_LAYER][:, 1]
                           - clean_cache[MEDIATION_LAYER][:, 1])
            null_output[null_idx] += float(random_effect.mean()) / 2.0
            null_last_cos[null_idx] += float(
                _cos_rows(random_last, native_last).mean()) / 2.0

        log(f"query {query}: n={len(group)} "
            f"add={float(add_effect.mean()):+.2f} "
            f"natural={float(natural_effect.mean()):+.2f}")

    clean_acc = float(np.mean(clean_hits))
    cf_acc = float(np.mean(cf_hits))
    g0 = clean_acc >= 0.80 and cf_acc >= 0.80
    summary = _summarize_metrics(store)
    output_effect = float(np.mean(add_effect_rows))
    natural_effect = float(np.mean(natural_effect_rows))
    output_ratio = output_effect / natural_effect if natural_effect > EPS else -np.inf
    output_p = permutation_pvalue(output_effect, null_output, "greater")
    l2_cos = summary["L2_slot"]["cos"]
    l2_error = summary["L2_slot"]["error"]
    l2_frac = float(np.mean(np.asarray(store["L2_slot"]["cos"]) >= 0.50))
    wrong_cos = float(np.mean(wrong_cos_rows))
    l8_last_cos = summary["L8_last"]["cos"]
    l8_last_frac = float(np.mean(
        np.asarray(store["L8_last"]["cos"]) >= 0.25))
    last_p = permutation_pvalue(l8_last_cos, null_last_cos, "greater")
    positive_frac = float(np.mean(np.asarray(add_effect_rows) > 0))
    other_effect = float(np.mean(other_effect_rows))
    patch_add = float(np.mean(patch_add_rows))
    patch_cf = float(np.mean(patch_cf_rows))
    patch_ratio = patch_add / patch_cf if patch_cf > EPS else -np.inf
    blocked = float(np.mean(blocked_rows))
    block_fraction = ((output_effect - blocked) / output_effect
                      if output_effect > EPS else -np.inf)
    natural_blocked = float(np.mean(natural_blocked_rows))
    natural_block_fraction = (
        (natural_effect - natural_blocked) / natural_effect
        if natural_effect > EPS else -np.inf)
    block_fraction_gap = abs(block_fraction - natural_block_fraction)

    gates = {
        "G0": bool(g0),
        "A1": bool(l2_cos >= 0.80 and l2_error <= 0.60
                   and l2_frac >= 0.80 and l2_cos > wrong_cos),
        "Q1": bool(l8_last_cos >= 0.50 and l8_last_frac >= 0.80
                   and last_p < 0.01),
        "O1": bool(output_effect > 0 and output_p < 0.01
                   and output_ratio >= 0.70 and positive_frac >= 0.80),
        "R1": bool(abs(other_effect) <= 0.20 * max(abs(output_effect), EPS)),
        "M1": bool(patch_add > 0 and patch_cf > 0
                   and 0.70 <= patch_ratio <= 1.30),
        "M2": bool(block_fraction >= 0.70
                   and natural_block_fraction >= 0.70
                   and block_fraction_gap <= 0.20),
        "D1": bool(
            0.80 <= summary["L2_slot"]["norm_ratio"] <= 1.20
            and 0.80 <= summary["L8_slot"]["norm_ratio"] <= 1.20),
        "S1": bool(secondary["S1"]),
    }
    if not gates["G0"]:
        verdict = "OPERATOR_INELICITABLE"
    elif all(gates[key] for key in ("A1", "Q1", "O1", "R1", "M1", "M2", "D1")):
        verdict = "AFFINE_COUNTERFACTUAL_OPERATOR"
    elif all(gates[key] for key in ("A1", "Q1", "O1", "M1", "M2", "D1")):
        verdict = "STATE_EQUIVALENT_NOT_ROLE_CLEAN"
    elif gates["S1"] and not (gates["A1"] and gates["Q1"] and gates["O1"]):
        verdict = "PAIR_OR_CONTEXT_SPECIFIC"
    else:
        verdict = "NOT_AFFINE"

    result = {
        "stage": "delta_operator",
        "model_path": model_path,
        "values": values,
        "donor_names": list(DONOR_NAMES),
        "heldout_name": HELDOUT_NAME,
        "n_trials": len(rows),
        "n_null": int(n_null),
        "secondary_single": secondary,
        "behavior": {"clean_acc": clean_acc, "cf_acc": cf_acc},
        "trajectory": summary,
        "primary": {
            "l2_cos": l2_cos,
            "l2_error": l2_error,
            "l2_cos_fraction_ge_0_5": l2_frac,
            "l2_wrong_cos": wrong_cos,
            "l8_last_cos": l8_last_cos,
            "l8_last_fraction_ge_0_25": l8_last_frac,
            "l8_last_null_mean": float(null_last_cos.mean()),
            "l8_last_p": float(last_p),
            "output_effect": output_effect,
            "natural_effect": natural_effect,
            "output_ratio": float(output_ratio),
            "output_positive_fraction": positive_frac,
            "output_null_mean": float(null_output.mean()),
            "output_p": float(output_p),
            "wrong_output_effect": float(np.mean(wrong_effect_rows)),
            "other_slot_effect": other_effect,
            "patch_add_effect": patch_add,
            "patch_cf_effect": patch_cf,
            "patch_ratio": float(patch_ratio),
            "blocked_effect": blocked,
            "block_fraction": float(block_fraction),
            "natural_blocked_effect": natural_blocked,
            "natural_block_fraction": float(natural_block_fraction),
            "block_fraction_gap": float(block_fraction_gap),
        },
        "rows": row_records,
        "null_draws": {
            "output_effect": null_output.tolist(),
            "l8_last_cos": null_last_cos.tolist(),
        },
        "gates": gates,
        "verdict": verdict,
    }
    with open(os.path.join(out_dir, "results_delta_operator.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    log(f"gates={gates}")
    log(f"VERDICT: {verdict}")
    return result
