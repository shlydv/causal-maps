"""Experiment stages, dispatched by name. Runs on Kaggle GPU.

Stages:
  plumbing / p0_binding     — instrument calibration (done)
  *_hand10                  — Phase 2 hand-10 pilots (done)
  x50_behav                 — Completion + Variable ×50 behavioral gate (7B)
  completion_p1 / variable_p1 — IE sweep + Gate P1 (set model_size 7b|1.5b)

Outputs under out_dir:
  results_<stage>.json, ie_<stage>.npz, heatmap_<stage>.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch

from . import binding_pairs, completion_pairs, instruction_pairs, rule_world, variable_pairs
from .logutil import log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, logit_diff,
                          single_token_id, validate_single_token)
from .nulls import (localization_stats, map_entropy_norm, mass_weighted_mean_layer,
                    mean_map, pair_partition_null, permutation_pvalue,
                    random_position_null, spearman_grid)
from .patching import sweep_ie
from .tensorize import tensorize_pairs

DEFAULT_MODELS = {
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}

STAGE_SPEC = {
    "plumbing":   {"build": rule_world.make_copy_pairs,
                   "pools": {"codes": rule_world.CODES},
                   "require": ("val_slot",), "expected_role": "val_slot",
                   "size": "1.5b", "n_pairs": 40, "keep": 20},
    "p0_binding": {"build": binding_pairs.make_binding_pairs,
                   "pools": {"names": binding_pairs.NAMES, "cities": binding_pairs.CITIES},
                   "require": ("a1_slot", "a2_slot"), "expected_role": "a1_slot",
                   "size": "7b", "n_pairs": 80, "keep": 30},
    "completion_hand10": {
        "build": completion_pairs.make_completion_pairs, "pools": {},
        "require": ("bit_slot",), "expected_role": "bit_slot",
        "size": "7b", "n_pairs": 10, "keep": 10,
        "build_kw": {"hand": completion_pairs.HAND10}},
    "variable_hand10": {
        "build": variable_pairs.make_variable_pairs, "pools": {},
        "require": ("val_slot",), "expected_role": "val_slot",
        "size": "7b", "n_pairs": 10, "keep": 10,
        "build_kw": {"hand": variable_pairs.HAND10}},
    "instruction_hand10": {
        "build": instruction_pairs.make_instruction_pairs, "pools": {},
        "require": (), "expected_role": "frame_slot",
        "size": "7b", "n_pairs": 10, "keep": 10, "loose_behav": True},
    # ×50 libraries
    "completion_x50": {
        "build": completion_pairs.make_completion_pairs, "pools": {},
        "require": ("bit_slot",), "expected_role": "bit_slot",
        "size": "7b", "n_pairs": 50, "keep": 50},
    "variable_x50": {
        "build": variable_pairs.make_variable_pairs, "pools": {},
        "require": ("val_slot",), "expected_role": "val_slot",
        "size": "7b", "n_pairs": 50, "keep": 50},
    # P1 sweeps (model_size overridden via config). Completion uses explicit
    # families only (implicit failed ×50 behav — caveat stands for the note).
    "completion_p1": {
        "build": completion_pairs.make_completion_pairs, "pools": {},
        "require": ("bit_slot",), "expected_role": "bit_slot",
        "size": "7b", "n_pairs": 40, "keep": 40, "do_p1": True,
        "build_kw": {"families": ["explicit_A", "explicit_B",
                                  "explicit_C", "explicit_D"]}},
    "variable_p1": {
        "build": variable_pairs.make_variable_pairs, "pools": {},
        "require": ("val_slot",), "expected_role": "val_slot",
        "size": "7b", "n_pairs": 50, "keep": 50, "do_p1": True},
}
X50_STAGES = ("completion_x50", "variable_x50")


@torch.no_grad()
def behavioral_check(model, batch):
    dev = input_device(model)
    cl = last_token_logits(model, batch["clean"]["input_ids"].to(dev),
                           batch["clean"]["attention_mask"].to(dev))
    fl = last_token_logits(model, batch["cf"]["input_ids"].to(dev),
                           batch["cf"]["attention_mask"].to(dev))
    pos = batch["pos_ids"].to(dev)
    neg = batch["neg_ids"].to(dev)
    clean_ld = logit_diff(cl, pos, neg)
    cf_ld = logit_diff(fl, pos, neg)
    behave = (clean_ld < 0) & (cf_ld > 0)
    return {
        "behave": behave.cpu().numpy(),
        "clean_ld": clean_ld.float().cpu().numpy(),
        "cf_ld": cf_ld.float().cpu().numpy(),
        "clean_greedy_hit": (cl.argmax(-1) == neg).cpu().numpy(),
        "cf_greedy_hit": (fl.argmax(-1) == pos).cpu().numpy(),
    }


@torch.no_grad()
def behavioral_check_loose(model, tok, pairs):
    """Per-pair check (unequal lengths OK). Used for dropped/instr cases."""
    dev = input_device(model)
    behave, clean_ld, cf_ld = [], [], []
    clean_hit, cf_hit = [], []
    kept, excl_multi = [], 0
    for i, p in enumerate(pairs):
        try:
            neg = single_token_id(tok, p["answer_clean"])
            pos = single_token_id(tok, p["answer_cf"])
        except ValueError:
            excl_multi += 1
            continue
        ci = tok.encode(p["clean_text"], add_special_tokens=False)
        fi = tok.encode(p["cf_text"], add_special_tokens=False)
        ct = torch.tensor([ci], dtype=torch.long, device=dev)
        ft = torch.tensor([fi], dtype=torch.long, device=dev)
        cl = last_token_logits(model, ct)
        fl = last_token_logits(model, ft)
        cld = float(logit_diff(cl, pos, neg)[0])
        fld = float(logit_diff(fl, pos, neg)[0])
        clean_ld.append(cld); cf_ld.append(fld)
        behave.append(cld < 0 and fld > 0)
        clean_hit.append(int(cl.argmax(-1).item()) == neg)
        cf_hit.append(int(fl.argmax(-1).item()) == pos)
        kept.append(i)
        m = min(len(ci), len(fi))
        shared = sum(1 for a, b in zip(ci, fi) if a == b)
        # recount shared prefix properly
        shared = 0
        for a, b in zip(ci, fi):
            if a != b:
                break
            shared += 1
        frac = shared / max(len(ci), len(fi), 1)
        log(f"  pair[{i}] {p.get('meta', {}).get('id', i)}: "
            f"len={len(ci)}/{len(fi)} prefix={shared}/{max(len(ci), len(fi))} "
            f"({frac:.0%}) clean_ld={cld:+.2f} cf_ld={fld:+.2f} "
            f"behave={behave[-1]} greedy={clean_hit[-1]}/{cf_hit[-1]}")
    return {
        "behave": np.array(behave, dtype=bool),
        "clean_ld": np.array(clean_ld, dtype=np.float64),
        "cf_ld": np.array(cf_ld, dtype=np.float64),
        "clean_greedy_hit": np.array(clean_hit, dtype=bool),
        "cf_greedy_hit": np.array(cf_hit, dtype=bool),
        "kept_indices": kept,
        "exclusions": {"answer_multitoken": excl_multi},
        "S": None, "anchors": {},
    }


def _subset_batch(batch, idx):
    t = torch.tensor(idx, dtype=torch.long)
    out = dict(batch)
    out["clean"] = {"input_ids": batch["clean"]["input_ids"][t],
                    "attention_mask": batch["clean"]["attention_mask"][t]}
    out["cf"] = {"input_ids": batch["cf"]["input_ids"][t],
                 "attention_mask": batch["cf"]["attention_mask"][t]}
    out["pos_ids"] = batch["pos_ids"][t]
    out["neg_ids"] = batch["neg_ids"][t]
    if "templates" in batch:
        out["templates"] = [batch["templates"][i] for i in idx]
    if "metas" in batch:
        out["metas"] = [batch["metas"][i] for i in idx]
    return out


def _p0_stats(ie_all, exp_axis_idx, cand_axis_idx, seed):
    real, real_layer, null = random_position_null(
        ie_all, expected_positions=[exp_axis_idx],
        candidate_positions=cand_axis_idx, n_draws=1000, seed=seed)
    return {"real": float(real), "layer": int(real_layer),
            "null_mean": float(np.mean(null)), "null_p95": float(np.percentile(null, 95)),
            "p": float(permutation_pvalue(real, null, "greater"))}


def _template_disjoint_halves(templates):
    """Split unique templates into two disjoint sets (sorted for determinism).
    Returns (idx_a, idx_b, templates_a, templates_b)."""
    by_t = defaultdict(list)
    for i, t in enumerate(templates):
        by_t[t].append(i)
    keys = sorted(by_t.keys())
    if len(keys) < 2:
        # fallback: pair split
        n = len(templates); h = n // 2
        return list(range(h)), list(range(h, 2 * h)), ["pair_A"], ["pair_B"]
    mid = len(keys) // 2
    ta, tb = keys[:mid], keys[mid:]
    if not ta:  # odd single left
        ta, tb = keys[:1], keys[1:]
    idx_a = [i for t in ta for i in by_t[t]]
    idx_b = [i for t in tb for i in by_t[t]]
    return idx_a, idx_b, ta, tb


def _p1_stats(ie_all, templates, seed=0):
    idx_a, idx_b, ta, tb = _template_disjoint_halves(templates)
    map_a = mean_map(ie_all, idx_a)
    map_b = mean_map(ie_all, idx_b)
    rho = spearman_grid(map_a, map_b)
    null = pair_partition_null(ie_all, len(idx_a), len(idx_b), n_draws=1000, seed=seed)
    p = permutation_pvalue(rho, null, "greater")
    return {
        "rho": float(rho),
        "p": float(p),
        "null_mean": float(np.nanmean(null)),
        "null_p95": float(np.nanpercentile(null, 95)),
        "n_a": len(idx_a), "n_b": len(idx_b),
        "templates_a": list(ta), "templates_b": list(tb),
        "pass": bool(rho > 0.5 and p < 0.01),
    }


def _contrast_stats(ie_mean, layers, positions, expected_idx):
    loc = localization_stats(ie_mean)
    Hn = map_entropy_norm(ie_mean)
    Lbar = mass_weighted_mean_layer(ie_mean, layers)
    nL = len(layers)
    if Lbar < nL / 3:
        band = "early"
    elif Lbar > 2 * nL / 3:
        band = "late"
    else:
        band = "mid"
    # mass on expected-position column
    if expected_idx in positions:
        pi = positions.index(expected_idx)
        w = np.abs(ie_mean)
        col = float(w[:, pi].sum() / (w.sum() + 1e-12))
    else:
        col = float("nan")
    return {"localization": loc, "H_norm": float(Hn), "mean_layer": float(Lbar),
            "layer_band": band, "expected_col_mass": col}


def _per_template_behav(templates, behave, clean_g, cf_g):
    by_t = defaultdict(lambda: {"n": 0, "behave": 0, "clean_g": 0, "cf_g": 0})
    for t, b, cg, fg in zip(templates, behave, clean_g, cf_g):
        by_t[t]["n"] += 1
        by_t[t]["behave"] += int(b)
        by_t[t]["clean_g"] += int(cg)
        by_t[t]["cf_g"] += int(fg)
    out = {}
    for t, d in by_t.items():
        n = max(d["n"], 1)
        out[t] = {"n": d["n"], "behave_rate": d["behave"] / n,
                  "clean_greedy": d["clean_g"] / n, "cf_greedy": d["cf_g"] / n,
                  "gate_80": (d["behave"] / n) >= 0.80}
    return out


def _save_heatmap(ie_mean, positions, layers, expected_idx, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(5, 1 + 0.35 * len(positions)),
                                    max(3, 0.22 * len(layers))))
    vmax = float(np.nanmax(np.abs(ie_mean))) or 1.0
    im = ax.imshow(ie_mean, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="lower")
    ax.set_xlabel("position"); ax.set_ylabel("layer")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions, fontsize=6)
    if expected_idx in positions:
        ax.axvline(positions.index(expected_idx), color="k", ls="--", lw=1.0)
    fig.colorbar(im, ax=ax, label="mean IE (logit-diff)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


@torch.no_grad()
def _debug_predictions(model, tok, batch, n=4):
    dev = input_device(model)
    for kind in ("clean", "cf"):
        ids = batch[kind]["input_ids"][:n].to(dev)
        am = batch[kind]["attention_mask"][:n].to(dev)
        logits = last_token_logits(model, ids, am)
        topv, topi = logits.topk(6, dim=-1)
        for j in range(ids.shape[0]):
            top = [(tok.decode([t]), round(v, 2))
                   for t, v in zip(topi[j].tolist(), topv[j].tolist())]
            a1 = batch["neg_ids"][j].item(); a2 = batch["pos_ids"][j].item()
            tail = tok.decode(ids[j].tolist())[-45:]
            log(f"DBG {kind}[{j}] …{tail!r} | a1={tok.decode([a1])!r}({logits[j,a1]:.2f}) "
                f"a2={tok.decode([a2])!r}({logits[j,a2]:.2f}) | top6={top}")


def _build_pairs(spec, tok, n_pairs, seed):
    pools = {k: list(validate_single_token(tok, v).keys())
             for k, v in spec.get("pools", {}).items()}
    kw = dict(spec.get("build_kw") or {})
    return spec["build"](n_pairs, seed, tok=tok, **pools, **kw)


def run_stage(stage, model_path, out_dir, n_pairs, keep, expected_role, seed=0,
              hb_every=30.0, device_map=None, cache_to_cpu=False,
              behav_only=False, debug=True, quantization=None,
              model=None, tok=None):
    os.makedirs(out_dir, exist_ok=True)
    spec = STAGE_SPEC[stage]
    if model is None or tok is None:
        model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                              quantization=quantization)
    pairs = _build_pairs(spec, tok, n_pairs, seed)

    loose = bool(spec.get("loose_behav")) or any(p.get("allow_len_mismatch") for p in pairs)
    if loose and behav_only:
        bc = behavioral_check_loose(model, tok, pairs)
        behave_rate = float(bc["behave"].mean()) if len(bc["behave"]) else 0.0
        log(f"behavioral(loose): rate={behave_rate:.2%} | "
            f"clean_greedy={bc['clean_greedy_hit'].mean():.2%} "
            f"cf_greedy={bc['cf_greedy_hit'].mean():.2%} | "
            f"n_behaving={int(bc['behave'].sum())}/{len(bc['behave'])}")
        result = {
            "stage": stage, "behave_rate": behave_rate,
            "n_behaving": int(bc["behave"].sum()),
            "n_checked": int(len(bc["behave"])),
            "clean_greedy": float(bc["clean_greedy_hit"].mean()) if len(bc["behave"]) else 0.0,
            "cf_greedy": float(bc["cf_greedy_hit"].mean()) if len(bc["behave"]) else 0.0,
            "n_pairs_generated": len(pairs),
            "exclusions": bc["exclusions"], "loose_behav": True,
            "gate_80": bool(behave_rate >= 0.80),
            "per_pair_behave": bc["behave"].astype(bool).tolist(),
        }
        with open(os.path.join(out_dir, f"results_{stage}_behav.json"), "w") as f:
            json.dump(result, f, indent=2, default=float)
        log(f"behav_only(loose): gate_80={'PASS' if result['gate_80'] else 'FAIL'} "
            f"({behave_rate:.0%})")
        return result

    # Prefer strict tensorize; if it fails (e.g. mixed-length families), fall back
    # to loose per-pair behav so we still get per-family rates.
    try:
        batch = tensorize_pairs(tok, pairs, require_anchor_roles=spec["require"])
    except ValueError as e:
        log(f"tensorize failed ({e}); falling back to loose behav")
        if not behav_only:
            raise
        bc = behavioral_check_loose(model, tok, pairs)
        behave_rate = float(bc["behave"].mean()) if len(bc["behave"]) else 0.0
        result = {
            "stage": stage, "behave_rate": behave_rate,
            "n_behaving": int(bc["behave"].sum()) if len(bc["behave"]) else 0,
            "n_checked": int(len(bc["behave"])),
            "clean_greedy": float(bc["clean_greedy_hit"].mean()) if len(bc["behave"]) else 0.0,
            "cf_greedy": float(bc["cf_greedy_hit"].mean()) if len(bc["behave"]) else 0.0,
            "n_pairs_generated": len(pairs),
            "exclusions": bc["exclusions"], "loose_behav": True,
            "gate_80": bool(behave_rate >= 0.80),
            "tensorize_error": str(e),
        }
        with open(os.path.join(out_dir, f"results_{stage}_behav.json"), "w") as f:
            json.dump(result, f, indent=2, default=float)
        log(f"behav_only(loose-fallback): gate_80="
            f"{'PASS' if result['gate_80'] else 'FAIL'} ({behave_rate:.0%})")
        return result

    bc = behavioral_check(model, batch)
    behave_rate = float(bc["behave"].mean())
    cg = float(bc["clean_greedy_hit"].mean())
    fg = float(bc["cf_greedy_hit"].mean())
    log(f"behavioral: rate={behave_rate:.2%} | clean_greedy={cg:.2%} "
        f"cf_greedy={fg:.2%} | n_behaving={int(bc['behave'].sum())}/"
        f"{len(bc['behave'])} tensorized={len(batch['kept_indices'])}/"
        f"{len(pairs)}")
    per_t = _per_template_behav(
        batch["templates"], bc["behave"], bc["clean_greedy_hit"], bc["cf_greedy_hit"])
    for t, d in sorted(per_t.items()):
        log(f"  template {t}: behave={d['behave_rate']:.0%} "
            f"greedy={d['clean_greedy']:.0%}/{d['cf_greedy']:.0%} n={d['n']} "
            f"gate80={'PASS' if d['gate_80'] else 'FAIL'}")
    if debug:
        _debug_predictions(model, tok, batch)

    # Also loose-check any pairs dropped by tensorize (e.g. implicit length)
    dropped = [pairs[i] for i in range(len(pairs)) if i not in set(batch["kept_indices"])]
    dropped_behav = None
    if dropped:
        log(f"loose-checking {len(dropped)} tensorize-dropped pairs")
        dropped_behav = behavioral_check_loose(model, tok, dropped)

    if behav_only:
        # Gate: overall behave≥80%, greedy≥80%, and every surviving template ≥80%
        templates_ok = all(d["gate_80"] for d in per_t.values()) if per_t else False
        gate = (behave_rate >= 0.80 and cg >= 0.80 and fg >= 0.80 and templates_ok)
        result = {
            "stage": stage, "behave_rate": behave_rate,
            "n_behaving": int(bc["behave"].sum()),
            "n_checked": int(len(bc["behave"])),
            "clean_greedy": cg, "cf_greedy": fg,
            "n_tensorized": int(len(batch["kept_indices"])),
            "n_pairs_generated": len(pairs),
            "exclusions": batch["exclusions"], "S": batch["S"],
            "anchors": batch["anchors"],
            "per_template": per_t,
            "gate_80": bool(gate),
            "templates_balanced": bool(templates_ok),
            "dropped_n": len(dropped),
            "dropped_behav_rate": (
                float(dropped_behav["behave"].mean()) if dropped_behav and len(dropped_behav["behave"])
                else None),
            "implicit_note": (
                "see per_template / dropped for completion_implicit family"),
        }
        with open(os.path.join(out_dir, f"results_{stage}_behav.json"), "w") as f:
            json.dump(result, f, indent=2, default=float)
        log(f"behav_only: gate={'PASS' if gate else 'FAIL'} "
            f"(behave={behave_rate:.0%} greedy={cg:.0%}/{fg:.0%} "
            f"templates_ok={templates_ok})")
        return result

    keep_idx = [i for i, b in enumerate(bc["behave"]) if b][:keep]
    if len(keep_idx) < keep:
        log(f"WARN only {len(keep_idx)} behaving pairs (< requested {keep})")
    rb = _subset_batch(batch, keep_idx)

    ie_mean, ie_all, clean_ld, meta = sweep_ie(
        model, rb["clean"], rb["cf"], rb["pos_ids"], rb["neg_ids"],
        out_dir=out_dir, tag=stage, hb_every=hb_every, cache_to_cpu=cache_to_cpu)

    positions, layers, S = meta["positions"], meta["layers"], meta["S"]
    exp_idx = batch["anchors"][expected_role]
    exp_axis = positions.index(exp_idx)
    cand_axis = [positions.index(p) for p in positions if p != exp_idx and p != S - 1]

    full = _p0_stats(ie_all, exp_axis, cand_axis, seed)
    p1 = _p1_stats(ie_all, rb["templates"], seed=seed + 10) if spec.get("do_p1") else None
    contrasts = _contrast_stats(ie_mean, layers, positions, exp_idx)

    _save_heatmap(ie_mean, positions, layers, exp_idx,
                  os.path.join(out_dir, f"heatmap_{stage}.png"),
                  f"{stage}: mean IE  expected {expected_role}@{exp_idx}")

    li, pi = np.unravel_index(np.nanargmax(ie_mean), ie_mean.shape)
    result = {
        "stage": stage, "model_path": model_path,
        "n_pairs_generated": len(pairs), "n_behaving": int(bc["behave"].sum()),
        "behave_rate": behave_rate, "clean_greedy": cg, "cf_greedy": fg,
        "n_used": len(keep_idx), "S": S,
        "kept_behaving_indices": keep_idx,
        "templates_used": rb.get("templates"),
        "anchors": batch["anchors"], "exclusions": batch["exclusions"],
        "expected_role": expected_role, "expected_idx": int(exp_idx),
        "clean_ld_mean": float(np.mean(clean_ld)),
        "site_stats": full,
        "p1": p1,
        "contrasts": contrasts,
        "top_site": {"layer": int(layers[li]), "position": int(positions[pi]),
                     "IE": float(ie_mean[li, pi])},
    }
    with open(os.path.join(out_dir, f"results_{stage}.json"), "w") as f:
        json.dump(result, f, indent=2, default=lambda o: float(o))
    np.savez(os.path.join(out_dir, f"ie_{stage}.npz"),
             ie_mean=ie_mean, ie_all=ie_all, positions=np.array(positions),
             layers=np.array(layers), clean_ld=clean_ld,
             behav_clean_ld=bc["clean_ld"], behav_cf_ld=bc["cf_ld"],
             templates=np.array(rb.get("templates", []), dtype=object))
    msg = (f"RESULT {stage}: site={full['real']:+.3f}@L{full['layer']} "
           f"p={full['p']:.4f} top={result['top_site']}")
    if p1:
        msg += (f" | P1 rho={p1['rho']:.3f} p={p1['p']:.4f} "
                f"pass={p1['pass']}")
    msg += (f" | loc={contrasts['localization']['localized']} "
            f"Hn={contrasts['H_norm']:.3f} band={contrasts['layer_band']}")
    log(msg)
    return result


def run_x50_behav(model_path, out_dir, quantization="8bit", device_map=None):
    """One 7B load; Completion + Variable ×50 behavioral gates."""
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    summary = {"stage": "x50_behav", "skills": {}}
    for st in X50_STAGES:
        spec = STAGE_SPEC[st]
        log(f"--- x50 skill={st} ---")
        res = run_stage(
            st, model_path, out_dir,
            n_pairs=spec["n_pairs"], keep=spec["keep"],
            expected_role=spec["expected_role"],
            behav_only=True, debug=True, quantization=quantization,
            model=model, tok=tok)
        summary["skills"][st] = {
            "behave_rate": res["behave_rate"],
            "clean_greedy": res["clean_greedy"],
            "cf_greedy": res["cf_greedy"],
            "n_behaving": res["n_behaving"],
            "n_checked": res["n_checked"],
            "gate_80": res["gate_80"],
            "per_template": res.get("per_template"),
            "dropped_n": res.get("dropped_n"),
            "dropped_behav_rate": res.get("dropped_behav_rate"),
        }
    n_pass = sum(1 for s in summary["skills"].values() if s["gate_80"])
    summary["n_skills_pass"] = n_pass
    with open(os.path.join(out_dir, "results_x50_behav.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"X50 SUMMARY: {n_pass}/2 skills clear behav gate")
    return summary


def run_hand10_behav(model_path, out_dir, quantization="8bit", device_map=None):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    stages = ("completion_hand10", "variable_hand10", "instruction_hand10")
    summary = {"stage": "hand10_behav", "skills": {}}
    for st in stages:
        spec = STAGE_SPEC[st]
        log(f"--- hand10 skill={st} ---")
        res = run_stage(
            st, model_path, out_dir,
            n_pairs=spec["n_pairs"], keep=spec["keep"],
            expected_role=spec["expected_role"],
            behav_only=True, debug=True, quantization=quantization,
            model=model, tok=tok)
        summary["skills"][st] = {
            "behave_rate": res["behave_rate"],
            "clean_greedy": res["clean_greedy"],
            "cf_greedy": res["cf_greedy"],
            "n_behaving": res["n_behaving"],
            "n_checked": res["n_checked"],
            "gate_80": res["gate_80"],
        }
    n_pass = sum(1 for s in summary["skills"].values() if s["gate_80"])
    summary["n_skills_pass_80"] = n_pass
    with open(os.path.join(out_dir, "results_hand10_behav.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"HAND10 SUMMARY: {n_pass}/3 skills clear ≥80% behave")
    return summary


def run_p1_both(model_path, out_dir, quantization=None, device_map=None,
                model_size="7b"):
    """One model load; Completion + Variable P1 sweeps."""
    os.makedirs(out_dir, exist_ok=True)
    if quantization is None and model_size == "7b":
        quantization = "8bit"
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    cache_to_cpu = model_size == "7b"
    summary = {"stage": f"p1_both_{model_size}", "skills": {}}
    for st in ("completion_p1", "variable_p1"):
        spec = STAGE_SPEC[st]
        log(f"--- P1 skill={st} size={model_size} ---")
        res = run_stage(
            st, model_path, out_dir,
            n_pairs=spec["n_pairs"], keep=spec["keep"],
            expected_role=spec["expected_role"],
            behav_only=False, debug=True, quantization=quantization,
            cache_to_cpu=cache_to_cpu, model=model, tok=tok)
        summary["skills"][st] = {
            "behave_rate": res.get("behave_rate"),
            "n_used": res.get("n_used"),
            "p1": res.get("p1"),
            "site_stats": res.get("site_stats"),
            "contrasts": res.get("contrasts"),
            "top_site": res.get("top_site"),
        }
    with open(os.path.join(out_dir, f"results_p1_both_{model_size}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    for st, s in summary["skills"].items():
        p1 = s.get("p1") or {}
        log(f"P1 SUMMARY {st}: pass={p1.get('pass')} rho={p1.get('rho')}")
    return summary


def main(stage, config=None, out_dir="/kaggle/working"):
    config = config or {}
    log(f"=== main stage={stage} config={config} ===")
    if stage == "hand10_behav":
        size = config.get("model_size", "7b")
        res = run_hand10_behav(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None))
        log("=== done ===")
        return res
    if stage == "x50_behav":
        size = config.get("model_size", "7b")
        res = run_x50_behav(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None))
        log("=== done ===")
        return res
    if stage in ("p1_7b", "p1_1p5b"):
        size = "7b" if stage == "p1_7b" else "1.5b"
        res = run_p1_both(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            model_size=size)
        log("=== done ===")
        return res
    if stage == "delta_transfer":
        from .direction_transfer import run_delta_transfer, PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_transfer(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_var_robust":
        from .delta_robust import run_var_robust
        size = config.get("model_size", "7b")
        res = run_var_robust(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_var_shufflefix":
        from .delta_shufflefix import run_var_shufflefix
        size = config.get("model_size", "7b")
        res = run_var_shufflefix(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_completion":
        from .delta_completion import run_delta_completion
        size = config.get("model_size", "7b")
        res = run_delta_completion(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_var_1p5b":
        from .delta_1p5b import run_delta_var_1p5b
        size = "1.5b"
        res = run_delta_var_1p5b(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_var_crosspos":
        from .delta_crosspos import run_delta_var_crosspos
        size = config.get("model_size", "7b")
        res = run_delta_var_crosspos(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_decompose":
        from .delta_decompose import run_delta_decompose
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_decompose(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_centroid":
        from .delta_centroid import run_delta_centroid
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_centroid(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_multislot":
        from .delta_multislot import run_delta_multislot
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_multislot(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_trials=config.get("n_trials", 12))
        log("=== done ===")
        return res
    if stage == "delta_crossskill":
        from .delta_crossskill import run_delta_crossskill
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_crossskill(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_trials=config.get("n_trials", 10))
        log("=== done ===")
        return res
    if stage == "delta_transform":
        from .delta_transform import run_delta_transform
        size = config.get("model_size", "7b")
        res = run_delta_transform(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            layers=config.get("layers", None),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_capacity":
        from .delta_capacity import run_delta_capacity
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_capacity(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_select":
        from .delta_select import run_delta_select
        size = config.get("model_size", "7b")
        res = run_delta_select(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            layers=config.get("layers", None),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_instruction":
        from .delta_instruction import run_delta_instruction
        size = config.get("model_size", "7b")
        res = run_delta_instruction(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            layers=config.get("layers", None),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_instruction_s3":
        from .delta_instruction import run_delta_instruction_stage3
        size = config.get("model_size", "7b")
        res = run_delta_instruction_stage3(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            layers=config.get("layers", None),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_typology":
        from .delta_typology import run_delta_typology
        size = config.get("model_size", "7b")
        res = run_delta_typology(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            layers=config.get("layers", None),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_explicit":
        from .delta_explicit import run_delta_explicit
        size = config.get("model_size", "7b")
        res = run_delta_explicit(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            layers=config.get("layers", None),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_protocol":
        from .delta_protocol import run_delta_protocol
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_protocol(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_trials=config.get("n_trials", 12),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_chain":
        from .delta_chain import run_delta_chain
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_chain(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_trials=config.get("n_trials", 12),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_necessity":
        from .delta_necessity import run_delta_necessity
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_necessity(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_asymmetry":
        from .delta_asymmetry import run_delta_asymmetry
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_asymmetry(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_bindmiss":
        from .delta_bindmiss import run_delta_bindmiss
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_bindmiss(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_router_read":
        from .delta_router_read import run_delta_router_read
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_router_read(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_trials=config.get("n_trials", 12),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_router_ood":
        from .delta_router_ood import run_delta_router_ood
        from .direction_transfer import PRIMARY_LAYER
        size = config.get("model_size", "7b")
        res = run_delta_router_ood(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            layer=config.get("layer", PRIMARY_LAYER),
            seed=config.get("seed", 0),
            n_trials=config.get("n_trials", 12))
        log("=== done ===")
        return res
    if stage == "delta_trajectory":
        from .delta_trajectory import run_delta_trajectory
        size = config.get("model_size", "7b")
        res = run_delta_trajectory(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage in ("delta_operator", "delta_operator_v11"):
        from .delta_operator import run_delta_operator
        size = config.get("model_size", "7b")
        res = run_delta_operator(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_operator_readout":
        from .delta_operator_readout import run_delta_operator_readout
        size = config.get("model_size", "7b")
        res = run_delta_operator_readout(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_operator_content":
        from .delta_operator_content import run_delta_operator_content
        size = config.get("model_size", "7b")
        res = run_delta_operator_content(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_reachability":
        from .delta_reachability import run_delta_reachability
        size = config.get("model_size", "7b")
        res = run_delta_reachability(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage in ("delta_instruction_validate", "delta_instruction_validate_v2"):
        from .delta_instruction_validate import run_delta_instruction_validate
        size = config.get("model_size", "7b")
        res = run_delta_instruction_validate(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_select_quotient":
        from .delta_select_quotient import run_delta_select_quotient
        size = config.get("model_size", "7b")
        res = run_delta_select_quotient(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_reasoning_controller":
        from .delta_reasoning_controller import run_delta_reasoning_controller
        size = config.get("model_size", "7b")
        res = run_delta_reasoning_controller(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_reasoning_screen":
        from .delta_reasoning_screen import run_delta_reasoning_screen
        size = config.get("model_size", "7b")
        res = run_delta_reasoning_screen(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_arithmetic_controller":
        from .delta_arithmetic_controller import run_delta_arithmetic_controller
        size = config.get("model_size", "7b")
        res = run_delta_arithmetic_controller(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_orchestration_screen":
        from .delta_orchestration_screen import run_delta_orchestration_screen
        size = config.get("model_size", "7b")
        res = run_delta_orchestration_screen(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_orchestration_controller":
        from .delta_orchestration_controller import run_delta_orchestration_controller
        size = config.get("model_size", "7b")
        res = run_delta_orchestration_controller(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_orchestration_cross_model_confirmation":
        from .delta_orchestration_controller import run_delta_orchestration_controller
        res = run_delta_orchestration_controller(
            config.get("model_path", "mistralai/Mistral-7B-Instruct-v0.3"),
            out_dir, quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None), seed=0, n_null=100,
            confirmation=True)
        log("=== done ===")
        return res
    if stage == "delta_evidence_arbitration":
        from .delta_evidence_arbitration import run_delta_evidence_arbitration
        res = run_delta_evidence_arbitration(
            config.get("model_path", "mistralai/Mistral-7B-Instruct-v0.3"),
            out_dir, quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None), seed=0, n_null=100)
        log("=== done ===")
        return res
    if stage == "delta_multiturn_evidence_bridge":
        from .delta_multiturn_evidence_bridge import (
            run_delta_multiturn_evidence_bridge)
        res = run_delta_multiturn_evidence_bridge(
            config.get("model_path", "mistralai/Mistral-7B-Instruct-v0.3"),
            out_dir, quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None), seed=0, n_null=100)
        log("=== done ===")
        return res
    if stage == "delta_compositional_agent_control":
        from .delta_compositional_agent_control import (
            run_delta_compositional_agent_control)
        res = run_delta_compositional_agent_control(
            config.get("model_path", "mistralai/Mistral-7B-Instruct-v0.3"),
            out_dir, quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None), seed=0, n_null=100)
        log("=== done ===")
        return res
    if stage == "delta_orchestration_lexical":
        from .delta_orchestration_lexical import run_delta_orchestration_lexical
        size = config.get("model_size", "7b")
        res = run_delta_orchestration_lexical(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_orchestration_label_transfer":
        from .delta_orchestration_label_transfer import run_delta_orchestration_label_transfer
        size = config.get("model_size", "7b")
        res = run_delta_orchestration_label_transfer(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_agent_workspace_screen":
        from .delta_agent_workspace_screen import run_delta_agent_workspace_screen
        size = config.get("model_size", "7b")
        res = run_delta_agent_workspace_screen(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            trust_remote_code=config.get("trust_remote_code", False))
        log("=== done ===")
        return res
    if stage == "delta_answer_turn_control":
        from .delta_continuous_orchestration import run_delta_answer_turn_control
        size = config.get("model_size", "7b")
        res = run_delta_answer_turn_control(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_agent_policy_broadcast":
        from .delta_agent_policy_broadcast import run_delta_agent_policy_broadcast
        size = config.get("model_size", "7b")
        res = run_delta_agent_policy_broadcast(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_policy_slot_readers":
        from .delta_policy_slot_readers import run_delta_policy_slot_readers
        size = config.get("model_size", "7b")
        res = run_delta_policy_slot_readers(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_policy_head_routing":
        from .delta_policy_head_routing import run_delta_policy_head_routing
        size = config.get("model_size", "7b")
        res = run_delta_policy_head_routing(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_component_convergence":
        from .delta_binding_component_convergence import (
            run_delta_binding_component_convergence)
        size = config.get("model_size", "7b")
        res = run_delta_binding_component_convergence(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_binding_causal_subspace":
        from .delta_binding_causal_subspace import (
            run_delta_binding_causal_subspace)
        size = config.get("model_size", "7b")
        res = run_delta_binding_causal_subspace(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 100))
        log("=== done ===")
        return res
    if stage == "delta_binding_causal_state_timeline":
        from .delta_binding_causal_state_timeline import (
            run_delta_binding_causal_state_timeline)
        size = config.get("model_size", "7b")
        res = run_delta_binding_causal_state_timeline(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_slot_transport":
        from .delta_binding_slot_transport import run_delta_binding_slot_transport
        size = config.get("model_size", "7b")
        res = run_delta_binding_slot_transport(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_slot_broadcast":
        from .delta_binding_slot_broadcast import run_delta_binding_slot_broadcast
        size = config.get("model_size", "7b")
        res = run_delta_binding_slot_broadcast(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_slot_bridge":
        from .delta_binding_slot_bridge import run_delta_binding_slot_bridge
        size = config.get("model_size", "7b")
        res = run_delta_binding_slot_bridge(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_backup_formation":
        from .delta_binding_backup_formation import (
            run_delta_binding_backup_formation)
        size = config.get("model_size", "7b")
        res = run_delta_binding_backup_formation(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_backup_reversed_confirmation":
        from .delta_binding_backup_reversed_confirmation import (
            run_delta_binding_backup_reversed_confirmation)
        size = config.get("model_size", "7b")
        res = run_delta_binding_backup_reversed_confirmation(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_cross_model_gate":
        from .delta_binding_cross_model_gate import run_delta_binding_cross_model_gate
        size = config.get("model_size", "7b")
        res = run_delta_binding_cross_model_gate(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            trust_remote_code=config.get("trust_remote_code", False))
        log("=== done ===")
        return res
    if stage == "delta_mistral_binding_backup_port":
        from .delta_mistral_binding_backup_port import run_delta_mistral_binding_backup_port
        size = config.get("model_size", "7b")
        res = run_delta_mistral_binding_backup_port(
            config.get("model_path", "mistralai/Mistral-7B-Instruct-v0.3"), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_surface_operator":
        from .delta_binding_surface_operator import run_delta_binding_surface_operator
        size = config.get("model_size", "7b")
        res = run_delta_binding_surface_operator(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_cross_surface_transfer":
        from .delta_binding_cross_surface_transfer import (
            run_delta_binding_cross_surface_transfer)
        size = config.get("model_size", "7b")
        res = run_delta_binding_cross_surface_transfer(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_composition":
        from .delta_binding_composition import run_delta_binding_composition
        size = config.get("model_size", "7b")
        res = run_delta_binding_composition(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_binding_neutral_carrier":
        from .delta_binding_neutral_carrier import run_delta_binding_neutral_carrier
        size = config.get("model_size", "7b")
        res = run_delta_binding_neutral_carrier(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_addressed_reasoning_ladder":
        from .delta_addressed_reasoning_ladder import (
            run_delta_addressed_reasoning_ladder)
        size = config.get("model_size", "7b")
        res = run_delta_addressed_reasoning_ladder(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_addressed_arithmetic_state":
        from .delta_addressed_arithmetic_state import (
            run_delta_addressed_arithmetic_state)
        size = config.get("model_size", "7b")
        res = run_delta_addressed_arithmetic_state(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0))
        log("=== done ===")
        return res
    if stage == "delta_addressed_arithmetic_confirmation":
        from .delta_addressed_arithmetic_state import (
            run_delta_addressed_arithmetic_state)
        res = run_delta_addressed_arithmetic_state(
            config.get("model_path", "mistralai/Mistral-7B-Instruct-v0.3"),
            out_dir, quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None), seed=config.get("seed", 0),
            workspace_layers=(30,), confirmation=True)
        log("=== done ===")
        return res
    if stage == "delta_dual_readout_workspace":
        from .delta_dual_readout_workspace import (
            run_delta_dual_readout_workspace)
        size = config.get("model_size", "7b")
        res = run_delta_dual_readout_workspace(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_random=config.get("n_random", 20))
        log("=== done ===")
        return res
    if stage == "delta_workspace_matrix":
        from .delta_workspace_matrix import run_delta_workspace_matrix
        size = config.get("model_size", "7b")
        res = run_delta_workspace_matrix(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            model_key=config.get("model_key", "model"),
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            layer=config.get("layer", 2),
            layer_candidates=config.get("layer_candidates", None),
            n_null=config.get("n_null", 50))
        log("=== done ===")
        return res
    if stage == "delta_consequence_law":
        from .delta_consequence_law import run_delta_consequence_law
        size = config.get("model_size", "7b")
        res = run_delta_consequence_law(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            model_key=config.get("model_key", "model"),
            quantization=config.get(
                "quantization", "8bit" if size == "7b" else None),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            layer=config.get("layer", 2),
            n_null=config.get("n_null", 30))
        log("=== done ===")
        return res
    if stage == "delta_atlas":
        from .delta_atlas import run_delta_atlas
        res = run_delta_atlas(
            config["qwen_path"], config["mistral_path"], out_dir,
            quantization=config.get("quantization", "8bit"),
            layer=config.get("layer", 2),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 30))
        log("=== done ===")
        return res
    if stage == "delta_entity_matrix":
        from .delta_entity_matrix import run_delta_entity_matrix
        res = run_delta_entity_matrix(
            config.get("qwen_path"), config.get("mistral_path"), out_dir,
            quantization=config.get("quantization", "8bit"),
            layer=config.get("layer", 2),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 30),
            models=config.get("models"),
            max_memory=config.get("max_memory"))
        log("=== done ===")
        return res
    if stage == "delta_structured_workspace":
        from .delta_structured_workspace import run_delta_structured_workspace
        size = config.get("model_size", "7b")
        res = run_delta_structured_workspace(
            config.get("model_path", DEFAULT_MODELS[size]), out_dir,
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 30),
            report_only=config.get("report_only", False))
        log("=== done ===")
        return res
    if stage == "delta_anchor_write":
        from .delta_anchor_write import run_delta_anchor_write
        res = run_delta_anchor_write(
            config["model_path"], out_dir,
            quantization=config.get("quantization", "awq"),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 30))
        log("=== done ===")
        return res
    if stage == "delta_verbalization":
        from .delta_verbalization import run_delta_verbalization
        res = run_delta_verbalization(
            config["model_path"], out_dir,
            quantization=config.get("quantization", "awq"),
            device_map=config.get("device_map", None),
            seed=config.get("seed", 0),
            n_null=config.get("n_null", 30))
        log("=== done ===")
        return res
    if stage == "delta_preprint_v2_preflight":
        from .delta_preprint_battery import run_delta_preprint_v2_preflight
        res = run_delta_preprint_v2_preflight(
            config["model_path"], out_dir,
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_preprint_battery":
        from .delta_preprint_battery import run_delta_preprint_battery
        res = run_delta_preprint_battery(
            config["model_path"], out_dir,
            model_key=config.get("model_key", "model"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            seeds=config.get("seeds", [0, 1, 2]),
            layer=config.get("layer", 2),
            layer_candidates=config.get("layer_candidates", None),
            max_memory=config.get("max_memory", None),
            n_matrix=config.get("n_matrix", 30),
            n_entity=config.get("n_entity", 30),
            n_world=config.get("n_world", 30),
            checkpoint_layers=config.get("checkpoint_layers", None),
            matrix_null=config.get("matrix_null", 50),
            entity_null=config.get("entity_null", 30),
            anchor_null=config.get("anchor_null", 99),
            run_probe=config.get("run_probe", False),
            probe_reps=config.get("probe_reps", 6),
            run_quorum=config.get("run_quorum", False),
            skip=config.get("skip", []))
        log("=== done ===")
        return res
    if stage == "delta_preprint_probe":
        from .delta_preprint_probe import run_delta_preprint_probe
        res = run_delta_preprint_probe(
            config["model_path"], out_dir,
            quantization=config.get("quantization", "awq"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            layers=config.get("layers", None),
            n_reps=config.get("n_reps", 6),
            seed=config.get("seed", 2718))
        log("=== done ===")
        return res
    if stage == "delta_preprint_locus":
        from .delta_preprint_locus import run_delta_preprint_locus
        res = run_delta_preprint_locus(
            config["model_path"], out_dir,
            model_key=config.get("model_key", "qwen14b_locus"),
            quantization=config.get("quantization", "awq"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30),
            layers=config.get("layers", None),
            n_random_loci=config.get("n_random_loci", 3),
            random_seed=config.get("random_seed", 2718))
        log("=== done ===")
        return res
    if stage == "delta_preprint_locus_preflight":
        from .delta_preprint_locus import run_delta_preprint_locus_preflight
        res = run_delta_preprint_locus_preflight(
            config["model_path"], out_dir,
            n_world=config.get("n_world", 30),
            n_random_loci=config.get("n_random_loci", 3),
            random_seed=config.get("random_seed", 2718))
        log("=== done ===")
        return res
    if stage == "delta_paper1_closeout_preflight":
        from .delta_paper1_closeout import run_delta_paper1_closeout_preflight
        res = run_delta_paper1_closeout_preflight(
            config["model_path"], out_dir,
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_paper1_closeout":
        from .delta_paper1_closeout import run_delta_paper1_closeout
        res = run_delta_paper1_closeout(
            config["model_path"], out_dir,
            model_key=config.get("model_key", "qwen14b_closeout"),
            quantization=config.get("quantization", "awq"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30),
            layers=config.get("layers", None),
            n_null=config.get("n_null", 20),
            random_seed=config.get("random_seed", 8849))
        log("=== done ===")
        return res
    if stage == "delta_shared_component":
        from .delta_shared_component import run_delta_shared_component
        res = run_delta_shared_component(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "deepseek_shared_component_d1"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30),
            n_donor=config.get("n_donor", 15),
            layers=config.get("layers", None),
            random_seed=config.get("random_seed", 7319))
        log("=== done ===")
        return res
    if stage == "delta_binding_divergence":
        from .delta_binding_divergence import run_delta_binding_divergence
        res = run_delta_binding_divergence(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "deepseek_direct_binding_diagnostic"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_rows=config.get("n_rows", 30))
        log("=== done ===")
        return res
    if stage == "delta_content_aliasing":
        from .delta_content_aliasing import run_delta_content_aliasing
        res = run_delta_content_aliasing(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "deepseek_content_aliasing_d1"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_rows=config.get("n_rows", 30),
            loads=config.get("loads", (1, 2, 3)))
        log("=== done ===")
        return res
    if stage == "delta_sparse_transport":
        from .delta_sparse_transport import run_delta_sparse_transport
        res = run_delta_sparse_transport(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_sparse_transport_d1"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30),
            n_donor=config.get("n_donor", 15),
            layers=config.get("layers", (21, 22, 23, 24)),
            top_k=config.get("top_k", (1, 2, 4, 8)),
            n_random=config.get("n_random", 5),
            random_seed=config.get("random_seed", 4283))
        log("=== done ===")
        return res
    if stage == "delta_sparse_transport_confirmation":
        from .delta_sparse_transport_confirmation import (
            run_delta_sparse_transport_confirmation,
        )
        res = run_delta_sparse_transport_confirmation(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_sparse_transport_confirmation"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30),
            n_random=config.get("n_random", 99),
            random_seed=config.get("random_seed", 9407))
        log("=== done ===")
        return res
    if stage == "delta_source_head_mediation":
        from .delta_source_head_mediation import (
            run_delta_source_head_mediation,
        )
        res = run_delta_source_head_mediation(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_source_head_mediation"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30),
            n_random=config.get("n_random", 39),
            random_seed=config.get("random_seed", 6113))
        log("=== done ===")
        return res
    if stage == "delta_conditional_backup":
        from .delta_conditional_backup import run_delta_conditional_backup
        res = run_delta_conditional_backup(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_conditional_backup"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30),
            n_donor=config.get("n_donor", 15),
            n_random=config.get("n_random", 9),
            random_seed=config.get("random_seed", 7759))
        log("=== done ===")
        return res
    if stage == "delta_operation_handoff_depth":
        from .delta_operation_handoff_depth import (
            run_delta_operation_handoff_depth,
        )
        res = run_delta_operation_handoff_depth(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_operation_handoff_depth"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_depth_replication":
        from .delta_depth_replication import run_delta_depth_replication
        res = run_delta_depth_replication(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_depth_replication"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_semantic_command_factor":
        from .delta_semantic_command_factor import (
            run_delta_semantic_command_factor,
        )
        res = run_delta_semantic_command_factor(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_semantic_command_factor"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_prompt_factorial":
        from .delta_prompt_factorial import run_delta_prompt_factorial
        res = run_delta_prompt_factorial(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_prompt_factorial"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_token_length_ladder":
        from .delta_token_length_ladder import (
            run_delta_token_length_ladder,
        )
        res = run_delta_token_length_ladder(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_token_length_ladder"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_position_matched_label":
        from .delta_position_matched_label import (
            run_delta_position_matched_label,
        )
        res = run_delta_position_matched_label(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_position_matched_label"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_label_meaning_codebook":
        from .delta_label_meaning_codebook import (
            run_delta_label_meaning_codebook,
        )
        res = run_delta_label_meaning_codebook(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_label_meaning_codebook"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_lexical_class":
        from .delta_lexical_class import run_delta_lexical_class
        res = run_delta_lexical_class(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_lexical_class"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_label_route_switch":
        from .delta_label_route_switch import (
            run_delta_label_route_switch,
        )
        res = run_delta_label_route_switch(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_label_route_switch"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_paired_route_transplant":
        from .delta_paired_route_transplant import (
            run_delta_paired_route_transplant,
        )
        res = run_delta_paired_route_transplant(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_paired_route_transplant"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_distributed_label_transplant":
        from .delta_distributed_label_transplant import (
            run_delta_distributed_label_transplant,
        )
        res = run_delta_distributed_label_transplant(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_distributed_label_transplant"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_distributed_replication":
        from .delta_distributed_replication import (
            run_delta_distributed_replication,
        )
        res = run_delta_distributed_replication(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen14b_distributed_replication"),
            quantization=config.get("quantization", "awq"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30),
            source_layer=config.get("source_layer", 32),
            route_start=config.get("route_start", 33),
            route_checkpoints=config.get(
                "route_checkpoints",
                [34, 36, 38, 40, 41, 42, 44, 46, 47]),
            decision_layer=config.get("decision_layer", 41),
            replication_spec=config.get("replication_spec", "qwen14b"))
        log("=== done ===")
        return res
    if stage == "delta_replication_behavior_screen":
        from .delta_replication_behavior_screen import (
            run_delta_replication_behavior_screen,
        )
        res = run_delta_replication_behavior_screen(
            config["model_path"], out_dir,
            model_key=config.get("model_key", "qwen14b_behavior_screen"),
            quantization=config.get("quantization", "awq"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_synonym_prefix_transfer":
        from .delta_synonym_prefix_transfer import (
            run_delta_synonym_prefix_transfer,
        )
        res = run_delta_synonym_prefix_transfer(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_synonym_prefix_transfer"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_content_cancelled_controller":
        from .delta_content_cancelled_controller import (
            run_delta_content_cancelled_controller,
        )
        res = run_delta_content_cancelled_controller(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_content_cancelled_controller"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_cross_domain_controller":
        from .delta_cross_domain_controller import (
            run_delta_cross_domain_controller,
        )
        res = run_delta_cross_domain_controller(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_cross_domain_controller"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_controller_matrix":
        from .delta_controller_matrix import (
            run_delta_controller_matrix,
        )
        res = run_delta_controller_matrix(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_controller_matrix"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_leave_color_out_shared":
        from .delta_leave_color_out_shared import (
            run_delta_leave_color_out_shared,
        )
        res = run_delta_leave_color_out_shared(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_leave_color_out_shared"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_shared_adapter_decomposition":
        from .delta_shared_adapter_decomposition import (
            run_delta_shared_adapter_decomposition,
        )
        res = run_delta_shared_adapter_decomposition(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_shared_adapter_decomposition"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_residual_only_confirmation":
        from .delta_residual_only_confirmation import (
            run_delta_residual_only_confirmation,
        )
        res = run_delta_residual_only_confirmation(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_residual_only_confirmation"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_endogenous_residual_necessity":
        from .delta_endogenous_residual_necessity import (
            run_delta_endogenous_residual_necessity,
        )
        res = run_delta_endogenous_residual_necessity(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_endogenous_residual_necessity"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_endogenous_controller_factorial":
        from .delta_endogenous_controller_factorial import (
            run_delta_endogenous_controller_factorial,
        )
        res = run_delta_endogenous_controller_factorial(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_endogenous_controller_factorial"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_causal_rank_spectrum":
        from .delta_causal_rank_spectrum import (
            run_delta_causal_rank_spectrum,
        )
        res = run_delta_causal_rank_spectrum(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_causal_rank_spectrum"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 30))
        log("=== done ===")
        return res
    if stage == "delta_prospective_causal_sensitivity":
        from .delta_prospective_causal_sensitivity import (
            run_delta_prospective_causal_sensitivity,
        )
        res = run_delta_prospective_causal_sensitivity(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key",
                "qwen7b_prospective_causal_sensitivity"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 50))
        log("=== done ===")
        return res
    if stage == "delta_heterogeneous_family_screen":
        from .delta_heterogeneous_family_screen import (
            run_delta_heterogeneous_family_screen,
        )
        res = run_delta_heterogeneous_family_screen(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_heterogeneous_family_screen"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 15))
        log("=== done ===")
        return res
    if stage == "delta_cross_family_causal_subspace":
        from .delta_cross_family_causal_subspace import (
            run_delta_cross_family_causal_subspace,
        )
        res = run_delta_cross_family_causal_subspace(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_cross_family_causal_subspace"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 50))
        log("=== done ===")
        return res
    if stage == "delta_exact_transplant_locus_diagnostic":
        from .delta_exact_transplant_locus_diagnostic import (
            run_delta_exact_transplant_locus_diagnostic,
        )
        res = run_delta_exact_transplant_locus_diagnostic(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key",
                "qwen7b_exact_transplant_locus_diagnostic"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 12))
        log("=== done ===")
        return res
    if stage == "delta_predictive_conditional_transport":
        from .delta_predictive_conditional_transport import (
            run_delta_predictive_conditional_transport,
        )
        res = run_delta_predictive_conditional_transport(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key",
                "qwen7b_predictive_conditional_transport"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 12),
            self_test_only=config.get("self_test_only", False))
        log("=== done ===")
        return res
    if stage == "delta_within_family_conditional_transport":
        from .delta_within_family_conditional_transport import (
            run_delta_within_family_conditional_transport,
        )
        res = run_delta_within_family_conditional_transport(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key",
                "qwen7b_within_family_conditional_transport"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 12),
            self_test_only=config.get("self_test_only", False))
        log("=== done ===")
        return res
    if stage == "delta_context_geometry_width_screen":
        from .delta_context_geometry_width_screen import (
            run_delta_context_geometry_width_screen,
        )
        res = run_delta_context_geometry_width_screen(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key",
                "qwen7b_context_geometry_width_screen"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 8),
            self_test_only=config.get("self_test_only", False))
        log("=== done ===")
        return res
    if stage == "delta_heldout_inverse_control":
        from .delta_heldout_inverse_control import (
            run_delta_heldout_inverse_control,
        )
        res = run_delta_heldout_inverse_control(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_heldout_inverse_control"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 6),
            self_test_only=config.get("self_test_only", False))
        log("=== done ===")
        return res
    if stage == "delta_controller_circuit_epistasis":
        from .delta_controller_circuit_epistasis import (
            run_delta_controller_circuit_epistasis,
        )
        res = run_delta_controller_circuit_epistasis(
            config["model_path"], out_dir,
            model_key=config.get(
                "model_key", "qwen7b_controller_circuit_epistasis"),
            quantization=config.get("quantization", "8bit"),
            device_map=config.get("device_map", None),
            max_memory=config.get("max_memory", None),
            n_world=config.get("n_world", 60))
        log("=== done ===")
        return res
    spec = STAGE_SPEC[stage]
    size = config.get("model_size", spec["size"])
    model_path = config.get("model_path", DEFAULT_MODELS[size])
    device_map = config.get("device_map", None)
    quant = config.get("quantization", "8bit" if size == "7b" else None)
    cache_to_cpu = config.get("cache_to_cpu", size == "7b")
    res = run_stage(
        stage, model_path, out_dir,
        n_pairs=config.get("n_pairs", spec["n_pairs"]),
        keep=config.get("keep", spec["keep"]),
        expected_role=config.get("expected_role", spec["expected_role"]),
        seed=config.get("seed", 0), hb_every=config.get("hb_every", 30.0),
        device_map=device_map, cache_to_cpu=cache_to_cpu,
        behav_only=config.get("behav_only", False), debug=config.get("debug", True),
        quantization=quant)
    log("=== done ===")
    return res
