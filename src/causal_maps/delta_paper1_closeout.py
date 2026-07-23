"""Frozen final experimental closeout for the token-anchored Paper 1."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import (_anchor_position, _neutral_states, _resolve)
from .delta_preprint_battery import _compatible_world_rows
from .delta_preprint_locus import _forward_multi_patch
from .delta_structured_workspace import (LOCATIONS, MARKER, QUERY, _accuracy,
                                         _batch, _counterfactual, _locations,
                                         _rows, _switch_metrics, _switch_pass)
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import (input_device, load_model_and_tokenizer,
                          model_num_hidden_layers)
from .nulls import permutation_pvalue

PROTOCOL_VERSION = "2026-07-22-v1"
LAYERS = (2, 4, 8, 12, 16, 20, 24, 26, 32, 36, 41, 46)
G0 = 0.80
RATIO_GATE = (0.60, 1.40)
NAT_SOURCE, NAT_TARGET = "Tokyo", "Delhi"
N_ROWS = 30
N_NULL = 20
RANDOM_SEED = 8849
STYLES = ("case_note", "witness_transcript", "curator_narrative")


def _naturalized_rows(n_rows=N_ROWS):
    rows = _rows(NAT_SOURCE, NAT_TARGET, "ac", "train", n_rows=n_rows)
    groups = {style: [] for style in STYLES}
    for index, row in enumerate(rows):
        groups[STYLES[index % len(STYLES)]].append(row)
    return rows, groups


def _naturalized_user(row, query, style):
    question, command, _field = QUERY[query]
    if style == "case_note":
        rule = (
            "For this case, a person's belief follows the most recent move "
            "of an object that person actually witnessed. A move made after "
            "they left changes reality but not what they believe."
        )
        history = (
            f"The museum incident note says Alice alone watched a courier set "
            f"the cube down in {row['ac']}. After Alice signed out, Bob alone "
            f"saw the cube transferred to {row['bc']}. On a separate round, "
            f"Alice was the sole witness when the sphere went to {row['as']}; "
            f"later, with Alice absent, Bob watched it move to {row['bs']}. "
            f"Once both witnesses had left the building, staff made two final "
            f"unseen transfers: the cube to {row['tc']} and the sphere to "
            f"{row['ts']}. The lighting inspection mentioned in the note moved "
            "neither object."
        )
    elif style == "witness_transcript":
        rule = (
            "Read the transcript chronologically. Each witness updates only "
            "from placements they personally observe; an unobserved later "
            "placement determines the true location without revising that "
            "witness's private belief."
        )
        history = (
            f"Transcript: Alice reports, 'I watched the cube being left in "
            f"{row['ac']} before I departed.' The next entry records Bob, and "
            f"not Alice, observing that cube arrive in {row['bc']}. Alice then "
            f"personally observes the sphere placed in {row['as']}. A later "
            f"entry has Bob alone see the sphere carried to {row['bs']}. The "
            f"closing security record, witnessed by neither person, places the "
            f"cube in {row['tc']} and the sphere in {row['ts']}. Administrative "
            "comments between entries contain no object movements."
        )
    elif style == "curator_narrative":
        rule = (
            "Track objective locations separately from the characters' "
            "knowledge. What a character believes is fixed by the last "
            "placement that character saw, even if later unseen handling "
            "changes where the object really is."
        )
        history = (
            f"In the curator's account, Alice remains in the room long enough "
            f"to see the cube placed in {row['ac']}, then walks away. Bob later "
            f"enters without Alice and sees the cube moved to {row['bc']}. In "
            f"another gallery Alice watches the sphere reach {row['as']} and "
            f"leaves; afterward Bob sees that sphere taken to {row['bs']}. "
            f"Much later, after both have gone, an overnight crew silently "
            f"finishes with the cube in {row['tc']} and the sphere in "
            f"{row['ts']}. A discussion of tomorrow's opening time is irrelevant "
            "to every placement."
        )
    else:
        raise ValueError(f"unknown naturalized style: {style}")
    return (
        f"Maintain each private belief and the true world. {rule} {history} "
        f"Work out all six locations silently before the question. {MARKER}.\n"
        f"Question: {question} Reply with exactly {command}, one space, and "
        "the location name. Do not add anything else."
    )


def _naturalized_render(tok, row, query, style):
    return tok.apply_chat_template(
        [{"role": "user", "content": _naturalized_user(row, query, style)}],
        tokenize=False, add_generation_prompt=True)


def _naturalized_batch(tok, rows, query, style, dev):
    return _batch(
        tok, rows, query, "narrative", dev,
        render_fn=lambda row: _naturalized_render(tok, row, query, style))


def _aggregate_switch(parts):
    natural_rows = np.concatenate(
        [np.asarray(part["natural_rows"], dtype=float) for part in parts])
    effect_rows = np.concatenate(
        [np.asarray(part["effect_rows"], dtype=float) for part in parts])
    total = sum(part["n"] for part in parts)
    natural = float(natural_rows.mean())
    effect = float(effect_rows.mean())
    ratio = effect / natural if abs(natural) > 1e-8 else None
    out = {
        "natural_effect": natural,
        "effect": effect,
        "ratio": ratio,
        "target_acc": float(sum(part["target_acc"] * part["n"]
                                for part in parts) / total),
        "positive_fraction": float((effect_rows > 0).mean()),
        "effect_rows": effect_rows.tolist(),
        "natural_rows": natural_rows.tolist(),
        "n_rows": int(total),
        "g0_clean": float(sum(part["g0_clean"] * part["n"]
                              for part in parts) / total),
        "g0_natural": float(sum(part["g0_natural"] * part["n"]
                                for part in parts) / total),
    }
    out["pass"] = bool(min(out["g0_clean"], out["g0_natural"]) >= G0
                       and _switch_pass(out))
    return out


def _causal_cell(clean_ld, natural_ld, forward_ld, reverse_ld,
                 forward_acc, reverse_acc):
    natural_rows = natural_ld - clean_ld
    forward_rows = forward_ld - clean_ld
    reverse_rows = natural_ld - reverse_ld
    natural = float(natural_rows.mean())
    f_ratio = float(forward_rows.mean()) / natural if abs(natural) > 1e-8 else None
    r_ratio = float(reverse_rows.mean()) / natural if abs(natural) > 1e-8 else None
    lo, hi = RATIO_GATE
    sufficient = bool(f_ratio is not None and r_ratio is not None
                      and lo <= f_ratio <= hi and lo <= r_ratio <= hi
                      and forward_acc >= G0 and reverse_acc >= G0)
    return {
        "forward_ratio": f_ratio, "reverse_ratio": r_ratio,
        "forward_target_acc": float(forward_acc),
        "reverse_clean_acc": float(reverse_acc),
        "natural_effect_rows": natural_rows.tolist(),
        "forward_effect_rows": forward_rows.tolist(),
        "reverse_effect_rows": reverse_rows.tolist(),
        "forward_ld_rows": forward_ld.tolist(),
        "reverse_ld_rows": reverse_ld.tolist(),
        "sufficient": sufficient,
    }


@torch.no_grad()
def _exact_ac_only(model, tok, dev, layers, n_world):
    rows, indices = _compatible_world_rows(tok, torch.device("cpu"), n_world)
    natural_rows = _counterfactual(rows, {"ac": "Rome"})
    cb = _batch(tok, rows, "belief_ac", "narrative", dev)
    nb = _batch(tok, natural_rows, "belief_ac", "narrative", dev)
    anchor = _anchor_position(cb, nb)
    cl, cc = _forward(model, cb["ids"], cb["am"], (anchor,), tuple(layers))
    nl, nc = _forward(model, nb["ids"], nb["am"], (anchor,), tuple(layers))
    src, tgt = _locations(rows, "belief_ac"), _locations(natural_rows, "belief_ac")
    sid = torch.tensor([cb["amap"][x] for x in src])
    tid = torch.tensor([cb["amap"][x] for x in tgt])
    clean_ld, natural_ld = _ld(cl, tid, sid), _ld(nl, tid, sid)
    out = {
        "n_rows": len(rows), "indices_from_30": indices, "anchor": anchor,
        "g0_clean": _accuracy(cl, cb, src),
        "g0_natural": _accuracy(nl, nb, tgt), "per_layer": {},
    }
    for layer_i, layer in enumerate(layers):
        fwd = _forward_multi_patch(model, cb["ids"], cb["am"], layer,
                                   [anchor], nc[layer][:, [0], :])
        rev = _forward_multi_patch(model, nb["ids"], nb["am"], layer,
                                   [anchor], cc[layer][:, [0], :])
        cell = _causal_cell(
            clean_ld, natural_ld, _ld(fwd, tid, sid), _ld(rev, tid, sid),
            _accuracy(fwd, cb, tgt), _accuracy(rev, nb, src))
        out["per_layer"][layer] = cell
        log(f"  [exact ac L{layer}] fwd={cell['forward_ratio']:.3f}/"
            f"{cell['forward_target_acc']:.0%} rev={cell['reverse_ratio']:.3f}/"
            f"{cell['reverse_clean_acc']:.0%}")
    out["sufficient_layers"] = [int(layer) for layer, cell in out["per_layer"].items()
                                if cell["sufficient"]]
    out["verdict"] = ("AC_ONLY_SUFFICIENT" if out["sufficient_layers"]
                      else "AC_ONLY_NOT_SUFFICIENT")
    return out


@torch.no_grad()
def _naturalized_arm(model, tok, dev, layers, n_rows, n_null, random_seed):
    rows, groups = _naturalized_rows(n_rows)
    natural_groups = {
        style: _counterfactual(group, {"ac": NAT_TARGET})
        for style, group in groups.items()
    }
    delta_states = _neutral_states(
        model, tok, dev, 2, (NAT_SOURCE, NAT_TARGET))
    delta = delta_states[NAT_TARGET] - delta_states[NAT_SOURCE]
    result = {
        "source": NAT_SOURCE, "target": NAT_TARGET, "n_rows": len(rows),
        "styles": {name: len(group) for name, group in groups.items()},
        "layer": 2, "delta_norm": float(delta.norm()),
        "consequences": {},
    }
    belief_cache = []
    for query in ("belief_ac", "tell_ac"):
        parts = []
        for style in STYLES:
            clean_rows, natural_rows = groups[style], natural_groups[style]
            cb = _naturalized_batch(tok, clean_rows, query, style, dev)
            nb = _naturalized_batch(tok, natural_rows, query, style, dev)
            anchor = _anchor_position(cb, nb)
            cl, _ = _forward(model, cb["ids"], cb["am"], (anchor,))
            nl, _ = _forward(model, nb["ids"], nb["am"], (anchor,))
            add, _ = _forward(model, cb["ids"], cb["am"], (anchor,),
                              add=(2, anchor, delta))
            src = _locations(clean_rows, query)
            tgt = _locations(natural_rows, query)
            metrics = _switch_metrics(cl, nl, add, cb, src, tgt)
            metrics.update({"n": len(clean_rows),
                            "g0_clean": _accuracy(cl, cb, src),
                            "g0_natural": _accuracy(nl, nb, tgt)})
            parts.append(metrics)
            if query == "belief_ac":
                belief_cache.append({"style": style, "batch": cb, "base": cl,
                                     "anchor": anchor, "source": src,
                                     "target": tgt})
        result["consequences"][query] = _aggregate_switch(parts)
        m = result["consequences"][query]
        log(f"  [naturalized {query}] g0={m['g0_clean']:.0%}/"
            f"{m['g0_natural']:.0%} ratio={m['ratio']:.3f} "
            f"acc={m['target_acc']:.0%} pass={m['pass']}")

    # Unrelated Bob-belief specificity under the Alice-anchor write.
    clean_correct = add_correct = total = 0.0
    invariant_rows = []
    for style in STYLES:
        clean_rows, natural_rows = groups[style], natural_groups[style]
        b = _naturalized_batch(tok, clean_rows, "belief_bc", style, dev)
        n = _naturalized_batch(tok, natural_rows, "belief_bc", style, dev)
        anchor = _anchor_position(b, n)
        base, _ = _forward(model, b["ids"], b["am"], (anchor,))
        add, _ = _forward(model, b["ids"], b["am"], (anchor,),
                          add=(2, anchor, delta))
        expected = _locations(clean_rows, "belief_bc")
        count = len(clean_rows)
        clean_correct += _accuracy(base, b, expected) * count
        add_correct += _accuracy(add, b, expected) * count
        total += count
        pool = torch.tensor([b["amap"][x] for x in LOCATIONS])
        invariant_rows.extend((add[:, pool].argmax(-1)
                               == torch.tensor([LOCATIONS.index(x)
                                                for x in expected])).tolist())
    result["invariant_belief_bc"] = {
        "clean_acc": float(clean_correct / total),
        "add_acc": float(add_correct / total),
        "correct_rows": invariant_rows,
        "pass": bool(add_correct / total >= G0),
    }

    # Seeded norm-matched random directions on the belief consequence.
    rng = np.random.default_rng(random_seed)
    nulls = []
    norm = float(delta.norm().clamp(min=1e-8))
    for null_i in range(n_null):
        random = torch.from_numpy(
            rng.normal(size=delta.numel()).astype(np.float32))
        random = random / random.norm().clamp(min=1e-8) * norm
        effects = []
        for cached in belief_cache:
            b, base = cached["batch"], cached["base"]
            lg, _ = _forward(model, b["ids"], b["am"], (cached["anchor"],),
                             add=(2, cached["anchor"], random))
            sid = torch.tensor([b["amap"][x] for x in cached["source"]])
            tid = torch.tensor([b["amap"][x] for x in cached["target"]])
            effects.extend((_ld(lg, tid, sid) - _ld(base, tid, sid)).tolist())
        nulls.append(float(np.mean(effects)))
        log(f"  naturalized null {null_i + 1}/{n_null}: {nulls[-1]:+.3f}")
    observed = result["consequences"]["belief_ac"]["effect"]
    p = permutation_pvalue(observed, np.asarray(nulls), "greater")
    result["null"] = {"n": n_null, "p": float(p),
                      "observed": observed, "values": nulls,
                      "mean": float(np.mean(nulls)), "pass": bool(p < .05)}

    # Bidirectional marker/readout trajectory, aggregated across prose styles.
    trajectory_parts = {layer: {site: [] for site in ("checkpoint", "readout")}
                        for layer in layers}
    trajectory_sites = {}
    for style in STYLES:
        clean_rows, natural_rows = groups[style], natural_groups[style]
        cb = _naturalized_batch(tok, clean_rows, "belief_ac", style, dev)
        nb = _naturalized_batch(tok, natural_rows, "belief_ac", style, dev)
        sites = {"checkpoint": cb["marker"],
                 "readout": int(cb["ids"].shape[1] - 1)}
        trajectory_sites[style] = sites
        positions = list(sites.values())
        cl, cc = _forward(model, cb["ids"], cb["am"], positions, tuple(layers))
        nl, nc = _forward(model, nb["ids"], nb["am"], positions, tuple(layers))
        src, tgt = _locations(clean_rows, "belief_ac"), _locations(natural_rows,
                                                                        "belief_ac")
        sid = torch.tensor([cb["amap"][x] for x in src])
        tid = torch.tensor([cb["amap"][x] for x in tgt])
        clean_ld, natural_ld = _ld(cl, tid, sid), _ld(nl, tid, sid)
        for site_i, (site, position) in enumerate(sites.items()):
            for layer in layers:
                fwd = _forward_multi_patch(
                    model, cb["ids"], cb["am"], layer, [position],
                    nc[layer][:, [site_i], :])
                rev = _forward_multi_patch(
                    model, nb["ids"], nb["am"], layer, [position],
                    cc[layer][:, [site_i], :])
                trajectory_parts[layer][site].append({
                    "natural": (natural_ld - clean_ld).tolist(),
                    "forward": (_ld(fwd, tid, sid) - clean_ld).tolist(),
                    "reverse": (natural_ld - _ld(rev, tid, sid)).tolist(),
                    "forward_acc": _accuracy(fwd, cb, tgt),
                    "reverse_acc": _accuracy(rev, nb, src),
                    "n": len(clean_rows),
                })

    trajectory = {"sites_by_style": trajectory_sites, "per_layer": {}}
    for layer in layers:
        trajectory["per_layer"][layer] = {}
        for site in ("checkpoint", "readout"):
            parts = trajectory_parts[layer][site]
            natural = np.concatenate([np.asarray(x["natural"]) for x in parts])
            forward = np.concatenate([np.asarray(x["forward"]) for x in parts])
            reverse = np.concatenate([np.asarray(x["reverse"]) for x in parts])
            total = sum(x["n"] for x in parts)
            denominator = float(natural.mean())
            f_ratio = float(forward.mean()) / denominator
            r_ratio = float(reverse.mean()) / denominator
            f_acc = sum(x["forward_acc"] * x["n"] for x in parts) / total
            r_acc = sum(x["reverse_acc"] * x["n"] for x in parts) / total
            lo, hi = RATIO_GATE
            cell = {
                "forward_ratio": f_ratio, "reverse_ratio": r_ratio,
                "forward_target_acc": float(f_acc),
                "reverse_clean_acc": float(r_acc),
                "natural_effect_rows": natural.tolist(),
                "forward_effect_rows": forward.tolist(),
                "reverse_effect_rows": reverse.tolist(),
                "sufficient": bool(lo <= f_ratio <= hi and lo <= r_ratio <= hi
                                   and f_acc >= G0 and r_acc >= G0),
            }
            trajectory["per_layer"][layer][site] = cell
            log(f"  [naturalized {site} L{layer}] fwd={f_ratio:.3f}/"
                f"{f_acc:.0%} rev={r_ratio:.3f}/{r_acc:.0%}")
    checkpoint_values = [abs(block["checkpoint"][direction])
                         for block in trajectory["per_layer"].values()
                         for direction in ("forward_ratio", "reverse_ratio")]
    trajectory["max_abs_checkpoint_ratio"] = float(max(checkpoint_values))
    trajectory["readout_sufficient_layers"] = [
        int(layer) for layer, block in trajectory["per_layer"].items()
        if block["readout"]["sufficient"]]
    trajectory["verdict"] = (
        "NATURALIZED_TRAJECTORY_CONFIRMED"
        if trajectory["max_abs_checkpoint_ratio"] < .30
        and trajectory["readout_sufficient_layers"]
        else "NATURALIZED_TRAJECTORY_NOT_CONFIRMED")
    result["trajectory"] = trajectory
    anchor_ok = (all(x["pass"] for x in result["consequences"].values())
                 and result["invariant_belief_bc"]["pass"]
                 and result["null"]["pass"])
    result["verdict"] = (
        "NATURALIZED_CONFIRMED" if anchor_ok
        and trajectory["verdict"] == "NATURALIZED_TRAJECTORY_CONFIRMED"
        else "NATURALIZED_NOT_CONFIRMED")
    return result


def run_delta_paper1_closeout_preflight(model_path, out_dir, n_world=N_ROWS):
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    resolved = _resolve(model_path)
    tok = AutoTokenizer.from_pretrained(resolved)
    dev = torch.device("cpu")
    standard, indices = _compatible_world_rows(tok, dev, n_world)
    standard_nat = _counterfactual(standard, {"ac": "Rome"})
    std_clean = _batch(tok, standard, "belief_ac", "narrative", dev)
    std_nat = _batch(tok, standard_nat, "belief_ac", "narrative", dev)
    std_anchor = _anchor_position(std_clean, std_nat)
    rows, groups = _naturalized_rows(n_world)
    contracts = {}
    for style in STYLES:
        natural = _counterfactual(groups[style], {"ac": NAT_TARGET})
        query_contracts = {}
        for query in ("belief_ac", "tell_ac", "belief_bc"):
            cb = _naturalized_batch(tok, groups[style], query, style, dev)
            nb = _naturalized_batch(tok, natural, query, style, dev)
            query_contracts[query] = {
                "n": len(groups[style]), "sequence_length": int(cb["ids"].shape[1]),
                "marker": cb["marker"], "anchor": _anchor_position(cb, nb),
            }
        anchors = {x["anchor"] for x in query_contracts.values()}
        if len(anchors) != 1:
            raise ValueError(f"naturalized anchor varies by query: {style}")
        contracts[style] = query_contracts
    result = {
        "stage": "delta_paper1_closeout_preflight",
        "protocol_version": PROTOCOL_VERSION,
        "model_path": model_path, "resolved_model_path": resolved,
        "standard": {"n": len(standard), "indices": indices,
                     "anchor": std_anchor,
                     "sequence_length": int(std_clean["ids"].shape[1])},
        "naturalized": {"n": len(rows), "source": NAT_SOURCE,
                        "target": NAT_TARGET, "contracts": contracts},
        "verdict": "PREFLIGHT_PASS",
    }
    path = os.path.join(out_dir, "results_paper1_closeout_preflight.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"PAPER1 CLOSEOUT PREFLIGHT PASS standard={len(standard)} "
        f"naturalized={len(rows)} styles={list(contracts)}")
    return result


@torch.no_grad()
def run_delta_paper1_closeout(model_path, out_dir, model_key="qwen14b_closeout",
                              quantization="awq", device_map=None,
                              max_memory=None, n_world=N_ROWS, layers=LAYERS,
                              n_null=N_NULL, random_seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    requested = LAYERS if layers is None else layers
    layers = [int(x) for x in requested
              if 0 <= int(x) < model_num_hidden_layers(model)]
    if not layers:
        raise ValueError("no requested closeout layers exist")
    log("=== PAPER1 CLOSEOUT exact ac-only ===")
    exact = _exact_ac_only(model, tok, dev, layers, n_world)
    log("=== PAPER1 CLOSEOUT naturalized surface ===")
    naturalized = _naturalized_arm(
        model, tok, dev, layers, n_world, n_null, random_seed)
    verdict = ("PAPER1_EXPERIMENTS_CLOSED" if
               exact["verdict"] == "AC_ONLY_SUFFICIENT"
               and naturalized["verdict"] == "NATURALIZED_CONFIRMED"
               else "PAPER1_CLOSEOUT_BOUNDARY")
    result = {
        "stage": "delta_paper1_closeout",
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key, "model_path": model_path,
        "quantization": quantization, "layers": layers,
        "n_world": n_world, "n_null": n_null,
        "random_seed": random_seed, "exact_ac_only": exact,
        "naturalized": naturalized, "verdict": verdict,
    }
    path = os.path.join(out_dir, f"results_delta_paper1_closeout_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"PAPER1 CLOSEOUT verdict={verdict} artifact={path}")
    return result
