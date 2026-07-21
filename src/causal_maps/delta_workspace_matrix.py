"""Workspace-write matrix — the decisive generality test (one kernel per model).

Pre-registered CAUSAL_MAPS_LOG.md 2026-07-14. Claim: a value-state extracted
from a NEUTRAL carrier ("Here is a token: v.") and written at the value address
of a task prompt is CONSUMED BY DOWNSTREAM COMPUTATION as if the prompt said v —
across tasks that compute over the value (not merely retrieve it) and across
model families.

Distinction from function vectors (Todd et al.): FV installs the PROGRAM; this
installs the DATA the in-context program consumes. Our program-write attempt
(add→subtract) was NULL; the data-write claim is the live one.

Cells (digits only — single tokens across families):
  retrieve  X = v. What is the value of X?                     -> v      (control)
  add2      X = v. Y = X + 2. What is the value of Y?          -> v+2
  sub1      X = v. Y = X - 1. What is the value of Y?          -> v-1
  max5      X = v. Y = 5. Which is larger ... ? (number)       -> max(v,5)
  gt5label  X = v. If X > 5 answer north else south.           -> north/south

Per row (a -> b, wrong c): CLEAN, NATURAL, ADD (z_b - z_a at value token,
layer L), WRONG (z_c - z_a; must track its OWN f(c)), EMB (embedding diff,
report-only), 50 shared norm-matched random writes. Per-cell G0 (a dead cell
never kills the kernel). Gates frozen in the log entry.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs, forward_with_add

N_ROWS = 12
N_NULL = 50
PRIMER = "Answer:"          # base cut point (no trailing space)
JOINER = "Answer: "         # canonical natural rendering of primer + answer
GATES = {"g0": 0.90, "w1_ratio": 0.70, "w2_acc": 0.80, "w3_p": 0.02,
         "w4_acc": 0.80}

DIGITS = list(range(1, 10))


# --------------------------------------------------------------------------- #
# task definitions
# --------------------------------------------------------------------------- #
def _t_retrieve(v):
    return f"X = {v}. What is the value of X?", str(v)


def _t_add2(v):
    return f"X = {v}. Y = X + 2. What is the value of Y?", str(v + 2)


def _t_sub1(v):
    return f"X = {v}. Y = X - 1. What is the value of Y?", str(v - 1)


def _t_max5(v):
    return (f"X = {v}. Y = 5. Which is larger, X or Y? Answer with the number.",
            str(max(v, 5)))


def _t_gt5(v):
    return (f"X = {v}. If X is greater than 5, answer north. Otherwise answer "
            f"south.", "north" if v > 5 else "south")


TASKS = {
    # name: (fn, pool of legal v, kind)
    "retrieve": (_t_retrieve, DIGITS, "compute_ne"),   # f(c) != f(a),f(b) possible
    "add2": (_t_add2, [v for v in DIGITS if v + 2 <= 9], "compute_ne"),
    "sub1": (_t_sub1, [v for v in DIGITS if v - 1 >= 1], "compute_ne"),
    "max5": (_t_max5, [v for v in DIGITS if v != 5], "max5"),
    "gt5label": (_t_gt5, [v for v in DIGITS if v != 5], "same_side"),
}
COMPUTE_CELLS = ("add2", "sub1", "max5", "gt5label")


def _make_rows(name, rng, n_rows=N_ROWS):
    """Rows (a, b, c) with f(a) != f(b); wrong c per task kind."""
    fn, pool, kind = TASKS[name]
    rows, seen = [], set()
    guard = 0
    while len(rows) < n_rows and guard < 20000:
        guard += 1
        if kind == "max5":
            lo = [v for v in pool if v < 5]
            hi = [v for v in pool if v > 5]
            if rng.random() < 0.5:
                a, b = int(rng.choice(lo)), int(rng.choice(hi))
            else:
                a, b = int(rng.choice(hi)), int(rng.choice(lo))
        elif kind == "same_side":
            lo = [v for v in pool if v < 5]
            hi = [v for v in pool if v > 5]
            if rng.random() < 0.5:
                a, b = int(rng.choice(lo)), int(rng.choice(hi))
            else:
                a, b = int(rng.choice(hi)), int(rng.choice(lo))
        else:
            a, b = (int(x) for x in rng.choice(pool, size=2, replace=False))
        fa, fb = fn(a)[1], fn(b)[1]
        if fa == fb or (a, b) in seen:
            continue
        # wrong value c
        if kind == "same_side":
            side = [v for v in pool if (v > 5) == (a > 5) and v != a]
            if not side:
                continue
            c = int(rng.choice(side))                # f(c) == f(a): no-flip gate
        else:
            cands = [v for v in pool
                     if v not in (a, b) and fn(v)[1] not in (fa, fb)]
            if not cands:
                continue
            c = int(rng.choice(cands))
        seen.add((a, b))
        rows.append((a, b, c))
    assert len(rows) == n_rows, f"{name}: only {len(rows)} rows"
    return rows


# --------------------------------------------------------------------------- #
# tokenizer contract (mechanical; no model forwards)
# --------------------------------------------------------------------------- #
def _chat(tok, user, primer):
    t = tok.apply_chat_template([{"role": "user", "content": user}],
                                tokenize=False, add_generation_prompt=True)
    return t + primer


def _enc(tok, text):
    return tok.encode(text, add_special_tokens=False)


def _common_prefix(seqs):
    p = []
    for toks in zip(*seqs):
        if len(set(toks)) != 1:
            break
        p.append(toks[0])
    return p


def tokenizer_preflight(tok):
    """Mechanical answer contract, no model forwards.

    For each task: base = enc(chat + "Answer:"); for each answer a in the
    task's answer set, cont(a) = enc(chat + "Answer: " + a)[len(base):]
    (the canonical BPE/SP path of the natural string). The COMMON token
    prefix across the answer set (e.g. Qwen's standalone space 220) is
    appended to every prompt; scoring happens at the first token where the
    answers diverge. Requires: base is a prefix of every full encoding, and
    the diverging ids are distinct within the task."""
    report = {"primer": repr(PRIMER), "joiner": repr(JOINER), "tasks": {}}
    neut = [_enc(tok, _chat(tok, f"Here is a token: {v}.", "")) for v in DIGITS]
    lens = {len(e) for e in neut}
    assert len(lens) == 1, f"neutral pool non-uniform: {lens}"
    diffs = [i for i, (x, y) in enumerate(zip(neut[0], neut[1])) if x != y]
    assert len(diffs) == 1, f"neutral prompts differ at {len(diffs)} positions"
    report["neutral_pos"] = diffs[0]
    for name, (fn, pool, _k) in TASKS.items():
        e = [_enc(tok, _chat(tok, fn(v)[0], PRIMER)) for v in pool]
        lens = {len(x) for x in e}
        assert len(lens) == 1, f"{name}: non-uniform lengths {lens}"
        d = [i for i, (x, y) in enumerate(zip(e[0], e[1])) if x != y]
        assert len(d) == 1, f"{name}: prompts differ at {len(d)} positions"
        chat0 = _chat(tok, fn(pool[0])[0], "")
        base = _enc(tok, chat0 + PRIMER)
        conts = {}
        for a in sorted({fn(v)[1] for v in pool}):
            full = _enc(tok, chat0 + JOINER + a)
            assert full[:len(base)] == base, f"{name}/{a}: base not a prefix"
            conts[a] = full[len(base):]
        common = _common_prefix(list(conts.values()))
        amap = {}
        for a, c in conts.items():
            assert len(c) > len(common), f"{name}/{a}: no diverging token"
            amap[a] = c[len(common)]
        assert len(set(amap.values())) == len(amap), f"{name}: ids collide"
        report["tasks"][name] = {"S": len(e[0]), "val_pos": d[0],
                                 "common_prefix": common, "answer_ids": amap}
    return report


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_delta_workspace_matrix(model_path, out_dir, model_key="model",
                               quantization="8bit", device_map=None, seed=0,
                               layer=2, layer_candidates=None, n_null=N_NULL,
                               n_rows=N_ROWS, model=None, tok=None):
    os.makedirs(out_dir, exist_ok=True)
    if "*" in model_path:                       # Kaggle model mounts vary in
        import glob as _glob                    # nesting; resolve via glob
        hits = sorted(_glob.glob(model_path, recursive=True))
        assert hits, f"model_path glob matched nothing: {model_path}"
        mp = hits[0]
        if os.path.basename(mp) == "config.json":
            mp = os.path.dirname(mp)
        log(f"model_path glob {model_path} -> {mp}")
        model_path = mp
    if model is None or tok is None:            # battery injects a live model
        model, tok = load_model_and_tokenizer(model_path,
                                              device_map=device_map,
                                              quantization=quantization)
    dev = input_device(model)
    rng = np.random.default_rng(seed)
    torch_rng = torch.Generator().manual_seed(seed + 130363)
    pre = tokenizer_preflight(tok)
    log(f"workspace_matrix[{model_key}]: contract OK "
        f"(common prefixes { {n: d['common_prefix'] for n, d in pre['tasks'].items()} }) "
        f"neutral_pos={pre['neutral_pos']} layer={layer} "
        f"candidates={layer_candidates}")

    # ---- neutral donor states z_v at every layer we might use --------------
    neut_texts = [_chat(tok, f"Here is a token: {v}.", "") for v in DIGITS]
    ids_n = torch.tensor([_enc(tok, t) for t in neut_texts], dtype=torch.long,
                         device=dev)
    cache_n = cache_layer_outputs(model, ids_n, torch.ones_like(ids_n),
                                  to_cpu=True)
    npos = pre["neutral_pos"]
    layers_needed = sorted(set([layer] + list(layer_candidates or [])))
    Z = {L: {v: cache_n[L][i, npos, :].float()
             for i, v in enumerate(DIGITS)} for L in layers_needed}
    emb_w = model.get_input_embeddings().weight.detach()

    @torch.no_grad()
    def margins(ids, am, fa_ids, fb_ids, ans_ids, delta=None, L=None, pos=None):
        """Return (margin fb-fa per row, argmax-token per row over ans_ids)."""
        if delta is None:
            lg = last_token_logits(model, ids, am)
        else:
            lg = forward_with_add(model, ids, am, L, pos, delta)
        lg = lg.float()
        n = ids.shape[0]
        ar = torch.arange(n, device=lg.device)
        m = (lg[ar, fb_ids] - lg[ar, fa_ids]).cpu().numpy()
        sub = lg[:, ans_ids]                     # [B, nAns]
        pick = sub.argmax(-1).cpu().numpy()      # index into ans_ids
        return m, pick

    def run_cell(name, L):
        fn, pool, kind = TASKS[name]
        rows = _make_rows(name, rng, n_rows)
        info = pre["tasks"][name]
        vpos = info["val_pos"]
        common = list(info["common_prefix"])
        amap = info["answer_ids"]
        ans_keys = sorted(amap)
        ans_ids = torch.tensor([amap[k] for k in ans_keys], device=dev)
        kidx = {k: i for i, k in enumerate(ans_keys)}

        def batch(vs):
            e = [_enc(tok, _chat(tok, fn(v)[0], PRIMER)) + common for v in vs]
            t = torch.tensor(e, dtype=torch.long, device=dev)
            return t, torch.ones_like(t)

        A = [r[0] for r in rows]
        B = [r[1] for r in rows]
        C = [r[2] for r in rows]
        ids_c, am_c = batch(A)
        ids_nat, am_nat = batch(B)
        fa = torch.tensor([amap[fn(a)[1]] for a in A], device=dev)
        fb = torch.tensor([amap[fn(b)[1]] for b in B], device=dev)
        fa_k = np.array([kidx[fn(a)[1]] for a in A])
        fb_k = np.array([kidx[fn(b)[1]] for b in B])
        fc_k = np.array([kidx[fn(c)[1]] for c in C])

        m_clean, p_clean = margins(ids_c, am_c, fa, fb, ans_ids)
        m_nat, p_nat = margins(ids_nat, am_nat, fa, fb, ans_ids)
        g0_clean = float((p_clean == fa_k).mean())
        g0_nat = float((p_nat == fb_k).mean())
        cell = {"rows": rows, "layer": int(L),
                "g0_clean": g0_clean, "g0_natural": g0_nat}
        if min(g0_clean, g0_nat) < GATES["g0"]:
            cell.update({"verdict": "INELICITABLE"})
            log(f"  [{name}] G0 fail clean={g0_clean:.0%} nat={g0_nat:.0%}")
            return cell

        d_add = torch.stack([Z[L][b] - Z[L][a] for a, b, _ in rows])
        d_wrong = torch.stack([Z[L][c] - Z[L][a] for a, _, c in rows])
        tok_a = ids_c[:, vpos]
        tok_b = ids_nat[:, vpos]
        d_emb = (emb_w[tok_b] - emb_w[tok_a]).float().cpu()

        m_add, p_add = margins(ids_c, am_c, fa, fb, ans_ids, d_add, L, vpos)
        m_wr, p_wr = margins(ids_c, am_c, fa, fb, ans_ids, d_wrong, L, vpos)
        m_emb, p_emb = margins(ids_c, am_c, fa, fb, ans_ids, d_emb, L, vpos)

        nat_rows = m_nat - m_clean
        add_rows = m_add - m_clean
        nat_eff = float(nat_rows.mean())
        add_eff = float(add_rows.mean())
        wr_own = float((p_wr == fc_k).mean())
        norms = d_add.norm(dim=1, keepdim=True)
        nulls = []
        for _ in range(n_null):
            r = torch.randn(d_add.shape, generator=torch_rng)
            r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-8) * norms
            nm, _ = margins(ids_c, am_c, fa, fb, ans_ids, r, L, vpos)
            nulls.append(float((nm - m_clean).mean()))
            hb.step()
        pv = permutation_pvalue(add_eff, np.asarray(nulls), "greater")
        ratio = add_eff / nat_eff if abs(nat_eff) > 1e-6 else float("nan")
        w1 = np.isfinite(ratio) and ratio >= GATES["w1_ratio"]
        w2 = float((p_add == fb_k).mean())
        w3 = pv < GATES["w3_p"]
        w4 = wr_own >= GATES["w4_acc"]
        cell.update({
            "natural_effect": nat_eff, "add_effect": add_eff,
            "natural_effect_rows": nat_rows.tolist(),
            "add_effect_rows": add_rows.tolist(),
            "ratio": float(ratio), "add_target_acc": w2,
            "p": float(pv), "null_mean": float(np.mean(nulls)),
            "wrong_own_target_acc": wr_own,
            "emb_effect": float((m_emb - m_clean).mean()),
            "emb_target_acc": float((p_emb == fb_k).mean()),
            "gates": {"W1": bool(w1), "W2": bool(w2 >= GATES["w2_acc"]),
                      "W3": bool(w3), "W4": bool(w4)},
        })
        cell["verdict"] = ("PASS" if (w1 and w2 >= GATES["w2_acc"] and w3 and w4)
                           else "FAIL")
        log(f"  [{name}] {cell['verdict']} nat={nat_eff:+.1f} add={add_eff:+.1f} "
            f"ratio={ratio:.2f} acc={w2:.0%} p={pv:.3f} wrong_own={wr_own:.0%} "
            f"emb={cell['emb_effect']:+.2f}/{cell['emb_target_acc']:.0%}")
        return cell

    # ---- optional layer selection on retrieve train rows (Phi etc.) --------
    chosen = layer
    if layer_candidates:
        fn, pool, _ = TASKS["retrieve"]
        tr = _make_rows("retrieve", np.random.default_rng(seed + 99))[:4]
        info = pre["tasks"]["retrieve"]
        amap = info["answer_ids"]
        e = [_enc(tok, _chat(tok, fn(a)[0], PRIMER)) + list(info["common_prefix"])
             for a, _, _ in tr]
        ids_t = torch.tensor(e, dtype=torch.long, device=dev)
        am_t = torch.ones_like(ids_t)
        fa = torch.tensor([amap[fn(a)[1]] for a, _, _ in tr], device=dev)
        fb = torch.tensor([amap[fn(b)[1]] for _, b, _ in tr], device=dev)
        ans_ids = torch.tensor([amap[k] for k in sorted(amap)], device=dev)
        base, _ = margins(ids_t, am_t, fa, fb, ans_ids)
        best, best_eff = None, -1e9
        for L in layer_candidates:
            d = torch.stack([Z[L][b] - Z[L][a] for a, b, _ in tr])
            m, _ = margins(ids_t, am_t, fa, fb, ans_ids, d, L,
                           info["val_pos"])
            eff = float((m - base).mean())
            log(f"  layer-select L{L}: retrieve-train effect {eff:+.2f}")
            if eff > best_eff:
                best, best_eff = L, eff
        chosen = best
        log(f"  layer selected: L{chosen} (train rows excluded from scoring)")

    hb = Heartbeat(len(TASKS) * n_null, f"workspace_{model_key}", every_sec=20,
                   out_dir=out_dir)
    cells = {}
    for name in TASKS:
        cells[name] = run_cell(name, chosen)
    hb.done()

    compute_pass = [n for n in COMPUTE_CELLS if cells[n].get("verdict") == "PASS"]
    retrieve_ok = cells["retrieve"].get("verdict") == "PASS"
    inelig = [n for n in TASKS if cells[n].get("verdict") == "INELICITABLE"]
    if retrieve_ok and len(compute_pass) >= 3:
        verdict = "WORKSPACE_GENERAL"
    elif retrieve_ok and len(compute_pass) >= 1:
        verdict = "WORKSPACE_PARTIAL"
    elif retrieve_ok:
        verdict = "RETRIEVE_ONLY"
    else:
        verdict = "WORKSPACE_INELICITABLE" if inelig else "WORKSPACE_DEAD"

    results = {"stage": "delta_workspace_matrix", "model_key": model_key,
               "model_path": model_path, "layer_used": int(chosen),
               "seed": seed, "n_rows": n_rows, "n_null": n_null,
               "gates": GATES, "preflight": pre, "cells": cells,
               "compute_cells_pass": compute_pass,
               "ineligible_cells": inelig, "verdict": verdict}
    fp = os.path.join(out_dir, f"results_delta_workspace_matrix_{model_key}.json")
    with open(fp, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT workspace_matrix[{model_key}]: {verdict} | "
        f"compute_pass={compute_pass} inelig={inelig} L={chosen}")
    return results
