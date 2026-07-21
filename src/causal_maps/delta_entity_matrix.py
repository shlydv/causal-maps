"""Entity / world-state editing with inference over the edit.

Pre-registered CAUSAL_MAPS_LOG.md 2026-07-14. The significance test: does the
workspace-write mechanism extend from symbols to WORLD STATE, where the answer
is reachable only THROUGH an in-context rule (edited vocab ⊥ answer vocab)?

  KEYS: Alice has the {a} key. Bob has the {b} key. The {a} key opens the
        {oa}. The {b} key opens the {ob}. The {w} key opens the {ow}.
    retrieve: What color is Alice's key?   -> a      (control)
    twohop:   What can Alice open?         -> oa     (THE cell)
    other:    What can Bob open?           -> ob     (edit Alice; must not move)
  CITY: Alice is in {a}. Bob is in {b}. People in {a} say {sa}. ...
    retrieve: Where is Alice? / twohop: What does Alice say? / other: Bob.

Write = neutral-carrier state z(w) - z(a) at Alice's possession/location
token (rule lines untouched). Natural CF: text with only that token swapped.
WRONG write z(b) - z(a) must produce ITS OWN consequence. One kernel runs
Qwen then Mistral sequentially (mounted paths). Per-cell G0; frozen gates.
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import torch

from .delta_workspace_matrix import _chat, _common_prefix, _enc, JOINER, PRIMER
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs, forward_with_add

N_NULL = 30
N_ROWS = 10
G0 = 0.90
GATE_ACC = 0.80
GATE_RATIO = 0.70
GATE_P = 0.04
GATE_OTHER = 0.25

COLORS = ["red", "blue", "green", "black", "white", "brown", "pink", "gray"]
OBJECTS = ["shed", "barn", "gate", "door", "box", "safe", "chest", "cage"]
CITIES = ["Paris", "Rome", "Tokyo", "Delhi", "Cairo", "Oslo", "Lima", "Miami"]
SAYWORDS = ["north", "south", "left", "right", "true", "false", "cat", "dog"]

FAMILIES = {
    "keys": {
        "world": ("Alice has the {A} key. Bob has the {b} key. "
                  "The {a} key opens the {oa}. The {b} key opens the {ob}. "
                  "The {w} key opens the {ow}."),
        "vals": "colors", "cons": "objects",
        "q_retrieve": "What color is Alice's key?",
        "q_twohop": "What can Alice open?",
        "q_other": "What can Bob open?",
    },
    "city": {
        "world": ("Alice is in {A}. Bob is in {b}. "
                  "People in {a} say {oa}. People in {b} say {ob}. "
                  "People in {w} say {ow}."),
        "vals": "cities", "cons": "saywords",
        "q_retrieve": "Where is Alice?",
        "q_twohop": "What does Alice say?",
        "q_other": "What does Bob say?",
    },
}
POOLS = {"colors": COLORS, "objects": OBJECTS,
         "cities": CITIES, "saywords": SAYWORDS}


def _resolve(model_path):
    if "*" in model_path:
        import glob as _glob
        hits = sorted(_glob.glob(model_path, recursive=True))
        assert hits, f"model_path glob matched nothing: {model_path}"
        mp = hits[0]
        if os.path.basename(mp) == "config.json":
            mp = os.path.dirname(mp)
        return mp
    return model_path


def _text(fam, row, question, alice_val=None):
    """World text + question. {A} = Alice's live token (editable); {a} = the
    rule mention of her clean value (NEVER swapped — rules stay fixed)."""
    spec = FAMILIES[fam]
    A = alice_val if alice_val is not None else row["a"]
    world = spec["world"].format(A=A, a=row["a"], b=row["b"], w=row["w"],
                                 oa=row["oa"], ob=row["ob"], ow=row["ow"])
    return f"{world} {question}"


def _filter_pools(tok, fam):
    """Mechanical: value words must be single-token IN CONTEXT (swap preserves
    positions/length); answer words need distinct first tokens (contract)."""
    spec = FAMILIES[fam]
    vals = []
    base_row = None
    for cand in POOLS[spec["vals"]]:
        pool0 = [v for v in POOLS[spec["vals"]] if v != cand][:2]
        cons0 = POOLS[spec["cons"]][:3]
        row = {"a": cand, "b": pool0[0], "w": pool0[1],
               "oa": cons0[0], "ob": cons0[1], "ow": cons0[2]}
        t1 = _enc(tok, _chat(tok, _text(fam, row, spec["q_retrieve"]), PRIMER))
        row2 = dict(row)
        t2 = _enc(tok, _chat(tok, _text(fam, row2, spec["q_retrieve"],
                                        alice_val=pool0[0]), PRIMER))
        if len(t1) == len(t2) and sum(1 for x, y in zip(t1, t2) if x != y) == 1:
            vals.append(cand)
            base_row = base_row or row
    return vals, base_row


def _contract(tok, fam, question, answers, row):
    """Canonical-continuation answer ids for `answers` on this family/question."""
    chat0 = _chat(tok, _text(fam, row, question), "")
    base = _enc(tok, chat0 + PRIMER)
    conts = {}
    for a in answers:
        full = _enc(tok, chat0 + JOINER + a)
        assert full[:len(base)] == base, f"{fam}/{a}: base not a prefix"
        conts[a] = full[len(base):]
    common = _common_prefix(list(conts.values()))
    amap = {}
    for a, c in conts.items():
        assert len(c) > len(common), f"{fam}/{a}: no diverging token"
        amap[a] = c[len(common)]
    assert len(set(amap.values())) == len(amap), f"{fam}: answer ids collide"
    return {"common": common, "amap": amap}


def _make_rows(fam, vals, tok, rng, n_rows=N_ROWS):
    spec = FAMILIES[fam]
    cons_pool = POOLS[spec["cons"]]
    rows = []
    seen = set()
    guard = 0
    while len(rows) < n_rows and guard < 20000:
        guard += 1
        if len(vals) < 3:
            raise RuntimeError(f"{fam}: <3 usable value words: {vals}")
        a, b, w = (str(x) for x in rng.choice(vals, size=3, replace=False))
        oa, ob, ow = (str(x) for x in rng.choice(cons_pool, size=3,
                                                 replace=False))
        key = (a, b, w, oa, ob, ow)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"a": a, "b": b, "w": w, "oa": oa, "ob": ob, "ow": ow})
    assert len(rows) == n_rows, f"{fam}: only {len(rows)} unique worlds"
    return rows


def _run_model(model_key, model_path, quantization, layer, n_null, seed, hb,
               max_memory=None, n_rows=N_ROWS, model=None, tok=None):
    owns_model = model is None or tok is None
    if owns_model:
        model, tok = load_model_and_tokenizer(_resolve(model_path),
                                              quantization=quantization,
                                              max_memory=max_memory)
    dev = input_device(model)
    rng = np.random.default_rng(seed)
    torch_rng = torch.Generator().manual_seed(seed + 104729)
    out = {"model_key": model_key, "quantization": quantization,
           "families": {}}

    # neutral donors for every value word that survives filtering (per family)
    def donors(words):
        texts = [_chat(tok, f"Here is a word: {v}.", "") for v in words]
        enc = [_enc(tok, t) for t in texts]
        assert len({len(e) for e in enc}) == 1, "neutral pool non-uniform"
        ids = torch.tensor(enc, dtype=torch.long, device=dev)
        cache = cache_layer_outputs(model, ids, torch.ones_like(ids),
                                    to_cpu=True)
        d = [i for i, (x, y) in enumerate(zip(enc[0], enc[1])) if x != y]
        assert len(d) == 1
        return {v: cache[layer][i, d[0], :].float()
                for i, v in enumerate(words)}

    @torch.no_grad()
    def lg_of(ids, am, delta=None, pos=None):
        if delta is None:
            return last_token_logits(model, ids, am).float()
        return forward_with_add(model, ids, am, layer, pos, delta).float()

    for fam, spec in FAMILIES.items():
        vals, base_row = _filter_pools(tok, fam)
        log(f"  [{model_key}/{fam}] usable values: {vals}")
        if len(vals) < 3:
            out["families"][fam] = {"verdict": "INELICITABLE",
                                    "reason": f"pool too small: {vals}"}
            continue
        Z = donors(vals)
        rows = _make_rows(fam, vals, tok, rng, n_rows)

        # value-slot position: diff clean vs natural on row 0
        q_ret = spec["q_retrieve"]
        e_c = _enc(tok, _chat(tok, _text(fam, rows[0], q_ret), PRIMER))
        e_n = _enc(tok, _chat(tok, _text(fam, rows[0], q_ret,
                                         alice_val=rows[0]["w"]), PRIMER))
        dpos = [i for i, (x, y) in enumerate(zip(e_c, e_n)) if x != y]
        assert len(dpos) == 1, f"{fam}: swap changed {len(dpos)} positions"
        vpos = dpos[0]

        fam_res = {"rows": rows, "val_pos": vpos, "values": vals}
        cells = {}
        for cell in ("retrieve", "twohop", "other"):
            q = spec[f"q_{cell}"]
            if cell == "retrieve":
                answers = vals
                tgt_clean = [r["a"] for r in rows]
                tgt_nat = [r["w"] for r in rows]
                tgt_wrong = [r["b"] for r in rows]
            else:
                answers = sorted({r[k] for r in rows
                                  for k in ("oa", "ob", "ow")})
                if cell == "twohop":
                    tgt_clean = [r["oa"] for r in rows]
                    tgt_nat = [r["ow"] for r in rows]
                    tgt_wrong = [r["ob"] for r in rows]
                else:                          # other-entity: Bob's answer
                    tgt_clean = [r["ob"] for r in rows]
                    tgt_nat = [r["ob"] for r in rows]   # must NOT move
                    tgt_wrong = [r["ob"] for r in rows]
            con = _contract(tok, fam, q, answers, rows[0])
            amap = con["amap"]
            pool_ids = torch.tensor([amap[a] for a in answers], device=dev)
            kidx = {a: i for i, a in enumerate(answers)}

            def batch(alice_vals):
                e = [_enc(tok, _chat(tok, _text(fam, r, q, alice_val=v),
                                     PRIMER)) + list(con["common"])
                     for r, v in zip(rows, alice_vals)]
                assert len({len(x) for x in e}) == 1
                t = torch.tensor(e, dtype=torch.long, device=dev)
                return t, torch.ones_like(t)

            ids_c, am_c = batch([r["a"] for r in rows])
            ids_n, am_n = batch([r["w"] for r in rows])
            n = len(rows)
            ar = torch.arange(n, device=dev)
            t_c = torch.tensor([amap[x] for x in tgt_clean], device=dev)
            t_n = torch.tensor([amap[x] for x in tgt_nat], device=dev)
            k_c = np.array([kidx[x] for x in tgt_clean])
            k_n = np.array([kidx[x] for x in tgt_nat])
            k_w = np.array([kidx[x] for x in tgt_wrong])

            def margin(lg):
                return (lg[ar, t_n] - lg[ar, t_c]).cpu().numpy()

            def pick(lg):
                return lg[:, pool_ids].argmax(-1).cpu().numpy()

            lg_c, lg_n = lg_of(ids_c, am_c), lg_of(ids_n, am_n)
            g0c = float((pick(lg_c) == k_c).mean())
            g0n = float((pick(lg_n) == k_n).mean())
            c_res = {"g0_clean": g0c, "g0_natural": g0n, "n_rows": n}
            if min(g0c, g0n) < G0:
                c_res["verdict"] = "INELICITABLE"
                cells[cell] = c_res
                log(f"  [{model_key}/{fam}/{cell}] G0 fail "
                    f"{g0c:.0%}/{g0n:.0%}")
                continue
            d_add = torch.stack([Z[r["w"]] - Z[r["a"]] for r in rows])
            d_wr = torch.stack([Z[r["b"]] - Z[r["a"]] for r in rows])
            lg_add = lg_of(ids_c, am_c, d_add, vpos)
            lg_wr = lg_of(ids_c, am_c, d_wr, vpos)
            m_c = margin(lg_c)
            nat_rows = margin(lg_n) - m_c
            add_rows = margin(lg_add) - m_c
            nat_eff = float(nat_rows.mean())
            add_eff = float(add_rows.mean())
            acc_add = float((pick(lg_add) == k_n).mean())
            acc_wr = float((pick(lg_wr) == k_w).mean())
            if cell == "other":
                # specificity: Bob's answer margin must not move
                shift = abs(add_eff)
                c_res.update({"bob_shift": shift})
                c_res["verdict"] = "PENDING_TWOHOP_SCALE"
                cells[cell] = c_res
                log(f"  [{model_key}/{fam}/other] bob_shift={shift:.2f}")
                continue
            ratio = add_eff / nat_eff if abs(nat_eff) > 1e-6 else float("nan")
            norms = d_add.norm(dim=1, keepdim=True)
            nulls = []
            for _ in range(n_null):
                rnd = torch.randn(d_add.shape, generator=torch_rng)
                rnd = (rnd / rnd.norm(dim=1, keepdim=True).clamp(min=1e-8)
                       * norms)
                nulls.append(float((margin(lg_of(ids_c, am_c, rnd, vpos))
                                    - m_c).mean()))
                hb.step()
            p = permutation_pvalue(add_eff, np.asarray(nulls), "greater")
            ok = (acc_add >= GATE_ACC and np.isfinite(ratio)
                  and ratio >= GATE_RATIO and p < GATE_P
                  and acc_wr >= GATE_ACC)
            c_res.update({"natural_effect": nat_eff, "add_effect": add_eff,
                          "natural_effect_rows": nat_rows.tolist(),
                          "add_effect_rows": add_rows.tolist(),
                          "ratio": float(ratio), "add_target_acc": acc_add,
                          "wrong_own_target_acc": acc_wr, "p": float(p),
                          "verdict": "PASS" if ok else "FAIL"})
            cells[cell] = c_res
            log(f"  [{model_key}/{fam}/{cell}] {c_res['verdict']} "
                f"nat={nat_eff:+.1f} add={add_eff:+.1f} ratio={ratio:.2f} "
                f"acc={acc_add:.0%} wrong_own={acc_wr:.0%} p={p:.3f}")

        # score other-entity against the twohop effect scale
        if (cells.get("other", {}).get("verdict") == "PENDING_TWOHOP_SCALE"
                and "add_effect" in cells.get("twohop", {})):
            scale = abs(cells["twohop"]["add_effect"])
            shift = cells["other"]["bob_shift"]
            ok = scale > 1e-6 and shift <= GATE_OTHER * scale
            cells["other"]["shift_over_twohop"] = (shift / scale
                                                   if scale > 1e-6 else None)
            cells["other"]["verdict"] = "PASS" if ok else "FAIL"
            log(f"  [{model_key}/{fam}/other] {cells['other']['verdict']} "
                f"shift/twohop={cells['other']['shift_over_twohop']}")
        fam_res["cells"] = cells
        fam_res["pass"] = all(cells.get(c, {}).get("verdict") == "PASS"
                              for c in ("retrieve", "twohop", "other"))
        out["families"][fam] = fam_res

    passed = [f for f, r in out["families"].items() if r.get("pass")]
    if len(passed) == 2:
        out["verdict"] = "WORLD_STATE_GENERAL"
    elif len(passed) == 1:
        out["verdict"] = "WORLD_STATE_PARTIAL"
    elif any(r.get("cells", {}).get("retrieve", {}).get("verdict") == "PASS"
             for r in out["families"].values()):
        out["verdict"] = "RETRIEVE_ONLY"
    else:
        out["verdict"] = "WORLD_STATE_DEAD_OR_INELICITABLE"
    log(f"  MODEL VERDICT [{model_key}]: {out['verdict']} "
        f"(families passed: {passed})")
    if owns_model:                      # never free a battery-injected model
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return out


def run_delta_entity_matrix(qwen_path=None, mistral_path=None, out_dir=".",
                            quantization="8bit", layer=2, seed=0,
                            n_null=N_NULL, models=None, max_memory=None):
    """models: optional list of {"key","path"} dicts (e.g. a single 14B run
    under the identical frozen protocol); default = the original Qwen+Mistral
    pair. Protocol, pools, gates, and seed are unchanged either way."""
    os.makedirs(out_dir, exist_ok=True)
    model_list = models or [{"key": "qwen7b", "path": qwen_path},
                            {"key": "mistral7b", "path": mistral_path}]
    hb = Heartbeat(len(model_list) * 2 * 2 * n_null, "entity_matrix",
                   every_sec=20, out_dir=out_dir)
    results = {"stage": "delta_entity_matrix", "layer": layer, "seed": seed,
               "n_null": n_null, "quantization": quantization, "models": {}}
    for i, m in enumerate(model_list):
        log(f"=== entity matrix phase {i + 1}: {m['key']} ===")
        results["models"][m["key"]] = _run_model(m["key"], m["path"],
                                                 quantization, layer, n_null,
                                                 seed, hb,
                                                 max_memory=max_memory)
    hb.done()
    verdicts = {k: v["verdict"] for k, v in results["models"].items()}
    if (set(verdicts) == {"qwen7b", "mistral7b"}
            and all(v == "WORLD_STATE_GENERAL" for v in verdicts.values())):
        results["verdict"] = "WORLD_STATE_EDITING_CONFIRMED"
    else:
        results["verdict"] = "|".join(f"{k}:{v}" for k, v in verdicts.items())
    with open(os.path.join(out_dir, "results_delta_entity_matrix.json"),
              "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT entity_matrix: {results['verdict']}")
    return results
