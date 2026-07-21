"""Structured-workspace causal interchange for bound, endogenous beliefs.

This experiment asks whether a transformer exposes a query-independent state
checkpoint containing *bound relations*, rather than only an unordered bag of
active concepts.  Event histories imply four private beliefs and two true
locations.  A checkpoint occurs before the question, so its activation cannot
depend on the requested readout.

The protocol is deliberately gated:

1. the model must solve CLEAN and natural-counterfactual histories;
2. matched checkpoint states must interchange in both directions;
3. a donor-mean relational delta must transfer to held-out histories and a
   held-out surface, with wrong-address and invariant-state controls;
4. two independently learned relation edits must compose;
5. an unbound location-concept direction and norm-matched random directions
   are reported as rival explanations.

No causal conclusion is drawn when an earlier gate fails.  See
STRUCTURED_WORKSPACE_PROTOCOL.md for the frozen interpretation.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_orchestration_screen import _normalize
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer
from .nulls import permutation_pvalue


PROTOCOL_VERSION = "2026-07-14-v1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCATIONS = ("Paris", "Rome", "Tokyo", "Delhi",
             "Cairo", "Oslo", "Lima", "Miami")
ADDRESSES = ("ac", "bc", "as", "bs")
MARKER = "STATECHECK"
N_ROWS = 5
N_NULL = 30
G0 = .80
G_ACC = .80
G_RATIO = (.60, 1.40)
LAYERS = (8, 12, 16, 20, 24, 26)
EPS = 1e-8


QUERY = {
    "belief_ac": ("Where does Alice believe the cube is?", "BELIEF", "ac"),
    "search_ac": ("Alice must retrieve the cube using only her own "
                  "information. Where does she search?", "SEARCH", "ac"),
    "tell_ac": ("Alice tells a teammate where she believes the cube is. "
                "What does she report?", "TELL", "ac"),
    "belief_as": ("Where does Alice believe the sphere is?", "BELIEF", "as"),
    "belief_bc": ("Where does Bob believe the cube is?", "BELIEF", "bc"),
    "belief_bs": ("Where does Bob believe the sphere is?", "BELIEF", "bs"),
    "search_bs": ("Bob must retrieve the sphere using only his own "
                  "information. Where does he search?", "SEARCH", "bs"),
    "truth_cube": ("Where is the cube actually?", "TRUTH", "tc"),
    "truth_sphere": ("Where is the sphere actually?", "TRUTH", "ts"),
}


def _counterfactual(rows, changes):
    """Return copied rows with only the specified latent variables changed."""
    out = []
    for row in rows:
        changed = dict(row)
        changed.update(changes)
        out.append(changed)
    return out


def _rows(source, target, address, split, n_rows=None):
    """Balanced histories for one address-specific source -> target edit.

    Three belief registers contain ``source`` and a fourth contains ``target``.
    Thus both concepts remain elsewhere after the edit.  A value-only/global
    replacement would incorrectly alter at least two untouched registers.

    n_rows=None keeps the frozen N_ROWS behaviour byte-for-byte. Larger
    n_rows (preprint widening, pre-registered) extends with the remaining
    ordered (tc, ts) pairs from `other`, never duplicating a history.
    """
    if source == target or source not in LOCATIONS or target not in LOCATIONS:
        raise ValueError("invalid source/target")
    if address not in ADDRESSES or split not in ("train", "test"):
        raise ValueError("invalid address/split")
    target_present = next(key for key in ("bs", "bc", "as", "ac")
                          if key != address)
    other = [x for x in LOCATIONS if x not in (source, target)]
    offset = 0 if split == "train" else 3
    n = N_ROWS if n_rows is None else n_rows
    base_pairs = [(other[(offset + i) % len(other)],
                   other[(offset + i + 2) % len(other)])
                  for i in range(min(n, N_ROWS))]
    if n > N_ROWS:
        extra = [(a, b) for a in other for b in other
                 if a != b and (a, b) not in base_pairs]
        base_pairs += extra[:n - len(base_pairs)]
    assert len(base_pairs) == n, f"only {len(base_pairs)} distinct histories"
    rows = []
    for tc, ts in base_pairs:
        row = {key: source for key in ADDRESSES}
        row[target_present] = target
        row["tc"], row["ts"] = tc, ts
        if row["tc"] == row["ts"]:
            raise AssertionError("truth locations collided")
        rows.append(row)
    return rows


def _composition_rows(split):
    """Worlds for simultaneous ac: Paris->Rome and bs: Tokyo->Delhi."""
    fillers = ("Cairo", "Oslo", "Lima", "Miami")
    offset = 0 if split == "train" else 1
    rows = []
    for i in range(N_ROWS):
        p = [(offset + i + j) % len(fillers) for j in range(4)]
        rows.append({
            "ac": "Paris", "bs": "Tokyo",
            "as": fillers[p[0]], "bc": fillers[p[1]],
            "tc": fillers[p[2]], "ts": fillers[p[3]],
        })
    return rows


def _expected(row, query):
    _question, command, field = QUERY[query]
    return f"{command} {row[field]}"


def _user(row, query, surface):
    question, command, _field = QUERY[query]
    rule = (
        "An observer believes an object's location is the last placement of "
        "that object the observer personally saw. Placements the observer did "
        "not see do not update that belief. The actual location is the final "
        "placement, whether observed or not."
    )
    if surface == "ledger":
        history = (
            f"Private event ledger: Alice alone saw the cube placed in {row['ac']}. "
            f"Bob alone later saw the cube placed in {row['bc']}. "
            f"Alice alone saw the sphere placed in {row['as']}. "
            f"Bob alone later saw the sphere placed in {row['bs']}. "
            f"Finally, while both were absent, the cube was placed in {row['tc']} "
            f"and the sphere was placed in {row['ts']}."
        )
    elif surface == "narrative":
        history = (
            f"Alice watched a worker leave the cube in {row['ac']}, then left. "
            f"Later Bob, without Alice present, watched it go to {row['bc']}. "
            f"Alice separately watched the sphere go to {row['as']}, then left. "
            f"Bob later watched the sphere go to {row['bs']}. "
            f"After both agents departed, unseen final moves put the cube in "
            f"{row['tc']} and the sphere in {row['ts']}."
        )
    else:
        raise ValueError(f"unknown surface: {surface}")
    return (
        f"Maintain the true world and each agent's private beliefs. {rule} "
        f"{history} Silently compute all six state variables now. {MARKER}.\n"
        f"Question: {question} Reply with exactly {command}, one space, and "
        "the location name. Do not add anything else."
    )


def _render(tok, row, query, surface):
    return tok.apply_chat_template(
        [{"role": "user", "content": _user(row, query, surface)}],
        tokenize=False, add_generation_prompt=True)


def _marker_position(tok, text):
    """Last token overlapping MARKER, using offsets with a safe fallback."""
    start = text.index(MARKER)
    end = start + len(MARKER)
    try:
        encoded = tok(text, add_special_tokens=False,
                      return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        hits = [i for i, (a, b) in enumerate(offsets)
                if a < end and b > start]
        if hits:
            return hits[-1]
    except (TypeError, KeyError, NotImplementedError):
        pass
    full = tok.encode(text, add_special_tokens=False)
    for needle in (" " + MARKER, MARKER):
        sub = tok.encode(needle, add_special_tokens=False)
        matches = [i for i in range(len(full) - len(sub) + 1)
                   if full[i:i + len(sub)] == sub]
        if len(matches) == 1:
            return matches[0] + len(sub) - 1
    raise ValueError("could not locate unique state marker")


def _common_prefix(seqs):
    out = []
    for values in zip(*seqs):
        if len(set(values)) != 1:
            break
        out.append(values[0])
    return out


def _batch(tok, rows, query, surface, dev, render_fn=None):
    """Teacher-force the answer prefix through the first location token.
    render_fn(row) optionally overrides the default prompt rendering (used by
    the verbalization protocol to insert a V line); contract logic unchanged."""
    texts = [(render_fn(row) if render_fn else _render(tok, row, query, surface))
             for row in rows]
    prefixes, maps, marker_positions = [], [], []
    command = QUERY[query][1]
    for text in texts:
        base = tok.encode(text, add_special_tokens=False)
        continuations = {}
        for location in LOCATIONS:
            full = tok.encode(text + f"{command} {location}",
                              add_special_tokens=False)
            if full[:len(base)] != base:
                raise ValueError(f"answer contract resegmented for {query}")
            continuations[location] = full[len(base):]
        common = _common_prefix(list(continuations.values()))
        amap = {}
        for location, continuation in continuations.items():
            if len(continuation) <= len(common):
                raise ValueError(f"no diverging answer token for {location}")
            amap[location] = continuation[len(common)]
        if len(set(amap.values())) != len(LOCATIONS):
            raise ValueError(f"location answer tokens collide for {query}")
        prefixes.append(base + common)
        maps.append(amap)
        marker_positions.append(_marker_position(tok, text))
    if len({len(x) for x in prefixes}) != 1:
        raise ValueError(f"nonuniform {surface}/{query} batch")
    if len(set(marker_positions)) != 1:
        raise ValueError(f"nonuniform marker positions: {marker_positions}")
    if any(amap != maps[0] for amap in maps[1:]):
        raise ValueError("answer-token map varies across rows")
    ids = torch.tensor(prefixes, dtype=torch.long, device=dev)
    return {
        "texts": texts, "ids": ids, "am": torch.ones_like(ids),
        "marker": marker_positions[0], "amap": maps[0],
    }


def _locations(rows, query):
    field = QUERY[query][2]
    return [row[field] for row in rows]


def _accuracy(logits, batch, expected):
    pool = torch.tensor([batch["amap"][x] for x in LOCATIONS])
    chosen = logits[:, pool].argmax(-1)
    gold = torch.tensor([LOCATIONS.index(x) for x in expected])
    return float((chosen == gold).float().mean())


def _switch_metrics(clean_logits, natural_logits, candidate_logits, batch,
                    source, target):
    source_ids = torch.tensor([batch["amap"][x] for x in source])
    target_ids = torch.tensor([batch["amap"][x] for x in target])
    clean_ld = _ld(clean_logits, target_ids, source_ids)
    natural_rows = _ld(natural_logits, target_ids, source_ids) - clean_ld
    effect_rows = _ld(candidate_logits, target_ids, source_ids) - clean_ld
    natural = float(natural_rows.mean())
    effect = float(effect_rows.mean())
    ratio = effect / natural if abs(natural) > EPS else None
    return {
        "natural_effect": natural, "effect": effect, "ratio": ratio,
        "target_acc": _accuracy(candidate_logits, batch, target),
        "positive_fraction": float((effect_rows > 0).float().mean()),
        "effect_rows": effect_rows.tolist(),
        "natural_rows": natural_rows.tolist(),
    }


def _switch_pass(metrics):
    ratio = metrics["ratio"]
    return bool(metrics["target_acc"] >= G_ACC
                and metrics["positive_fraction"] >= G_ACC
                and ratio is not None and G_RATIO[0] <= ratio <= G_RATIO[1])


def _encode_pair(model, tok, dev, clean_rows, natural_rows, query, surface,
                 capture_layers=()):
    clean = _batch(tok, clean_rows, query, surface, dev)
    natural = _batch(tok, natural_rows, query, surface, dev)
    if clean["ids"].shape != natural["ids"].shape:
        raise ValueError("clean/natural prompt shapes differ")
    if clean["marker"] != natural["marker"] or clean["amap"] != natural["amap"]:
        raise ValueError("clean/natural contract differs")
    positions = (clean["marker"],)
    cl, cc = _forward(model, clean["ids"], clean["am"], positions,
                      capture_layers)
    nl, nc = _forward(model, natural["ids"], natural["am"], positions,
                      capture_layers)
    return clean, natural, cl, nl, cc, nc


def _behavior(model, tok, dev, clean_rows, natural_rows, queries, surface):
    out = {}
    for query in queries:
        clean, natural, cl, nl, _cc, _nc = _encode_pair(
            model, tok, dev, clean_rows, natural_rows, query, surface)
        source = _locations(clean_rows, query)
        target = _locations(natural_rows, query)
        out[query] = {
            "clean_acc": _accuracy(cl, clean, source),
            "natural_acc": _accuracy(nl, natural, target),
        }
        log(f"structured G0 {surface}/{query}: "
            f"{out[query]['clean_acc']:.0%}/{out[query]['natural_acc']:.0%}")
    return out


def _behavior_pass(result):
    return all(min(row["clean_acc"], row["natural_acc"]) >= G0
               for group in result.values() for row in group.values())


def _unbound_direction(model, tok, dev, source, target, layer):
    """Processed unbound-concept rival; deliberately not claimed as J-lens."""
    def text(location):
        user = (
            f"Silently hold the unbound location concept {location} in mind. "
            f"{MARKER}. Then repeat it. Reply exactly BELIEF {location}."
        )
        return tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True)
    left, right = text(source), text(target)
    li = torch.tensor([tok.encode(left, add_special_tokens=False)],
                      dtype=torch.long, device=dev)
    ri = torch.tensor([tok.encode(right, add_special_tokens=False)],
                      dtype=torch.long, device=dev)
    lp, rp = _marker_position(tok, left), _marker_position(tok, right)
    if li.shape != ri.shape or lp != rp:
        raise ValueError("unbound concept prompts are not aligned")
    _, lc = _forward(model, li, torch.ones_like(li), (lp,), (layer,))
    _, rc = _forward(model, ri, torch.ones_like(ri), (rp,), (layer,))
    return rc[layer][0, 0] - lc[layer][0, 0]


def _candidate(model, batch, layer, delta=None, patch=None):
    add = None if delta is None else (layer, batch["marker"], delta)
    p = None if patch is None else (layer, batch["marker"], patch)
    logits, _ = _forward(model, batch["ids"], batch["am"],
                         (batch["marker"],), add=add, patch=p)
    return logits


def _evaluate_direction(model, tok, dev, clean_rows, natural_rows, queries,
                        surface, layer, delta):
    out = {}
    for query in queries:
        clean, natural, cl, nl, _cc, _nc = _encode_pair(
            model, tok, dev, clean_rows, natural_rows, query, surface)
        add = _candidate(model, clean, layer, delta=delta)
        out[query] = _switch_metrics(
            cl, nl, add, clean, _locations(clean_rows, query),
            _locations(natural_rows, query))
        out[query]["pass"] = _switch_pass(out[query])
    return out


def _invariants(model, tok, dev, rows, queries, surface, layer, delta):
    out = {}
    for query in queries:
        batch = _batch(tok, rows, query, surface, dev)
        base = _candidate(model, batch, layer)
        add = _candidate(model, batch, layer, delta=delta)
        expected = _locations(rows, query)
        out[query] = {
            "clean_acc": _accuracy(base, batch, expected),
            "add_acc": _accuracy(add, batch, expected),
            "pass": _accuracy(add, batch, expected) >= G_ACC,
        }
    return out


def _generate(model, tok, dev, rows, query, surface, layer=None, delta=None,
              max_new_tokens=5):
    texts = [_render(tok, row, query, surface) for row in rows]
    encoded = [tok.encode(text, add_special_tokens=False) for text in texts]
    if len({len(x) for x in encoded}) != 1:
        raise ValueError("generation batch is nonuniform")
    marker = _marker_position(tok, texts[0])
    if any(_marker_position(tok, text) != marker for text in texts):
        raise ValueError("generation markers differ")
    ids = torch.tensor(encoded, dtype=torch.long, device=dev)
    am = torch.ones_like(ids)
    start = ids.shape[1]
    finished = torch.zeros(ids.shape[0], dtype=torch.bool)
    eos = {int(tok.eos_token_id)}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end >= 0:
        eos.add(int(im_end))
    for _ in range(max_new_tokens):
        add = None
        if delta is not None:
            add = (layer, marker, delta.unsqueeze(0).expand(ids.shape[0], -1))
        logits, _ = _forward(model, ids, am, (marker,), add=add)
        nxt = logits.argmax(-1).long()
        nxt[finished] = int(tok.eos_token_id)
        finished |= torch.tensor([int(x) in eos for x in nxt])
        ids = torch.cat([ids, nxt.to(dev).unsqueeze(1)], dim=1)
        am = torch.cat([am, torch.ones((am.shape[0], 1), dtype=am.dtype,
                                       device=dev)], dim=1)
        if bool(finished.all()):
            break
    raw = [tok.decode(row, skip_special_tokens=False)
           for row in ids[:, start:].detach().cpu().tolist()]
    expected = [_expected(row, query) for row in rows]
    normalized = [_normalize(x) for x in raw]
    return {
        "accuracy": sum(a == b for a, b in zip(normalized, expected)) / len(rows),
        "answers": normalized, "expected": expected, "raw": raw,
    }


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_structured_workspace.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result


@torch.no_grad()
def run_delta_structured_workspace(model_path, out_dir, quantization="8bit",
                                   device_map=None, seed=0, n_null=N_NULL,
                                   report_only=False):
    # The freeze pins CHECKPOINT x QUANTIZATION pairs, not storage locations.
    # 7B int8: the original discovery config (HF id or its validated Kaggle
    # mount). 14B AWQ: the pre-registered scale follow-up after the 7B
    # BEHAVIORALLY_INELIGIBLE verdict (2026-07-14) — official pre-quantized
    # checkpoint via the loader path validated by the entity-matrix AWQ run.
    _FROZEN_PAIRS = {
        (MODEL, "8bit"),
        ("/kaggle/input/datasets/ragnar123/qwen2-5-7b-instruct", "8bit"),
        ("/kaggle/input/**/14b-instruct-awq/**/config.json", "awq"),
    }
    if (model_path, quantization) not in _FROZEN_PAIRS or seed != 0:
        raise ValueError("frozen structured-workspace discovery config mismatch")
    # Report-only variant (pre-registered 2026-07-14 after two-scale G0
    # anatomy): drops ONLY the search_* readouts — the action-under-
    # counterfactual class that failed G0 at both scales — keeping the full
    # multi-register discriminative design. Frozen to the 14B-AWQ pair (7B
    # also failed belief_bs cells, so report-only could not clear G0 there).
    if report_only and quantization != "awq":
        raise ValueError("report_only variant is frozen to the 14B-AWQ pair")

    def _rq(queries):
        return tuple(q for q in queries
                     if not (report_only and q.startswith("search")))
    if "*" in model_path:                     # Kaggle model mounts vary in
        import glob as _glob                  # nesting; resolve mechanically
        hits = sorted(_glob.glob(model_path, recursive=True))
        assert hits, f"model_path glob matched nothing: {model_path}"
        mp = hits[0]
        if os.path.basename(mp) == "config.json":
            mp = os.path.dirname(mp)
        log(f"model_path glob -> {mp}")
        model_path = mp
    if n_null != N_NULL:
        raise ValueError(f"frozen n_null={N_NULL}, got {n_null}")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    layers = tuple(x for x in LAYERS
                   if x < int(model.config.num_hidden_layers))
    result = {
        "stage": "delta_structured_workspace",
        "protocol_variant": ("report_only_v1" if report_only else "full"),
        "protocol_version": PROTOCOL_VERSION, "model_path": model_path,
        "quantization": quantization, "seed": seed, "n_null": n_null,
        "layers": list(layers), "n_rows_per_split": N_ROWS,
        "status": "discovery_only",
    }

    # Frozen direction specifications.  The first and fourth also define the
    # two factors in the joint-composition test.
    specs = {
        "ac_paris_rome": ("ac", "Paris", "Rome"),
        "as_paris_rome": ("as", "Paris", "Rome"),
        "bc_paris_rome": ("bc", "Paris", "Rome"),
        "bs_tokyo_delhi": ("bs", "Tokyo", "Delhi"),
        "ac_tokyo_delhi": ("ac", "Tokyo", "Delhi"),
        "ac_cairo_oslo": ("ac", "Cairo", "Oslo"),
    }
    datasets = {}
    for name, (address, source, target) in specs.items():
        train = _rows(source, target, address, "train")
        test = _rows(source, target, address, "test")
        datasets[name] = {
            "train_clean": train,
            "train_natural": _counterfactual(train, {address: target}),
            "test_clean": test,
            "test_natural": _counterfactual(test, {address: target}),
        }
    result["design"] = {
        "direction_specs": {
            name: {"address": address, "source": source, "target": target}
            for name, (address, source, target) in specs.items()},
        "rows": datasets,
        "queries": {name: {"question": question, "command": command,
                            "field": field}
                    for name, (question, command, field) in QUERY.items()},
        "train_surface": "ledger", "test_surface": "narrative",
        "marker": MARKER,
    }

    comp_clean = _composition_rows("test")
    comp_ac = _counterfactual(comp_clean, {"ac": "Rome"})
    comp_bs = _counterfactual(comp_clean, {"bs": "Delhi"})
    comp_joint = _counterfactual(
        comp_clean, {"ac": "Rome", "bs": "Delhi"})

    # G0 includes donor and held-out surfaces, target consequences, address
    # controls, and the joint natural counterfactual.  No intervention runs if
    # any required cell is behaviorally ineligible.
    behavior = {}
    primary_queries = _rq(("belief_ac", "search_ac", "tell_ac"))
    for name, queries in (
            ("ac_paris_rome", primary_queries),
            ("bs_tokyo_delhi", _rq(("belief_bs", "search_bs"))),
            ("ac_tokyo_delhi", _rq(("belief_ac", "search_ac"))),
            ("ac_cairo_oslo", _rq(("belief_ac", "search_ac"))),
            ("as_paris_rome", ("belief_as",)),
            ("bc_paris_rome", ("belief_bc",))):
        data = datasets[name]
        behavior[name + "_train"] = _behavior(
            model, tok, dev, data["train_clean"], data["train_natural"],
            queries, "ledger")
        behavior[name + "_test"] = _behavior(
            model, tok, dev, data["test_clean"], data["test_natural"],
            queries, "narrative")
    behavior["primary_invariants"] = _behavior(
        model, tok, dev, datasets["ac_paris_rome"]["test_clean"],
        datasets["ac_paris_rome"]["test_clean"],
        ("belief_as", "belief_bc", "truth_cube", "truth_sphere"),
        "narrative")
    behavior["composition"] = _behavior(
        model, tok, dev, comp_clean, comp_joint,
        _rq(("belief_ac", "search_ac", "belief_bs", "search_bs",
             "belief_as", "belief_bc", "truth_cube", "truth_sphere")),
        "narrative")
    result["behavior"] = behavior
    result["behavior_pass"] = _behavior_pass(behavior)
    if not result["behavior_pass"]:
        result["verdict"] = "STRUCTURED_WORKSPACE_BEHAVIORALLY_INELIGIBLE"
        return _write(out_dir, result)

    # Cache query-independent checkpoint states.  Causality guarantees that a
    # checkpoint before the question is identical across readouts; using one
    # canonical question avoids redundant cache passes.
    caches = {}
    for name, (address, _source, _target) in specs.items():
        data = datasets[name]
        canonical = "belief_" + address
        clean, natural, _cl, _nl, cc, nc = _encode_pair(
            model, tok, dev, data["train_clean"], data["train_natural"],
            canonical, "ledger", layers)
        caches[name] = {"clean": clean, "natural": natural,
                        "clean_cache": cc, "natural_cache": nc}

    # Donor-only bidirectional interchange chooses the earliest valid layer.
    # This is a native-state gate, not a search for the strongest steering site.
    discovery = []
    discovery_specs = {
        "ac_paris_rome": _rq(("belief_ac", "search_ac")),
        "bs_tokyo_delhi": _rq(("belief_bs", "search_bs")),
    }
    for layer in layers:
        layer_rows, layer_pass = {}, True
        for name, queries in discovery_specs.items():
            data, cache = datasets[name], caches[name]
            qrows = {}
            for query in queries:
                clean, natural, cl, nl, _cc, _nc = _encode_pair(
                    model, tok, dev, data["train_clean"],
                    data["train_natural"], query, "ledger")
                forward = _candidate(
                    model, clean, layer,
                    patch=cache["natural_cache"][layer][:, 0])
                reverse = _candidate(
                    model, natural, layer,
                    patch=cache["clean_cache"][layer][:, 0])
                switch = _switch_metrics(
                    cl, nl, forward, clean,
                    _locations(data["train_clean"], query),
                    _locations(data["train_natural"], query))
                reverse_acc = _accuracy(
                    reverse, natural, _locations(data["train_clean"], query))
                passed = _switch_pass(switch) and reverse_acc >= G_ACC
                qrows[query] = {"forward": switch,
                                "reverse_clean_acc": reverse_acc,
                                "pass": passed}
                layer_pass &= passed
            layer_rows[name] = qrows
        discovery.append({"layer": layer, "specs": layer_rows,
                          "pass": bool(layer_pass)})
        log(f"structured workspace L{layer}: interchangeable={layer_pass}")
    selected = next((x["layer"] for x in discovery if x["pass"]), None)
    result["workspace_discovery"] = discovery
    result["selected_layer"] = selected
    if selected is None:
        result["verdict"] = "NO_BIDIRECTIONAL_STRUCTURED_CHECKPOINT"
        return _write(out_dir, result)

    directions = {
        name: (cache["natural_cache"][selected][:, 0]
               - cache["clean_cache"][selected][:, 0]).mean(0)
        for name, cache in caches.items()
    }

    # Held-out surface and histories: value breadth at the same frozen layer.
    value_tests = {}
    for name in ("ac_paris_rome", "ac_tokyo_delhi", "ac_cairo_oslo"):
        data = datasets[name]
        value_tests[name] = _evaluate_direction(
            model, tok, dev, data["test_clean"], data["test_natural"],
            primary_queries if name == "ac_paris_rome"
            else _rq(("belief_ac", "search_ac")),
            "narrative", selected, directions[name])
    result["value_transfer"] = value_tests

    primary = datasets["ac_paris_rome"]
    invariants = _invariants(
        model, tok, dev, primary["test_clean"],
        ("belief_as", "belief_bc", "truth_cube", "truth_sphere"),
        "narrative", selected, directions["ac_paris_rome"])

    # Wrong-address directions must edit their own relation and preserve ac.
    wrong_address = {}
    for name, own_query in (("as_paris_rome", "belief_as"),
                            ("bc_paris_rome", "belief_bc")):
        own_natural = _counterfactual(
            primary["test_clean"], {specs[name][0]: "Rome"})
        own = _evaluate_direction(
            model, tok, dev, primary["test_clean"], own_natural,
            (own_query,), "narrative", selected, directions[name])
        preserve = _invariants(
            model, tok, dev, primary["test_clean"], ("belief_ac",),
            "narrative", selected, directions[name])
        wrong_address[name] = {"own_relation": own,
                               "target_relation_preserved": preserve}
    result["address_specificity"] = {
        "target_invariants": invariants, "wrong_address": wrong_address}

    # Rival: a processed unbound Paris->Rome concept at the same checkpoint.
    unbound = _unbound_direction(
        model, tok, dev, "Paris", "Rome", selected)
    unbound_target = _evaluate_direction(
        model, tok, dev, primary["test_clean"], primary["test_natural"],
        primary_queries, "narrative", selected, unbound)
    unbound_invariants = _invariants(
        model, tok, dev, primary["test_clean"],
        ("belief_as", "belief_bc", "truth_cube", "truth_sphere"),
        "narrative", selected, unbound)
    result["unbound_concept_baseline"] = {
        "note": "processed concept rival, not an exact Jacobian-lens vector",
        "target": unbound_target, "invariants": unbound_invariants,
    }

    # Factorial composition on a new world distribution.
    d_ac = directions["ac_paris_rome"]
    d_bs = directions["bs_tokyo_delhi"]
    composition = {}
    comp_queries = _rq(("belief_ac", "search_ac", "belief_bs", "search_bs"))
    for query in comp_queries:
        clean, joint, cl, jl, _cc, _jc = _encode_pair(
            model, tok, dev, comp_clean, comp_joint, query, "narrative")
        add_ac = _candidate(model, clean, selected, delta=d_ac)
        add_bs = _candidate(model, clean, selected, delta=d_bs)
        add_joint = _candidate(model, clean, selected, delta=d_ac + d_bs)
        target_rows = comp_ac if query.endswith("ac") else comp_bs
        own = add_ac if query.endswith("ac") else add_bs
        other = add_bs if query.endswith("ac") else add_ac
        own_nat_batch = _batch(tok, target_rows, query, "narrative", dev)
        # Natural logits for the relevant single edit provide the proper upper
        # bound; the joint natural is used for the composed edit.
        own_natural = _candidate(model, own_nat_batch, selected)
        own_metrics = _switch_metrics(
            cl, own_natural, own, clean, _locations(comp_clean, query),
            _locations(target_rows, query))
        other_preserve = _accuracy(
            other, clean, _locations(comp_clean, query))
        joint_metrics = _switch_metrics(
            cl, jl, add_joint, clean, _locations(comp_clean, query),
            _locations(comp_joint, query))
        composition[query] = {
            "own_single": own_metrics,
            "other_single_preserve_acc": other_preserve,
            "joint": joint_metrics,
            "pass": (_switch_pass(own_metrics)
                     and other_preserve >= G_ACC
                     and _switch_pass(joint_metrics)),
        }
    comp_invariants = _invariants(
        model, tok, dev, comp_clean,
        ("belief_as", "belief_bc", "truth_cube", "truth_sphere"),
        "narrative", selected, d_ac + d_bs)
    result["composition"] = {"targets": composition,
                             "invariants": comp_invariants}

    # Norm-matched random controls use the mean normalized causal effect across
    # the three primary readouts.  Thirty nulls make p=.032 the strongest
    # attainable result, matching the frozen p<.04 gate used elsewhere here.
    # Cache clean and natural forwards once.  Each random null therefore costs
    # only one intervention forward per readout, not three full conditions.
    null_contexts = {}
    for query in primary_queries:
        clean, _natural, cl, nl, _cc, _nc = _encode_pair(
            model, tok, dev, primary["test_clean"], primary["test_natural"],
            query, "narrative")
        null_contexts[query] = {
            "batch": clean, "clean_logits": cl, "natural_logits": nl,
            "source": _locations(primary["test_clean"], query),
            "target": _locations(primary["test_natural"], query),
        }

    def score(delta):
        ratios = []
        for query in primary_queries:
            context = null_contexts[query]
            add = _candidate(model, context["batch"], selected, delta=delta)
            metrics = _switch_metrics(
                context["clean_logits"], context["natural_logits"], add,
                context["batch"], context["source"], context["target"])
            ratios.append(metrics["ratio"] if metrics["ratio"] is not None
                          else -1e9)
        return float(np.mean(ratios))
    learned_score = score(d_ac)
    generator = torch.Generator().manual_seed(seed + 4903)
    null_scores = []
    for i in range(n_null):
        random = torch.randn(d_ac.shape, generator=generator)
        random = random / random.norm().clamp(min=EPS) * d_ac.norm()
        null_scores.append(score(random))
        if (i + 1) % 5 == 0:
            log(f"structured random directions {i + 1}/{n_null}")
    p = permutation_pvalue(learned_score, null_scores, "greater")
    result["random_null"] = {"learned_score": learned_score,
                             "scores": null_scores, "p": float(p)}

    # Full greedy continuations at the frozen layer.  Token-level effects alone
    # cannot earn the headline verdict.
    generation = {"primary": {}, "composition": {}}
    for query in primary_queries:
        generation["primary"][query] = {
            "clean": _generate(model, tok, dev, primary["test_clean"], query,
                               "narrative"),
            "natural": _generate(model, tok, dev, primary["test_natural"], query,
                                 "narrative"),
            "add": _generate(model, tok, dev, primary["test_clean"], query,
                             "narrative", selected, d_ac),
        }
    for query in comp_queries:
        generation["composition"][query] = {
            "natural": _generate(model, tok, dev, comp_joint, query,
                                 "narrative"),
            "add": _generate(model, tok, dev, comp_clean, query,
                             "narrative", selected, d_ac + d_bs),
        }
    result["generation"] = generation

    value_pass = sum(all(row["pass"] for row in tests.values())
                     for tests in value_tests.values()) >= 2
    invariant_pass = all(x["pass"] for x in invariants.values())
    wrong_pass = all(
        all(x["pass"] for x in row["own_relation"].values())
        and all(x["pass"] for x in row["target_relation_preserved"].values())
        for row in wrong_address.values())
    composition_pass = (
        all(x["pass"] for x in composition.values())
        and all(x["pass"] for x in comp_invariants.values()))
    generation_pass = (
        all(row["clean"]["accuracy"] >= G_ACC
            and row["natural"]["accuracy"] >= G_ACC
            and row["add"]["accuracy"] >= G_ACC
            for row in generation["primary"].values())
        and all(row["natural"]["accuracy"] >= G_ACC
                and row["add"]["accuracy"] >= G_ACC
                for row in generation["composition"].values()))
    gates = {
        "value_breadth_2_of_3": bool(value_pass),
        "target_invariants": bool(invariant_pass),
        "wrong_address": bool(wrong_pass),
        "composition": bool(composition_pass),
        "random_p_lt_04": bool(p < .04),
        "full_generation": bool(generation_pass),
    }
    gates["pass"] = all(gates.values())
    result["gates"] = gates
    result["verdict"] = (
        "FACTORIZED_RELATIONAL_WORKSPACE" if gates["pass"]
        else "STRUCTURED_WORKSPACE_PARTIAL_OR_NULL")
    return _write(out_dir, result)
