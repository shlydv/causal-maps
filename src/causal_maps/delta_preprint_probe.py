"""Grouped, cross-surface probe audit for the structured-state checkpoint.

This is a diagnostic, not evidence of a causal buffer. It asks whether Alice's
cube belief is linearly decodable at the source anchor, STATECHECK, and final
readout. Layer selection is nested inside replicate-held-out evaluation.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from .delta_anchor_write import _anchor_position, _resolve
from .delta_preprint_battery import _full_depth_layers
from .delta_structured_workspace import (LOCATIONS, _accuracy, _batch,
                                         _locations)
from .delta_trajectory import _forward
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer


def _balanced_rows(n_reps=6, seed=2718):
    """Eight balanced labels per replicate with independently permuted nuisances."""
    rng = np.random.default_rng(seed)
    fields = ("bc", "as", "bs", "tc", "ts")
    rows, labels, reps = [], [], []
    for rep in range(n_reps):
        perms = {field: rng.permutation(len(LOCATIONS)) for field in fields}
        while np.any(perms["tc"] == perms["ts"]):
            perms["ts"] = rng.permutation(len(LOCATIONS))
        for label, value in enumerate(LOCATIONS):
            row = {"ac": value}
            row.update({field: LOCATIONS[int(perms[field][label])]
                        for field in fields})
            rows.append(row)
            labels.append(label)
            reps.append(rep)
    return rows, np.asarray(labels), np.asarray(reps)


def _ridge_predict(x_train, y_train, x_test, n_classes):
    """Linear ridge classifier in the sample-space dual (D can be thousands)."""
    x_train = x_train.double()
    x_test = x_test.double()
    mean = x_train.mean(0, keepdim=True)
    x_train = x_train - mean
    x_test = x_test - mean
    gram = x_train @ x_train.T
    scale = float(torch.diagonal(gram).mean().clamp(min=1e-8))
    alpha = 1e-3 * scale
    target = torch.nn.functional.one_hot(
        torch.as_tensor(y_train, dtype=torch.long), n_classes).double()
    dual = torch.linalg.solve(
        gram + alpha * torch.eye(gram.shape[0], dtype=torch.double), target)
    return (x_test @ x_train.T @ dual).argmax(-1).numpy()


def _cv_accuracy(x, y, groups):
    pred = np.empty_like(y)
    for group in sorted(set(groups.tolist())):
        train, test = groups != group, groups == group
        pred[test] = _ridge_predict(x[train], y[train], x[test], len(LOCATIONS))
    return float((pred == y).mean())


def _select_layer(x_by_layer, y, groups, layers):
    scores = [_cv_accuracy(x_by_layer[:, i], y, groups)
              for i in range(len(layers))]
    best = int(np.argmax(scores))
    return best, scores


def _nested_score(x_by_layer, y, groups, layers):
    """Outer grouped test; layer selected only from the outer training rows."""
    pred = np.empty_like(y)
    selected = []
    for group in sorted(set(groups.tolist())):
        outer_train, outer_test = groups != group, groups == group
        inner_groups = groups[outer_train]
        candidate_scores = []
        for li in range(len(layers)):
            candidate_scores.append(_cv_accuracy(
                x_by_layer[outer_train, li], y[outer_train], inner_groups))
        chosen = int(np.argmax(candidate_scores))
        selected.append(int(layers[chosen]))
        pred[outer_test] = _ridge_predict(
            x_by_layer[outer_train, chosen], y[outer_train],
            x_by_layer[outer_test, chosen], len(LOCATIONS))
    accuracy = float((pred == y).mean())
    return {"accuracy": accuracy, "selected_layers": selected,
            "n": int(len(y)), "correct": int((pred == y).sum())}


def _binomial_greater(k, n, p):
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(k, n + 1)))


@torch.no_grad()
def run_delta_preprint_probe(model_path, out_dir, quantization="awq",
                             device_map=None, max_memory=None, layers=None,
                             n_reps=6, seed=2718, model=None, tok=None):
    os.makedirs(out_dir, exist_ok=True)
    if model is None or tok is None:
        model, tok = load_model_and_tokenizer(
            _resolve(model_path), quantization=quantization,
            device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    layers = _full_depth_layers(int(model.config.num_hidden_layers), layers)
    rows, labels, reps = _balanced_rows(n_reps, seed)
    changed = [dict(row, ac=LOCATIONS[(labels[i] + 1) % len(LOCATIONS)])
               for i, row in enumerate(rows)]
    sites = ("anchor", "checkpoint", "readout")
    activations = {}
    behavior = {}

    for surface in ("ledger", "narrative"):
        batch = _batch(tok, rows, "belief_ac", surface, dev)
        changed_batch = _batch(tok, changed, "belief_ac", surface, dev)
        anchor = _anchor_position(batch, changed_batch)
        positions = (anchor, batch["marker"], int(batch["ids"].shape[1] - 1))
        logits, cache = _forward(model, batch["ids"], batch["am"], positions,
                                 tuple(layers))
        behavior[surface] = _accuracy(
            logits, batch, _locations(rows, "belief_ac"))
        for site_idx, site in enumerate(sites):
            activations[(surface, site)] = torch.stack(
                [cache[layer][:, site_idx] for layer in layers], dim=1)
        log(f"probe capture {surface}: n={len(rows)} behavior="
            f"{behavior[surface]:.0%} positions={positions}")

    probe = {}
    for surface in ("ledger", "narrative"):
        probe[surface] = {}
        for site in sites:
            score = _nested_score(activations[(surface, site)], labels,
                                  reps, layers)
            score["p_vs_chance"] = _binomial_greater(
                score["correct"], score["n"], 1 / len(LOCATIONS))
            probe[surface][site] = score

    cross_surface = {}
    for source, target in (("ledger", "narrative"),
                           ("narrative", "ledger")):
        key = f"{source}_to_{target}"
        cross_surface[key] = {}
        for site in sites:
            x_src = activations[(source, site)]
            chosen, cv_scores = _select_layer(x_src, labels, reps, layers)
            pred = _ridge_predict(x_src[:, chosen], labels,
                                  activations[(target, site)][:, chosen],
                                  len(LOCATIONS))
            cross_surface[key][site] = {
                "layer": int(layers[chosen]),
                "source_cv_by_layer": {str(layer): float(score)
                                       for layer, score in zip(layers, cv_scores)},
                "accuracy": float((pred == labels).mean()),
            }

    dump = {"labels": labels, "replicates": reps,
            "layers": np.asarray(layers), "sites": np.asarray(sites)}
    for (surface, site), value in activations.items():
        dump[f"{surface}_{site}"] = value.numpy().astype(np.float16)
    np.savez_compressed(os.path.join(out_dir, "preprint_probe_activations.npz"),
                        **dump)

    result = {"stage": "delta_preprint_probe", "model_path": model_path,
              "quantization": quantization, "n_reps": n_reps,
              "n_rows": int(len(rows)), "chance": 1 / len(LOCATIONS),
              "layers": layers, "behavior": behavior, "probe": probe,
              "cross_surface": cross_surface,
              "interpretation": ("Decodability is diagnostic only; causal "
                                 "load is evaluated by checkpoint patching.")}
    with open(os.path.join(out_dir, "results_delta_preprint_probe.json"), "w") as f:
        json.dump(result, f, indent=2)
    log("probe audit complete")
    return result
