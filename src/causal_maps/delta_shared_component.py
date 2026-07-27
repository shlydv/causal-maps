"""Paper 2 discovery pilot for a shared causal component of address spillover.

The experiment is intentionally interpretation-neutral.  It asks whether the
natural-minus-synthetic state difference in the DeepSeek spillover case
contains a donor-learned component that repairs spillover on held-out worlds
and induces it in reverse.  Passing the experiment is evidence for a causal
address-specificity component; calling that component "routing" requires the
subsequent path and generalization studies.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import LAYER as CONTENT_LAYER
from .delta_anchor_write import SOURCE, TARGET, _neutral_states, _resolve
from .delta_preprint_battery import _compatible_world_rows
from .delta_preprint_locus import _uniform_locus_positions
from .delta_structured_workspace import (
    LOCATIONS,
    _accuracy,
    _batch,
    _counterfactual,
    _locations,
)
from .delta_trajectory import _forward, _ld
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-23-p2-shared-component-v1"
PROTOCOL_SHA256 = (
    "F1D7D3B7AC31F44B01B3BC6A3035A3FCDAA50F2D7582606AE7D6BEEBC0D6AE10")
DEFAULT_LAYERS = (2, 8, 16, 24, 30)
G_ACC = 0.80
RANDOM_SEED = 7319
EPS = 1e-8


def _fixed_split(rows, n_donor=15):
    """Mechanical donor/evaluation split; order comes from tokenizer selection."""
    if not 1 <= int(n_donor) < len(rows):
        raise ValueError("n_donor must leave nonempty donor and evaluation sets")
    return list(rows[:int(n_donor)]), list(rows[int(n_donor):])


def _random_positions(marker, excluded, size, seed):
    candidates = sorted(set(range(int(marker) + 1)) - set(excluded))
    if len(candidates) < int(size):
        raise ValueError("insufficient random-locus candidates")
    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(candidates, size=int(size), replace=False).tolist())


def _build_component_loci(marker, summary, anchors, seed=RANDOM_SEED):
    """Frozen primary, matched-negative, and size-matched control loci."""
    all_anchors = sorted(anchors.values())
    belief = sorted(anchors[key] for key in ("ac", "bc", "as", "bs"))
    excluded = set(all_anchors) | set(summary)
    loci = {
        "edited_anchor": [anchors["ac"]],
        "belief_anchors": belief,
        "source_anchors": all_anchors,
        "summary_span": list(summary),
        "marker_only": [int(marker)],
    }
    loci["random_single"] = _random_positions(
        marker, excluded, 1, int(seed))
    loci["random_belief_size"] = _random_positions(
        marker, excluded, len(belief), int(seed) + 1)
    loci["random_source_size"] = _random_positions(
        marker, excluded, len(all_anchors), int(seed) + 2)
    loci["random_summary_size"] = _random_positions(
        marker, excluded, len(summary), int(seed) + 3)
    return loci


def _component_stats(residual):
    """Descriptive donor alignment and spectrum for [N,P,D] residual rows."""
    flat = residual.float().reshape(residual.shape[0], -1)
    row_energy = flat.square().sum(1)
    mean = flat.mean(0)
    mean_fraction = float(
        mean.square().sum() / row_energy.mean().clamp(min=EPS))
    unit = flat / flat.norm(dim=1, keepdim=True).clamp(min=EPS)
    gram = unit @ unit.T
    if flat.shape[0] > 1:
        mask = ~torch.eye(flat.shape[0], dtype=torch.bool)
        pairwise_cos = float(gram[mask].mean())
    else:
        pairwise_cos = 1.0
    singular = torch.linalg.svdvals(flat)
    energy = singular.square()
    total = energy.sum().clamp(min=EPS)
    top = {}
    for k in (1, 2, 4, 8):
        kk = min(k, len(singular))
        top[str(k)] = float(energy[:kk].sum() / total)
    return {
        "mean_energy_fraction": mean_fraction,
        "mean_pairwise_cosine": pairwise_cos,
        "top_k_energy_fraction": top,
        "mean_norm": float(mean.norm()),
        "mean_row_norm": float(flat.norm(dim=1).mean()),
    }


def _is_random_locus(name):
    return name.startswith("random_")


def _adjudicate(eligible, cells):
    if not eligible:
        return "BEHAVIORALLY_INELIGIBLE"
    primary_both = [
        (layer, locus) for layer, block in cells.items()
        for locus, cell in block.items()
        if not _is_random_locus(locus)
        and locus not in ("summary_span", "marker_only")
        and cell.get("bidirectional_pass", False)
    ]
    random_both = [
        (layer, locus) for layer, block in cells.items()
        for locus, cell in block.items()
        if _is_random_locus(locus) and cell.get("bidirectional_pass", False)
    ]
    primary_rescue = any(
        cell.get("rescue_pass", False)
        for block in cells.values() for locus, cell in block.items()
        if not _is_random_locus(locus)
        and locus not in ("summary_span", "marker_only"))
    any_nonlocal = any(
        cell.get("bidirectional_pass", False)
        for block in cells.values() for locus, cell in block.items()
        if _is_random_locus(locus) or locus in ("summary_span", "marker_only"))
    if primary_both and not random_both:
        return "SHARED_CAUSAL_COMPONENT"
    if primary_both and random_both:
        return "NONLOCAL_OR_NONSPECIFIC_COMPONENT"
    if primary_rescue:
        return "RESCUE_ONLY_COMPONENT"
    if any_nonlocal:
        return "NONLOCAL_OR_NONSPECIFIC_COMPONENT"
    return "NO_SHARED_CAUSAL_COMPONENT"


def _ids(batch, values):
    return torch.tensor([batch["amap"][value] for value in values])


def _endpoint(logits, batch, expected, rival):
    expected_ids = _ids(batch, expected)
    rival_ids = _ids(batch, rival)
    margin = _ld(logits, expected_ids, rival_ids)
    return {
        "accuracy": float(_accuracy(logits, batch, expected)),
        "margin_mean": float(margin.mean()),
        "margin_rows": margin.tolist(),
    }


def _run_component(model, batch, content_pos, content_delta, layer, positions,
                   component, *, synthetic, sign):
    additions = []
    if synthetic:
        additions.append((CONTENT_LAYER, int(content_pos), content_delta))
    signed = float(sign) * component
    for j, position in enumerate(positions):
        additions.append((int(layer), int(position), signed[j]))
    logits, _ = _forward(
        model, batch["ids"], batch["am"], (),
        add=additions)
    return logits


def _baseline_block(model, batches, content_pos, content_delta):
    """Run CLEAN, NATURAL, and SYNTHETIC endpoints once per query."""
    out = {}
    for query in ("belief_ac", "belief_bc", "belief_bs"):
        clean_batch, natural_batch = batches[query]
        clean_logits, _ = _forward(
            model, clean_batch["ids"], clean_batch["am"], ())
        natural_logits, _ = _forward(
            model, natural_batch["ids"], natural_batch["am"], ())
        synthetic_logits, _ = _forward(
            model, clean_batch["ids"], clean_batch["am"], (),
            add=(CONTENT_LAYER, content_pos, content_delta))
        if query == "belief_ac":
            clean_expected = [SOURCE] * clean_batch["ids"].shape[0]
            natural_expected = [TARGET] * clean_batch["ids"].shape[0]
            synthetic_expected = natural_expected
            clean_rival = [TARGET] * clean_batch["ids"].shape[0]
            natural_rival = clean_expected
            synthetic_rival = clean_expected
        elif query == "belief_bc":
            clean_expected = [SOURCE] * clean_batch["ids"].shape[0]
            natural_expected = clean_expected
            synthetic_expected = [TARGET] * clean_batch["ids"].shape[0]
            clean_rival = [TARGET] * clean_batch["ids"].shape[0]
            natural_rival = clean_rival
            synthetic_rival = clean_expected
        else:
            clean_expected = [TARGET] * clean_batch["ids"].shape[0]
            natural_expected = clean_expected
            synthetic_expected = clean_expected
            clean_rival = [SOURCE] * clean_batch["ids"].shape[0]
            natural_rival = clean_rival
            synthetic_rival = clean_rival
        out[query] = {
            "clean": _endpoint(
                clean_logits, clean_batch, clean_expected, clean_rival),
            "natural": _endpoint(
                natural_logits, natural_batch, natural_expected,
                natural_rival),
            "synthetic": _endpoint(
                synthetic_logits, clean_batch, synthetic_expected,
                synthetic_rival),
        }
    return out


def _eligible(baseline):
    required = (
        baseline["belief_ac"]["clean"]["accuracy"],
        baseline["belief_ac"]["natural"]["accuracy"],
        baseline["belief_ac"]["synthetic"]["accuracy"],
        baseline["belief_bc"]["clean"]["accuracy"],
        baseline["belief_bc"]["natural"]["accuracy"],
        baseline["belief_bc"]["synthetic"]["accuracy"],
        baseline["belief_bs"]["clean"]["accuracy"],
        baseline["belief_bs"]["natural"]["accuracy"],
        baseline["belief_bs"]["synthetic"]["accuracy"],
    )
    return bool(min(required) >= G_ACC)


@torch.no_grad()
def run_delta_shared_component(
        model_path, out_dir, model_key="deepseek_shared_component_d1",
        quantization="8bit", device_map=None, max_memory=None, n_world=30,
        n_donor=15, layers=DEFAULT_LAYERS, random_seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    donor_rows, eval_rows = _fixed_split(rows, n_donor=n_donor)
    donor_natural = _counterfactual(donor_rows, {"ac": TARGET})
    eval_natural = _counterfactual(eval_rows, {"ac": TARGET})

    donor_clean = _batch(
        tok, donor_rows, "belief_ac", "narrative", dev)
    donor_nat = _batch(
        tok, donor_natural, "belief_ac", "narrative", dev)
    eval_batches = {
        query: (
            _batch(tok, eval_rows, query, "narrative", dev),
            _batch(tok, eval_natural, query, "narrative", dev),
        )
        for query in ("belief_ac", "belief_bc", "belief_bs")
    }
    summary, anchors = _uniform_locus_positions(
        tok, donor_clean, donor_rows)
    natural_summary, natural_anchors = _uniform_locus_positions(
        tok, donor_nat, donor_natural)
    if summary != natural_summary or anchors != natural_anchors:
        raise ValueError("natural and synthetic locus positions differ")
    for clean_batch, natural_batch in eval_batches.values():
        if clean_batch["marker"] != donor_clean["marker"]:
            raise ValueError("donor/evaluation marker positions differ")
        if clean_batch["ids"].shape != natural_batch["ids"].shape:
            raise ValueError("evaluation natural batch is not aligned")
        eval_summary, eval_anchors = _uniform_locus_positions(
            tok, clean_batch, eval_rows)
        eval_natural_summary, eval_natural_anchors = (
            _uniform_locus_positions(tok, natural_batch, eval_natural))
        if (eval_summary != summary or eval_anchors != anchors
                or eval_natural_summary != summary
                or eval_natural_anchors != anchors):
            raise ValueError("donor/evaluation locus positions differ")

    component_loci = _build_component_loci(
        donor_clean["marker"], summary, anchors, seed=random_seed)
    content_pos = int(anchors["ac"])
    states = _neutral_states(
        model, tok, dev, CONTENT_LAYER, (SOURCE, TARGET))
    content_delta = states[TARGET] - states[SOURCE]
    n_model_layers = model_num_hidden_layers(model)
    layers = DEFAULT_LAYERS if layers is None else layers
    selected_layers = sorted({
        int(layer) for layer in layers
        if 0 <= int(layer) < n_model_layers
    })
    if not selected_layers:
        raise ValueError("no requested component layer exists in model")

    baseline = _baseline_block(
        model, eval_batches, content_pos, content_delta)
    eligible = _eligible(baseline)
    log(
        "P2 component G0 "
        f"ac={baseline['belief_ac']['synthetic']['accuracy']:.0%} "
        f"bc_spill={baseline['belief_bc']['synthetic']['accuracy']:.0%} "
        f"bs_same_value={baseline['belief_bs']['synthetic']['accuracy']:.0%} "
        f"eligible={eligible}")

    result = {
        "stage": "delta_shared_component",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "n_donor": len(donor_rows),
            "n_evaluation": len(eval_rows),
            "donor_indices_from_selected": list(range(len(donor_rows))),
            "evaluation_indices_from_selected": list(
                range(len(donor_rows), len(rows))),
            "donor_rows": donor_rows,
            "evaluation_rows": eval_rows,
        },
        "layers": selected_layers,
        "positions": {
            "marker": donor_clean["marker"],
            "summary": summary,
            "anchors": anchors,
            "loci": component_loci,
        },
        "baseline": baseline,
        "eligible": eligible,
        "per_layer": {},
    }
    if not eligible:
        result["verdict"] = "BEHAVIORALLY_INELIGIBLE"
    else:
        total = len(selected_layers) * len(component_loci)
        hb = Heartbeat(total, "shared_component", every_sec=30,
                       out_dir=out_dir)
        capture_positions = tuple(range(int(donor_clean["marker"]) + 1))
        for layer in selected_layers:
            _, natural_cache = _forward(
                model, donor_nat["ids"], donor_nat["am"],
                capture_positions, (layer,))
            _, synthetic_cache = _forward(
                model, donor_clean["ids"], donor_clean["am"],
                capture_positions, (layer,),
                add=(CONTENT_LAYER, content_pos, content_delta))
            residual = natural_cache[layer] - synthetic_cache[layer]
            layer_out = {}
            for locus, positions in component_loci.items():
                component_rows = residual[:, positions, :]
                component = component_rows.mean(0)
                stats = _component_stats(component_rows)

                rescue = {}
                necessity = {}
                specifications = {
                    "belief_ac": ([TARGET] * len(eval_rows),
                                  [SOURCE] * len(eval_rows)),
                    "belief_bc": ([SOURCE] * len(eval_rows),
                                  [TARGET] * len(eval_rows)),
                    "belief_bs": ([TARGET] * len(eval_rows),
                                  [SOURCE] * len(eval_rows)),
                }
                for query, (expected, rival) in specifications.items():
                    clean_batch, natural_batch = eval_batches[query]
                    rescued = _run_component(
                        model, clean_batch, content_pos, content_delta,
                        layer, positions, component,
                        synthetic=True, sign=+1)
                    induced = _run_component(
                        model, natural_batch, content_pos, content_delta,
                        layer, positions, component,
                        synthetic=False, sign=-1)
                    rescue[query] = _endpoint(
                        rescued, clean_batch, expected, rival)
                    necessity_expected = (
                        [TARGET] * len(eval_rows)
                        if query in ("belief_ac", "belief_bc", "belief_bs")
                        else expected)
                    necessity_rival = (
                        [SOURCE] * len(eval_rows)
                        if query != "belief_bs" else [SOURCE] * len(eval_rows))
                    necessity[query] = _endpoint(
                        induced, natural_batch,
                        necessity_expected, necessity_rival)

                rescue_pass = bool(
                    min(rescue[q]["accuracy"] for q in rescue) >= G_ACC)
                necessity_pass = bool(
                    min(necessity[q]["accuracy"] for q in necessity) >= G_ACC)
                cell = {
                    "component": stats,
                    "rescue": rescue,
                    "necessity": necessity,
                    "rescue_pass": rescue_pass,
                    "necessity_pass": necessity_pass,
                    "bidirectional_pass": (
                        rescue_pass and necessity_pass),
                }
                layer_out[locus] = cell
                hb.step(extra=(
                    f"L{layer}/{locus} rescue={rescue_pass} "
                    f"necessity={necessity_pass} "
                    f"bc={rescue['belief_bc']['accuracy']:.0%}/"
                    f"{necessity['belief_bc']['accuracy']:.0%}"))
            result["per_layer"][str(layer)] = layer_out
            del natural_cache, synthetic_cache, residual
        hb.done()
        result["verdict"] = _adjudicate(
            eligible, result["per_layer"])

    path = os.path.join(
        out_dir, f"results_delta_shared_component_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"P2 shared-component verdict={result['verdict']} artifact={path}")
    return result
