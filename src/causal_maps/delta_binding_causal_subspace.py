"""Held-out causal rank curve for the distributed binding representation.

Frozen design: BINDING_CAUSAL_SUBSPACE_PROTOCOL.md.  This tests a linear L8
residual subspace, not individual heads or an upstream circuit.
"""
import json
import os

import numpy as np
import torch

from .delta_binding_component_convergence import _group_data, _split_trials
from .delta_operator import (
    DONOR_NAMES, INJECT_LAYER, MEDIATION_LAYER, _directions,
    _encode_uniform, _single_text, _trials, _values)
from .delta_trajectory import EPS, _cos_rows, _forward, _ld
from .logutil import log
from .model_utils import get_decoder_layers, input_device, load_model_and_tokenizer
from .nulls import permutation_pvalue
from .patching import _split_output

RANKS = (1, 2, 4, 8, 16)
N_NULL = 100
PROTOCOL_VERSION = "2026-07-13-v1"


def _effect(clean_logits, changed_logits, pos_ids, neg_ids):
    return _ld(changed_logits, pos_ids, neg_ids) - _ld(
        clean_logits, pos_ids, neg_ids)


@torch.no_grad()
def _capture_conditions(model, groups):
    """Capture matched L8 CLEAN/NATURAL/ADD states for each query group."""
    observations = []
    natural_rows, add_rows, clean_hits, natural_hits, add_positive, cos_rows = (
        [], [], [], [], [], [])
    for group in groups:
        positions = (group["last"] - 1, group["last"])
        clean_logits, clean_cache = _forward(
            model, group["clean_ids"], group["clean_am"], positions,
            (MEDIATION_LAYER,))
        natural_logits, natural_cache = _forward(
            model, group["natural_ids"], group["natural_am"], positions,
            (MEDIATION_LAYER,))
        add_logits, add_cache = _forward(
            model, group["clean_ids"], group["clean_am"], positions,
            (MEDIATION_LAYER,),
            add=(INJECT_LAYER, group["slot"], group["direction"]))
        natural = _effect(
            clean_logits, natural_logits, group["pos_ids"], group["neg_ids"])
        add = _effect(clean_logits, add_logits, group["pos_ids"], group["neg_ids"])
        clean_state = clean_cache[MEDIATION_LAYER]
        natural_state = natural_cache[MEDIATION_LAYER]
        add_state = add_cache[MEDIATION_LAYER]
        natural_disp = natural_state[:, 1] - clean_state[:, 1]
        add_disp = add_state[:, 1] - clean_state[:, 1]
        cosine = _cos_rows(add_disp, natural_disp)
        observations.append({
            "group": group,
            "clean_logits": clean_logits,
            "clean_state": clean_state,
            "natural_disp": natural_disp,
            "add_disp": add_disp,
            "natural_effect_rows": natural,
            "add_effect_rows": add,
            "final_cos_rows": cosine,
        })
        natural_rows.extend(natural.tolist())
        add_rows.extend(add.tolist())
        clean_hits.extend((clean_logits.argmax(-1) == group["neg_ids"]).tolist())
        natural_hits.extend((natural_logits.argmax(-1) == group["pos_ids"]).tolist())
        add_positive.extend((add > 0).tolist())
        cos_rows.extend(cosine.tolist())
    return {
        "observations": observations,
        "natural_effect": float(np.mean(natural_rows)),
        "add_effect": float(np.mean(add_rows)),
        "clean_acc": float(np.mean(clean_hits)),
        "natural_acc": float(np.mean(natural_hits)),
        "add_positive_fraction": float(np.mean(add_positive)),
        "l8_final_cos": float(np.mean(cos_rows)),
    }


def _fit_basis(observations, max_rank=max(RANKS)):
    rows = []
    for observation in observations:
        rows.extend((observation["natural_disp"], observation["add_disp"]))
    matrix = torch.cat(rows, dim=0).float()
    _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    if vh.shape[0] < max_rank:
        raise ValueError(f"only {vh.shape[0]} basis dimensions available")
    return vh[:max_rank].T.contiguous(), singular


def _basis_energy(observations, basis):
    result = {}
    for key in ("natural_disp", "add_disp"):
        rows = torch.cat([item[key] for item in observations], dim=0).float()
        projection = (rows @ basis) @ basis.T
        numerator = projection.square().sum(dim=1)
        denominator = rows.square().sum(dim=1).clamp(min=EPS)
        result[key.replace("_disp", "_energy")] = float(
            (numerator / denominator).mean())
    return result


@torch.no_grad()
def _forward_remove_subspace(model, input_ids, attention_mask, inject_position,
                             clean_state, basis, position, direction=None):
    """Remove a basis projection of the matched-clean displacement at L8."""
    layers = get_decoder_layers(model)
    handles = []
    if direction is not None:
        def add_hook(_module, _inputs, output):
            hidden, rebuild = _split_output(output)
            edited = hidden.clone()
            delta = direction.to(device=edited.device, dtype=edited.dtype)
            if delta.ndim == 1:
                delta = delta.unsqueeze(0).expand(edited.shape[0], -1)
            edited[:, inject_position] += delta
            return rebuild(edited)
        handles.append(layers[INJECT_LAYER].register_forward_hook(add_hook))

    def remove_hook(_module, _inputs, output):
        hidden, rebuild = _split_output(output)
        reference = clean_state.to(device=hidden.device, dtype=hidden.dtype)
        subspace = basis.to(device=hidden.device, dtype=hidden.dtype)
        if reference.shape != (hidden.shape[0], hidden.shape[-1]):
            raise ValueError(
                f"clean-state shape {tuple(reference.shape)} incompatible with "
                f"hidden state {tuple(hidden.shape)}")
        displacement = hidden[:, position] - reference
        removed = (displacement @ subspace) @ subspace.T
        edited = hidden.clone()
        edited[:, position] -= removed
        return rebuild(edited)
    handles.append(layers[MEDIATION_LAYER].register_forward_hook(remove_hook))
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False)
        return output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def _subspace_measure(model, observations, basis, previous_position=False):
    """Held-out effect under a learned or random L8 displacement projection."""
    natural_rows, add_rows = [], []
    state_index = 0 if previous_position else 1
    for observation in observations:
        group = observation["group"]
        position = group["last"] - int(previous_position)
        reference = observation["clean_state"][:, state_index]
        natural_logits = _forward_remove_subspace(
            model, group["natural_ids"], group["natural_am"], group["slot"],
            reference, basis, position)
        add_logits = _forward_remove_subspace(
            model, group["clean_ids"], group["clean_am"], group["slot"],
            reference, basis, position, direction=group["direction"])
        natural_rows.extend(_effect(
            observation["clean_logits"], natural_logits,
            group["pos_ids"], group["neg_ids"]).tolist())
        add_rows.extend(_effect(
            observation["clean_logits"], add_logits,
            group["pos_ids"], group["neg_ids"]).tolist())
    return {"natural_effect": float(np.mean(natural_rows)),
            "add_effect": float(np.mean(add_rows))}


def _loss(base, ablated, key):
    if base[key] <= EPS:
        raise ValueError(f"non-positive baseline {key}: {base[key]}")
    return float((base[key] - ablated[key]) / base[key])


def _random_basis(hidden_size, rank, rng):
    matrix = rng.standard_normal((hidden_size, rank)).astype(np.float32)
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return torch.from_numpy(basis.copy())


def _summary(base, measurement):
    return {
        **measurement,
        "natural_loss": _loss(base, measurement, "natural_effect"),
        "add_loss": _loss(base, measurement, "add_effect"),
    }


def _inelicitable(out_dir, model_path, quantization, seed, reason, heldout):
    result = {
        "stage": "delta_binding_causal_subspace",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "failure_reason": reason,
        "heldout_base": heldout,
        "gates": {"G0": False},
        "verdict": "CAUSAL_SUBSPACE_INELICITABLE",
    }
    path = os.path.join(out_dir, "results_delta_binding_causal_subspace.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: CAUSAL_SUBSPACE_INELICITABLE ({reason})")
    return result


@torch.no_grad()
def run_delta_binding_causal_subspace(
        model_path, out_dir, quantization="8bit", device_map=None,
        seed=0, n_null=N_NULL):
    if n_null < 100:
        raise ValueError("shared-subspace p<.01 gate requires >=100 nulls")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    layers = get_decoder_layers(model)
    if len(layers) <= MEDIATION_LAYER:
        raise ValueError("model lacks frozen L8 mediator")
    values = _values(tok)
    rows = _trials(values)
    discovery_rows, heldout_rows = _split_trials(rows)

    donor_rows, donor_values = [], []
    for name in DONOR_NAMES:
        for value in values:
            donor_rows.append(_single_text(tok, name, value))
            donor_values.append(value)
    donor_ids, donor_am, donor_pos = _encode_uniform(tok, donor_rows)
    _, donor_cache = _forward(
        model, donor_ids.to(dev), donor_am.to(dev), (donor_pos,),
        (INJECT_LAYER,))
    donor_hidden = donor_cache[INJECT_LAYER][:, 0]
    prototypes = {
        value: donor_hidden[[i for i, observed in enumerate(donor_values)
                             if observed == value]].mean(0)
        for value in values
    }
    discovery_groups = _group_data(model, tok, discovery_rows, prototypes, dev)
    heldout_groups = _group_data(model, tok, heldout_rows, prototypes, dev)
    log("delta_binding_causal_subspace: "
        f"discovery={len(discovery_rows)} heldout={len(heldout_rows)} "
        f"ranks={RANKS} n_null={n_null}")

    discovery = _capture_conditions(model, discovery_groups)
    heldout = _capture_conditions(model, heldout_groups)
    effect_ratio = (heldout["add_effect"] / heldout["natural_effect"]
                    if heldout["natural_effect"] > EPS else float("nan"))
    g0 = bool(
        heldout["clean_acc"] >= .80
        and heldout["natural_acc"] >= .80
        and heldout["add_positive_fraction"] >= .80
        and .70 <= effect_ratio <= 1.30
        and heldout["l8_final_cos"] >= .50)
    base_summary = {key: heldout[key] for key in (
        "natural_effect", "add_effect", "clean_acc", "natural_acc",
        "add_positive_fraction", "l8_final_cos")}
    base_summary["add_to_natural_effect_ratio"] = float(effect_ratio)
    if not g0:
        return _inelicitable(
            out_dir, model_path, quantization, seed,
            "held-out operator or L8-convergence gate failed", base_summary)

    full_basis, singular = _fit_basis(discovery["observations"])
    hidden_size = int(full_basis.shape[0])
    rng = np.random.default_rng(seed + 3801)
    rank_results = []
    for rank in RANKS:
        basis = full_basis[:, :rank]
        learned = _summary(
            heldout, _subspace_measure(model, heldout["observations"], basis))
        previous = _summary(
            heldout, _subspace_measure(
                model, heldout["observations"], basis, previous_position=True))
        natural_null, add_null = [], []
        for index in range(n_null):
            random = _random_basis(hidden_size, rank, rng)
            measurement = _subspace_measure(model, heldout["observations"], random)
            natural_null.append(_loss(heldout, measurement, "natural_effect"))
            add_null.append(_loss(heldout, measurement, "add_effect"))
            if (index + 1) % 10 == 0 or index + 1 == n_null:
                log(f"rank={rank} random bases {index + 1}/{n_null}")
        natural_p = permutation_pvalue(
            learned["natural_loss"], np.asarray(natural_null), "greater")
        add_p = permutation_pvalue(
            learned["add_loss"], np.asarray(add_null), "greater")
        energy = _basis_energy(heldout["observations"], basis)
        shared = bool(
            learned["natural_loss"] >= .50 and learned["add_loss"] >= .50
            and natural_p < .01 and add_p < .01
            and abs(previous["natural_loss"]) <= .20
            and abs(previous["add_loss"]) <= .20)
        natural_only = bool(learned["natural_loss"] >= .50 and natural_p < .01)
        add_only = bool(learned["add_loss"] >= .50 and add_p < .01)
        rank_results.append({
            "rank": rank,
            "heldout_basis_energy": energy,
            "learned_basis": learned,
            "previous_position_control": previous,
            "random_natural_losses": natural_null,
            "random_add_losses": add_null,
            "natural_p": float(natural_p),
            "add_p": float(add_p),
            "shared_causal_rank": shared,
            "natural_only": natural_only,
            "add_only": add_only,
        })
        log(f"rank={rank} learned losses natural={learned['natural_loss']:+.3f} "
            f"add={learned['add_loss']:+.3f} p=({natural_p:.4f},{add_p:.4f}) "
            f"shared={shared}")

    shared_low = [row["rank"] for row in rank_results
                  if row["rank"] <= 8 and row["shared_causal_rank"]]
    shared_mid = [row["rank"] for row in rank_results
                  if row["rank"] == 16 and row["shared_causal_rank"]]
    one_sided = any(row["natural_only"] != row["add_only"]
                    for row in rank_results)
    if shared_low:
        verdict = "SHARED_LOW_RANK_CAUSAL_SUBSPACE"
    elif shared_mid:
        verdict = "SHARED_MIDRANK_CAUSAL_SUBSPACE"
    elif one_sided:
        verdict = "ONE_SIDED_OR_DIVERGENT_SUBSPACE"
    else:
        verdict = "HIGH_RANK_OR_NONLINEAR_DISTRIBUTED"
    result = {
        "stage": "delta_binding_causal_subspace",
        "model_path": model_path,
        "quantization": quantization,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "inject_layer": INJECT_LAYER,
        "mediation_layer": MEDIATION_LAYER,
        "position": "L8 final query/readout token",
        "ranks": list(RANKS),
        "n_trials": len(rows),
        "n_discovery": len(discovery_rows),
        "n_heldout": len(heldout_rows),
        "n_null": int(n_null),
        "discovery": {
            "base": {key: discovery[key] for key in (
                "natural_effect", "add_effect", "clean_acc", "natural_acc",
                "add_positive_fraction", "l8_final_cos")},
            "singular_values": singular.tolist(),
        },
        "heldout_base": base_summary,
        "rank_results": rank_results,
        "gates": {"G0": g0},
        "shared_low_ranks": shared_low,
        "shared_midranks": shared_mid,
        "verdict": verdict,
    }
    path = os.path.join(out_dir, "results_delta_binding_causal_subspace.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {verdict}")
    return result
