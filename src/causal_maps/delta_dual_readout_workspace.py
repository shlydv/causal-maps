"""Cross-fitted edit of a shared arithmetic state with two causal readouts.

Discovery is deliberately nested: donor rows select one layer and define one
direction; held-out rows are touched once.  See DUAL_READOUT_WORKSPACE_PROTOCOL.md.
"""
import json
import os

import numpy as np
import torch

from .delta_reasoning_controller import _continuation_token_id
from .delta_reasoning_screen import _chat
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer
from .tensorize import _anchor_token_index


PROTOCOL_VERSION = "2026-07-14-v1"
CANDIDATE_LAYERS = (4, 8, 12, 16, 20, 24, 26)
N_RANDOM = 20
EPS = 1e-8


def _rows(split):
    """Disjoint operand pairs; all source and counterfactual sums are one digit."""
    if split == "donor":
        pairs = ((1, 1), (2, 1), (3, 1), (4, 1),
                 (1, 2), (2, 2), (3, 2), (4, 2))
    elif split == "test":
        pairs = ((1, 3), (2, 3), (3, 3),
                 (1, 4), (2, 4), (3, 4),
                 (1, 5), (2, 5))
    else:
        raise ValueError(f"unknown split: {split}")
    return [{"a": a, "b": b, "target_a": a + 1} for a, b in pairs]


def _answer(row, readout, target=False):
    total = (row["target_a"] if target else row["a"]) + row["b"]
    if readout == "sum":
        return str(total)
    if readout == "parity":
        return "even" if total % 2 == 0 else "odd"
    raise ValueError(readout)


def _prompt(tok, row, readout, target=False):
    a = row["target_a"] if target else row["a"]
    question = (
        "What is the sum? Return only the integer answer."
        if readout == "sum" else
        "Is the sum even or odd? Return only the word even or odd."
    )
    user = (
        f"First number: {a}. Second number: {row['b']}. "
        "Add the two numbers mentally now. Computation checkpoint: READY. "
        f"Question: {question}"
    )
    return _chat(tok, user)


def _encode(tok, rows, readout, target=False):
    prompts = [_prompt(tok, row, readout, target) for row in rows]
    encoded = [tok.encode(text, add_special_tokens=False) for text in prompts]
    lengths = {len(ids) for ids in encoded}
    if len(lengths) != 1:
        raise ValueError(f"{readout} prompt lengths nonuniform: {sorted(lengths)}")
    marker_positions = []
    for text, ids in zip(prompts, encoded):
        offset = text.index("READY")
        pos = _anchor_token_index(tok, text, offset)
        candidates = [i for i in range(max(0, pos - 1), min(len(ids), pos + 3))
                      if "READY" in tok.decode([ids[i]])]
        if len(candidates) != 1:
            raise ValueError(f"READY token is not unique near marker: {candidates}")
        marker_positions.append(candidates[0])
    if len(set(marker_positions)) != 1:
        raise ValueError(f"marker positions nonuniform: {marker_positions}")
    answers = [_answer(row, readout, target) for row in rows]
    answer_ids = torch.tensor([
        _continuation_token_id(tok, prompt, answer)
        for prompt, answer in zip(prompts, answers)], dtype=torch.long)
    ids = torch.tensor(encoded, dtype=torch.long)
    return {
        "ids": ids, "mask": torch.ones_like(ids),
        "marker": marker_positions[0], "last": ids.shape[1] - 1,
        "answers": answers, "answer_ids": answer_ids, "prompts": prompts,
    }


def _aligned_pair(tok, rows, readout):
    clean = _encode(tok, rows, readout, target=False)
    natural = _encode(tok, rows, readout, target=True)
    if clean["ids"].shape != natural["ids"].shape:
        raise ValueError(f"{readout} clean/natural shape mismatch")
    if clean["marker"] != natural["marker"] or clean["last"] != natural["last"]:
        raise ValueError(f"{readout} clean/natural marker mismatch")
    changed = []
    for left, right in zip(clean["ids"], natural["ids"]):
        diff = (left != right).nonzero().flatten().tolist()
        if len(diff) != 1:
            raise ValueError(f"expected one changed operand token, got {diff}")
        changed.append(diff[0])
    if len(set(changed)) != 1:
        raise ValueError(f"operand positions nonuniform: {changed}")
    return clean, natural, changed[0]


def _effect(logits, clean_ld, target_ids, source_ids, natural_effect):
    rows = _ld(logits, target_ids, source_ids) - clean_ld
    value = float(rows.mean())
    return {
        "effect": value,
        "effect_ratio": value / natural_effect if abs(natural_effect) > EPS else None,
        "target_acc": float((logits.argmax(-1) == target_ids).float().mean()),
        "positive_fraction": float((rows > 0).float().mean()),
    }


def _passes(metric):
    ratio = metric["effect_ratio"]
    return bool(metric["target_acc"] >= .75 and
                metric["positive_fraction"] >= .75 and
                ratio is not None and .60 <= ratio <= 1.40)


def _to_dev(batch, dev):
    return batch["ids"].to(dev), batch["mask"].to(dev)


def _prediction_audit(tok, logits, expected_ids, k=5):
    """Exact-token diagnostics for behavioral failures; no gate uses this."""
    values, ids = logits.topk(k, dim=-1)
    predictions = logits.argmax(-1)
    rows = []
    for i in range(logits.shape[0]):
        expected = int(expected_ids[i])
        predicted = int(predictions[i])
        rows.append({
            "expected_id": expected,
            "expected_token": tok.decode([expected]),
            "prediction_id": predicted,
            "prediction_token": tok.decode([predicted]),
            "hit": predicted == expected,
            "expected_minus_top_logit": float(logits[i, expected] - values[i, 0]),
            "top5": [
                {"id": int(ids[i, j]), "token": tok.decode([int(ids[i, j])]),
                 "logit": float(values[i, j])}
                for j in range(k)
            ],
        })
    return rows


@torch.no_grad()
def run_delta_dual_readout_workspace(model_path, out_dir,
                                     quantization="8bit", device_map=None,
                                     seed=0, n_random=N_RANDOM):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    dev = input_device(model)
    layers = tuple(x for x in CANDIDATE_LAYERS
                   if x < int(model.config.num_hidden_layers))
    split_rows = {name: _rows(name) for name in ("donor", "test")}
    batches = {}
    operand_positions = {}
    for split in ("donor", "test"):
        batches[split] = {}
        for readout in ("sum", "parity"):
            clean, natural, operand = _aligned_pair(
                tok, split_rows[split], readout)
            batches[split][readout] = {"clean": clean, "natural": natural}
            operand_positions[(split, readout)] = operand

    result = {
        "stage": "delta_dual_readout_workspace",
        "protocol_version": PROTOCOL_VERSION, "model_path": model_path,
        "candidate_layers": list(layers), "n_random": int(n_random),
        "split_rows": split_rows, "behavior": {}, "donor_localization": [],
    }

    # G0 and sparse caches.  The marker precedes the differing questions, so
    # its state must be exactly identical across readouts for the same prefix.
    runs = {}
    behavior_pass = True
    for split in ("donor", "test"):
        runs[split] = {}
        result["behavior"][split] = {}
        for readout in ("sum", "parity"):
            pair = batches[split][readout]
            ci, cam = _to_dev(pair["clean"], dev)
            ni, nam = _to_dev(pair["natural"], dev)
            marker, last = pair["clean"]["marker"], pair["clean"]["last"]
            clean_logits, clean_cache = _forward(
                model, ci, cam, (marker,), layers)
            natural_logits, natural_cache = _forward(
                model, ni, nam, (marker,), layers)
            source_ids = pair["clean"]["answer_ids"]
            target_ids = pair["natural"]["answer_ids"]
            clean_acc = float((clean_logits.argmax(-1) == source_ids).float().mean())
            natural_acc = float((natural_logits.argmax(-1) == target_ids).float().mean())
            eligible = clean_acc >= .875 and natural_acc >= .875
            behavior_pass = behavior_pass and eligible
            clean_ld = _ld(clean_logits, target_ids, source_ids)
            natural_effect = float(
                (_ld(natural_logits, target_ids, source_ids) - clean_ld).mean())
            result["behavior"][split][readout] = {
                "clean_acc": clean_acc, "natural_acc": natural_acc,
                "natural_effect": natural_effect, "eligible": bool(eligible),
                "marker": marker, "last": last,
                "operand_token": operand_positions[(split, readout)],
                "clean_prediction_audit": _prediction_audit(
                    tok, clean_logits, source_ids),
                "natural_prediction_audit": _prediction_audit(
                    tok, natural_logits, target_ids),
            }
            runs[split][readout] = {
                "clean_logits": clean_logits, "natural_logits": natural_logits,
                "clean_cache": clean_cache, "natural_cache": natural_cache,
                "clean_ld": clean_ld, "natural_effect": natural_effect,
                "source_ids": source_ids, "target_ids": target_ids,
                "ci": ci, "cam": cam, "ni": ni, "nam": nam,
                "marker": marker, "last": last,
            }
            log(f"G0 {split}/{readout}: {clean_acc:.0%}/{natural_acc:.0%} "
                f"effect={natural_effect:+.3f} eligible={eligible}")
    result["behavior_pass"] = bool(behavior_pass)
    if not behavior_pass:
        result["verdict"] = "DUAL_READOUT_BEHAVIORALLY_INELIGIBLE"
        return _write(out_dir, result)

    for layer in layers:
        entry = {"layer": layer, "readouts": {}}
        layer_pass = True
        for readout in ("sum", "parity"):
            run = runs["donor"][readout]
            forward_logits, _ = _forward(
                model, run["ci"], run["cam"], (run["last"],),
                patch=(layer, run["marker"], run["natural_cache"][layer][:, 0]))
            reverse_logits, _ = _forward(
                model, run["ni"], run["nam"], (run["last"],),
                patch=(layer, run["marker"], run["clean_cache"][layer][:, 0]))
            forward = _effect(forward_logits, run["clean_ld"],
                              run["target_ids"], run["source_ids"],
                              run["natural_effect"])
            reverse_acc = float(
                (reverse_logits.argmax(-1) == run["source_ids"]).float().mean())
            interchangeable = _passes(forward) and reverse_acc >= .75
            layer_pass = layer_pass and interchangeable
            entry["readouts"][readout] = {
                "forward": forward, "reverse_source_acc": reverse_acc,
                "interchangeable": bool(interchangeable),
            }
        entry["shared_interchangeable"] = bool(layer_pass)
        result["donor_localization"].append(entry)
        log(f"donor L{layer}: shared_interchangeable={layer_pass} "
            f"metrics={entry['readouts']}")
    selected = next((x["layer"] for x in result["donor_localization"]
                     if x["shared_interchangeable"]), None)
    result["selected_layer"] = selected
    if selected is None:
        result["verdict"] = "NO_SHARED_DONOR_WORKSPACE"
        return _write(out_dir, result)

    # The prefix through READY is identical for both query surfaces. Assert the
    # cached state agrees, then fit one direction using donor instances only.
    donor_sum = runs["donor"]["sum"]
    donor_parity = runs["donor"]["parity"]
    for key in ("clean_cache", "natural_cache"):
        if not torch.allclose(donor_sum[key][selected],
                              donor_parity[key][selected], atol=2e-3, rtol=2e-3):
            raise ValueError("shared pre-query marker differs across readouts")
    donor_deltas = (donor_sum["natural_cache"][selected][:, 0]
                    - donor_sum["clean_cache"][selected][:, 0])
    learned = donor_deltas.mean(0)
    learned_norm = float(learned.norm())
    result["learned_direction"] = {
        "norm": learned_norm, "donor_row_norm_mean": float(donor_deltas.norm(dim=1).mean())}

    heldout = {}
    embedding = model.get_input_embeddings().weight.detach().float().cpu()
    for readout in ("sum", "parity"):
        run = runs["test"][readout]
        operand = operand_positions[("test", readout)]
        embed_delta = torch.stack([
            embedding[int(run["ni"][i, operand])]
            - embedding[int(run["ci"][i, operand])]
            for i in range(len(split_rows["test"]))])
        variants = {"learned": learned, "reverse": -learned,
                    "raw_embedding": embed_delta}
        metrics = {}
        for name, delta in variants.items():
            logits, _ = _forward(
                model, run["ci"], run["cam"], (run["last"],),
                add=(selected, run["marker"], delta))
            metrics[name] = _effect(logits, run["clean_ld"],
                                    run["target_ids"], run["source_ids"],
                                    run["natural_effect"])
        blocked_logits, _ = _forward(
            model, run["ni"], run["nam"], (run["last"],),
            patch=(selected, run["marker"], run["clean_cache"][selected][:, 0]))
        metrics["clean_overwrite"] = {
            "source_acc": float(
                (blocked_logits.argmax(-1) == run["source_ids"]).float().mean()),
            "target_acc": float(
                (blocked_logits.argmax(-1) == run["target_ids"]).float().mean()),
        }
        heldout[readout] = metrics
        log(f"heldout {readout}: {metrics}")
    result["heldout"] = heldout

    generator = torch.Generator().manual_seed(seed + 4171)
    random_scores = []
    for draw in range(n_random):
        random = torch.randn(learned.shape, generator=generator)
        random = random / random.norm().clamp(min=EPS) * learned.norm()
        ratios = []
        for readout in ("sum", "parity"):
            run = runs["test"][readout]
            logits, _ = _forward(
                model, run["ci"], run["cam"], (run["last"],),
                add=(selected, run["marker"], random))
            metric = _effect(logits, run["clean_ld"], run["target_ids"],
                             run["source_ids"], run["natural_effect"])
            ratios.append(metric["effect_ratio"])
        random_scores.append(float(min(ratios)))
        log(f"random {draw + 1}/{n_random}: min_ratio={random_scores[-1]:+.3f}")
    real_score = min(heldout[x]["learned"]["effect_ratio"]
                     for x in ("sum", "parity"))
    result["random_control"] = {
        "score_definition": "minimum heldout effect ratio across readouts",
        "real": float(real_score), "scores": random_scores,
        "p_greater": float((1 + sum(x >= real_score for x in random_scores))
                           / (1 + len(random_scores))),
        "p95": float(np.percentile(random_scores, 95)),
    }
    learned_pass = all(_passes(heldout[x]["learned"])
                       for x in ("sum", "parity"))
    controls_pass = all(
        not _passes(heldout[x]["reverse"])
        and not _passes(heldout[x]["raw_embedding"])
        and heldout[x]["clean_overwrite"]["source_acc"] >= .75
        for x in ("sum", "parity"))
    random_pass = result["random_control"]["p_greater"] <= .05
    result["gates"] = {"learned_both_readouts": bool(learned_pass),
                       "directional_and_necessity_controls": bool(controls_pass),
                       "random_control": bool(random_pass)}
    result["verdict"] = (
        "CROSS_FITTED_DUAL_READOUT_WORKSPACE_EDIT"
        if learned_pass and controls_pass and random_pass else
        "DUAL_READOUT_WORKSPACE_NOT_ESTABLISHED")
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir, "results_delta_dual_readout_workspace.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
