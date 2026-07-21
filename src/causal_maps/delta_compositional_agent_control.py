"""Factor and compose workflow-phase and evidence-policy control states."""
import json
import os

import torch

from .delta_continuous_orchestration import _generate_answers, _output_metrics
from .delta_evidence_arbitration import _original_followups
from .delta_multiturn_evidence_bridge import _score_conflicts, _zero_rows
from .delta_orchestration_controller import (
    EPS, _assert_runtime, _encode_uniform, _safe_ratio)
from .delta_orchestration_screen import (
    _execute, _expected_call, _normalize, _parse_call, _render, _rows)
from .delta_reasoning_controller import _candidate_metrics
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer


PROTOCOL_VERSION = "2026-07-14-v1"
INJECT_LAYER = 2
LAYERS = (2, 9, 16, 23, 30)
MEDIATION_LAYER = 23
N_NULL = 100
PHASES = ("A", "B")
SOURCES = ("I", "E")


def _conflict_rows():
    rows = [row for row in _rows()
            if row["a"] + row["b"] != row["database_value"]]
    if len(rows) < 16:
        raise ValueError("insufficient frozen conflict rows")
    return rows[:16]


def _values(row, family):
    if family == "database":
        call = _expected_call(row, "lookup")
        internal = row["a"] + row["b"]
        external = row["database_value"]
    elif family == "calculator":
        call = _expected_call(row, "calculate")
        external = row["a"] + row["b"]
        internal = external % 9 + 1
    else:
        raise ValueError(f"unknown family: {family}")
    if internal == external or not (0 <= internal <= 9 and 0 <= external <= 9):
        raise ValueError(f"non-diagnostic values: {family} {row}")
    if _execute(_parse_call(call)) != str(external):
        raise ValueError(f"tool action does not execute to result: {call}")
    return call, str(internal), str(external)


def _content(row, family, phase, source):
    call, internal, external = _values(row, family)
    return (
        "Control a tool agent using two independent registers. "
        "Phase A means emit the listed tool action. "
        "Phase B means emit a final answer. "
        "Source I means a final answer uses the internal candidate. "
        "Source E means a final answer uses the executed tool result. "
        f"Listed tool action: [{call}]. "
        f"Internal candidate: {internal}. "
        f"Executed tool result: {external}. "
        f"Phase register: {phase}. Source register: {source}. "
        "If Phase A, copy only the text inside the listed-action brackets. "
        "If Phase B, output ANSWER, one space, and the selected integer. "
        "Do not output brackets, punctuation, or any other text."
    )


def _texts(tok, rows, family, phase, source):
    return [_render(tok, [{"role": "user", "content":
                           _content(row, family, phase, source)}])
            for row in rows]


def _expected(row, family, phase, source):
    call, internal, external = _values(row, family)
    if phase == "A":
        return call
    return f"ANSWER {internal if source == 'I' else external}"


def _exact_score(raw, expected):
    answers = [_normalize(x) for x in raw]
    rows = [a == b for a, b in zip(answers, expected)]
    return {"accuracy": sum(rows) / len(rows), "rows": rows,
            "answers": answers, "expected": expected}


def _grid(tok, rows, family, dev):
    texts, ids, masks = {}, {}, {}
    for phase in PHASES:
        for source in SOURCES:
            key = phase + source
            texts[key] = _texts(tok, rows, family, phase, source)
            ids[key], masks[key] = _encode_uniform(tok, texts[key], dev)
    lengths = {value.shape[1] for value in ids.values()}
    if len(lengths) != 1:
        raise ValueError(f"{family} factorial lengths differ: {lengths}")

    def changes(left, right):
        return [[i for i, (a, b) in enumerate(zip(x, y)) if a != b]
                for x, y in zip(ids[left].tolist(), ids[right].tolist())]

    phase_changes = changes("AI", "BI")
    source_changes = changes("BI", "BE")
    if (len({tuple(x) for x in phase_changes}) != 1
            or len(phase_changes[0]) != 1):
        raise ValueError(f"{family} phase alignment: {phase_changes}")
    if (len({tuple(x) for x in source_changes}) != 1
            or len(source_changes[0]) != 1):
        raise ValueError(f"{family} source alignment: {source_changes}")
    phase_pos, source_pos = phase_changes[0][0], source_changes[0][0]
    if phase_pos == source_pos:
        raise ValueError("phase and source register positions collide")
    return {
        "texts": texts, "ids": ids, "masks": masks,
        "last": next(iter(ids.values())).shape[1] - 1,
        "phase_position": phase_pos, "source_position": source_pos,
    }


def _choice_batch(tok, texts, negative, positive, dev):
    prefixes, neg_ids, pos_ids, common_lengths = [], [], [], []
    for text, neg, pos in zip(texts, negative, positive):
        prompt = tok.encode(text, add_special_tokens=False)
        left = tok.encode(text + neg, add_special_tokens=False)
        right = tok.encode(text + pos, add_special_tokens=False)
        common = 0
        for a, b in zip(left, right):
            if a != b:
                break
            common += 1
        if common < len(prompt) or common >= min(len(left), len(right)):
            raise ValueError(
                f"invalid contextual choice prefix: prompt={len(prompt)} "
                f"common={common}")
        prefixes.append(left[:common])
        neg_ids.append(left[common])
        pos_ids.append(right[common])
        common_lengths.append(common)
    if len(set(common_lengths)) != 1:
        raise ValueError(f"choice prefix lengths differ: {common_lengths}")
    if len({len(x) for x in prefixes}) != 1:
        raise ValueError("choice batches are not uniform")
    ids = torch.tensor(prefixes, dtype=torch.long, device=dev)
    return (ids, torch.ones_like(ids),
            torch.tensor(pos_ids, dtype=torch.long),
            torch.tensor(neg_ids, dtype=torch.long))


def _extract_directions(model, grid):
    last = grid["last"]
    caches = {}
    for key in ("AI", "BI", "BE"):
        _, caches[key] = _forward(
            model, grid["ids"][key], grid["masks"][key], (last,),
            (INJECT_LAYER,))
    phase_rows = (caches["BI"][INJECT_LAYER][:, 0]
                  - caches["AI"][INJECT_LAYER][:, 0])
    evidence_rows = (caches["BE"][INJECT_LAYER][:, 0]
                     - caches["BI"][INJECT_LAYER][:, 0])
    return phase_rows.mean(0), evidence_rows.mean(0), {
        "phase_rows": phase_rows, "evidence_rows": evidence_rows}


def _generate_edit(model, tok, dev, texts, direction, last):
    return _generate_answers(
        model, tok, texts, dev, direction=direction,
        inject_layer=INJECT_LAYER, mode_position=last, max_new_tokens=8)


def _natural_behavior(model, tok, dev, rows, family, grid):
    result = {}
    for phase in PHASES:
        for source in SOURCES:
            key = phase + source
            raw, token_ids = _generate_edit(
                model, tok, dev, grid["texts"][key], None, grid["last"])
            score = _exact_score(
                raw, [_expected(row, family, phase, source) for row in rows])
            result[key] = {"score": score, "raw": raw,
                           "token_ids": token_ids}
            log(f"composition G0 {family} {key}: {score['accuracy']:.0%}")
    return result


def _generation_suite(model, tok, dev, rows, family, grid,
                      phase_direction, evidence_direction):
    cases = {
        "phase_only_AI_to_BI": ("AI", phase_direction, "B", "I",
                                 grid["last"]),
        "evidence_only_AI_invariant": ("AI", evidence_direction, "A", "I",
                                        grid["last"]),
        "composed_AI_to_BE": ("AI", phase_direction + evidence_direction,
                               "B", "E", grid["last"]),
        "evidence_only_BI_to_BE": ("BI", evidence_direction, "B", "E",
                                    grid["last"]),
        "phase_only_AE_to_BE": ("AE", phase_direction, "B", "E",
                                 grid["last"]),
        "wrong_address_composed": (
            "AI", phase_direction + evidence_direction, "B", "E",
            grid["last"] - 8),
    }
    result = {}
    for name, (base, direction, target_phase, target_source, position) in cases.items():
        raw, token_ids = _generate_answers(
            model, tok, grid["texts"][base], dev, direction=direction,
            inject_layer=INJECT_LAYER, mode_position=position,
            max_new_tokens=8)
        expected = [_expected(row, family, target_phase, target_source)
                    for row in rows]
        result[name] = {
            "score": _exact_score(raw, expected), "raw": raw,
            "token_ids": token_ids, "position": position}
        log(f"composition {family} {name}: "
            f"{result[name]['score']['accuracy']:.0%}")
    return result


def _effect_inputs(tok, rows, family, grid, dev):
    ai = grid["texts"]["AI"]
    bi = grid["texts"]["BI"]
    be = grid["texts"]["BE"]
    call = [_expected(row, family, "A", "I") for row in rows]
    answer_i = [_expected(row, family, "B", "I") for row in rows]
    answer_e = [_expected(row, family, "B", "E") for row in rows]
    phase_clean = _choice_batch(tok, ai, call, answer_i, dev)
    phase_natural = _choice_batch(tok, bi, call, answer_i, dev)
    comp_clean = _choice_batch(tok, ai, call, answer_e, dev)
    comp_natural = _choice_batch(tok, be, call, answer_e, dev)
    evidence_clean = _choice_batch(tok, bi, answer_i, answer_e, dev)
    evidence_natural = _choice_batch(tok, be, answer_i, answer_e, dev)
    for clean, natural in ((phase_clean, phase_natural),
                           (comp_clean, comp_natural),
                           (evidence_clean, evidence_natural)):
        if not torch.equal(clean[2], natural[2]) or not torch.equal(
                clean[3], natural[3]):
            raise ValueError("choice tokens changed across natural prompts")
    return {"phase": (phase_clean, phase_natural),
            "evidence": (evidence_clean, evidence_natural),
            "composition": (comp_clean, comp_natural)}


def _metric_dict(value):
    return {k: (v.tolist() if torch.is_tensor(v) else v)
            for k, v in value.items()}


def _causal_metrics(model, grid, directions, effects, full=False,
                    generator=None, n_null=0):
    phase_direction, evidence_direction = directions
    candidate_dirs = {
        "phase": phase_direction,
        "evidence": evidence_direction,
        "composition": phase_direction + evidence_direction,
    }
    output, states, logits_store = {}, {}, {}
    for name in ("phase", "evidence", "composition"):
        clean, natural = effects[name]
        clean_ids, clean_am, pos_ids, neg_ids = clean
        natural_ids, natural_am, _, _ = natural
        position = grid["last"]
        clean_logits, clean_cache = _forward(
            model, clean_ids, clean_am, (position,), LAYERS)
        natural_logits, natural_cache = _forward(
            model, natural_ids, natural_am, (position,), LAYERS)
        add_logits, add_cache = _forward(
            model, clean_ids, clean_am, (position,), LAYERS,
            add=(INJECT_LAYER, position,
                 candidate_dirs[name].unsqueeze(0).expand(clean_ids.shape[0], -1)))
        output[name] = _output_metrics(
            clean_logits, natural_logits, add_logits,
            pos_ids, neg_ids, [True] * clean_ids.shape[0])
        trajectory = {}
        for layer in LAYERS:
            natural_delta = (natural_cache[layer][:, 0]
                             - clean_cache[layer][:, 0])
            induced_delta = add_cache[layer][:, 0] - clean_cache[layer][:, 0]
            metric = _candidate_metrics(induced_delta, natural_delta)
            trajectory[str(layer)] = {
                "cosine": metric["cosine"], "error": metric["error"]}
        states[name] = {"trajectory": trajectory,
                        "clean": clean_cache, "natural": natural_cache,
                        "add": add_cache}
        logits_store[name] = {"clean": clean_logits,
                              "natural": natural_logits, "add": add_logits}

    result = {
        "output": {k: _metric_dict(v) for k, v in output.items()},
        "trajectory": {k: v["trajectory"] for k, v in states.items()},
    }
    raw = {"states": states, "logits": logits_store}
    if not full:
        return result, raw

    # Composition mediation uses the phase-transition decision. Exact
    # generation separately requires the correct external value.
    clean, natural = effects["composition"]
    clean_ids, clean_am, pos_ids, neg_ids = clean
    natural_ids, natural_am, _, _ = natural
    position = grid["last"]
    batch = candidate_dirs["composition"].unsqueeze(0).expand(
        clean_ids.shape[0], -1)
    clean_logits = logits_store["composition"]["clean"]
    clean_ld = _ld(clean_logits, pos_ids, neg_ids)

    patch_add_logits, _ = _forward(
        model, clean_ids, clean_am, (position,),
        patch=(MEDIATION_LAYER, position,
               states["composition"]["add"][MEDIATION_LAYER][:, 0]))
    patch_nat_logits, _ = _forward(
        model, clean_ids, clean_am, (position,),
        patch=(MEDIATION_LAYER, position,
               states["composition"]["natural"][MEDIATION_LAYER][:, 0]))
    block_add_logits, _ = _forward(
        model, clean_ids, clean_am, (position,),
        add=(INJECT_LAYER, position, batch),
        patch=(MEDIATION_LAYER, position,
               states["composition"]["clean"][MEDIATION_LAYER][:, 0]))
    block_nat_logits, _ = _forward(
        model, natural_ids, natural_am, (position,),
        patch=(MEDIATION_LAYER, position,
               states["composition"]["clean"][MEDIATION_LAYER][:, 0]))

    def effect(logits):
        return float((_ld(logits, pos_ids, neg_ids) - clean_ld).mean())

    learned = output["composition"]["effect"]
    natural_effect = output["composition"]["natural_effect"]
    patch_add, patch_nat = effect(patch_add_logits), effect(patch_nat_logits)
    block_add, block_nat = effect(block_add_logits), effect(block_nat_logits)
    add_fraction = _safe_ratio(learned - block_add, learned)
    nat_fraction = _safe_ratio(natural_effect - block_nat, natural_effect)
    mediation = {
        "patch_add": patch_add, "patch_natural": patch_nat,
        "patch_ratio": _safe_ratio(patch_add, patch_nat),
        "blocked_add": block_add, "blocked_natural": block_nat,
        "add_block_fraction": add_fraction,
        "natural_block_fraction": nat_fraction,
        "block_gap": (abs(add_fraction - nat_fraction)
                      if add_fraction is not None and nat_fraction is not None
                      else None),
    }

    if generator is None or n_null != N_NULL:
        raise ValueError("full metrics require frozen null generator")
    evidence_clean, _ = effects["evidence"]
    e_ids, e_am, e_pos, e_neg = evidence_clean
    e_clean_logits = logits_store["evidence"]["clean"]
    learned_combined = (
        output["composition"]["effect"] + output["evidence"]["effect"])
    null_effects = []
    norm = candidate_dirs["composition"].norm()
    for i in range(n_null):
        random = torch.randn(candidate_dirs["composition"].shape,
                             generator=generator)
        random = random / random.norm().clamp(min=EPS) * norm
        random_batch = random.unsqueeze(0).expand(clean_ids.shape[0], -1)
        phase_logits, _ = _forward(
            model, clean_ids, clean_am, (position,),
            add=(INJECT_LAYER, position, random_batch))
        evidence_logits, _ = _forward(
            model, e_ids, e_am, (position,),
            add=(INJECT_LAYER, position, random_batch))
        phase_effect = float((
            _ld(phase_logits, pos_ids, neg_ids) - clean_ld).mean())
        evidence_effect = float((
            _ld(evidence_logits, e_pos, e_neg)
            - _ld(e_clean_logits, e_pos, e_neg)).mean())
        null_effects.append(phase_effect + evidence_effect)
        if (i + 1) % 10 == 0:
            log(f"composition random {i + 1}/{n_null}")
    null_exceed = sum(x >= learned_combined for x in null_effects)
    result["mediation"] = mediation
    result["null"] = {
        "learned_combined_effect": learned_combined,
        "effects": null_effects, "exceedances": null_exceed}
    raw.update({
        "mediation_logits": {
            "patch_add": patch_add_logits, "patch_natural": patch_nat_logits,
            "block_add": block_add_logits, "block_natural": block_nat_logits},
        "null_effects": torch.tensor(null_effects),
    })
    return result, raw


def _core_gate(generation, causal):
    required = (
        "phase_only_AI_to_BI", "evidence_only_AI_invariant",
        "composed_AI_to_BE", "evidence_only_BI_to_BE",
        "phase_only_AE_to_BE")
    behavior = all(generation[name]["score"]["accuracy"] >= .875
                   for name in required)
    trajectory = all(
        causal["trajectory"][name][str(MEDIATION_LAYER)]["cosine"] >= .80
        and causal["trajectory"][name][str(MEDIATION_LAYER)]["error"] <= .60
        for name in ("phase", "evidence", "composition"))
    output = all(
        causal["output"][name]["ratio"] is not None
        and .60 <= causal["output"][name]["ratio"] <= 1.40
        and causal["output"][name]["positive_fraction"] >= .75
        for name in ("phase", "evidence", "composition"))
    return {"behavior": behavior, "trajectory": trajectory,
            "output": output, "pass": behavior and trajectory and output}


@torch.no_grad()
def run_delta_compositional_agent_control(
        model_path, out_dir, quantization="8bit", device_map=None,
        seed=0, n_null=N_NULL):
    if quantization != "8bit" or seed != 0 or n_null != N_NULL:
        raise ValueError("frozen compositional-control config mismatch")
    os.makedirs(out_dir, exist_ok=True)
    runtime = _assert_runtime()
    model, tok = load_model_and_tokenizer(
        model_path, device_map=device_map, quantization=quantization)
    if int(model.config.num_hidden_layers) != 32:
        raise ValueError("frozen compositional control expects 32 layers")
    dev = input_device(model)
    rows = _conflict_rows()
    train_rows, test_rows = rows[::2], rows[1::2]
    result = {
        "stage": "delta_compositional_agent_control",
        "protocol_version": PROTOCOL_VERSION, "model_path": model_path,
        "runtime": runtime, "layers": list(LAYERS),
        "train_rows": train_rows, "test_rows": test_rows,
    }

    train_grids, test_grids, behavior, eligible = {}, {}, {}, {}
    for family in ("database", "calculator"):
        train_grids[family] = _grid(tok, train_rows, family, dev)
        test_grids[family] = _grid(tok, test_rows, family, dev)
        if (train_grids[family]["phase_position"]
                != test_grids[family]["phase_position"]
                or train_grids[family]["source_position"]
                != test_grids[family]["source_position"]):
            raise ValueError(f"{family} register positions differ by split")
        behavior[family] = {
            "train": _natural_behavior(
                model, tok, dev, train_rows, family, train_grids[family]),
            "test": _natural_behavior(
                model, tok, dev, test_rows, family, test_grids[family]),
        }
        eligible[family] = all(
            behavior[family][split][cell]["score"]["accuracy"] >= .875
            for split in ("train", "test")
            for cell in ("AI", "AE", "BI", "BE"))
    result.update({
        "register_positions": {
            family: {
                "phase": test_grids[family]["phase_position"],
                "source": test_grids[family]["source_position"],
                "decision_marker": test_grids[family]["last"]}
            for family in test_grids},
        "behavior": behavior, "eligibility": eligible,
    })
    if not any(eligible.values()):
        result["gates"] = {"G0_database": eligible["database"],
                           "G0_calculator": eligible["calculator"]}
        result["verdict"] = "COMPOSITIONAL_CONTROL_BEHAVIORALLY_INELIGIBLE"
        return _write(out_dir, result)

    directions, donor_artifacts = {}, {}
    for family in ("database", "calculator"):
        if not eligible[family]:
            continue
        phase, evidence, artifact = _extract_directions(
            model, train_grids[family])
        directions[family] = (phase, evidence)
        donor_artifacts[family] = artifact

    evaluations, raw = {}, {}
    settings = {}
    if eligible["database"]:
        settings["database_local"] = (
            "database", directions["database"], True)
    if eligible["calculator"]:
        settings["calculator_local"] = (
            "calculator", directions["calculator"], False)
    if all(eligible.values()):
        settings["database_to_calculator"] = (
            "calculator", directions["database"], False)
    generator = torch.Generator().manual_seed(seed + 14071)
    for name, (family, dirs, full) in settings.items():
        grid = test_grids[family]
        generation = _generation_suite(
            model, tok, dev, test_rows, family, grid, *dirs)
        effects = _effect_inputs(tok, test_rows, family, grid, dev)
        causal, raw_metrics = _causal_metrics(
            model, grid, dirs, effects, full=full,
            generator=generator if full else None,
            n_null=n_null if full else 0)
        evaluations[name] = {
            "family": family, "generation": generation,
            "causal": causal, "core_gate": _core_gate(generation, causal)}
        raw[name] = raw_metrics

    med_gate, controls_gate, database_pass = None, None, False
    if eligible["database"]:
        database_full = evaluations["database_local"]
        med = database_full["causal"]["mediation"]
        med_gate = bool(
            med["patch_ratio"] is not None
            and .70 <= med["patch_ratio"] <= 1.30
            and med["add_block_fraction"] is not None
            and med["natural_block_fraction"] is not None
            and med["add_block_fraction"] >= .70
            and med["natural_block_fraction"] >= .70
            and med["block_gap"] <= .20)
        controls_gate = bool(
            database_full["generation"]["wrong_address_composed"]["score"][
                "accuracy"] < .875
            and database_full["causal"]["null"]["exceedances"] <= 1)
        database_pass = bool(
            database_full["core_gate"]["pass"]
            and med_gate and controls_gate)
    calculator_pass = bool(
        eligible["calculator"]
        and evaluations["calculator_local"]["core_gate"]["pass"])
    transfer_pass = bool(
        all(eligible.values())
        and evaluations["database_to_calculator"]["core_gate"]["pass"])

    # Supplemental transfer to the untouched original zero-result transcripts.
    bridge_rows = _zero_rows()[1::2]
    bridge_texts, _ = _original_followups(tok, bridge_rows, "lookup")
    bridge_ids, _bridge_am = _encode_uniform(tok, bridge_texts, dev)
    bridge_last = bridge_ids.shape[1] - 1
    bridge_target = ["0"] * len(bridge_rows)
    bridge_source = [str(row["a"] + row["b"]) for row in bridge_rows]
    bridge = {}
    bridge_directions = [("base", None)]
    bridge_directions.extend(
        (f"{family}_evidence", directions[family][1])
        for family in ("database", "calculator") if family in directions)
    for name, direction in bridge_directions:
        raw_answers, token_ids = _generate_edit(
            model, tok, dev, bridge_texts, direction, bridge_last)
        bridge[name] = {
            "score": _score_conflicts(
                raw_answers, bridge_target, bridge_source),
            "raw": raw_answers, "token_ids": token_ids}

    gates = {
        "G0_database": eligible["database"],
        "G0_calculator": eligible["calculator"],
        "D_database_composition": database_pass,
        "D_mediation": med_gate, "D_controls": controls_gate,
        "L_calculator_local_replication": calculator_pass,
        "X_literal_cross_workflow_transfer": transfer_pass,
    }
    if database_pass and calculator_pass and transfer_pass:
        verdict = "PORTABLE_COMPOSITIONAL_AGENT_CONTROL"
    elif database_pass and calculator_pass:
        verdict = "COMPOSITIONAL_AGENT_CONTROL"
    elif database_pass or calculator_pass:
        verdict = "SINGLE_FAMILY_COMPOSITIONAL_CONTROL"
    else:
        verdict = "AGENT_CONTROL_NOT_DECOMPOSED"
    raw_artifact = "raw_delta_compositional_agent_control.pt"
    torch.save({
        "directions": directions, "donors": donor_artifacts,
        "evaluations": raw,
    }, os.path.join(out_dir, raw_artifact))
    result.update({
        "direction_norms": {
            family: {"phase": float(values[0].norm()),
                     "evidence": float(values[1].norm()),
                     "cosine": float(torch.nn.functional.cosine_similarity(
                         values[0], values[1], dim=0))}
            for family, values in directions.items()},
        "cross_family_direction_cosines": (
            {
                "phase": float(torch.nn.functional.cosine_similarity(
                    directions["database"][0], directions["calculator"][0],
                    dim=0)),
                "evidence": float(torch.nn.functional.cosine_similarity(
                    directions["database"][1], directions["calculator"][1],
                    dim=0)),
            } if all(eligible.values()) else None),
        "evaluations": evaluations, "original_bridge": bridge,
        "gates": gates, "raw_tensor_artifact": raw_artifact,
        "verdict": verdict,
    })
    return _write(out_dir, result)


def _write(out_dir, result):
    path = os.path.join(out_dir,
                        "results_delta_compositional_agent_control.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    log(f"VERDICT: {result['verdict']}")
    return result
