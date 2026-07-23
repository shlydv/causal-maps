#!/usr/bin/env python3
"""Build pilot-number macros from immutable copied JSON artifacts.

Confirmatory battery artifacts will supersede these macros without requiring
manual number transcription in the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "pilots"
CONFIRM = ROOT / "evidence" / "confirmatory"
OUT = Path(__file__).resolve().parent / "generated" / "pilot_results.tex"


def read(name):
    with (EVIDENCE / name).open() as handle:
        return json.load(handle)


def read_confirm(name):
    with (CONFIRM / name).open() as handle:
        return json.load(handle)


def fmt(x, digits=3):
    return f"{float(x):.{digits}f}"


def command(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}"


def main():
    qwen = read("workspace_qwen7b.json")
    mistral = read("workspace_mistral7b.json")
    phi = read("workspace_phi35.json")
    anchor = read("anchor_write_qwen14b.json")
    checkpoint = read("checkpoint_qwen14b.json")
    entity = read("entity_qwen14b.json")
    verbal = read("verbalization_qwen14b.json")
    qconf = read_confirm("results_delta_preprint_battery_qwen7b_confirm.json")
    mconf = read_confirm("results_delta_preprint_battery_mistral7b_confirm.json")
    q14 = read_confirm("results_delta_preprint_battery_qwen14b_headline.json")
    deepseek = read_confirm(
        "results_delta_preprint_battery_deepseek_r1_llama8b_confirm.json")
    gemma = read_confirm(
        "results_delta_preprint_battery_gemma3_12b_confirm.json")
    locus = read_confirm(
        "results_delta_preprint_locus_qwen14b_locus_v1.json")
    closeout_path = CONFIRM / "results_delta_paper1_closeout_qwen14b_closeout_v1.json"
    closeout = json.loads(closeout_path.read_text()) if closeout_path.exists() else None
    uncertainty = json.loads((Path(__file__).resolve().parent / "generated" /
                              "confirmatory_summary.json").read_text())

    ratios_q = [cell["ratio"] for cell in qwen["cells"].values()
                if "ratio" in cell]
    ratios_m = [cell["ratio"] for cell in mistral["cells"].values()
                if "ratio" in cell]
    checkpoint_ratios = []
    for layer in checkpoint["workspace_discovery"]:
        for spec in layer["specs"].values():
            for query in spec.values():
                checkpoint_ratios.append(abs(query["forward"]["ratio"]))
    city = entity["models"]["qwen14b_awq"]["families"]["city"]["cells"]
    belief = anchor["consequences"]["belief_ac"]
    vbelief = verbal["readouts"]["belief_ac"]

    def matrix_cells(result):
        return [cell for block in result["per_seed"].values()
                for cell in block["matrix"]["cells"].values()]

    qmc, mmc = matrix_cells(qconf), matrix_cells(mconf)
    qcity = [block["entity"]["families"]["city"]["cells"]
             for block in qconf["per_seed"].values()]
    q_twohop = [cells["twohop"] for cells in qcity]
    q_specificity_pass = sum(cells["other"]["verdict"] == "PASS"
                             for cells in qcity)
    q_specificity_ineligible = sum(cells["other"]["verdict"] == "INELICITABLE"
                                   for cells in qcity)
    qa, ma = qconf["anchor"], mconf["anchor"]
    qc, mc = qconf["checkpoint"], mconf["checkpoint"]
    q14a, q14c = q14["anchor"], q14["checkpoint"]
    q14p = q14["probe"]
    q14r = q14["reverse_base"]["readouts"]["belief_ac"]
    dmc = matrix_cells(deepseek)
    da, dc = deepseek["anchor"], deepseek["checkpoint"]
    gmc = matrix_cells(gemma)
    ga, gc = gemma["anchor"], gemma["checkpoint"]
    gcity = [block["entity"]["families"]["city"]["cells"]["twohop"]
             for block in gemma["per_seed"].values()]
    locus_layers = {int(layer): block
                    for layer, block in locus["per_layer"].items()}
    locus_sufficient = {layer: block["source_anchors"]
                        for layer, block in locus_layers.items()
                        if block["source_anchors"]["sufficient"]}
    locus_random = [abs(cell["forward_ratio"])
                    for block in locus_layers.values()
                    for name, cell in block.items()
                    if name.startswith("random_size_matched_")]

    lines = [
        "% Generated from evidence/pilots/*.json; do not hand edit.",
        command("PilotQwenMatrixPass", sum(c["verdict"] == "PASS"
                                           for c in qwen["cells"].values())),
        command("PilotMistralMatrixPass", sum(c["verdict"] == "PASS"
                                              for c in mistral["cells"].values())),
        command("PilotPhiComputePass", sum(phi["cells"][name]["verdict"] == "PASS"
                                           for name in ("add2", "sub1", "max5", "gt5label"))),
        command("PilotQwenRatioMin", fmt(min(ratios_q))),
        command("PilotQwenRatioMax", fmt(max(ratios_q))),
        command("PilotMistralRatioMin", fmt(min(ratios_m))),
        command("PilotMistralRatioMax", fmt(max(ratios_m))),
        command("PilotAnchorRatio", fmt(belief["ratio"])),
        command("PilotAnchorAccuracy", fmt(100 * belief["target_acc"], 0) + r"\%"),
        command("PilotCheckpointMax", fmt(max(checkpoint_ratios), 4)),
        command("PilotAnchorCheckpointGap",
                fmt(belief["ratio"] / max(checkpoint_ratios), 0) + r"$\times$"),
        command("PilotCityTwoHopRatio", fmt(city["twohop"]["ratio"])),
        command("PilotCityTwoHopAccuracy",
                fmt(100 * city["twohop"]["add_target_acc"], 0) + r"\%"),
        command("PilotOtherShift", fmt(city["other"]["shift_over_twohop"])),
        command("PilotVerbalHistoryNoV", fmt(vbelief["cond1_hist_noV"]["lam"])),
        command("PilotVerbalHistoryV", fmt(vbelief["cond2_hist_underV"]["lam"])),
        command("PilotVerbalBoth", fmt(vbelief["cond4_both"]["lam"])),
        command("ConfirmQwenMatrixPass",
                str(sum(cell["verdict"] == "PASS" for cell in qmc))),
        command("ConfirmMistralMatrixPass",
                str(sum(cell["verdict"] == "PASS" for cell in mmc))),
        command("ConfirmQwenRatioMin", fmt(min(c["ratio"] for c in qmc))),
        command("ConfirmQwenRatioMax", fmt(max(c["ratio"] for c in qmc))),
        command("ConfirmMistralRatioMin", fmt(min(c["ratio"] for c in mmc))),
        command("ConfirmMistralRatioMax", fmt(max(c["ratio"] for c in mmc))),
        command("ConfirmQwenAnchorN", str(qa["n_rows"])),
        command("ConfirmQwenAnchorBelief", fmt(qa["consequences"]["belief_ac"]["ratio"])),
        command("ConfirmQwenAnchorBeliefCI", "[" +
                fmt(uncertainty["qwen"]["anchor"]["belief_ac"]["low"]) + ", " +
                fmt(uncertainty["qwen"]["anchor"]["belief_ac"]["high"]) + "]"),
        command("ConfirmQwenAnchorTell", fmt(qa["consequences"]["tell_ac"]["ratio"])),
        command("ConfirmQwenInvariantMin", fmt(100 * min(
            value["add_acc"] for value in qa["invariants"].values()), 0) + r"\%"),
        command("ConfirmQwenCheckpointMax", fmt(qc["max_abs_checkpoint_lam"], 4)),
        command("ConfirmQwenReadoutMax", fmt(max(
            abs(value["lam"]) for value in qc["per_site"]["readout"].values()), 3)),
        command("ConfirmMistralAnchorN", str(ma["n_rows"])),
        command("ConfirmMistralAnchorBelief", fmt(ma["consequences"]["belief_ac"]["ratio"])),
        command("ConfirmMistralAnchorBeliefCI", "[" +
                fmt(uncertainty["mistral"]["anchor"]["belief_ac"]["low"]) + ", " +
                fmt(uncertainty["mistral"]["anchor"]["belief_ac"]["high"]) + "]"),
        command("ConfirmMistralAnchorTell", fmt(ma["consequences"]["tell_ac"]["ratio"])),
        command("ConfirmMistralCheckpointMax", fmt(mc["max_abs_checkpoint_lam"], 4)),
        command("ConfirmMistralReadoutMax", fmt(max(
            abs(value["lam"]) for value in mc["per_site"]["readout"].values()), 3)),
        command("ConfirmQwenFourteenAnchorN", str(q14a["n_rows"])),
        command("ConfirmQwenFourteenAnchorBelief",
                fmt(q14a["consequences"]["belief_ac"]["ratio"])),
        command("ConfirmQwenFourteenAnchorBeliefCI", "[" +
                fmt(uncertainty["qwen14"]["anchor"]["belief_ac"]["low"]) + ", " +
                fmt(uncertainty["qwen14"]["anchor"]["belief_ac"]["high"]) + "]"),
        command("ConfirmQwenFourteenAnchorTell",
                fmt(q14a["consequences"]["tell_ac"]["ratio"])),
        command("ConfirmQwenFourteenCheckpointMax",
                fmt(q14c["max_abs_checkpoint_lam"], 4)),
        command("ConfirmQwenFourteenReadoutMax", fmt(max(
            abs(value["lam"]) for value in q14c["per_site"]["readout"].values()), 3)),
        command("ConfirmQwenFourteenProbeCheckpointLedger",
                fmt(100 * q14p["probe"]["ledger"]["checkpoint"]["accuracy"], 0) + r"\%"),
        command("ConfirmQwenFourteenProbeCheckpointNarrative",
                fmt(100 * q14p["probe"]["narrative"]["checkpoint"]["accuracy"], 0) + r"\%"),
        command("ConfirmQwenFourteenProbeCheckpointCross", fmt(100 * min(
            q14p["cross_surface"]["ledger_to_narrative"]["checkpoint"]["accuracy"],
            q14p["cross_surface"]["narrative_to_ledger"]["checkpoint"]["accuracy"]), 1) +
                "--" + fmt(100 * max(
            q14p["cross_surface"]["ledger_to_narrative"]["checkpoint"]["accuracy"],
            q14p["cross_surface"]["narrative_to_ledger"]["checkpoint"]["accuracy"]), 1) + r"\%"),
        command("ConfirmReverseHistory", fmt(q14r["history_reverse"]["lam"])),
        command("ConfirmReverseVerbal", fmt(q14r["verbal_reverse"]["lam"])),
        command("ConfirmReverseBoth", fmt(q14r["both_reverse"]["lam"])),
        command("ConfirmReverseVerdict", q14["reverse_base"]["verdict"].replace("_", r"\_")),
        command("ConfirmDeepseekMatrixPass",
                str(sum(cell["verdict"] == "PASS" for cell in dmc))),
        command("ConfirmDeepseekMatrixEligible",
                str(sum(cell["verdict"] != "INELICITABLE" for cell in dmc))),
        command("ConfirmDeepseekRatioMin", fmt(min(
            cell["ratio"] for cell in dmc if cell["verdict"] == "PASS"))),
        command("ConfirmDeepseekRatioMax", fmt(max(
            cell["ratio"] for cell in dmc if cell["verdict"] == "PASS"))),
        command("ConfirmDeepseekAnchorBelief",
                fmt(da["consequences"]["belief_ac"]["ratio"])),
        command("ConfirmDeepseekAnchorBeliefCI", "[" +
                fmt(uncertainty["deepseek_llama"]["anchor"]["belief_ac"]["low"]) + ", " +
                fmt(uncertainty["deepseek_llama"]["anchor"]["belief_ac"]["high"]) + "]"),
        command("ConfirmDeepseekAnchorTell",
                fmt(da["consequences"]["tell_ac"]["ratio"])),
        command("ConfirmDeepseekInvariantBeliefBC",
                fmt(100 * da["invariants"]["belief_bc"]["add_acc"], 0) + r"\%"),
        command("ConfirmDeepseekCheckpointMax", fmt(dc["max_abs_checkpoint_lam"], 4)),
        command("ConfirmDeepseekReadoutMax", fmt(max(
            abs(value["lam"]) for value in dc["per_site"]["readout"].values()), 3)),
        command("ConfirmGemmaMatrixPass",
                str(sum(cell["verdict"] == "PASS" for cell in gmc))),
        command("ConfirmGemmaRatioMin", fmt(min(cell["ratio"] for cell in gmc))),
        command("ConfirmGemmaRatioMax", fmt(max(cell["ratio"] for cell in gmc))),
        command("ConfirmGemmaCityTwoHopMin", fmt(min(
            cell["ratio"] for cell in gcity))),
        command("ConfirmGemmaCityTwoHopMax", fmt(max(
            cell["ratio"] for cell in gcity))),
        command("ConfirmGemmaAnchorBelief",
                fmt(ga["consequences"]["belief_ac"]["ratio"])),
        command("ConfirmGemmaAnchorBeliefCI", "[" +
                fmt(uncertainty["gemma3"]["anchor"]["belief_ac"]["low"]) + ", " +
                fmt(uncertainty["gemma3"]["anchor"]["belief_ac"]["high"]) + "]"),
        command("ConfirmGemmaAnchorTell",
                fmt(ga["consequences"]["tell_ac"]["ratio"])),
        command("ConfirmGemmaCheckpointMax", fmt(gc["max_abs_checkpoint_lam"], 4)),
        command("ConfirmGemmaReadoutMax", fmt(max(
            abs(value["lam"]) for value in gc["per_site"]["readout"].values()), 3)),
        command("ConfirmLocusLastSufficientLayer", str(max(locus_sufficient))),
        command("ConfirmLocusSourceForwardMin", fmt(min(
            cell["forward_ratio"] for cell in locus_sufficient.values()))),
        command("ConfirmLocusSourceForwardMax", fmt(max(
            cell["forward_ratio"] for cell in locus_sufficient.values()))),
        command("ConfirmLocusSourceReverseMin", fmt(min(
            cell["reverse_ratio"] for cell in locus_sufficient.values()))),
        command("ConfirmLocusSourceReverseMax", fmt(max(
            cell["reverse_ratio"] for cell in locus_sufficient.values()))),
        command("ConfirmLocusMarkerMax", fmt(max(abs(
            block["marker_only"]["forward_ratio"])
            for block in locus_layers.values()), 4)),
        command("ConfirmLocusSummaryMax", fmt(max(abs(
            block["summary_span"]["forward_ratio"])
            for block in locus_layers.values()), 4)),
        command("ConfirmLocusNoAcMax", fmt(max(abs(
            block["anchors_without_ac"]["forward_ratio"])
            for block in locus_layers.values()), 3)),
        command("ConfirmLocusRandomMax", fmt(max(locus_random), 4)),
        command("ConfirmLocusLThirtySix", fmt(
            locus_layers[36]["source_anchors"]["forward_ratio"], 3)),
        command("ConfirmLocusLFortyOne", fmt(
            locus_layers[41]["source_anchors"]["forward_ratio"], 3)),
        command("ConfirmLocusLFortySix", fmt(
            locus_layers[46]["source_anchors"]["forward_ratio"], 3)),
        command("ConfirmAnchorP", fmt(ma["null"]["p"], 2)),
        command("ConfirmQwenCityTwoHopMin", fmt(min(x["ratio"] for x in q_twohop))),
        command("ConfirmQwenCityTwoHopMax", fmt(max(x["ratio"] for x in q_twohop))),
        command("ConfirmQwenCityTwoHopAccMin",
                fmt(100 * min(x["add_target_acc"] for x in q_twohop), 1) + r"\%"),
        command("ConfirmQwenSpecificityPass", str(q_specificity_pass)),
        command("ConfirmQwenSpecificityIneligible", str(q_specificity_ineligible)),
    ]
    if closeout is not None:
        exact = closeout["exact_ac_only"]
        naturalized = closeout["naturalized"]
        exact_pass = {int(layer): cell for layer, cell in exact["per_layer"].items()
                      if cell["sufficient"]}
        natural_readout = {
            int(layer): block["readout"]
            for layer, block in naturalized["trajectory"]["per_layer"].items()}
        lines += [
            command("ConfirmCloseoutVerdict", closeout["verdict"].replace("_", r"\_")),
            command("ConfirmCloseoutExactLastLayer", str(max(exact_pass))),
            command("ConfirmNaturalAnchorBelief", fmt(
                naturalized["consequences"]["belief_ac"]["ratio"])),
            command("ConfirmNaturalAnchorBeliefCI", "[" + fmt(
                uncertainty["qwen14_closeout"]["naturalized"]["consequences"]
                ["belief_ac"]["low"]) + ", " + fmt(
                uncertainty["qwen14_closeout"]["naturalized"]["consequences"]
                ["belief_ac"]["high"]) + "]"),
            command("ConfirmNaturalAnchorTell", fmt(
                naturalized["consequences"]["tell_ac"]["ratio"])),
            command("ConfirmNaturalTellCleanAcc", fmt(
                100 * naturalized["consequences"]["tell_ac"]["g0_clean"], 0)
                + r"\%"),
            command("ConfirmNaturalTellTargetAcc", fmt(
                100 * naturalized["consequences"]["tell_ac"]["g0_natural"], 0)
                + r"\%"),
            command("ConfirmNaturalInvariant", fmt(
                100 * naturalized["invariant_belief_bc"]["add_acc"], 0) + r"\%"),
            command("ConfirmNaturalInvariantClean", fmt(
                100 * naturalized["invariant_belief_bc"]["clean_acc"], 0)
                + r"\%"),
            command("ConfirmNaturalNullP", fmt(naturalized["null"]["p"], 3)),
            command("ConfirmNaturalCheckpointMax", fmt(
                naturalized["trajectory"]["max_abs_checkpoint_ratio"], 4)),
            command("ConfirmNaturalReadoutFirstLayer", str(min(
                naturalized["trajectory"]["readout_sufficient_layers"]))),
            command("ConfirmNaturalReadoutMax", fmt(max(
                cell["forward_ratio"] for cell in natural_readout.values()), 3)),
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
