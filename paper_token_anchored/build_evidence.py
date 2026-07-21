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
        command("ConfirmAnchorP", fmt(ma["null"]["p"], 2)),
        command("ConfirmQwenCityTwoHopMin", fmt(min(x["ratio"] for x in q_twohop))),
        command("ConfirmQwenCityTwoHopMax", fmt(max(x["ratio"] for x in q_twohop))),
        command("ConfirmQwenCityTwoHopAccMin",
                fmt(100 * min(x["add_target_acc"] for x in q_twohop), 1) + r"\%"),
        command("ConfirmQwenSpecificityPass", str(q_specificity_pass)),
        command("ConfirmQwenSpecificityIneligible", str(q_specificity_ineligible)),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
