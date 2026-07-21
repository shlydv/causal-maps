#!/usr/bin/env python3
"""Build publication figures and TeX macros from frozen run artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
GEN = OUT / "generated"


def load(relative: str) -> dict:
    with (RUNS / relative).open() as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )

    p1 = load("p1_7b/results_p1_both_7b.json")["skills"]
    frag_var = load("p1_7b/fragility_variable_p1.json")
    frag_comp = load("p1_7b/fragility_completion_p1.json")
    transfer = load("delta_transfer/results_delta_transfer.json")
    robust = load("delta_var_robust/results_delta_var_robust.json")
    controls = load("delta_var_shufflefix/results_delta_var_shufflefix.json")
    completion = load("delta_completion/results_delta_completion.json")
    scale = load("delta_var_1p5b/results_delta_var_1p5b.json")
    crosspos = load("delta_var_crosspos/results_delta_var_crosspos.json")

    # Fail loudly if the frozen scientific verdicts no longer match the paper.
    assert not p1["variable_p1"]["p1"]["pass"]
    assert not p1["completion_p1"]["p1"]["pass"]
    assert transfer["pass"] and transfer["verdict"] == "DIRECTION_REUSABLE"
    assert robust["embed_verdict"] == "NONTRIVIAL"
    assert controls["verdict"] == "GENERIC_BOOST"
    assert completion["pass"] and scale["pass"] and crosspos["pass"]

    # Figure 1: the coordinate-system contrast.
    labels = ["Variable", "Completion"]
    full_grid = [
        frag_var["mean_pairwise_rho"],
        frag_comp["mean_pairwise_rho"],
    ]
    expected_col = [
        frag_var["mean_expected_col_r"],
        frag_comp["mean_expected_col_r"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(4.7, 2.8))
    ax.bar(x - width / 2, full_grid, width, label="full layer×position map")
    ax.bar(x + width / 2, expected_col, width, label="expected-site column")
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1, label="P1 threshold")
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel("cross-template correlation")
    ax.set_title("Site-map replication depends on the scored object")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "coordinate_system")

    # Figure 2: every primary held-out transfer cell, grouped by generalization.
    groups = {
        "Variable\n7B": transfer["primary"]["rows"],
        "Completion\n7B": completion["slots"]["primary_62"]["rows"],
        "Variable\n1.5B": scale["layers"]["L2"]["rows"],
        "short→long\n7B": crosspos["directions"]["short_to_long"]["rows"],
        "long→short\n7B": crosspos["directions"]["long_to_short"]["rows"],
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    rng = np.random.default_rng(0)
    means = []
    for i, (label, rows) in enumerate(groups.items()):
        ratios = np.asarray([r["cross_over_within"] for r in rows])
        means.append(float(ratios.mean()))
        jitter = rng.uniform(-0.09, 0.09, size=len(ratios))
        ax.scatter(np.full(len(ratios), i) + jitter, ratios, s=28, zorder=3)
        ax.plot([i - 0.18, i + 0.18], [ratios.mean(), ratios.mean()],
                color="black", linewidth=1.5)
    ax.axhline(1.0, color="0.45", linewidth=1, label="within-template strength")
    ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1, label="PASS threshold")
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_ylim(0.35, 1.23)
    ax.set_ylabel("cross / within ΔIE")
    ax.set_title("Direction transfer replicates across surfaces, size, and position")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save_figure(fig, "transfer_ratios")

    # Figure 3: control result that sets the exact claim strength.
    control_names = ["matched Δ", "wrong-value Δ", "anti −Δ", "embedding Δ"]
    control_ie = [
        controls["matched_confirm"]["mean_cross_ie"],
        controls["wrong_value_control"]["mean_cross_ie"],
        controls["anti_delta_control"]["mean_cross_ie"],
        robust["embed_control"]["mean_cross_ie"],
    ]
    colors = ["#2878B5", "#F39C35", "#C23B3B", "#777777"]
    fig, ax = plt.subplots(figsize=(5.1, 2.9))
    bars = ax.bar(control_names, control_ie, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("mean ΔIE")
    ax.set_title("Controls identify a coarse, signed slot-update direction")
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in control_ie], padding=3, fontsize=8)
    ax.set_ylim(min(control_ie) - 0.35, max(control_ie) + 0.45)
    save_figure(fig, "controls")

    summary = {
        "p1": {
            "variable_gate_rho": p1["variable_p1"]["p1"]["rho"],
            "variable_gate_p": p1["variable_p1"]["p1"]["p"],
            "variable_pairwise_grid_rho": frag_var["mean_pairwise_rho"],
            "variable_expected_column_r": frag_var["mean_expected_col_r"],
            "completion_gate_rho": p1["completion_p1"]["p1"]["rho"],
            "completion_gate_p": p1["completion_p1"]["p1"]["p"],
            "variable_site_ie": p1["variable_p1"]["site_stats"]["real"],
            "variable_site_p": p1["variable_p1"]["site_stats"]["p"],
        },
        "transfer": {
            "variable_7b_mean_ratio": transfer["primary"]["mean_ratio"],
            "variable_7b_ratios": [
                r["cross_over_within"] for r in transfer["primary"]["rows"]
            ],
            "completion_mean_ratio": completion["slots"]["primary_62"]["mean_ratio"],
            "completion_flip_rate": mean(
                [r["flip_rate"] for r in completion["slots"]["primary_62"]["rows"]]
            ),
            "variable_1p5b_mean_ratio": scale["layers"]["L2"]["mean_ratio"],
            "short_to_long_mean_ratio": crosspos["directions"]["short_to_long"]["mean_ratio"],
            "long_to_short_mean_ratio": crosspos["directions"]["long_to_short"]["mean_ratio"],
        },
        "controls": {
            "embed_over_l2": robust["embed_control"]["cross_embed_over_cross_L2"],
            "matched_mean_ratio": controls["matched_confirm"]["mean_ratio"],
            "wrong_mean_ratio": controls["wrong_value_control"]["mean_ratio"],
            "matched_wrong_cosine": controls["cos_matched_wrong"],
            "anti_mean_ie": controls["anti_delta_control"]["mean_cross_ie"],
        },
    }
    # Lead figure: two-regime accessibility curves (H1).
    # Store = Variable site-IE at expected position (retrieve); others =
    # direction-add effects from boundary runs. Normalized per skill to max|·|.
    select = load("delta_select/results_delta_select.json")
    transform = load("delta_transform/results_delta_transform.json")
    instruction = load("delta_instruction/results_delta_instruction.json")
    ie_var = np.load(RUNS / "p1_7b/ie_variable_p1.npz", allow_pickle=True)
    ie_mean = ie_var["ie_mean"]
    ie_layers = ie_var["layers"].astype(int)
    ie_positions = ie_var["positions"].astype(int)
    var_meta = load("p1_7b/results_variable_p1.json")
    exp_pos = int(var_meta["expected_idx"])
    pos_i = int(np.where(ie_positions == exp_pos)[0][0])
    sweep_layers = [2, 8, 14, 20, 26]

    def _norm_curve(xs):
        m = max(abs(v) for v in xs) or 1.0
        return [v / m for v in xs]

    store_raw = []
    for L in sweep_layers:
        Li = int(np.where(ie_layers == L)[0][0])
        store_raw.append(float(ie_mean[Li, pos_i]))
    sel_raw = [float(r["route"]["effect"]) for r in select["per_layer"]]
    # Transform: computed-value selectivity (cc)
    xf_by_L = {int(r["layer"]): float(r["cc"]["selectivity"]) for r in transform["per_layer"]}
    xf_raw = [xf_by_L[L] for L in sweep_layers]
    instr_raw = [float(r["add_to_data"]["effect"]) for r in instruction["per_layer"]]

    curves = {
        "Store (retrieve, site-IE)": (_norm_curve(store_raw), "#2A6F97", [2]),
        "Select (route)": (_norm_curve(sel_raw), "#1B9E77", select["sig_layers"]),
        "Transform (create)": (_norm_curve(xf_raw), "#D95F02", transform["cc_significant_layers"]),
        "Instruction (designate)": (
            _norm_curve(instr_raw), "#7570B3", instruction["sig_layers"]),
    }
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.axvspan(-0.3, 2.3, color="#2A6F97", alpha=0.06, label="early/mid (≤L14)")
    ax.axvspan(2.7, 4.3, color="#D95F02", alpha=0.06, label="late (≥L20)")
    x = np.arange(len(sweep_layers))
    for name, (ys, color, sig) in curves.items():
        ax.plot(x, ys, marker="o", color=color, label=name, linewidth=1.8)
        for i, L in enumerate(sweep_layers):
            if L in set(sig or []):
                ax.scatter([i], [ys[i]], s=70, facecolors=color, edgecolors="black",
                           linewidths=0.8, zorder=5)
    ax.axhline(0, color="0.5", linewidth=0.7)
    ax.set_xticks(x, [f"L{L}" for L in sweep_layers])
    ax.set_ylabel("normalized effect")
    ax.set_title("Two regimes: retrieve/route early–strong; create/designate late–weak")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_ylim(-0.15, 1.15)
    save_figure(fig, "two_regimes")

    summary["boundary"] = {
        "store_sig_layers": [2, 8, 14],
        "select_sig_layers": select["sig_layers"],
        "transform_sig_layers": transform["cc_significant_layers"],
        "instruction_sig_layers": instruction["sig_layers"],
        "select_best_route": select["best_route"],
        "transform_best_ratio": transform["best_ratio_cc_over_ss"],
        "instruction_best_add": instruction["best_add_to_data"],
        "instruction_verdict": instruction["verdict"],
        "select_verdict": select["verdict"],
        "transform_verdict": transform["verdict"],
    }

    with (GEN / "results_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    macros = rf"""% Generated by paper/build_figures.py. Do not edit by hand.
\newcommand{{\VarGateRho}}{{{summary['p1']['variable_gate_rho']:.2f}}}
\newcommand{{\VarGateP}}{{{summary['p1']['variable_gate_p']:.2f}}}
\newcommand{{\VarPairRho}}{{{summary['p1']['variable_pairwise_grid_rho']:.2f}}}
\newcommand{{\VarColumnR}}{{{summary['p1']['variable_expected_column_r']:.3f}}}
\newcommand{{\CompGateRho}}{{{summary['p1']['completion_gate_rho']:.2f}}}
\newcommand{{\VarSiteIE}}{{{summary['p1']['variable_site_ie']:.1f}}}
\newcommand{{\VarTransferRatio}}{{{summary['transfer']['variable_7b_mean_ratio']:.2f}}}
\newcommand{{\CompTransferRatio}}{{{summary['transfer']['completion_mean_ratio']:.2f}}}
\newcommand{{\ScaleTransferRatio}}{{{summary['transfer']['variable_1p5b_mean_ratio']:.2f}}}
\newcommand{{\ShortLongRatio}}{{{summary['transfer']['short_to_long_mean_ratio']:.2f}}}
\newcommand{{\LongShortRatio}}{{{summary['transfer']['long_to_short_mean_ratio']:.2f}}}
\newcommand{{\WrongRatio}}{{{summary['controls']['wrong_mean_ratio']:.2f}}}
\newcommand{{\MatchedWrongCos}}{{{summary['controls']['matched_wrong_cosine']:.2f}}}
\newcommand{{\EmbedRatio}}{{{summary['controls']['embed_over_l2']:.3f}}}
\newcommand{{\AntiIE}}{{{summary['controls']['anti_mean_ie']:.2f}}}
\newcommand{{\SelectBestRoute}}{{{summary['boundary']['select_best_route']:.1f}}}
\newcommand{{\TransformRatio}}{{{summary['boundary']['transform_best_ratio']:.2f}}}
\newcommand{{\InstrBestAdd}}{{{summary['boundary']['instruction_best_add']:.1f}}}
"""
    (GEN / "results_macros.tex").write_text(macros)
    print(f"Wrote figures to {FIG}")
    print(f"Wrote frozen summary to {GEN / 'results_summary.json'}")


if __name__ == "__main__":
    main()
