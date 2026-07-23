#!/usr/bin/env python3
"""Build every Paper 1 figure from immutable confirmatory artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "evidence" / "confirmatory"
GENERATED = Path(__file__).resolve().parent / "generated"
OUT = Path(__file__).resolve().parent / "figures"

MODEL_FILES = {
    "Qwen2.5-7B": "results_delta_preprint_battery_qwen7b_confirm.json",
    "Mistral-7B": "results_delta_preprint_battery_mistral7b_confirm.json",
    "DeepSeek-Llama-8B":
        "results_delta_preprint_battery_deepseek_r1_llama8b_confirm.json",
    "Gemma-3-12B":
        "results_delta_preprint_battery_gemma3_12b_confirm.json",
    "Qwen2.5-14B": "results_delta_preprint_battery_qwen14b_headline.json",
}
TASKS = ("retrieve", "add2", "sub1", "max5", "gt5label")
TASK_LABELS = ("retrieve", "+2", "−1", "max(·,5)", ">5 label")
COLORS = ("#3567b7", "#d7832f", "#3a9464", "#9a65b5", "#c34e52")


def read(name: str) -> dict:
    return json.loads((CONFIRM / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / f'{stem}.pdf'}")


def matrix_means(result: dict) -> list[float]:
    means = []
    for task in TASKS:
        values = []
        for block in result["per_seed"].values():
            cell = block["matrix"]["cells"][task]
            ratio = cell.get("ratio")
            if ratio is not None and cell.get("verdict") != "INELICITABLE":
                values.append(float(ratio))
        means.append(float(np.mean(values)) if values else np.nan)
    return means


def mechanism_triplet(result: dict) -> list[float]:
    anchor = abs(float(result["anchor"]["consequences"]["belief_ac"]["ratio"]))
    checkpoint = abs(float(result["checkpoint"]["max_abs_checkpoint_lam"]))
    readout = max(abs(float(v["lam"]))
                  for v in result["checkpoint"]["per_site"]["readout"].values())
    return [anchor, checkpoint, readout]


def figure_family_overview(results: dict[str, dict]) -> None:
    matrix_models = list(MODEL_FILES)[:4]
    matrix = np.asarray([matrix_means(results[name]) for name in matrix_models])
    mechanism_models = list(MODEL_FILES)

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 2.85),
                             gridspec_kw={"width_ratios": [1.28, 1]})
    ax = axes[0]
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e1e1e1")
    im = ax.imshow(masked, aspect="auto", cmap=cmap,
                   norm=TwoSlopeNorm(vmin=.88, vcenter=1, vmax=1.12))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            label = "ineligible" if np.isnan(matrix[row, col]) else f"{matrix[row, col]:.3f}"
            ax.text(col, row, label, ha="center", va="center",
                    fontsize=7, color="#202020")
    ax.set_xticks(range(len(TASKS)), TASK_LABELS, rotation=15, ha="right",
                  fontsize=7.5)
    ax.set_yticks(range(len(matrix_models)), matrix_models)
    ax.set_title("a  Consequence matching across families", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    cbar.set_label("write effect / textual effect")

    ax = axes[1]
    sites = ("source\nanchor", "summary\ncheckpoint", "late\nreadout")
    offsets = np.linspace(-.25, .25, len(mechanism_models))
    for idx, (name, color, offset) in enumerate(
            zip(mechanism_models, COLORS, offsets)):
        values = mechanism_triplet(results[name])
        ax.scatter(np.arange(3) + offset, values, s=28, color=color,
                   marker=("o", "s", "^", "D", "P")[idx], label=name,
                   zorder=3)
    ax.axhline(1, color="#333333", lw=.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(8e-5, 1.6)
    ax.set_xticks(range(3), sites, fontsize=7.5)
    ax.set_ylabel("absolute effect ratio (log scale)")
    ax.set_title("b  Anchor–checkpoint–readout contrast", loc="left")
    ax.grid(axis="y", color="#dddddd", lw=.5)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    fig.tight_layout()
    save(fig, "family-overview")


def figure_causal_handoff(qwen14: dict, closeout: dict) -> None:
    exact = closeout["exact_ac_only"]["per_layer"]
    layers = np.asarray([int(x) for x in exact])
    forward = np.asarray([exact[str(x)]["forward_ratio"] for x in layers])
    reverse = np.asarray([exact[str(x)]["reverse_ratio"] for x in layers])
    source_mid = (forward + reverse) / 2
    source_low = np.minimum(forward, reverse)
    source_high = np.maximum(forward, reverse)
    checkpoint = np.asarray([
        qwen14["checkpoint"]["per_site"]["checkpoint"][str(x)]["lam"]
        for x in layers])
    readout = np.asarray([
        qwen14["checkpoint"]["per_site"]["readout"][str(x)]["lam"]
        for x in layers])

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.fill_between(layers, source_low, source_high, color=COLORS[0],
                    alpha=.18, label="source forward–reverse range")
    ax.plot(layers, source_mid, color=COLORS[0], marker="o", ms=3,
            label="edited anchor only")
    ax.plot(layers, readout, color=COLORS[1], marker="s", ms=3,
            label="late readout")
    ax.plot(layers, checkpoint, color="#555555", marker=".", ms=4,
            label="summary checkpoint")
    ax.axhline(1, color="#333333", lw=.8, ls="--")
    ax.axvline(32, color="#999999", lw=.7, ls=":")
    ax.set_xlim(layers.min() - 1, layers.max() + 1)
    ax.set_ylim(-.07, 1.14)
    ax.set_xticks((2, 8, 16, 24, 32, 36, 41, 46))
    ax.set_xlabel("layer")
    ax.set_ylabel("matched-state effect ratio")
    ax.set_title("a  Causal support hands off after L32", loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")

    probe = qwen14["probe"]
    site_names = ("anchor", "checkpoint", "readout")
    within = [np.mean([probe["probe"][surface][site]["accuracy"]
                       for surface in ("ledger", "narrative")])
              for site in site_names]
    cross = [np.mean([
        probe["cross_surface"]["ledger_to_narrative"][site]["accuracy"],
        probe["cross_surface"]["narrative_to_ledger"][site]["accuracy"]])
             for site in site_names]
    causal = [qwen14["anchor"]["consequences"]["belief_ac"]["ratio"],
              qwen14["checkpoint"]["max_abs_checkpoint_lam"],
              max(abs(v["lam"]) for v in
                  qwen14["checkpoint"]["per_site"]["readout"].values())]
    x = np.arange(3)
    width = .24
    ax = axes[1]
    ax.bar(x - width, within, width, color=COLORS[2], label="within-surface decode")
    ax.bar(x, cross, width, color=COLORS[3], label="cross-surface decode")
    ax.bar(x + width, causal, width, color=COLORS[0], label="causal effect")
    ax.axhline(.125, color="#777777", lw=.7, ls=":", label="probe chance")
    ax.set_xticks(x, site_names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy or effect ratio")
    ax.set_title("b  Decodability is not substitutability", loc="left")
    ax.legend(frameon=False, fontsize=6.7, loc="upper center", ncol=2,
              bbox_to_anchor=(.5, -.19))
    fig.tight_layout()
    fig.subplots_adjust(bottom=.23)
    save(fig, "causal-handoff")


def figure_naturalized_closeout(closeout: dict) -> None:
    summary = json.loads((GENERATED / "confirmatory_summary.json").read_text(
        encoding="utf-8"))["qwen14_closeout"]["naturalized"]["consequences"]
    natural = closeout["naturalized"]
    labels = ("belief", "report")
    estimates = np.asarray([summary["belief_ac"]["estimate"],
                            summary["tell_ac"]["estimate"]])
    lows = np.asarray([summary["belief_ac"]["low"], summary["tell_ac"]["low"]])
    highs = np.asarray([summary["belief_ac"]["high"], summary["tell_ac"]["high"]])

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.65),
                             gridspec_kw={"width_ratios": [1, 1.55]})
    ax = axes[0]
    x = np.arange(2)
    ax.errorbar(x, estimates, yerr=np.vstack((estimates - lows, highs - estimates)),
                fmt="o", color=COLORS[0], capsize=3, ms=5)
    ax.axhline(1, color="#333333", lw=.8, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(.975, 1.03)
    ax.set_ylabel("write effect / textual effect")
    ax.set_title("a  Naturalized effect matching", loc="left")
    ax.text(.04, .06, "p=.0476 vs random", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7)

    belief = natural["consequences"]["belief_ac"]
    report = natural["consequences"]["tell_ac"]
    invariant = natural["invariant_belief_bc"]
    gate_labels = ("belief\nclean", "belief\ntarget", "report\nclean",
                   "report\ntarget", "other\nclean", "other\nafter write")
    gate_values = np.asarray([belief["g0_clean"], belief["g0_natural"],
                              report["g0_clean"], report["g0_natural"],
                              invariant["clean_acc"], invariant["add_acc"]])
    gate_colors = [COLORS[2] if value >= .8 else COLORS[4]
                   for value in gate_values]
    ax = axes[1]
    bars = ax.bar(np.arange(len(gate_values)), gate_values, color=gate_colors)
    ax.axhline(.8, color="#333333", lw=.9, ls="--", label="frozen behavior gate")
    ax.set_xticks(range(len(gate_values)), gate_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("behavioral accuracy")
    ax.set_title("b  The prespecified surface boundary", loc="left")
    for bar, value in zip(bars, gate_values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .025,
                f"{value:.0%}", ha="center", va="bottom", fontsize=7)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "naturalized-closeout")


def main() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    results = {name: read(filename) for name, filename in MODEL_FILES.items()}
    closeout = read("results_delta_paper1_closeout_qwen14b_closeout_v1.json")
    figure_family_overview(results)
    figure_causal_handoff(results["Qwen2.5-14B"], closeout)
    figure_naturalized_closeout(closeout)


if __name__ == "__main__":
    main()
