#!/usr/bin/env python3
"""Pilot figures; confirmatory artifacts will replace their inputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "pilots"
CONFIRM = ROOT / "evidence" / "confirmatory"
OUT = Path(__file__).resolve().parent / "figures"


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def main():
    qwen = json.loads((CONFIRM /
        "results_delta_preprint_battery_qwen7b_confirm.json").read_text())
    mistral = json.loads((CONFIRM /
        "results_delta_preprint_battery_mistral7b_confirm.json").read_text())
    tasks = ("retrieve", "add2", "sub1", "max5", "gt5label")
    labels = ("retrieve", "+2", "$-1$", "max(.,5)", ">5 label")

    def task_ratios(result, task):
        return np.asarray([block["matrix"]["cells"][task]["ratio"]
                           for block in result["per_seed"].values()])

    def anchor_triplet(result):
        anchor = result["anchor"]["consequences"]["belief_ac"]["ratio"]
        checkpoint = result["checkpoint"]["max_abs_checkpoint_lam"]
        readout = max(abs(value["lam"]) for value in
                      result["checkpoint"]["per_site"]["readout"].values())
        return [anchor, checkpoint, readout]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(len(tasks))
    width = .36
    qvals = np.asarray([task_ratios(qwen, task) for task in tasks])
    mvals = np.asarray([task_ratios(mistral, task) for task in tasks])
    axes[0].bar(x - width / 2, qvals.mean(1), width, label="Qwen2.5-7B",
                color="#3567b7", yerr=np.vstack((qvals.mean(1)-qvals.min(1),
                                                  qvals.max(1)-qvals.mean(1))),
                capsize=2)
    axes[0].bar(x + width / 2, mvals.mean(1), width, label="Mistral-7B",
                color="#d7832f", yerr=np.vstack((mvals.mean(1)-mvals.min(1),
                                                  mvals.max(1)-mvals.mean(1))),
                capsize=2)
    axes[0].axhline(1, color="black", lw=.8, ls="--")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("write effect / textual effect")
    axes[0].set_ylim(0, 1.18)
    axes[0].set_title("Value writes match consequences")
    axes[0].legend(frameon=False, ncol=2, loc="lower left")

    categories = ("source\nanchor", "summary\ncheckpoint", "late\nreadout")
    qtriplet, mtriplet = anchor_triplet(qwen), anchor_triplet(mistral)
    x2 = np.arange(3)
    bars_q = axes[1].bar(x2 - width/2, qtriplet, width, color="#3567b7",
                         label="Qwen2.5-7B")
    bars_m = axes[1].bar(x2 + width/2, mtriplet, width, color="#d7832f",
                         label="Mistral-7B")
    axes[1].axhline(1, color="black", lw=.8, ls="--")
    axes[1].set_xticks(x2, categories)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Matched belief-state transport")
    axes[1].text(1, .04,
                 f"{qtriplet[1]:.3f} / {mtriplet[1]:.3f}",
                 ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "confirm_effect_matching.pdf", bbox_inches="tight")
    fig.savefig(OUT / "confirm_effect_matching.png", dpi=220,
                bbox_inches="tight")
    print(f"wrote {OUT / 'confirm_effect_matching.pdf'}")


if __name__ == "__main__":
    main()
