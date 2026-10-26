"""Plots for the README from results/summary.json and results/robustness.json.

    python make_plots.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import settings as cfg  # noqa: E402

RESULTS = cfg.ROOT / "results"
BLUE, ORANGE, GRAY, INK, INK2, SURFACE = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def percent(k, n):
    return 100 * k / n


def stages(summary):
    n = summary["stage1_one_policy"]["episodes"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1, 2.2]))
    values = [percent(summary["stage1_one_policy"]["success"], n), percent(summary["stage1_manager"]["success"], n)]
    ax1.bar([0, 1], values, color=[ORANGE, BLUE], width=0.6)
    for i, v in enumerate(values):
        ax1.text(i, v + 1.5, f"{v:.1f}", ha="center", color=INK)
    ax1.set_xticks([0, 1], ["one policy", "manager\n(rotation + pushing)"])
    ax1.set_ylim(0, 105), ax1.set_ylabel("success, %")
    ax1.set_title("Stage 1: block pose", loc="left", color=INK)
    cats = ["letter A", "letter R", "letter M", "word ARM"]
    s2 = [percent(summary["stage2_letters"][k]["success"], n) for k in "ARM"] + [percent(summary["stage2_word"]["word_success"], n)]
    s3 = [percent(summary["stage3_letters"][k]["success"], n) for k in "ARM"] + [percent(summary["stage3_word"]["word_success"], n)]
    x = np.arange(4)
    ax2.bar(x - 0.18, s2, width=0.34, color=BLUE, label="stage 2: three letter policies")
    ax2.bar(x + 0.18, s3, width=0.34, color=ORANGE, label="stage 3: one shared policy")
    for xi, a, b in zip(x, s2, s3):
        ax2.text(xi - 0.18, a + 1.5, f"{a:.1f}", ha="center", fontsize=9, color=INK)
        ax2.text(xi + 0.18, b + 1.5, f"{b:.1f}", ha="center", fontsize=9, color=INK)
    ax2.set_xticks(x, cats), ax2.set_ylim(0, 105)
    ax2.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax2.set_title(f"Stages 2 and 3: letters and the word ({n} tasks / scenes each)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "results.png", dpi=140)
    plt.close(fig)


def failures(summary):
    rows = []
    for stage, label in (("stage2_word", "stage 2"), ("stage3_word", "stage 3")):
        s = summary[stage]
        missing = s["scenes"] - s["word_success"] - sum(s["failures"].values())
        rows.append((label, dict(s["failures"], **{"letter not at goal at the end": missing})))
    keys = sorted({k for _, f in rows for k in f}, key=lambda k: -sum(f.get(k, 0) for _, f in rows))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(keys))[::-1]
    ax.barh(y + 0.18, [rows[0][1].get(k, 0) for k in keys], height=0.34, color=BLUE, label=rows[0][0])
    ax.barh(y - 0.18, [rows[1][1].get(k, 0) for k in keys], height=0.34, color=ORANGE, label=rows[1][0])
    ax.set_yticks(y, [k.replace("_", " ") for k in keys])
    ax.set_xlabel("failed scenes")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Why the word fails", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "word_failures.png", dpi=140)
    plt.close(fig)


def robustness(data):
    labels = {"nominal": "nominal", "friction_0.7": "friction x0.7", "friction_1.3": "friction x1.3",
              "mass_0.5": "mass x0.5", "mass_2": "mass x2", "noise_pos_2mm": "position noise 2 mm",
              "noise_pos_5mm": "position noise 5 mm", "noise_ang_1deg": "angle noise 1 deg",
              "noise_ang_3deg": "angle noise 3 deg", "order_RAM": "order R-A-M", "order_MRA": "order M-R-A",
              "layout_gap6": "gap 6 cm", "layout_y+5": "goal row y +5 cm", "layout_x-5": "goal row x -5 cm",
              "layout_y-5": "goal row y -5 cm", "delay_1": "action delay 1 step"}
    rows = [(labels[k], data[k]["stage2"]["word_success"], data[k]["stage3"]["word_success"]) for k in labels]
    spread = data["seed_spread"]["stage2"]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    y = np.arange(len(rows))[::-1]
    ax.axvspan(spread["mean"] - spread["std"], spread["mean"] + spread["std"], color="#e9e8e3",
               label="nominal spread over 5 scene sets")
    for yi, (_, a, b) in zip(y, rows):
        ax.plot([min(a, b), max(a, b)], [yi, yi], color="#c9c8c2", lw=1.5, zorder=1)
    ax.scatter([r[1] for r in rows], y, color=BLUE, s=48, zorder=2, edgecolor=SURFACE, lw=1.5, label="stage 2")
    ax.scatter([r[2] for r in rows], y, color=ORANGE, s=48, zorder=2, edgecolor=SURFACE, lw=1.5, label="stage 3")
    ax.set_yticks(y, [r[0] for r in rows]), ax.set_xlim(55, 100)
    ax.set_xlabel("word assembled, out of 100 new scenes")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3)
    ax.set_title("Robustness (evaluation only)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness.png", dpi=140)
    plt.close(fig)


def main():
    if (RESULTS / "summary.json").exists():
        summary = json.loads((RESULTS / "summary.json").read_text())
        stages(summary)
        failures(summary)
    if (RESULTS / "robustness.json").exists():
        robustness(json.loads((RESULTS / "robustness.json").read_text()))


if __name__ == "__main__":
    main()
