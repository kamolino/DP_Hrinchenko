"""Tables and plots for the Phase 6 ablations from results/<experiment>/... (summary.json + scenes.csv).

    python -m faza6.report
"""
import csv
import json
from pathlib import Path

import numpy as np

from common import settings as cfg
from faza6 import stats

RESULTS = cfg.ROOT / "results"
PLOTS = cfg.ROOT / "plots"
STAGES = {"stage2": "Stage 2 (3 policies)", "stage3": "Stage 3 (shared policy)"}
A1_VARIANTS = [("repair0", "A1-0"), ("repair1", "A1-1"), ("repair2", "A1-2"), ("repair4", "A1-4")]
A2_VARIANTS = [("segment0", "A2-direct"), ("segment0.06", "A2-6"), ("segment0.12", "A2-12"), ("segment0.24", "A2-24")]


def load(path):
    path = Path(path)
    if not (path / "summary.json").exists():
        return None
    summary = json.loads((path / "summary.json").read_text())
    with open(path / "scenes.csv") as f:
        scenes = list(csv.DictReader(f))
    return dict(summary=summary["summary"], meta=summary, scenes=scenes,
                success=np.array([int(r["word_success"]) for r in scenes]),
                disturbed=np.array([int(r["disturbed"]) for r in scenes]),
                ids=[int(r["id"]) for r in scenes])


def row(name, v):
    s, k, n = v["summary"], int(v["success"].sum()), len(v["success"])
    low, high = stats.wilson(k, n)
    d = int(v["disturbed"].sum())
    return dict(variant=name, success=f"{k}/{n}", percent=round(100 * k / n, 1),
                ci=f"{100 * low:.1f}–{100 * high:.1f}", mean_steps=round(s["mean_steps"]),
                median_steps=round(s["median_steps"]), disturbed=f"{d}/{n} ({100 * d / n:.0f} %)",
                repair_phases=round(s["mean_repair_phases"], 2), pieces=round(s["mean_pieces_first_pass"], 2),
                categories=", ".join(f"{c} {m}" for c, m in sorted(s["categories"].items(), key=lambda t: -t[1])
                                     if c != "success"))


def comparison(name, a, b):
    assert a["ids"] == b["ids"], "variants must use the same scenes"
    c = stats.compare(a["success"], b["success"])
    p = "< 0.001" if c["p_mcnemar"] < 0.001 else f"{c['p_mcnemar']:.3f}"
    return dict(comparison=name, diff_pp=round(c["diff_pp"], 1), ci_pp=f"{c['ci_pp'][0]:.1f} … {c['ci_pp'][1]:.1f}",
                p=p, discordant=f"{c['only_first']} / {c['only_second']}")


def markdown(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


MAIN = ["variant", "success", "percent", "ci", "mean_steps", "median_steps", "disturbed", "repair_phases", "pieces",
        "categories"]
COMPARE = ["comparison", "diff_pp", "ci_pp", "p", "discordant"]


def errors_table(variants):
    out = []
    for name, v in variants:
        s = v["summary"]
        out.append(dict(variant=name, **{f"{n}": "{:.2f} / {:.2f} cm, {:.1f} / {:.1f}°".format(
            *s[f"{n}_cm_median_p90"], *s[f"{n}_deg_median_p90"]) for n in "ARM"}))
    return out


def experiment(prefix, variants, reference, consecutive):
    text, data = [], {}
    for stage, title in STAGES.items():
        loaded = [(label, load(RESULTS / prefix / stage / key)) for key, label in variants]
        loaded = [(label, v) for label, v in loaded if v is not None]
        if not loaded:
            continue
        data[stage] = loaded
        text += [f"### {title}", "", markdown([row(l, v) for l, v in loaded], MAIN), ""]
        by = dict(loaded)
        pairs = []
        if consecutive:
            pairs += [(b, a) for (a, _), (b, _) in zip(loaded, loaded[1:])]
        pairs += [(l, reference) for l, _ in loaded if l != reference and (l, reference) not in pairs]
        comps = [comparison(f"{x} − {y}", by[x], by[y]) for x, y in pairs if x in by and y in by]
        if comps:
            text += ["Paired differences on the same scenes (95 % CI by Newcombe, exact McNemar; discordant = "
                     "scenes solved only by the first / only by the second):", "", markdown(comps, COMPARE), ""]
        text += ["Final errors, median / P90:", "", markdown(errors_table(loaded), ["variant", "A", "R", "M"]), ""]
    return text, data


def plot(data, labels, path, xlabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(labels))
    for stage, marker in (("stage2", "o"), ("stage3", "s")):
        if stage not in data:
            continue
        by = dict(data[stage])
        vs = [by.get(l) for l in labels]
        idx = [i for i, v in enumerate(vs) if v is not None]
        p = np.array([100 * vs[i]["success"].mean() for i in idx])
        ci = np.array([stats.wilson(int(vs[i]["success"].sum()), len(vs[i]["success"])) for i in idx]) * 100
        axes[0].errorbar(x[idx] + (0.05 if stage == "stage3" else -0.05), p, yerr=[p - ci[:, 0], ci[:, 1] - p],
                         marker=marker, capsize=4, label=STAGES[stage])
        axes[1].plot(x[idx], [vs[i]["summary"]["mean_steps"] for i in idx], marker=marker, label=STAGES[stage])
    for ax, ylabel in zip(axes, ["word ARM success [%] (95 % CI)", "mean control steps per scene"]):
        ax.set_xticks(x, labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


TRAIN_SEEDS = [0, 10000, 20000]
TRAINED = {"A3-start": "A3/start", "A3-single": "A3/single/seed{}", "A3-mixed": "A3/mixed/seed{}",
           "A4-bez": "A4/bez/seed{}", "A4-s (= A3-mixed)": "A3/mixed/seed{}",
           "A5-id": "A5/id/seed{}", "A5-noid": "A5/noid/seed{}"}


def load_trained(label):
    pattern = TRAINED[label]
    paths = [pattern] if "{}" not in pattern else [pattern.format(s) for s in TRAIN_SEEDS]
    runs = []
    for seed, p in zip(["-"] if "{}" not in pattern else TRAIN_SEEDS, paths):
        v = load(RESULTS / p)
        if v is None:
            continue
        letters = RESULTS / p / "letters.json"
        v["letters"] = json.loads(letters.read_text()) if letters.exists() else None
        v["seed"] = seed
        runs.append(v)
    return runs


def trained_rows(label, runs):
    out = []
    for v in runs:
        r = row(f"{label} seed {v['seed']}", v)
        r["letters"] = " / ".join(str(v["letters"][n]["success"]) for n in "ARM") if v["letters"] else "-"
        r["events"] = round(v["summary"]["disturbances"] / len(v["success"]), 2)
        out.append(r)
    if len(runs) > 1:
        mean, low, high = stats.seed_bootstrap([v["success"] for v in runs])
        rates = [100 * v["success"].mean() for v in runs]
        dist = [100 * v["disturbed"].mean() for v in runs]
        let = [np.mean([v["letters"][n]["success"] for v in runs if v["letters"]]) for n in "ARM"] \
            if all(v["letters"] for v in runs) else None
        out.append(dict(variant=f"**{label} mean of {len(runs)} seeds**", success="-",
                        percent=f"{np.mean(rates):.1f} ± {np.std(rates, ddof=1):.1f}",
                        ci=f"{100 * low:.1f}–{100 * high:.1f} (seeds+scenes)",
                        mean_steps=round(np.mean([v["summary"]["mean_steps"] for v in runs])),
                        median_steps="-", disturbed=f"{np.mean(dist):.0f} %", repair_phases=round(np.mean(
                            [v["summary"]["mean_repair_phases"] for v in runs]), 2), pieces="-", categories="-",
                        letters=" / ".join(f"{x:.0f}" for x in let) if let else "-",
                        events=round(np.mean([v["summary"]["disturbances"] / len(v["success"]) for v in runs]), 2)))
    return out


def trained_comparison(name, x_runs, y_runs, key="success"):
    diff, low, high = stats.seed_bootstrap_diff([v[key] for v in x_runs], [v[key] for v in y_runs])
    per_seed = []
    if len(y_runs) == 1:  # against the single start model: paired McNemar per training seed
        comps = [stats.compare(v[key], y_runs[0][key]) for v in x_runs]
        per_seed = [f"{c['diff_pp']:+.1f} (p {c['p_mcnemar']:.3f})" for c in comps]
    return dict(comparison=name, metric=key, diff_pp=round(100 * diff, 1), ci_pp=f"{100 * low:.1f} … {100 * high:.1f}",
                per_seed=", ".join(per_seed) or "-")


TRAINED_COLUMNS = ["variant", "success", "percent", "ci", "mean_steps", "disturbed", "events", "repair_phases",
                   "letters", "categories"]
TRAINED_COMPARE = ["comparison", "metric", "diff_pp", "ci_pp", "per_seed"]


def plot_trained(groups, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (label, runs) in enumerate(groups):
        for ax, key in zip(axes, ["success", "disturbed"]):
            values = [100 * v[key].mean() for v in runs]
            ax.scatter([i] * len(values), values, color="C0", zorder=3, label="training seed" if i == 0 else None)
            ax.hlines(np.mean(values), i - 0.25, i + 0.25, color="C3", zorder=2, label="mean" if i == 0 else None)
    for ax, ylabel in zip(axes, ["word ARM success [%]", "scenes with a disturbed letter [%]"]):
        ax.set_xticks(range(len(groups)), [g for g, _ in groups])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title + " - 500 final scenes, one point per training seed")
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trained_section(title, labels, comparisons, plot_path, extra=None):
    groups = [(label, load_trained(label)) for label in labels]
    groups = [(l, r) for l, r in groups if r]
    if not groups:
        return []
    text = [title, "", "Word ARM on the 500 final scenes; letters = single-letter tasks solved out of 300 "
            "(new sets 1301000 / 1311000 / 1321000); events = disturbance events per scene.", "",
            markdown([r for l, runs in groups for r in trained_rows(l, runs)], TRAINED_COLUMNS), ""]
    by = dict(groups)
    comps = [trained_comparison(f"{x} − {y}", by[x], by[y], key) for x, y, key in comparisons if x in by and y in by]
    if comps:
        text += ["Differences with a 95 % bootstrap interval that resamples training seeds and scenes; per seed = "
                 "difference and exact McNemar p of each trained model against the one start model.", "",
                 markdown(comps, TRAINED_COMPARE), ""]
    plot_trained(groups, plot_path, title.strip("# "))
    return text


def main():
    text = ["# Phase 6 results", ""]
    a1, d1 = experiment("A1", A1_VARIANTS, "A1-4", consecutive=True)
    if d1:
        text += ["## A1 - repair passes", ""] + a1
        plot(d1, [l for _, l in A1_VARIANTS], PLOTS / "A1_repairs.png", "maximum number of repair passes",
             "A1 - repair passes")
    a2, d2 = experiment("A2", A2_VARIANTS, "A2-12", consecutive=False)
    if d2:
        text += ["## A2 - route segments (one budget of 1600 policy steps per placement attempt)", ""] + a2
        plot(d2, [l for _, l in A2_VARIANTS], PLOTS / "A2_segments.png", "segment length", "A2 - route segments")
    text += trained_section("## A3 - training states (three letter policies, Stage 2)",
                            ["A3-start", "A3-single", "A3-mixed"],
                            [("A3-mixed", "A3-single", "success"), ("A3-single", "A3-start", "success"),
                             ("A3-mixed", "A3-start", "success"), ("A3-mixed", "A3-single", "disturbed")],
                            PLOTS / "A3_training_states.png")
    text += trained_section("## A4 - disturbance penalty (three letter policies, Stage 2)",
                            ["A4-bez", "A4-s (= A3-mixed)"],
                            [("A4-bez", "A4-s (= A3-mixed)", "disturbed"), ("A4-bez", "A4-s (= A3-mixed)", "success")],
                            PLOTS / "A4_disturbance_penalty.png")
    text += trained_section("## A5 - one-hot letter ID (shared policy, Stage 3)", ["A5-noid", "A5-id"],
                            [("A5-id", "A5-noid", "success"), ("A5-id", "A5-noid", "disturbed")], PLOTS / "A5_letter_id.png")
    (RESULTS / "RESULTS.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
