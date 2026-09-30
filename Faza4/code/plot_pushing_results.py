import argparse
import csv
import html
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    if not path.exists():
        return []
    with path.open() as file:
        return list(csv.DictReader(file))


def update_dashboard(folder, status):
    folder = Path(folder)
    rows = read_rows(folder / "episodes.csv")
    validations = read_rows(folder / "validation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    if rows:
        step = np.array([float(r["timesteps"]) for r in rows])
        for ax, field, title, scale in [
            (axes[0, 0], "reward", "Training reward (last 50 episodes)", 1),
            (axes[1, 0], "is_success", "Training success (last 50 episodes, %)", 100),
            (axes[1, 1], "object_error", "Final object error (last 50 episodes, cm)", 100),
        ]:
            values = np.array([float(r[field]) * scale for r in rows])
            window = min(50, len(values))
            ax.plot(step[window - 1:], np.convolve(values, np.ones(window) / window, mode="valid"), color="#2685be")
            ax.set_title(title)
    if validations:
        axes[0, 1].plot([float(r["timesteps"]) for r in validations],
                        [100 * float(r["success_rate"]) for r in validations], "o-", color="#278c61")
    axes[0, 1].set_title("Validation success (%) — not the final test")
    axes[0, 1].set_ylim(-2, 102)
    for ax in axes.flat:
        ax.set_xlabel("Training steps in this run")
        ax.grid(alpha=0.2)
    fig.savefig(folder / "learning_curves.tmp.png", dpi=125)
    plt.close(fig)
    (folder / "learning_curves.tmp.png").replace(folder / "learning_curves.png")
    status = dict(status, updated=time.strftime("%Y-%m-%d %H:%M:%S"), updated_unix=time.time(), episodes=len(rows))
    if validations:
        status["validation"] = validations[-1]
    (folder / "status.tmp.json").write_text(json.dumps(status, indent=2))
    (folder / "status.tmp.json").replace(folder / "status.json")
    last = validations[-1] if validations else None
    val_text = f'{100 * float(last["success_rate"]):.1f}%' if last else "Pending"
    error = f'{100 * float(last["mean_error"]):.2f} cm' if last else "Pending"
    version = time.time_ns()
    difficulty = status.get("curriculum_fraction")
    curriculum = f"Training range expansion: {100*difficulty:.0f}%. Validation uses the full range." if difficulty is not None else ""
    demo = '<p>The first policy replay will appear after validation.</p>'
    side_checks = ""
    if (folder / "side_goals.png").exists():
        side_checks = f'<h2>Small changes in goal position</h2><p>Same checkpoint and cube start; goal Y = 0, ±2, ±4 and ±6 cm. Development checks, not the final benchmark.</p><img src="side_goals.png?v={version}" alt="Seven fixed side-goal checks">'
    if (folder / "latest_demo.gif").exists():
        demo = f'<img src="latest_demo.gif?v={version}" alt="Latest validation policy replay">'
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30">
<title>FR3 PPO — training progress</title>
<style>body{{font:16px system-ui;background:#101b29;color:#e8eef6;margin:30px auto;max-width:1120px;padding:0 20px}}
h1{{font-size:30px}}.cards{{display:flex;flex-wrap:wrap;gap:14px}}.card{{background:#203247;padding:18px;flex:1;border-radius:8px}}
strong{{display:block;font-size:27px;margin-top:6px}}img{{max-width:100%;border-radius:8px}}.muted{{color:#aab9cb}}a{{color:#8dcfff}}</style>
<h1>Franka FR3 · PPO pushing</h1><p id="freshness"></p><p class="muted">{html.escape(folder.name)} · Stage {html.escape(str(status["stage"]))} · {html.escape(status["state"])} · Updated {status["updated"]}</p>
<div class="cards"><div class="card">Training steps<strong>{status["timesteps"]:,}</strong>Target: {status.get("target_steps", "see config.json")}</div>
<div class="card">Validation success<strong>{val_text}</strong></div><div class="card">Validation mean error<strong>{error}</strong></div></div>
<p>The orange cube must stay within 2 cm of the green goal for 0.3 s. Validation uses fixed seeds for development. These are not independent final-test results.</p>
<p>{curriculum}</p>
<p>Total steps in this stage, including earlier runs: {status.get("stage_steps", status["timesteps"]):,}. The charts below show this run only.</p>
<img src="learning_curves.png?v={version}" alt="Reward, success and object error curves">
{side_checks}
<h2>Latest validation replay</h2>{demo}
<p class="muted">The replay uses a fixed validation task. Failed attempts are shown too. The page refreshes every 30 seconds; policy replays update at validation checkpoints.</p>
<p><a href="status.json">Status data</a> · <a href="episodes.csv">Episode log</a> · <a href="validation.csv">Validation log</a></p>
<script>
const updated = {status["updated_unix"]};
const training = {str(status["state"] == "training").lower()};
function showAge() {{
 const age = Math.max(0, Math.floor(Date.now()/1000 - updated));
 document.getElementById("freshness").textContent = training && age > 180
  ? "No update for " + Math.floor(age/60) + " minutes. The process may be paused or stopped; check its log."
  : "Last data update: " + age + " seconds ago.";
}}
showAge(); setInterval(showAge, 1000);
</script></html>''' 
    (folder / "progress.tmp.html").write_text(text)
    (folder / "progress.tmp.html").replace(folder / "progress.html")
    overview = text.replace("<title>", f'<base href="{html.escape(folder.name)}/"><title>', 1)
    (folder.parent / "progress.tmp.html").write_text(overview)
    (folder.parent / "progress.tmp.html").replace(folder.parent / "progress.html")



def stage_rows(folder, filename, seen=None):
    folder = Path(folder)
    seen = set() if seen is None else seen
    if folder.resolve() in seen:
        raise ValueError("Training history contains a cycle.")
    seen.add(folder.resolve())
    config = json.loads((folder / "config.json").read_text())
    offset = config.get("stage_steps_before_run", 0)
    rows = []
    if config.get("previous_run"):
        previous = folder.parent / Path(config["previous_run"]).name
        if not previous.exists():
            previous = Path(config["previous_run"])
        rows = [row for row in stage_rows(previous, filename, seen)
                if float(row["timesteps"]) <= offset]
    for row in read_rows(folder / filename):
        row = dict(row)
        row["timesteps"] = offset + float(row["timesteps"])
        rows.append(row)
    return rows


def plot_training_history(folder, output):
    episodes = stage_rows(folder, "episodes.csv")
    validations = stage_rows(folder, "validation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    if episodes:
        step = np.array([float(row["timesteps"]) for row in episodes])
        for ax, key, title, scale in [
            (axes[0, 0], "reward", "Training reward", 1),
            (axes[1, 0], "length", "Episode length (control steps)", 1),
            (axes[1, 1], "object_error", "Final error (cm)", 100),
        ]:
            values = np.array([float(row[key])*scale for row in episodes])
            window = min(50, len(values))
            ax.plot(step[window-1:], np.convolve(values, np.ones(window)/window, mode="valid"))
            ax.set_title(title + " — mean of last 50 episodes")
        values = np.array([float(row["is_success"])*100 for row in episodes])
        window = min(50, len(values))
        axes[0, 1].plot(step[window-1:], np.convolve(values, np.ones(window)/window, mode="valid"),
                        alpha=.6, label="Training, last 50 episodes")
    if validations:
        axes[0, 1].plot([float(row["timesteps"]) for row in validations],
                        [float(row["success_rate"])*100 for row in validations], "o-", label="Fixed validation tasks")
    axes[0, 1].set(title="Success rate (%)", ylim=(-2, 102))
    axes[0, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Accumulated steps in this stage")
        ax.grid(alpha=.2)
    fig.savefig(output, dpi=150)
    plt.close(fig)



def plot_goal_coverage(folder):
    folder = Path(folder)
    episodes = json.loads((folder / "evaluation.json").read_text())["episodes"]
    sectors = {name: dict(successes=0, episodes=0) for name in ["Forward (+X)", "Side (+Y)", "Backward (-X)", "Side (-Y)"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for success, color, label in [(True, "#218d64", "Success"), (False, "#c34444", "Failure")]:
        group = [e for e in episodes if bool(e["is_success"]) == success]
        if group:
            offsets = np.array([np.asarray(e["goal"]) - np.asarray(e["object"])[:2] for e in group]) * 100
            axes[0].scatter(offsets[:, 0], offsets[:, 1], c=color, label=label, alpha=.7, s=25)
    for episode in episodes:
        dx, dy = np.asarray(episode["goal"]) - np.asarray(episode["object"])[:2]
        sector = ("Forward (+X)" if dx >= 0 else "Backward (-X)") if abs(dx) >= abs(dy) else ("Side (+Y)" if dy >= 0 else "Side (-Y)")
        sectors[sector]["episodes"] += 1
        sectors[sector]["successes"] += int(episode["is_success"])
    axes[0].scatter([0], [0], c="#e79824", marker="s", label="Cube start")
    axes[0].set(xlabel="Goal minus cube start: X (cm)", ylabel="Y (cm)", title="Frozen C benchmark: all saved tasks", aspect="equal")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    values = [100 * r["successes"] / r["episodes"] if r["episodes"] else 0 for r in sectors.values()]
    bars = axes[1].bar(list(sectors), values, color="#328bb2")
    axes[1].bar_label(bars, labels=[f'{r["successes"]}/{r["episodes"]}' for r in sectors.values()], padding=3)
    axes[1].set(ylim=(0, 110), ylabel="Success rate (%)", title="90-degree sectors; counts shown above bars")
    axes[1].tick_params(axis="x", labelsize=9)
    fig.savefig(folder / "goal_coverage.png", dpi=150)
    plt.close(fig)
    (folder / "direction_summary.json").write_text(json.dumps(sectors, indent=2))

def plot_evaluation(folder):
    folder = Path(folder)
    plot_goal_coverage(folder)
    results = json.loads((folder / "evaluation.json").read_text())
    episodes = results["episodes"]
    successes = [i for i, e in enumerate(episodes) if e["is_success"]]
    failures = [i for i, e in enumerate(episodes) if not e["is_success"]]
    # First, middle and last success; include the first failure when available.
    chosen = successes[::max(1, len(successes) // 2)][:3]
    if failures:
        chosen = chosen[:2] + failures[:1]
    for i in range(len(episodes)):
        if len(chosen) >= 3:
            break
        if i not in chosen:
            chosen.append(i)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, index in zip(axes, chosen):
        episode = episodes[index]
        obj = np.array(episode["object_trajectory"])
        tcp = np.array(episode["tcp_trajectory"])
        goal = episode["goal"]
        ax.plot(obj[:, 0], obj[:, 1], color="#df7319", label="Cube path")
        ax.plot(tcp[:, 0], tcp[:, 1], color="#2685be", label="Tool path", alpha=0.7)
        ax.scatter(*obj[0, :2], marker="s", color="#df7319")
        ax.add_patch(plt.Circle(goal, 0.02, color="#30b56d", alpha=0.3))
        ax.scatter(*goal, marker="+", color="#158344")
        ax.set_title(f'Task {index + 1}: {episode["reason"]}\nerror {episode["object_error"] * 100:.2f} cm')
        ax.set_aspect("equal")
        ax.set_xlim(0.24, 0.78)
        ax.set_ylim(-0.23, 0.23)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(folder / "trajectories.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3), layout="constrained")
    for ax, index in zip(axes, chosen):
        episode = episodes[index]
        tcp = np.array(episode["tcp_trajectory"])
        times = np.arange(len(tcp)) * 0.05
        times[-1] = episode["sim_time"]
        ax.plot(times, tcp[:, 2] * 100, color="#2685be")
        ax.axhline(23, color="#df7319", linestyle="--", label="Cube top")
        ax.set(title=f'Task {index+1}: {episode["reason"]}', xlabel="Simulation time (s)", ylabel="TCP height (cm)")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(folder / "tcp_height.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    plot_evaluation(args.folder)
