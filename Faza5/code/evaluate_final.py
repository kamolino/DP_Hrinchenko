"""Run every final test (300 tasks or scenes each) and write results/summary.json and results/SUMMARY.md.

    python evaluate_final.py [--quick]
"""
import argparse
import json
from functools import partial
from pathlib import Path

from common import rollout, settings as cfg
from stage1_cube.cube_env import CubeEnv
from stage1_cube.evaluate import POLICY as CUBE_POLICY
from stage1_cube.manager import Manager, summary as manager_summary
from stage1_cube.tasks import TEST_SEED, make_tasks as cube_tasks, turn_groups
from stage2_letters import word_manager
from stage2_letters.letter_env import TEST_SEEDS, LetterEnv, make_tasks as letter_tasks
from stage3_shared.shared_env import SharedLetterEnv

SHARED = cfg.MODELS / "stage3_shared.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="10 tasks / scenes per test (software check)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default=str(cfg.ROOT / "results"))
    args = parser.parse_args()
    n = 10 if args.quick else 300
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    tasks = cube_tasks(n, TEST_SEED)
    rows, _, _ = rollout.run(tasks, CubeEnv, student_path=CUBE_POLICY, workers=args.workers)
    results["stage1_one_policy"] = dict(rollout.summary(rows), groups=turn_groups(rows, tasks))
    print("stage 1, one policy:", results["stage1_one_policy"]["success"], flush=True)
    manager = Manager(retries=2)
    rows = [manager.run(t) for t in tasks]
    results["stage1_manager"] = manager_summary(rows)
    print("stage 1, manager:", results["stage1_manager"]["success"], flush=True)

    for stage, shared in (("stage2", None), ("stage3", SHARED)):
        letters = {}
        for k, letter in enumerate("ARM"):
            tasks = letter_tasks(letter, n, TEST_SEEDS[letter])
            if shared:
                tasks = [dict(t, id=k * 100000 + t["id"]) for t in tasks]
                rows, _, _ = rollout.run(tasks, SharedLetterEnv, student_path=shared, workers=args.workers)
            else:
                rows, _, _ = rollout.run(tasks, partial(LetterEnv, letter),
                                         student_path=word_manager.LETTER_POLICIES[letter], workers=args.workers)
            letters[letter] = rollout.summary(rows)
        results[f"{stage}_letters"] = letters
        rows = word_manager.evaluate(word_manager.make_scenes(n, word_manager.TEST_SEED), shared=shared,
                                     workers=args.workers)
        (out / f"{stage}_word_episodes.json").write_text(json.dumps(rows, indent=1))
        results[f"{stage}_word"] = word_manager.summarize(rows)
        print(f"{stage}: letters", {k: v["success"] for k, v in letters.items()},
              "word", results[f"{stage}_word"]["word_success"], flush=True)

    (out / "summary.json").write_text(json.dumps(results, indent=1))
    pct = lambda k, total: f"{k}/{total} ({100 * k / total:.1f} %)"  # noqa: E731
    lines = ["# Final test", "", f"{n} tasks or scenes per line, seeds: block {TEST_SEED}, letters "
             f"{TEST_SEEDS['A']}/{TEST_SEEDS['R']}/{TEST_SEEDS['M']}, word {word_manager.TEST_SEED}.", "",
             "| Stage | Task | Model | Success |", "| --- | --- | --- | ---: |",
             f"| 1 | block pose | one policy | {pct(results['stage1_one_policy']['success'], n)} |",
             f"| 1 | block pose | manager: rotation + pushing | {pct(results['stage1_manager']['success'], n)} |"]
    for stage, model in (("stage2", "3 letter policies"), ("stage3", "1 shared policy")):
        for letter in "ARM":
            lines.append(f"| {stage[-1]} | letter {letter} | {model} | {pct(results[f'{stage}_letters'][letter]['success'], n)} |")
        lines.append(f"| {stage[-1]} | **word ARM** | {model} + manager | **{pct(results[f'{stage}_word']['word_success'], n)}** |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
