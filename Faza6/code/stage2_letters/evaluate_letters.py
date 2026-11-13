"""Single-letter tasks for the three letter policies (stage 2) or the shared policy (stage 3).

    python -m stage2_letters.evaluate_letters --set test
    python -m stage2_letters.evaluate_letters --set test --shared ../models/stage3_shared.zip
"""
import argparse
import json
from functools import partial
from pathlib import Path

from common import rollout, settings as cfg
from stage2_letters.letter_env import DEV_SEEDS, TEST_SEEDS, LetterEnv, make_tasks
from stage2_letters.word_manager import LETTER_POLICIES

FAZA6_SEEDS = {"A": 1301000, "R": 1311000, "M": 1321000}


def evaluate_letter(letter, policy_path, tasks, workers):
    rows, _, _ = rollout.run(tasks, partial(LetterEnv, letter), student_path=policy_path, workers=workers)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["dev", "test", "faza6"], default="dev")
    parser.add_argument("--tasks", type=int, default=300)
    parser.add_argument("--shared")
    for n in "ARM":
        parser.add_argument(f"--policy-{n}", default=str(LETTER_POLICIES[n]))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default=str(cfg.ROOT / "results" / "letters.json"))
    args = parser.parse_args()
    seeds = dict(test=TEST_SEEDS, dev=DEV_SEEDS, faza6=FAZA6_SEEDS)[args.set]
    result = {}
    for k, letter in enumerate("ARM"):
        tasks = make_tasks(letter, args.tasks, seeds[letter])
        if args.shared:
            from stage3_shared.shared_env import SharedLetterEnv
            tasks = [dict(t, id=k * 100000 + t["id"]) for t in tasks]
            rows, _, _ = rollout.run(tasks, SharedLetterEnv, student_path=args.shared, workers=args.workers)
        else:
            rows = evaluate_letter(letter, getattr(args, f"policy_{letter}"), tasks, args.workers)
        result[letter] = dict(seed=seeds[letter], **rollout.summary(rows),
                              per_task=[int(r["success"]) for r in sorted(rows, key=lambda r: r["id"])])
        print(letter, result[letter]["success"], "/", result[letter]["episodes"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
