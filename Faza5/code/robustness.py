"""Word ARM under changed conditions (evaluation only), stages 2 and 3, 100 scenes each.

    python robustness.py [--scenes 100] [--seed 994000]
"""
import argparse
import json
import statistics
from pathlib import Path

from common import settings as cfg
from stage2_letters import conditions, word_manager

SHARED = cfg.MODELS / "stage3_shared.zip"
SPREAD_SEEDS = [995000, 996000, 997000, 998000, 999000]


def run(condition, seed, scenes, shared, workers):
    conditions.apply(condition)
    rows = word_manager.evaluate(word_manager.make_scenes(scenes, seed), shared=shared, workers=workers,
                                 condition=condition)
    return word_manager.summarize(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=994000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default=str(cfg.ROOT / "results" / "robustness.json"))
    args = parser.parse_args()
    results = {}
    for condition in conditions.NAMES:
        results[condition] = {stage: run(condition, args.seed, args.scenes, shared, args.workers)
                              for stage, shared in (("stage2", None), ("stage3", SHARED))}
        print(condition, {k: v["word_success"] for k, v in results[condition].items()}, flush=True)
    spread = {stage: [run("nominal", s, args.scenes, shared, args.workers)["word_success"] for s in SPREAD_SEEDS]
              for stage, shared in (("stage2", None), ("stage3", SHARED))}
    results["seed_spread"] = {k: dict(values=v, mean=statistics.mean(v), std=statistics.stdev(v)) for k, v in spread.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(json.dumps(results["seed_spread"]))


if __name__ == "__main__":
    main()
