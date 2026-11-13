"""Record one scene of any Phase 6 variant as a labelled GIF (scene view + top view), for the failure analysis.

    python -m faza6.replay --index 17 --gif ../plots/failures/a1_17.gif               # A1-4, Stage 2
    python -m faza6.replay --index 17 --repair 0 --gif ...                              # A1-0
    python -m faza6.replay --index 17 --segment-length 0 --attempt-budget 1600 --gif ...  # A2-direct
    python -m faza6.replay --index 17 --policies ../models/A3-single/seed0 --gif ...     # a trained A3/A4 model
"""
import argparse
import json
from pathlib import Path

import torch

from demo import Recorder
from stage2_letters import word_manager as W
from stage2_letters.word_sim import ORDER, WordSim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1202000)
    parser.add_argument("--shared", help="Stage 3 shared policy")
    parser.add_argument("--policies", help="folder with stage2_letter_{A,R,M}.zip")
    parser.add_argument("--repair", type=int, default=4)
    parser.add_argument("--segment-length", type=float, default=W.SEGMENT_LENGTH)
    parser.add_argument("--attempt-budget", type=int)
    parser.add_argument("--title", default="")
    parser.add_argument("--every", type=int, default=4, help="record every n-th control step")
    parser.add_argument("--gif", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = {n: f"{args.policies}/stage2_letter_{n}.zip" for n in ORDER} if args.policies else None
    sim = WordSim()
    recorder = Recorder(sim, f"{args.title or 'Word ARM'} | scene {args.index}", every=args.every)
    sim.act = recorder.wrap(sim.act)
    manager = W.Manager(sim, W.load_policies(paths, args.shared), args.repair,
                        segment_length=args.segment_length, attempt_budget=args.attempt_budget)
    result = manager.run_scene(W.make_scenes(args.index + 1, args.seed)[args.index])
    recorder.save(args.gif, "WORD ARM BUILT" if result["word_success"] else f"FAILED: {result['category']}")
    recorder.frames[-1].save(Path(args.gif).with_suffix(".png"))
    print(json.dumps({k: result[k] for k in ("word_success", "category", "failure", "steps", "disturbed")}))


if __name__ == "__main__":
    main()
