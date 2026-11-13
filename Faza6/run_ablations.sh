#!/usr/bin/env bash
# Phase 6 ablations. Usage: bash Faza6/run_ablations.sh control|A1|A2|A3|A4|A5|all
# Finished variants are skipped (their summary.json exists), so the script can be restarted.
set -euo pipefail
cd "$(dirname "$0")/code"
PY="${PY:-../../Faza5/.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SHARED=../../Faza5/models/stage3_shared.zip
DEV=(--scenes 150 --seed 1102000)
FINAL=(--scenes 500 --seed 1202000)
BUDGET=1600   # A2: policy steps per placement attempt, the same for every segment length

word() {  # word <result dir> <args...>
  local out="../results/$1"; shift
  if [ -f "$out/summary.json" ]; then echo "skip $out"; return; fi
  echo "$(date +%H:%M) $out"
  "$PY" -m stage2_letters.word_manager "$@" --out "$out" > /dev/null
}

stage_args() { [ "$1" = stage3 ] && echo "--shared $SHARED" || true; }

control() {
  for st in stage2 stage3; do word "control/$st" "${DEV[@]}" $(stage_args $st); done
}

A1() {
  for st in stage2 stage3; do
    for r in 0 1 2 4; do word "A1/$st/repair$r" "${FINAL[@]}" --repair $r $(stage_args $st); done
  done
}

A2() {
  for st in stage2 stage3; do
    for seg in 0 0.06 0.12 0.24; do
      word "A2/$st/segment$seg" "${FINAL[@]}" --repair 4 --segment-length $seg --attempt-budget $BUDGET $(stage_args $st)
    done
  done
}

SEEDS=(0 10000 20000)

train() {  # train <variant...>: all letters and seeds, 4 runs in parallel (3 envs each)
  mkdir -p ../runs/train
  for v in "$@"; do for s in "${SEEDS[@]}"; do for l in A R M; do echo "$v $s $l"; done; done; done \
    | xargs -P 4 -L 1 bash faza6/train_job.sh
}

evaluate_policies() {  # evaluate_policies <result dir> <model dir>
  local p=(--policy-A "$2/stage2_letter_A.zip" --policy-R "$2/stage2_letter_R.zip" --policy-M "$2/stage2_letter_M.zip")
  word "$1" "${FINAL[@]}" "${p[@]}"
  if [ ! -f "../results/$1/letters.json" ]; then
    "$PY" -m stage2_letters.evaluate_letters --set faza6 "${p[@]}" --out "../results/$1/letters.json" > /dev/null
  fi
}

A3() {
  mkdir -p ../models/A3-start
  for l in A R M; do cp --update=none "../models/start/stage2_letter_${l}_single.zip" "../models/A3-start/stage2_letter_$l.zip"; done
  evaluate_policies A3/start ../models/A3-start
  train A3-single A3-mixed
  for v in single mixed; do for s in "${SEEDS[@]}"; do evaluate_policies "A3/$v/seed$s" "../models/A3-$v/seed$s"; done; done
}

A4() {  # A4-s is A3-mixed (same configuration and seeds); only A4-bez is trained
  train A4-bez
  for s in "${SEEDS[@]}"; do evaluate_policies "A4/bez/seed$s" "../models/A4-bez/seed$s"; done
}

A5() {  # shared policy with / without the one-hot ID: same DAgger recipe, 4 iterations, last one used
  for arm in id noid; do
    for s in "${SEEDS[@]}"; do
      local model="../models/A5-$arm/seed$s/stage3_shared.zip" out="../results/A5/$arm/seed$s"
      if [ ! -f "$model" ]; then
        local run="../runs/train/A5-$arm/seed$s" extra=()
        [ $arm = noid ] && extra=(--no-id)
        mkdir -p "$run" "$(dirname "$model")"
        echo "$(date +%H:%M) DAgger A5-$arm seed$s"
        "$PY" -m stage3_shared.dagger --iterations 4 --seed "$s" --out "$run" "${extra[@]}" > "$run.log" 2>&1
        cp "$run/student_iter3.zip" "$model"
      fi
      word "A5/$arm/seed$s" "${FINAL[@]}" --shared "$model"
      if [ ! -f "$out/letters.json" ]; then
        "$PY" -m stage2_letters.evaluate_letters --set faza6 --shared "$model" --out "$out/letters.json" > /dev/null
      fi
    done
  done
}

case "${1:-all}" in
  control) control ;;
  A1) A1 ;;
  A2) A2 ;;
  A3) A3 ;;
  A4) A4 ;;
  A5) A5 ;;
  all) control; A1; A2; A3; A4; A5 ;;
  *) echo "unknown: $1"; exit 1 ;;
esac
