#!/usr/bin/env bash
# One A3/A4 training run: train_job.sh <variant> <seed offset> <letter>  (run from Faza6/code)
set -euo pipefail
variant=$1; offset=$2; letter=$3
PY="${PY:-../../Faza5/.venv/bin/python}"
case $variant in
  A3-single) extra=(--single-fraction 1.0) ;;
  A3-mixed)  extra=(--single-fraction 0.25) ;;
  A4-bez)    extra=(--single-fraction 0.25 --disturb-position 0 --disturb-angle 0) ;;
  *) echo "unknown variant $variant"; exit 1 ;;
esac
model="../models/$variant/seed$offset/stage2_letter_$letter.zip"
[ -f "$model" ] && { echo "skip $model"; exit 0; }
run="../runs/train/$variant/seed$offset/$letter"
echo "$(date +%H:%M) start $variant seed$offset $letter"
mkdir -p "$(dirname "$run")"
"$PY" -m stage2_letters.train_word_states --letter "$letter" --seed-offset "$offset" --out "$run" "${extra[@]}" \
  > "$run.log" 2>&1
mkdir -p "$(dirname "$model")"
cp "$run/last.zip" "$model"
echo "$(date +%H:%M) done  $variant seed$offset $letter"
