#!/usr/bin/env bash
set -euo pipefail
python -m src.eval.run_campaign \
  --config configs/rl_finetune.yaml \
  --seeds data/seed_prompts.txt \
  --campaign_id "${1:-demo_run_1}" "${@:2}"
