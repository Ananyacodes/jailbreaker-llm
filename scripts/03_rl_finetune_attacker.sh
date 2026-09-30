#!/usr/bin/env bash
set -euo pipefail
python -m src.attacker.rl_finetune --config configs/rl_finetune.yaml "$@"
