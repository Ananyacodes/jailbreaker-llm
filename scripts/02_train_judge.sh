#!/usr/bin/env bash
set -euo pipefail
python -m src.judge.train_judge \
  --train data/judge_train/train.jsonl \
  --val data/judge_train/val.jsonl \
  --out_dir checkpoints/judge_model

