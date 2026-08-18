#!/usr/bin/env bash
# Warm-starts the base model by default (configs/base_model.yaml:
# pretrain.warm_start_from). To run a from-scratch pretrain instead, set
# that field to null and run scripts/01b_train_tokenizer.sh first.
set -euo pipefail
python -m src.model.pretrain --config configs/base_model.yaml
