#!/usr/bin/env bash
# Only needed for a from-scratch run -- skip this if base_model.yaml's
# pretrain.warm_start_from is set (the warm-started base brings its own
# tokenizer).
set -euo pipefail
python -m src.tokenizer.train_tokenizer --config configs/tokenizer.yaml
