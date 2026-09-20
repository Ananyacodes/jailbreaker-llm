#!/usr/bin/env bash
set -euo pipefail
python -m src.tokenizer.train_tokenizer --config configs/tokenizer.yaml
