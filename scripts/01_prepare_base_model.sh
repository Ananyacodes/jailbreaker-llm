#!/usr/bin/env bash
set -euo pipefail
python -m src.model.pretrain --config configs/base_model.yaml
