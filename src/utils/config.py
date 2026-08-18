"""Tiny YAML config loader shared across entry points -- kept in one place
so every script resolves relative paths (checkpoints, data) the same way."""
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"{path} is empty")
    return cfg
