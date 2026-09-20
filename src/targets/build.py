from src.targets.api_target import APITarget
from src.targets.local_target import LocalTarget
from src.targets.target_interface import TargetModel


def build_target(cfg: dict) -> TargetModel:
    tcfg = cfg['target']
    breaker_threshold = tcfg.get('circuit_breaker_threshold', 5)
    if tcfg['mode'] == 'local':
        return LocalTarget(tcfg['local_model_name'], circuit_breaker_threshold=breaker_threshold)
    elif tcfg['mode'] == 'api':
        return APITarget(tcfg['api_base_url'], tcfg['api_key_env_var'], tcfg['local_model_name'], circuit_breaker_threshold=breaker_threshold)
    raise ValueError(f"unknown target mode {tcfg['mode']!r}")

