from src.targets.api_target import APITarget
from src.targets.local_target import LocalTarget
from src.targets.target_interface import TargetModel


def build_target(cfg: dict) -> TargetModel:
    tcfg = cfg['target']
    breaker_threshold = tcfg.get('circuit_breaker_threshold', 5)
    if tcfg['mode'] == 'local':
        return LocalTarget(
            tcfg['local_model_name'],
            tokenizer_name=tcfg.get('local_tokenizer_name'),
            device=tcfg.get('device'),
            dtype=tcfg.get('dtype', 'auto'),
            quantization=tcfg.get('quantization'),
            attn_implementation=tcfg.get('attn_implementation', 'sdpa'),
            max_input_tokens=tcfg.get('max_input_tokens', 2048),
            max_new_tokens=tcfg.get('max_new_tokens', 128),
            batch_size=tcfg.get('batch_size', 4),
            temperature=tcfg.get('temperature', 0.7),
            top_p=tcfg.get('top_p', 0.8),
            top_k=tcfg.get('top_k', 20),
            repetition_penalty=tcfg.get('repetition_penalty', 1.05),
            timeout_s=tcfg.get('timeout_s', 60.0),
            circuit_breaker_threshold=breaker_threshold,
        )
    elif tcfg['mode'] == 'api':
        return APITarget(tcfg['api_base_url'], tcfg['api_key_env_var'], tcfg['local_model_name'], circuit_breaker_threshold=breaker_threshold)
    raise ValueError(f"unknown target mode {tcfg['mode']!r}")
