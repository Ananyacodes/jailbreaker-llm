import os
import time

import requests

from src.targets.allowlist import CircuitBreaker, check_allowlisted
from src.targets.target_interface import TargetModel


class APITarget(TargetModel):
    def __init__(self, base_url: str, api_key_env_var: str, model: str, timeout: int = 30, max_retries: int = 3, backoff_base: float = 2.0, allowlist_path: str = 'configs/allowlist.yaml', circuit_breaker_threshold: int = 5):
        check_allowlisted(base_url, allowlist_path)
        self.base_url = base_url.rstrip('/')
        self.api_key = os.environ.get(api_key_env_var, '')
        if not self.api_key:
            raise ValueError(f'env var {api_key_env_var} is empty -- refusing to call an API target without a key')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._breaker = CircuitBreaker(threshold=circuit_breaker_threshold)

    def respond(self, conversation: list[dict]) -> str:
        self._breaker.before_call()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(f'{self.base_url}/chat/completions', headers={'Authorization': f'Bearer {self.api_key}'}, json={'model': self.model, 'messages': conversation}, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                text = data['choices'][0]['message']['content'].strip()
            except (requests.RequestException, KeyError, IndexError, ValueError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt)
            else:
                self._breaker.record_success()
                return text
        self._breaker.record_failure()
        raise RuntimeError(f'target API failed after {self.max_retries} attempts: {last_err}')
