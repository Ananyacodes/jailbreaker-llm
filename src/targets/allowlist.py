from urllib.parse import urlparse

import yaml


def check_allowlisted(base_url: str, allowlist_path: str = 'configs/allowlist.yaml') -> None:
    if not base_url:
        raise ValueError('api_base_url is empty -- nothing to allowlist-check')
    with open(allowlist_path) as f:
        cfg = yaml.safe_load(f) or {}
    allowed = cfg.get('allowed_hosts', [])
    host = urlparse(base_url).netloc
    if not host:
        raise PermissionError(f"'{base_url}' doesn't parse as a URL with a host (missing scheme?) -- refusing to allowlist-check an ambiguous value")
    if host not in allowed:
        raise PermissionError(f"'{host}' is not in {allowlist_path}. Add it explicitly under allowed_hosts -- and make sure you actually have written authorization to test it -- before pointing the attacker at it.")


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self._consecutive_failures = 0

    def before_call(self):
        if self._consecutive_failures >= self.threshold:
            raise CircuitOpenError(f'{self._consecutive_failures} consecutive failures against this target (threshold={self.threshold}) -- halting instead of continuing to retry. Check target availability / authorization before resuming.')

    def record_success(self):
        self._consecutive_failures = 0

    def record_failure(self):
        self._consecutive_failures += 1
