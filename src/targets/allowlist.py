"""
Two pieces of defense-in-depth for anything that talks to a target:

1. ScopeGate.check_allowlisted -- refuses to construct an APITarget unless
   its host is explicitly allowlisted in configs/allowlist.yaml. This lives
   INSIDE APITarget.__init__ (src/targets/api_target.py), not just in one
   factory function -- every construction path is checked, including a
   test, a notebook, or a future script that builds an APITarget directly.

2. CircuitBreaker -- halts a campaign after too many consecutive
   errors/auth failures instead of retrying forever. A previous version of
   this pipeline treated every failed call as "an empty response, penalize
   it and keep going," which meant a target that started rejecting every
   request (revoked auth, target taken down, wrong scope) would get
   hammered for the rest of a multi-hour run instead of the run stopping
   to tell you something's wrong.

Turns the README's "responsible use" promise into something the code
actually enforces rather than an honor system.
"""
from urllib.parse import urlparse

import yaml


def check_allowlisted(base_url: str, allowlist_path: str = "configs/allowlist.yaml") -> None:
    if not base_url:
        raise ValueError("api_base_url is empty -- nothing to allowlist-check")

    with open(allowlist_path) as f:
        cfg = yaml.safe_load(f) or {}
    allowed = cfg.get("allowed_hosts", [])

    host = urlparse(base_url).netloc
    if not host:
        raise PermissionError(
            f"'{base_url}' doesn't parse as a URL with a host (missing scheme?) -- "
            "refusing to allowlist-check an ambiguous value"
        )
    if host not in allowed:
        raise PermissionError(
            f"'{host}' is not in {allowlist_path}. Add it explicitly under allowed_hosts "
            "-- and make sure you actually have written authorization to test it -- "
            "before pointing the attacker at it."
        )


class CircuitOpenError(RuntimeError):
    """Raised when a target has failed too many times in a row. Stops the
    campaign instead of continuing to hammer a target that's signaling
    something is wrong (revoked auth, taken down, out of scope)."""


class CircuitBreaker:
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self._consecutive_failures = 0

    def before_call(self):
        if self._consecutive_failures >= self.threshold:
            raise CircuitOpenError(
                f"{self._consecutive_failures} consecutive failures against this target "
                f"(threshold={self.threshold}) -- halting instead of continuing to retry. "
                "Check target availability / authorization before resuming."
            )

    def record_success(self):
        self._consecutive_failures = 0

    def record_failure(self):
        self._consecutive_failures += 1
