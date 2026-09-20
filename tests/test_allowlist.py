import pytest
import yaml

from src.targets.allowlist import CircuitBreaker, CircuitOpenError, check_allowlisted


def test_allowed_host_passes(tmp_path):
    allowlist_path = tmp_path / 'allowlist.yaml'
    allowlist_path.write_text(yaml.dump({'allowed_hosts': ['localhost:8000']}))
    check_allowlisted('http://localhost:8000', str(allowlist_path))


def test_disallowed_host_raises(tmp_path):
    allowlist_path = tmp_path / 'allowlist.yaml'
    allowlist_path.write_text(yaml.dump({'allowed_hosts': ['localhost:8000']}))
    with pytest.raises(PermissionError):
        check_allowlisted('https://some-random-production-api.com', str(allowlist_path))


def test_empty_base_url_raises(tmp_path):
    allowlist_path = tmp_path / 'allowlist.yaml'
    allowlist_path.write_text(yaml.dump({'allowed_hosts': ['localhost:8000']}))
    with pytest.raises(ValueError):
        check_allowlisted('', str(allowlist_path))


def test_schemeless_url_raises_instead_of_silently_matching(tmp_path):
    allowlist_path = tmp_path / 'allowlist.yaml'
    allowlist_path.write_text(yaml.dump({'allowed_hosts': ['localhost:8000']}))
    with pytest.raises(PermissionError):
        check_allowlisted('localhost:8000', str(allowlist_path))


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(threshold=3)
    breaker.before_call()
    breaker.record_failure()
    breaker.record_failure()
    breaker.before_call()
    breaker.record_failure()
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_circuit_breaker_resets_on_success():
    breaker = CircuitBreaker(threshold=2)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.before_call()
