import yaml

from src.mutation.cluster_transcripts import _still_authorized, cluster_and_report


def test_local_target_rows_are_always_authorized():
    row = {'target_mode': 'local', 'target_host': None}
    assert _still_authorized(row, 'unused.yaml')


def test_api_target_row_authorized_when_host_allowlisted(tmp_path):
    allowlist_path = tmp_path / 'allowlist.yaml'
    allowlist_path.write_text(yaml.dump({'allowed_hosts': ['localhost:8000']}))
    row = {'target_mode': 'api', 'target_host': 'localhost:8000'}
    assert _still_authorized(row, str(allowlist_path))


def test_api_target_row_dropped_when_authorization_revoked(tmp_path):
    allowlist_path = tmp_path / 'allowlist.yaml'
    allowlist_path.write_text(yaml.dump({'allowed_hosts': []}))
    row = {'target_mode': 'api', 'target_host': 'localhost:8000'}
    assert not _still_authorized(row, str(allowlist_path))


def test_cluster_and_report_short_circuits_below_min_samples():
    rows = [{'attacker_prompt': 'only one row'}]
    assert cluster_and_report(rows, min_samples=2) == []
