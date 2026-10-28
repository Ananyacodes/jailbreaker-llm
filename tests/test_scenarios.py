import numpy as np
import pytest

from src.analysis import scenarios


@pytest.mark.parametrize('key', list(scenarios.SCENARIOS))
def test_scenario_is_valid_and_reproducible(key):
    a, b = scenarios.simulate(key), scenarios.simulate(key)
    assert a.equals(b)
    cols = ['judge_refusal', 'judge_partial_compliance', 'judge_full_violation']
    assert np.allclose(a[cols].sum(axis=1), 1.0)
    assert set(a['true_compliance_state']) <= set(scenarios.STATES)


def test_scenarios_actually_differ():
    d = {k: scenarios.simulate(k) for k in scenarios.SCENARIOS}
    assert d['stealthy'].groupby('conversation_id').size().max() > d['noisy'].groupby('conversation_id').size().max()
    assert len({len(v) for v in d.values()}) == 3
