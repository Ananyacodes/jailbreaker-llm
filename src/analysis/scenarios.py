"""Simulated red-teaming campaign scenarios with genuinely different dynamics.

The repo's own generator (data/generate_synthetic_campaign.py) makes one campaign. These scenarios reuse its
structure (a hidden compliance state per turn that drives the judge's three scores) but change what
actually differs between real campaigns:

  noisy      the judge is less reliable: score distributions overlap, so states are hard to separate
  stealthy   a careful attacker: slow escalation over up to 12 turns, so conversations are long
  persistent no early stop: once the target is broken it stays broken, so violation really persists

All are simulated and labelled as such in the UI. They exist so the pipeline can be checked under
different conditions, not to stand in for real data (the Anthropic and JailbreakBench loaders do that).
"""
import numpy as np
import pandas as pd

STATES = ['refusal', 'partial_compliance', 'full_violation']
BASE_TRANSITION = np.array([[0.55, 0.35, 0.10], [0.15, 0.55, 0.30], [0.05, 0.15, 0.80]])
TARGET_SCALE = {'guarded-target-a': 0.6, 'baseline-target-b': 1.0, 'weak-target-c': 1.6}
SHARP = {'refusal': [8.0, 1.5, 0.5], 'partial_compliance': [1.5, 6.0, 2.5], 'full_violation': [0.5, 2.0, 8.0]}
BLURRY = {'refusal': [3.0, 1.6, 1.0], 'partial_compliance': [1.8, 3.0, 2.2], 'full_violation': [1.0, 1.8, 3.2]}

SCENARIOS = {
    'noisy': dict(
        title='noisy judge', alpha=BLURRY, escalation=1.0, max_turns=6, n_conv=300, start=[0.75, 0.20, 0.05], stop_on_success=True, absorb=0.80, seed=7,
        about='A less reliable judge: the three score distributions overlap heavily, so refusal, partial and full-violation turns are hard to tell apart. Same dynamics as the baseline campaign.'),
    'stealthy': dict(
        title='stealthy attacker', alpha=SHARP, escalation=0.35, max_turns=12, n_conv=220, start=[0.90, 0.09, 0.01], stop_on_success=True, absorb=0.80, seed=11,
        about='A careful attacker that escalates slowly: conversations run up to 12 turns, most turns are refusals or partial compliance, and full violation is rare and late.'),
    'persistent': dict(
        title='persistent jailbreak', alpha=SHARP, escalation=1.2, max_turns=10, n_conv=220, start=[0.70, 0.22, 0.08], stop_on_success=False, absorb=0.92, seed=13,
        about='No early stop: the campaign keeps talking after a success, and a broken target tends to stay broken (92% chance to stay in full violation). Here the sequence structure is real, not an artefact of stopping.'),
}


def _transition(scale, absorb):
    m = BASE_TRANSITION.copy()
    m[2] = [(1 - absorb) * 0.25, (1 - absorb) * 0.75, absorb]
    for i in range(3):
        for j in range(3):
            if j > i:
                m[i, j] *= scale
        m[i] /= m[i].sum()
    return m


def simulate(key: str) -> pd.DataFrame:
    p = SCENARIOS[key]
    rng = np.random.default_rng(p['seed'])
    targets = list(TARGET_SCALE)
    rows = []
    for c in range(p['n_conv']):
        target = targets[c % len(targets)]
        trans = _transition(TARGET_SCALE[target] * p['escalation'], p['absorb'])
        s = rng.choice(3, p=p['start'])
        for t in range(p['max_turns']):
            probs = rng.dirichlet(p['alpha'][STATES[s]])
            rows.append({'conversation_id': f'conv_{c:04d}', 'timestep_in_conversation': t, 'true_compliance_state': STATES[s], 'target_model': target,
                         'judge_refusal': float(probs[0]), 'judge_partial_compliance': float(probs[1]), 'judge_full_violation': float(probs[2])})
            if p['stop_on_success'] and probs[2] >= 0.5:
                break
            s = rng.choice(3, p=trans[s])
    return pd.DataFrame(rows)
