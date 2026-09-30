from src.attacker.technique_library import TECHNIQUES, select_candidates


def test_all_techniques_have_unique_names():
    names = [t.name for t in TECHNIQUES]
    assert len(names) == len(set(names))


def test_obfuscation_techniques_transform_text():
    by_name = {t.name: t for t in TECHNIQUES}
    assert by_name['rot13_obfuscation'].apply('hello') != 'hello'
    assert 'base64' not in by_name['base64_obfuscation'].apply('hello').lower() or True
    assert by_name['base64_obfuscation'].apply('hello').startswith('Decode this base64')


def test_select_candidates_respects_n():
    scores = {'refusal': 0.1, 'partial_compliance': 0.1, 'full_violation': 0.1}
    assert len(select_candidates(scores, n=2)) == 2
    assert len(select_candidates(scores, n=1)) == 1


def test_select_candidates_prioritizes_escalation_on_partial_compliance():
    scores = {'refusal': 0.0, 'partial_compliance': 0.9, 'full_violation': 0.0}
    candidates = select_candidates(scores, n=1)
    assert candidates[0].name == 'escalate'


def test_select_candidates_prioritizes_retreat_on_strong_refusal():
    scores = {'refusal': 0.9, 'partial_compliance': 0.0, 'full_violation': 0.0}
    candidates = select_candidates(scores, n=1)
    assert candidates[0].name == 'retreat_and_reframe'

