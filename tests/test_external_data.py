import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import external_data as ext


def mock_redteam(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rating = int(rng.integers(0, 5))
        turns = int(rng.integers(1, 6))
        tr = ''.join(f'\n\nHuman: q{j}\n\nAssistant: ' + 'a' * int(rng.integers(5, 60)) for j in range(turns))
        rows.append({'transcript': tr, 'min_harmlessness_score_transcript': float(rng.normal(-rating * 0.3, 1)),
                     'task_description_harmlessness_score': float(rng.normal(0, 1)), 'num_params': str(rng.choice(['2.7B', '13B', '52B'])),
                     'model_type': str(rng.choice(['plain LM', 'context distillation', 'rejection sampling', 'RLHF'])),
                     'rating': float(rating), 'is_upworker': bool(rng.integers(0, 2)), 'tags': ['a', 'b'][: int(rng.integers(0, 3))]})
    return pd.DataFrame(rows)


def write_mock_jbb(root, n=60, seed=0):
    rng = np.random.default_rng(seed)
    for attack in ('PAIR', 'GCG'):
        d = root / 'attack-artifacts' / attack / 'black_box'
        d.mkdir(parents=True)
        items = []
        for i in range(n):
            jb = bool(rng.integers(0, 2))
            items.append({'index': i, 'goal': 'g', 'behavior': 'b', 'category': 'c', 'prompt': 'p' * int(rng.integers(10, 200)),
                          'response': ('Sure, here' if jb else "I'm sorry, I cannot help") + ' x' * 5, 'number_of_queries': int(rng.integers(1, 100)),
                          'queries_to_jailbreak': int(rng.integers(1, 100)) if jb else None, 'prompt_tokens': 20, 'response_tokens': 30, 'jailbroken': jb})
        (d / 'model.json').write_text(json.dumps({'parameters': {}, 'jailbreaks': items}))


def test_redteam_features():
    X, y, classes, about = ext.build_redteam_features(mock_redteam())
    assert len(X) == len(y) == 400 and set(np.unique(y)) <= set(range(5)) and len(classes) == 5
    assert {'n_turns', 'transcript_chars', 'mean_assistant_chars', 'model_size_b'} <= set(X.columns)
    assert 'rating' not in X.columns and not X.isna().any().any()
    assert X['n_turns'].min() >= 1


def test_redteam_bad_schema_gives_clear_message():
    with pytest.raises(ext.DatasetUnavailable, match='Columns found'):
        ext.build_redteam_features(pd.DataFrame({'foo': [1]}))


def test_jbb_features_exclude_leaky_field(tmp_path):
    write_mock_jbb(tmp_path)
    X, y, classes, about = ext.build_jbb_features(str(tmp_path))
    assert len(X) == 120 and set(y) == {0, 1} and classes == ['not jailbroken', 'jailbroken']
    assert 'queries_to_jailbreak' not in X.columns
    assert {'attack_PAIR', 'attack_GCG', 'refusal_phrase'} <= set(X.columns)


def test_jbb_empty_dir_gives_clear_message(tmp_path):
    with pytest.raises(ext.DatasetUnavailable, match='No JailbreakBench attack records'):
        ext.build_jbb_features(str(tmp_path))


def test_parse_transcript_and_raw_alignment(tmp_path, monkeypatch):
    assert ext.parse_transcript("\n\nHuman: hi\n\nAssistant: yo\nthere\n\nHuman: bye") == [('Human', 'hi'), ('Assistant', 'yo\nthere'), ('Human', 'bye')]
    raw = mock_redteam(50)
    raw.loc[3, 'rating'] = float('nan')  # a row without a rating is dropped by both paths
    p = tmp_path / 'raw.jsonl'
    raw.to_json(p, orient='records', lines=True)
    monkeypatch.setattr(ext, 'REDTEAM_CACHE', str(p))
    X, y, _, _ = ext.build_redteam_features(ext.pd.read_json(p, lines=True))
    assert len(ext.load_redteam_raw()) == len(X) == 49
