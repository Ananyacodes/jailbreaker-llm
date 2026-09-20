import glob
import json
import os

import numpy as np
import pandas as pd

OUTPUTS_DIR = 'outputs/analysis'
COMPLIANCE_STATES = ['refusal', 'partial_compliance', 'full_violation']


def load_transcript_rows(transcripts_dir: str = 'data/transcripts') -> pd.DataFrame:
    rows = []
    for path in glob.glob(f'{transcripts_dir}/*.jsonl'):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f'no rows found under {transcripts_dir}/*.jsonl -- run `python data/generate_synthetic_campaign.py` (or a real campaign) first')
    df = pd.DataFrame(rows)
    for prefix, col in [('judge', 'judge_scores'), ('eval', 'eval_ensemble')]:
        for state in COMPLIANCE_STATES:
            df[f'{prefix}_{state}'] = df[col].apply(lambda d: d.get(state, np.nan) if isinstance(d, dict) else np.nan)
    if 'conversation_id' not in df.columns:
        df['conversation_id'] = df['campaign_id'].astype(str) + '::' + df['seed_prompt'].astype(str)
    if 'timestep_in_conversation' not in df.columns:
        df['timestep_in_conversation'] = df.groupby('conversation_id').cumcount()
    return df


def ensure_outputs_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save_metrics(name: str, metrics: dict):
    ensure_outputs_dir()
    path = os.path.join(OUTPUTS_DIR, 'metrics.json')
    all_metrics = {}
    if os.path.exists(path):
        with open(path) as f:
            all_metrics = json.load(f)
    all_metrics[name] = metrics
    with open(path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'[{name}] ' + '  '.join((f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in metrics.items())))


def conversation_train_test_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    conv_ids = np.array(df['conversation_id'].unique().tolist(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(conv_ids)
    n_test = int(len(conv_ids) * test_frac)
    test_ids = set(conv_ids[:n_test])
    test_mask = df['conversation_id'].isin(test_ids)
    return (df[~test_mask].copy(), df[test_mask].copy())
