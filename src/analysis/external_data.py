"""Loaders for real red-teaming datasets, for the transparency UI (app.py).

Both datasets are used through their OWN numeric scores and tags: no judge
model is run, and no text is displayed or embedded.

1. Anthropic red-team-attempts (Hugging Face: Anthropic/hh-rlhf, data_dir="red-team-attempts"):
   ~39k real multi-turn red-team conversations, each with a human success rating (0-4).
   Label = the human rating. Features = the dataset's automatic harmlessness scores,
   conversation shape (turns, lengths), target model size/type and red-teamer tags.

2. JailbreakBench attack artifacts (GitHub: JailbreakBench/artifacts):
   real PAIR / GCG / etc. attack prompts against real target models, each with a
   `jailbroken` verdict. Label = jailbroken. Features = query counts, token counts,
   lengths, a refusal-phrase flag and the attack method.
   `queries_to_jailbreak` is deliberately NOT a feature: it is empty exactly when the
   attack failed, so it would leak the label.

Fetch once (needs internet):
    python -m src.analysis.external_data redteam
    python -m src.analysis.external_data jbb
"""
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

EXTERNAL_DIR = 'data/external'
REDTEAM_CACHE = os.path.join(EXTERNAL_DIR, 'red_team_attempts_raw.jsonl')
JBB_DIR = os.path.join(EXTERNAL_DIR, 'jailbreakbench_artifacts')

REFUSAL_PHRASES = ("i cannot", "i can't", "i can not", "i'm sorry", "i am sorry", "i apologize", "as an ai", "i'm not able", "i am not able", "cannot assist", "can't assist", "unable to")


class DatasetUnavailable(Exception):
    """Raised with a message that tells the user exactly how to get the data."""


# ---------------------------------------------------------------- Anthropic red-team attempts

def fetch_redteam():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DatasetUnavailable('Install the Hugging Face datasets package first: pip install datasets') from e
    try:
        ds = load_dataset('Anthropic/hh-rlhf', data_dir='red-team-attempts', split='train')
    except Exception as e:  # network, auth, schema
        raise DatasetUnavailable(f'Could not download Anthropic/hh-rlhf (red-team-attempts): {e}') from e
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    with open(REDTEAM_CACHE, 'w') as f:
        for row in ds:
            f.write(json.dumps(row, default=str) + '\n')
    return len(ds)


def _parse_params(v):
    m = re.match(r'\s*([\d.]+)\s*([BbMm]?)', str(v))
    if not m:
        return np.nan
    x = float(m.group(1))
    return x / 1000.0 if m.group(2).lower() == 'm' else x


def build_redteam_features(raw: pd.DataFrame):
    """raw rows (HF schema) -> (X features DataFrame, y rating codes, class names, about text)."""
    need = {'transcript', 'rating'}
    if not need.issubset(raw.columns):
        raise DatasetUnavailable(f'Unexpected red-team-attempts schema. Columns found: {list(raw.columns)}; expected at least {sorted(need)}.')
    df = raw.dropna(subset=['rating']).copy()
    tr = df['transcript'].astype(str)
    X = pd.DataFrame(index=df.index)
    for col in ('min_harmlessness_score_transcript', 'task_description_harmlessness_score'):
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors='coerce')
    X['n_turns'] = tr.str.count(r'\n\nHuman:')
    X['transcript_chars'] = tr.str.len()
    X['mean_assistant_chars'] = tr.apply(lambda t: np.mean([len(p) for p in t.split('\n\nAssistant:')[1:]] or [0]))
    if 'num_params' in df.columns:
        X['model_size_b'] = df['num_params'].apply(_parse_params)
    if 'is_upworker' in df.columns:
        X['is_upworker'] = df['is_upworker'].astype(float)
    if 'tags' in df.columns:
        X['n_tags'] = df['tags'].apply(lambda t: len(t) if isinstance(t, (list, tuple)) else (len([x for x in re.split(r'[,;]', str(t)) if x.strip()]) if t not in (None, '', 'None', 'nan') else 0))
    if 'model_type' in df.columns:
        for mt, col in pd.get_dummies(df['model_type'].astype(str), prefix='model').items():
            X[mt] = col.astype(float)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y_raw = df['rating'].astype(float).round().astype(int).clip(0, 4).values
    classes = [f'rating {i}' for i in range(5)]
    about = ('Real multi-turn red-team conversations from Anthropic (hh-rlhf, red-team-attempts). The label is the human red-teamer\'s success rating '
             '(0 = attack failed, 4 = very successful). Features are the dataset\'s own automatic harmlessness scores, conversation shape, target-model size/type and tag count. '
             'No conversation text is used or shown.')
    return X.reset_index(drop=True), y_raw, classes, about


def load_redteam():
    if not os.path.exists(REDTEAM_CACHE):
        raise DatasetUnavailable('Anthropic red-team-attempts is not downloaded yet.')
    raw = pd.read_json(REDTEAM_CACHE, lines=True)
    return build_redteam_features(raw)


def load_redteam_raw():
    """Raw rows in exactly the order of build_redteam_features (rows without a rating dropped), so row i here is row i of X."""
    if not os.path.exists(REDTEAM_CACHE):
        raise DatasetUnavailable('Anthropic red-team-attempts is not downloaded yet.')
    return pd.read_json(REDTEAM_CACHE, lines=True).dropna(subset=['rating']).reset_index(drop=True)


def parse_transcript(t):
    """'\\n\\nHuman: ...\\n\\nAssistant: ...' -> [('Human', text), ('Assistant', text), ...]"""
    out = []
    for m in re.finditer(r'(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|\Z)', str(t), flags=re.S):
        out.append((m.group(1), m.group(2).strip()))
    return out


# ---------------------------------------------------------------- JailbreakBench artifacts

def fetch_jbb():
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    if os.path.isdir(JBB_DIR):
        return 0
    try:
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/JailbreakBench/artifacts', JBB_DIR], check=True, capture_output=True, text=True)
    except Exception as e:
        raise DatasetUnavailable(f'Could not clone JailbreakBench/artifacts: {getattr(e, "stderr", e)}') from e
    return 1


def build_jbb_features(root):
    rows = []
    for path in glob.glob(os.path.join(root, '**', '*.json'), recursive=True):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        items = data.get('jailbreaks') if isinstance(data, dict) else None
        if not isinstance(items, list):
            continue
        rel = os.path.relpath(path, root).split(os.sep)
        attack = next((p for p in rel if p.upper() in ('PAIR', 'GCG', 'JBC', 'PROMPT_WITH_RANDOM_SEARCH', 'DSN')), rel[-3] if len(rel) >= 3 else rel[0])
        for it in items:
            if not isinstance(it, dict) or 'jailbroken' not in it:
                continue
            prompt, response = str(it.get('prompt') or ''), str(it.get('response') or '')
            rows.append({
                'attack': attack,
                'number_of_queries': it.get('number_of_queries'),
                'prompt_tokens': it.get('prompt_tokens'),
                'response_tokens': it.get('response_tokens'),
                'prompt_chars': len(prompt),
                'response_chars': len(response),
                'refusal_phrase': float(any(p in response.lower() for p in REFUSAL_PHRASES)),
                'jailbroken': bool(it.get('jailbroken')),
            })
    if not rows:
        raise DatasetUnavailable(f'No JailbreakBench attack records found under {root}. Expected JSON files with a "jailbreaks" list of records that have a "jailbroken" field.')
    df = pd.DataFrame(rows)
    X = pd.DataFrame({c: pd.to_numeric(df[c], errors='coerce') for c in ('number_of_queries', 'prompt_tokens', 'response_tokens', 'prompt_chars', 'response_chars', 'refusal_phrase')})
    for a, col in pd.get_dummies(df['attack'], prefix='attack').items():
        X[a] = col.astype(float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = df['jailbroken'].astype(int).values
    about = ('Real attack prompts from JailbreakBench artifacts (PAIR, GCG and others) against real target models. The label is the benchmark\'s own jailbroken verdict. '
             'Features: query count, token counts, lengths, a refusal-phrase flag (a simple keyword heuristic added here) and the attack method. '
             '`queries_to_jailbreak` is excluded because it is empty exactly when the attack failed, which would leak the label.')
    return X.reset_index(drop=True), y, ['not jailbroken', 'jailbroken'], about


def load_jbb():
    if not os.path.isdir(JBB_DIR):
        raise DatasetUnavailable('JailbreakBench artifacts are not downloaded yet.')
    return build_jbb_features(JBB_DIR)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else ''
    if which == 'redteam':
        print(f'cached {fetch_redteam()} red-team attempts to {REDTEAM_CACHE}')
    elif which == 'jbb':
        fetch_jbb()
        print(f'JailbreakBench artifacts at {JBB_DIR}')
    else:
        print('usage: python -m src.analysis.external_data [redteam|jbb]')
