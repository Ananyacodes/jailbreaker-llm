import argparse
import glob
import json

import numpy as np
from sklearn.cluster import DBSCAN

from src.targets.allowlist import check_allowlisted
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _still_authorized(row: dict, allowlist_path: str) -> bool:
    if row.get('target_mode') != 'api':
        return True
    host = row.get('target_host')
    if not host:
        return False
    try:
        check_allowlisted(f'https://{host}', allowlist_path)
        return True
    except PermissionError:
        return False


def load_successful_attacks(transcripts_dir: str, allowlist_path: str = 'configs/allowlist.yaml') -> list[dict]:
    rows = []
    dropped_revoked = 0
    for path in glob.glob(f'{transcripts_dir}/*.jsonl'):
        for line in open(path):
            row = json.loads(line)
            if row.get('judge_label') != 'full_violation':
                continue
            if not _still_authorized(row, allowlist_path):
                dropped_revoked += 1
                continue
            rows.append(row)
    if dropped_revoked:
        log.info('dropped %d rows whose target authorization has since been revoked', dropped_revoked)
    return rows


def cluster_and_report(rows: list[dict], eps: float = 0.25, min_samples: int = 2, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    if len(rows) < min_samples:
        log.info('only %d successful attacks logged so far, need more before clustering is meaningful', len(rows))
        return []
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [r['attacker_prompt'] for r in rows]
    X = model.encode(texts, normalize_embeddings=True)
    dist = 1 - X @ X.T
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(dist)
    clusters = {}
    for row, label in zip(rows, labels):
        clusters.setdefault(label, []).append(row)
    report = []
    for label, members in clusters.items():
        if label == -1:
            continue
        target_models = {m.get('target_model', 'unknown') for m in members}
        techniques = {m.get('technique', 'unknown') for m in members}
        report.append({'cluster_id': int(label), 'size': len(members), 'transfers_across_targets': len(target_models) > 1, 'target_models_hit': sorted(target_models), 'techniques_in_cluster': sorted(techniques), 'example_prompt': members[0]['attacker_prompt'][:200]})
    report.sort(key=lambda r: (r['transfers_across_targets'], r['size']), reverse=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts_dir', default='data/transcripts')
    parser.add_argument('--allowlist_path', default='configs/allowlist.yaml')
    args = parser.parse_args()
    rows = load_successful_attacks(args.transcripts_dir, args.allowlist_path)
    report = cluster_and_report(rows)
    for r in report:
        print(r)
