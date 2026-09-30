import json
from collections import Counter


def build_report(transcript_path: str, campaign_id: str) -> dict:
    rows = [json.loads(l) for l in open(transcript_path)]
    if not rows:
        print('no rows in transcript, nothing to report')
        return {}
    label_counts = Counter((r['judge_label'] for r in rows))
    total = len(rows)
    success_rate = label_counts.get('full_violation', 0) / total
    partial_rate = label_counts.get('partial_compliance', 0) / total
    review_flagged = sum((1 for r in rows if r.get('eval_needs_human_review')))
    worst_examples = sorted(rows, key=lambda r: r.get('eval_ensemble', r['judge_scores']).get('full_violation', 0), reverse=True)[:5]
    report = {
        'campaign_id': campaign_id,
        'target_model': rows[0].get('target_model', 'unknown'),
        'target_mode': rows[0].get('target_mode', 'unknown'),
        'total_attempts': total,
        'full_violation_rate': round(success_rate, 3),
        'partial_compliance_rate': round(partial_rate, 3),
        'refusal_rate': round(label_counts.get('refusal', 0) / total, 3),
        'flagged_for_human_review': review_flagged,
        'single_judge_fallback': any((r.get('eval_single_judge_fallback') for r in rows)),
        'top_reproducible_examples': [{'attacker_prompt': r['attacker_prompt'], 'target_response': r['target_response'][:300], 'technique': r.get('technique', 'unknown'), 'ensemble_full_violation_score': round(r.get('eval_ensemble', r['judge_scores']).get('full_violation', 0), 3), 'needs_human_review': r.get('eval_needs_human_review', False)} for r in worst_examples],
    }
    out_path = f'data/transcripts/{campaign_id}_report.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'report written to {out_path}')
    print(f'full-violation rate: {success_rate:.1%}  |  partial: {partial_rate:.1%}  |  flagged for review: {review_flagged}')
    return report
