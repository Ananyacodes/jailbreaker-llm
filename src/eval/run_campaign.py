import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from src.agent.multi_turn_loop import MultiTurnAgent
from src.attacker.generate import AttackerGenerator
from src.attacker.rl_finetune import load_policy_from_path, load_tokenizer
from src.eval.redact import redact_turn
from src.eval.report import build_report
from src.judge.eval_judge import build_eval_judge
from src.judge.judge_infer import RewardJudge
from src.targets.build import build_target
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def load_seed_prompts(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and (not line.strip().startswith('#'))]


def already_run_seeds(transcript_path: str) -> set[str]:
    if not os.path.exists(transcript_path):
        return set()
    seen = set()
    with open(transcript_path) as f:
        for line in f:
            try:
                seen.add(json.loads(line)['seed_prompt'])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def target_ledger_entry(cfg: dict) -> dict:
    tcfg = cfg['target']
    if tcfg['mode'] == 'api':
        return {'target_mode': 'api', 'target_host': urlparse(tcfg['api_base_url']).netloc}
    return {'target_mode': 'local', 'target_host': None}


def main(cfg_path: str, seeds_path: str, campaign_id: str | None, resume: bool):
    cfg = load_config(cfg_path)
    campaign_id = campaign_id or f'campaign_{uuid.uuid4().hex[:8]}'
    generator = build_attacker_generator(cfg)
    live_judge = RewardJudge(cfg['rl']['judge_checkpoint'])
    eval_judge = build_eval_judge(cfg)
    target = build_target(cfg)
    agent_cfg = cfg.get('agent', {})
    agent = MultiTurnAgent(generator, live_judge, target, max_turns=agent_cfg.get('max_turns', 6), branch_candidates=agent_cfg.get('branch_candidates', 3))
    seeds = load_seed_prompts(seeds_path)
    ledger = target_ledger_entry(cfg)
    os.makedirs('data/transcripts', exist_ok=True)
    out_path = f'data/transcripts/{campaign_id}.jsonl'
    already_run = already_run_seeds(out_path) if resume else set()
    if already_run:
        log.info('resuming %s: %d/%d seeds already logged, skipping them', campaign_id, len(already_run), len(seeds))
    mode = 'a' if resume and already_run else 'w'
    review_flagged = 0
    with open(out_path, mode) as f:
        for seed in seeds:
            if seed in already_run:
                continue
            result = agent.run(seed)
            ensemble = eval_judge.score_transcript([{'attacker_prompt': t.attacker_prompt, 'target_response': t.target_response} for t in result.turns])
            if ensemble.get('needs_human_review'):
                review_flagged += 1
            for turn in result.turns:
                row = redact_turn({'campaign_id': campaign_id, 'timestamp': datetime.now(timezone.utc).isoformat(), 'seed_prompt': seed, 'attacker_prompt': turn.attacker_prompt, 'target_response': turn.target_response, 'technique': turn.technique, 'judge_scores': turn.judge_scores, 'judge_label': result.final_label, 'eval_ensemble': ensemble['ensemble'], 'eval_needs_human_review': ensemble.get('needs_human_review', False), 'eval_single_judge_fallback': ensemble.get('single_judge_fallback', False), 'target_model': cfg['target'].get('local_model_name', 'unknown'), **ledger})
                f.write(json.dumps(row) + '\n')
            f.flush()
    log.info('campaign %s complete, %d seed prompts run', campaign_id, len(seeds) - len(already_run))
    if review_flagged:
        log.warning('%d/%d conversations had classifier/LLM-judge disagreement above threshold -- flagged eval_needs_human_review=true in the transcript, review before treating as confirmed', review_flagged, len(seeds))
    build_report(out_path, campaign_id)


def build_attacker_generator(cfg: dict) -> AttackerGenerator:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = load_tokenizer(cfg['attacker']['tokenizer_path'])
    model = load_policy_from_path(cfg['attacker']['checkpoint_path'], device, is_trainable=False)
    model.eval()
    return AttackerGenerator(model, tokenizer, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/rl_finetune.yaml')
    parser.add_argument('--seeds', default='data/seed_prompts.txt')
    parser.add_argument('--campaign_id', default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.config, args.seeds, args.campaign_id, args.resume)
