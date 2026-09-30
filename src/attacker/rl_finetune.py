import argparse
import os
import random

import torch
from tqdm import tqdm

from src.attacker.generate import _decode
from src.attacker.ppo_trainer import SimplePPOTrainer
from src.attacker.reward import DiversityTracker, compute_reward
from src.attacker.value_head import GPTWithValueHead
from src.judge.judge_infer import RewardJudge
from src.targets.allowlist import CircuitOpenError
from src.targets.build import build_target
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
_SENSITIVE_CONFIG_KEYS = {'api_base_url', 'api_key_env_var'}


def _redact_config(cfg: dict) -> dict:
    import copy
    redacted = copy.deepcopy(cfg)
    tcfg = redacted.get('target', {})
    for key in _SENSITIVE_CONFIG_KEYS:
        if tcfg.get(key):
            tcfg[key] = '<redacted>'
    return redacted


def load_tokenizer(tokenizer_path: str):
    if os.path.isfile(tokenizer_path):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tokenizer_path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path)


def load_policy_from_path(path: str, device: str, peft_config: dict | None = None, is_trainable: bool = True) -> GPTWithValueHead:
    if os.path.isdir(path):
        adapter_config = os.path.join(path, 'adapter_config.json')
        if os.path.exists(adapter_config):
            return GPTWithValueHead.from_peft_adapter(path, device, is_trainable=is_trainable)
        latest_pt = os.path.join(path, 'latest.pt')
        if os.path.exists(latest_pt):
            return GPTWithValueHead.from_pretrained_gpt(latest_pt, device)
        return GPTWithValueHead.from_warm_start(path, device, peft_config=peft_config)
    if os.path.isfile(path):
        return GPTWithValueHead.from_pretrained_gpt(path, device)
    return GPTWithValueHead.from_warm_start(path, device, peft_config=peft_config)


def load_policy(rl_cfg: dict, device: str) -> GPTWithValueHead:
    peft_config = rl_cfg.get('peft', {})
    if not peft_config.get('enabled', False):
        peft_config = None
    return load_policy_from_path(rl_cfg['base_checkpoint'], device, peft_config=peft_config)


def load_seed_prompts(path: str) -> list[str]:
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip() and (not line.strip().startswith('#'))]
    if not prompts:
        raise ValueError(f'no seed prompts found in {path} -- populate it before running RL fine-tuning')
    return prompts


def safe_target_respond(target, conversation, on_error_text=''):
    try:
        return target.respond(conversation)
    except CircuitOpenError:
        raise
    except Exception as e:
        log.warning('target call failed, treating as degenerate response: %s', e)
        return on_error_text


def safe_target_respond_many(target, conversations, on_error_text=''):
    try:
        respond_many = getattr(target, 'respond_many', None)
        if respond_many is None:
            return [safe_target_respond(target, conversation, on_error_text) for conversation in conversations]
        return respond_many(conversations)
    except CircuitOpenError:
        raise
    except Exception as e:
        log.warning('target batch failed, treating every response as degenerate: %s', e)
        return [on_error_text] * len(conversations)


def main(cfg_path: str, resume: bool = False):
    cfg = load_config(cfg_path)
    rl_cfg = cfg['rl']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = load_tokenizer(rl_cfg['tokenizer_path'])
    resume_path = os.path.join(rl_cfg['out_dir'], 'trainer_state.pt')
    should_resume = resume and os.path.exists(resume_path)
    if should_resume:
        policy = load_policy_from_path(rl_cfg['out_dir'], device)
    else:
        policy = load_policy(rl_cfg, device)
    trainer = SimplePPOTrainer(policy, tokenizer, device, lr=rl_cfg['learning_rate'], kl_coef=rl_cfg['kl_coef'], ppo_epochs=rl_cfg['ppo_epochs'], clip_range=rl_cfg.get('clip_range', 0.2), vf_coef=rl_cfg.get('vf_coef', 0.5), lam=rl_cfg.get('gae_lambda', 0.95), micro_batch_size=rl_cfg.get('ppo_micro_batch_size', 4))
    judge = RewardJudge(rl_cfg['judge_checkpoint'], device=device)
    target = build_target(cfg)
    diversity_coef = rl_cfg['reward'].get('diversity_coef', 0.0)
    diversity_tracker = DiversityTracker(buffer_size=rl_cfg['reward'].get('diversity_buffer_size', 200)) if diversity_coef > 0 else None
    seed_prompts = load_seed_prompts(rl_cfg.get('seed_prompts_path', 'data/seed_prompts.txt'))
    start_step = 0
    if should_resume:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        trainer.optimizer.load_state_dict(state['optimizer'])
        start_step = state['step'] + 1
        log.info('resumed RL fine-tuning from step %d (weights loaded from %s)', start_step, rl_cfg['out_dir'])
    tracking_cfg = rl_cfg.get('tracking', {})
    use_wandb = False
    if tracking_cfg.get('enabled', False):
        try:
            import wandb
            wandb.init(
                project=tracking_cfg.get('project', 'jailbreaker-llm'),
                name=tracking_cfg.get('run_name', 'rl_finetune'),
                config=_redact_config(cfg),
            )
            use_wandb = True
        except Exception as exc:
            log.warning('experiment tracking disabled after initialization failure: %s', exc)
    total_steps = rl_cfg['num_rollouts'] // rl_cfg['batch_size']
    for step in tqdm(range(start_step, total_steps), initial=start_step, total=total_steps):
        batch_prompts = random.choices(seed_prompts, k=rl_cfg['batch_size'])
        pairs = trainer.generate(batch_prompts, max_new_tokens=rl_cfg['max_new_tokens'])
        attacker_outputs = [_decode(tokenizer, response.tolist()) for _, response in pairs]
        target_responses = safe_target_respond_many(
            target,
            [[{'role': 'user', 'content': attack_prompt}] for attack_prompt in attacker_outputs],
        )
        rewards = []
        for attack_prompt, target_response in zip(attacker_outputs, target_responses):
            judge_scores = judge.score(attack_prompt, target_response)
            reward = compute_reward(judge_scores, attack_prompt, rl_cfg, diversity_tracker=diversity_tracker)
            if diversity_tracker is not None:
                diversity_tracker.add_if_successful(attack_prompt, judge_scores)
            rewards.append(reward)
        stats = trainer.step(pairs, rewards)
        mean_reward = sum(rewards) / len(rewards)
        if use_wandb:
            wandb.log({'step': step, 'mean_reward': mean_reward, **stats})
        if step % 50 == 0:
            trainer.save_pretrained(rl_cfg['out_dir'])
            os.makedirs(rl_cfg['out_dir'], exist_ok=True)
            torch.save({'optimizer': trainer.optimizer.state_dict(), 'step': step}, resume_path)
            log.info('step %d: mean_reward=%.3f policy_loss=%.4f mean_kl=%.4f', step, mean_reward, stats['policy_loss'], stats['mean_kl'])
    trainer.save_pretrained(rl_cfg['out_dir'])
    torch.save({'optimizer': trainer.optimizer.state_dict(), 'step': total_steps - 1}, resume_path)
    log.info('attacker RL fine-tuning done, saved to %s', rl_cfg['out_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/rl_finetune.yaml')
    parser.add_argument('--resume', action='store_true', help='resume from out_dir/trainer_state.pt if present')
    args = parser.parse_args()
    main(args.config, args.resume)
