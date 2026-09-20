import torch
import torch.nn.functional as F

from src.attacker.value_head import GPTWithValueHead


class SimplePPOTrainer:
    def __init__(self, policy: GPTWithValueHead, tokenizer, device: str, lr: float = 1e-05, kl_coef: float = 0.05, ppo_epochs: int = 4, clip_range: float = 0.2, vf_coef: float = 0.5, gamma: float = 1.0, lam: float = 0.95, max_grad_norm: float = 1.0):
        self.policy = policy
        self.ref = policy.make_reference(device)
        self.tokenizer = tokenizer
        self.device = device
        self.kl_coef = kl_coef
        self.ppo_epochs = ppo_epochs
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 128, temperature: float = 0.9, top_k: int = 50):
        pairs = []
        for p in prompts:
            enc = self.tokenizer.encode(p)
            ids = enc.ids if hasattr(enc, 'ids') else enc
            query = torch.tensor(ids, dtype=torch.long, device=self.device)
            out = self.policy.generate(query.unsqueeze(0), max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            response = out[0, len(ids):]
            pairs.append((query, response))
        return pairs

    def _logprobs_and_values(self, full_ids: torch.Tensor, prompt_len: int, use_ref: bool = False):
        shifted_input = full_ids[:-1]
        targets = full_ids[1:]
        if use_ref:
            logits = self.ref.logits(shifted_input).squeeze(0)
            values = None
        else:
            logits, values_full = self.policy.forward_logits_and_values(shifted_input)
            logits = logits.squeeze(0)
            values = values_full.squeeze(0)
        logp = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        resp_start = prompt_len - 1
        resp_logp = token_logp[resp_start:]
        resp_values = values[resp_start:] if values is not None else None
        return (resp_logp, resp_values)

    def step(self, pairs: list, external_rewards: list[float]) -> dict:
        all_full_ids = [torch.cat([q, r]) for q, r in pairs]
        all_prompt_lens = [len(q) for q, _ in pairs]
        old_logps, values_list, rewards_list = ([], [], [])
        mean_kl = 0.0
        with torch.no_grad():
            for full_ids, plen, ext_r in zip(all_full_ids, all_prompt_lens, external_rewards):
                logp, values = self._logprobs_and_values(full_ids, plen)
                ref_logp, _ = self._logprobs_and_values(full_ids, plen, use_ref=True)
                kl = logp - ref_logp
                per_token_reward = -self.kl_coef * kl
                per_token_reward[-1] = per_token_reward[-1] + ext_r
                old_logps.append(logp)
                values_list.append(values)
                rewards_list.append(per_token_reward)
                mean_kl += kl.mean().item() / len(pairs)
        advantages_list, returns_list = ([], [])
        for values, rewards in zip(values_list, rewards_list):
            T = len(rewards)
            adv = torch.zeros(T, device=self.device)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                nextvalue = values[t + 1] if t + 1 < T else 0.0
                delta = rewards[t] + self.gamma * nextvalue - values[t]
                lastgaelam = delta + self.gamma * self.lam * lastgaelam
                adv[t] = lastgaelam
            advantages_list.append(adv)
            returns_list.append(adv + values)
        flat_adv = torch.cat(advantages_list)
        adv_mean, adv_std = (flat_adv.mean(), flat_adv.std() + 1e-08)
        advantages_list = [(adv - adv_mean) / adv_std for adv in advantages_list]
        policy_loss_avg, value_loss_avg = (0.0, 0.0)
        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            total_policy_loss, total_value_loss = (0.0, 0.0)
            for full_ids, plen, old_logp, adv, ret in zip(all_full_ids, all_prompt_lens, old_logps, advantages_list, returns_list):
                new_logp, new_values = self._logprobs_and_values(full_ids, plen)
                ratio = torch.exp(new_logp - old_logp.detach())
                surr1 = ratio * adv.detach()
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, ret.detach())
                (policy_loss + self.vf_coef * value_loss).backward()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            policy_loss_avg = total_policy_loss / len(pairs)
            value_loss_avg = total_value_loss / len(pairs)
        return {'policy_loss': policy_loss_avg, 'value_loss': value_loss_avg, 'mean_kl': mean_kl}

    def save_pretrained(self, out_dir: str):
        self.policy.save_pretrained(out_dir)
