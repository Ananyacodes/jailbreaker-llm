import torch
import torch.nn.functional as F

from src.attacker.generate import _encode
from src.attacker.value_head import GPTWithValueHead


class SimplePPOTrainer:
    def __init__(self, policy: GPTWithValueHead, tokenizer, device: str, lr: float = 1e-05, kl_coef: float = 0.05, ppo_epochs: int = 4, clip_range: float = 0.2, vf_coef: float = 0.5, gamma: float = 1.0, lam: float = 0.95, max_grad_norm: float = 1.0, micro_batch_size: int = 4):
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
        self.micro_batch_size = max(1, micro_batch_size)
        trainable_parameters = [p for p in self.policy.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError('policy has no trainable parameters')
        self.optimizer = torch.optim.AdamW(trainable_parameters, lr=lr)

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 128, temperature: float = 0.9, top_k: int = 50):
        was_training = self.policy.training
        self.policy.eval()
        try:
            return self._generate(prompts, max_new_tokens, temperature, top_k)
        finally:
            self.policy.train(was_training)

    def _generate(self, prompts: list[str], max_new_tokens: int, temperature: float, top_k: int):
        if self.policy.backend == 'hf' and len(prompts) > 1:
            encoded = [_encode(self.tokenizer, prompt) for prompt in prompts]
            max_prompt_len = max(len(ids) for ids in encoded)
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            if pad_token_id is None:
                pad_token_id = eos_token_id if isinstance(eos_token_id, int) else 0
            input_ids = torch.full(
                (len(encoded), max_prompt_len),
                pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, ids in enumerate(encoded):
                input_ids[row, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask[row, -len(ids):] = 1
            out = self.policy.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                attention_mask=attention_mask,
            )
            eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
            pairs = []
            for ids, sequence in zip(encoded, out):
                response = sequence[max_prompt_len:]
                if eos_ids:
                    eos_positions = [i for i, token in enumerate(response.tolist()) if token in eos_ids]
                    if eos_positions:
                        response = response[:eos_positions[0] + 1]
                query = torch.tensor(ids, dtype=torch.long, device=self.device)
                pairs.append((query, response))
            return pairs

        pairs = []
        for p in prompts:
            ids = _encode(self.tokenizer, p)
            query = torch.tensor(ids, dtype=torch.long, device=self.device)
            batched_query = query.unsqueeze(0)
            out = self.policy.generate(
                batched_query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                attention_mask=torch.ones_like(batched_query),
            )
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
        # PPO ratios, KL, advantages, and value loss are numerically sensitive.
        # Keep them in FP32 even when the transformer runs in BF16/FP16.
        logp = F.log_softmax(logits.float(), dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        resp_start = prompt_len - 1
        resp_logp = token_logp[resp_start:]
        resp_values = values[resp_start:].float() if values is not None else None
        return (resp_logp, resp_values)

    def _batched_logprobs_and_values(self, full_ids_list: list[torch.Tensor], prompt_lens: list[int], use_ref: bool = False):
        max_steps = max(len(full_ids) - 1 for full_ids in full_ids_list)
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_token_id is None:
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else 0
        inputs = torch.full(
            (len(full_ids_list), max_steps),
            pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        targets = torch.full_like(inputs, pad_token_id)
        attention_mask = torch.zeros_like(inputs)
        valid_steps = []
        for row, full_ids in enumerate(full_ids_list):
            steps = len(full_ids) - 1
            inputs[row, :steps] = full_ids[:-1]
            targets[row, :steps] = full_ids[1:]
            attention_mask[row, :steps] = 1
            valid_steps.append(steps)
        if use_ref:
            logits = self.ref.logits(inputs, attention_mask=attention_mask)
            values = None
        else:
            logits, values = self.policy.forward_logits_and_values(inputs, attention_mask=attention_mask)
        logp = F.log_softmax(logits.float(), dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        response_logps, response_values = [], []
        for row, (prompt_len, steps) in enumerate(zip(prompt_lens, valid_steps)):
            response_start = prompt_len - 1
            response_logps.append(token_logp[row, response_start:steps])
            if values is not None:
                response_values.append(values[row, response_start:steps].float())
        return response_logps, response_values if values is not None else None

    def _score_sequences(self, full_ids_list: list[torch.Tensor], prompt_lens: list[int], use_ref: bool = False):
        if self.policy.backend == 'hf' and len(full_ids_list) > 1:
            return self._batched_logprobs_and_values(full_ids_list, prompt_lens, use_ref=use_ref)
        logps, values = [], []
        for full_ids, prompt_len in zip(full_ids_list, prompt_lens):
            logp, value = self._logprobs_and_values(full_ids, prompt_len, use_ref=use_ref)
            logps.append(logp)
            if value is not None:
                values.append(value)
        return logps, values if not use_ref else None

    def step(self, pairs: list, external_rewards: list[float]) -> dict:
        if len(pairs) != len(external_rewards):
            raise ValueError('pairs and external_rewards must have the same length')
        if not pairs:
            raise ValueError('cannot run a PPO step with an empty batch')
        self.policy.eval()
        all_full_ids = [torch.cat([q, r]) for q, r in pairs]
        all_prompt_lens = [len(q) for q, _ in pairs]
        old_logps, values_list, rewards_list = ([], [], [])
        mean_kl = 0.0
        with torch.no_grad():
            for start in range(0, len(pairs), self.micro_batch_size):
                stop = start + self.micro_batch_size
                chunk_ids = all_full_ids[start:stop]
                chunk_lens = all_prompt_lens[start:stop]
                chunk_logps, chunk_values = self._score_sequences(chunk_ids, chunk_lens)
                chunk_ref_logps, _ = self._score_sequences(chunk_ids, chunk_lens, use_ref=True)
                for logp, values, ref_logp, ext_r in zip(chunk_logps, chunk_values, chunk_ref_logps, external_rewards[start:stop]):
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
        adv_mean, adv_std = (flat_adv.mean(), flat_adv.std(unbiased=False) + 1e-08)
        advantages_list = [(adv - adv_mean) / adv_std for adv in advantages_list]
        policy_loss_avg, value_loss_avg = (0.0, 0.0)
        for _ in range(self.ppo_epochs):
            self.policy.train()
            self.optimizer.zero_grad()
            total_policy_loss, total_value_loss = (0.0, 0.0)
            for start in range(0, len(pairs), self.micro_batch_size):
                stop = start + self.micro_batch_size
                chunk_logps, chunk_values = self._score_sequences(
                    all_full_ids[start:stop],
                    all_prompt_lens[start:stop],
                )
                chunk_loss = 0.0
                for new_logp, new_values, old_logp, adv, ret in zip(
                    chunk_logps,
                    chunk_values,
                    old_logps[start:stop],
                    advantages_list[start:stop],
                    returns_list[start:stop],
                ):
                    ratio = torch.exp(new_logp - old_logp.detach())
                    surr1 = ratio * adv.detach()
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv.detach()
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(new_values, ret.detach())
                    chunk_loss = chunk_loss + policy_loss + self.vf_coef * value_loss
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            policy_loss_avg = total_policy_loss / len(pairs)
            value_loss_avg = total_value_loss / len(pairs)
        return {'policy_loss': policy_loss_avg, 'value_loss': value_loss_avg, 'mean_kl': mean_kl}

    def save_pretrained(self, out_dir: str):
        self.policy.save_pretrained(out_dir)
