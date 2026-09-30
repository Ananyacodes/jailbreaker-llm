import torch

from src.attacker.ppo_trainer import SimplePPOTrainer


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [10, 11] if messages[0]['content'] == 'short' else [20, 21, 22]


class _Policy(torch.nn.Module):
    backend = 'hf'

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0
        self.score_calls = 0
        self.last_attention_mask = None
        self.score_attention_masks = []
        self.score_training_modes = []
        self.reference = _Reference()

    def make_reference(self, device):
        return self.reference

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        self.last_attention_mask = attention_mask
        continuation = torch.tensor([[7, 2, 2], [8, 9, 2]], device=input_ids.device)
        return torch.cat([input_ids, continuation], dim=1)

    def forward_logits_and_values(self, input_ids, attention_mask=None):
        self.score_calls += 1
        self.score_attention_masks.append(attention_mask.clone())
        self.score_training_modes.append(self.training)
        batch, steps = input_ids.shape
        logits = self.weight.expand(batch, steps, 32)
        values = self.weight.expand(batch, steps)
        return logits, values


class _Reference:
    def __init__(self):
        self.calls = 0

    def logits(self, input_ids, attention_mask=None):
        self.calls += 1
        batch, steps = input_ids.shape
        return torch.zeros(batch, steps, 32)


def test_hf_prompts_are_generated_in_one_padded_batch():
    policy = _Policy()
    trainer = SimplePPOTrainer(policy, _Tokenizer(), 'cpu')
    pairs = trainer.generate(['short', 'long'])

    assert policy.calls == 1
    assert policy.last_attention_mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    assert pairs[0][0].tolist() == [10, 11]
    assert pairs[1][0].tolist() == [20, 21, 22]
    assert pairs[0][1].tolist() == [7, 2]
    assert pairs[1][1].tolist() == [8, 9, 2]
    assert policy.training


def test_ppo_scores_variable_length_hf_sequences_in_micro_batches():
    policy = _Policy()
    trainer = SimplePPOTrainer(policy, _Tokenizer(), 'cpu', ppo_epochs=1, micro_batch_size=2)
    pairs = [
        (torch.tensor([10, 11]), torch.tensor([7, 2])),
        (torch.tensor([20, 21, 22]), torch.tensor([8, 9, 2])),
    ]

    stats = trainer.step(pairs, [0.1, 0.2])

    assert policy.score_calls == 2
    assert policy.reference.calls == 1
    assert policy.score_attention_masks[0].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert policy.score_training_modes == [False, True]
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())
