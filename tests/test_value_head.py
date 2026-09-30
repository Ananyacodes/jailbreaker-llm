import pytest
import torch

from src.attacker.value_head import GPTWithValueHead
from src.attacker.ppo_trainer import SimplePPOTrainer
from src.model.transformer import GPT, GPTConfig


def _small_cfg():
    return GPTConfig(vocab_size=50, n_layer=2, n_head=2, n_embd=16, block_size=16)


def test_requires_exactly_one_of_cfg_or_hf_model_name():
    with pytest.raises(ValueError):
        GPTWithValueHead()
    with pytest.raises(ValueError):
        GPTWithValueHead(cfg=_small_cfg(), hf_model_name='some/model')


def test_scratch_backend_forward_shapes():
    model = GPTWithValueHead(cfg=_small_cfg())
    x = torch.randint(0, 50, (8,))
    logits, values = model.forward_logits_and_values(x)
    assert logits.shape == (1, 8, 50)
    assert values.shape == (1, 8)


def test_scratch_backend_forward_accepts_batches():
    model = GPTWithValueHead(cfg=_small_cfg())
    x = torch.randint(0, 50, (3, 8))
    logits, values = model.forward_logits_and_values(x)
    assert logits.shape == (3, 8, 50)
    assert values.shape == (3, 8)


def test_scratch_backend_generate_extends_sequence():
    model = GPTWithValueHead(cfg=_small_cfg())
    x = torch.randint(0, 50, (1, 4))
    out = model.generate(x, max_new_tokens=3)
    assert out.shape == (1, 7)


def test_from_pretrained_gpt_roundtrip(tmp_path):
    cfg = _small_cfg()
    gpt = GPT(cfg)
    ckpt_path = tmp_path / 'latest.pt'
    torch.save({'model': gpt.state_dict(), 'config': cfg, 'iter': 0}, ckpt_path)
    loaded = GPTWithValueHead.from_pretrained_gpt(str(ckpt_path), 'cpu')
    assert loaded.backend == 'scratch'
    assert loaded.gpt.cfg.vocab_size == cfg.vocab_size
    for p1, p2 in zip(gpt.parameters(), loaded.gpt.parameters()):
        assert torch.equal(p1, p2)


def test_make_reference_is_frozen():
    model = GPTWithValueHead(cfg=_small_cfg())
    ref = model.make_reference('cpu')
    assert all((not p.requires_grad for p in ref.parameters()))
    x = torch.randint(0, 50, (6,))
    logits = ref.logits(x)
    assert logits.shape == (1, 6, 50)


def test_ppo_statistics_stay_float32_with_bfloat16_policy():
    model = GPTWithValueHead(cfg=_small_cfg()).to(dtype=torch.bfloat16)
    trainer = SimplePPOTrainer(model, tokenizer=None, device='cpu', ppo_epochs=1)
    full_ids = torch.randint(0, 50, (8,))
    logp, values = trainer._logprobs_and_values(full_ids, prompt_len=4)
    assert logp.dtype == torch.float32
    assert values.dtype == torch.float32
