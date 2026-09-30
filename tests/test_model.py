import torch

from src.model.transformer import GPT, GPTConfig


def _small_cfg():
    return GPTConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16)


def test_forward_shape():
    model = GPT(_small_cfg())
    x = torch.randint(0, 100, (2, 10))
    logits, loss = model(x)
    assert logits.shape == (2, 10, 100)
    assert loss is None


def test_forward_with_targets_computes_loss():
    model = GPT(_small_cfg())
    x = torch.randint(0, 100, (2, 10))
    y = torch.randint(0, 100, (2, 10))
    _, loss = model(x, y)
    assert loss.item() > 0


def test_generate_extends_sequence():
    model = GPT(_small_cfg())
    x = torch.randint(0, 100, (1, 5))
    out = model.generate(x, max_new_tokens=5)
    assert out.shape == (1, 10)
