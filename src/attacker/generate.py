"""Wraps a trained attacker (GPTWithValueHead, either backend) + its
tokenizer for text-in/text-out generation -- used both during RL rollouts
and later at campaign time. Works with either the from-scratch tokenizer
(tokenizers.Tokenizer) or a warm-started HF tokenizer, so callers don't
need to know which backend the attacker was built on."""
import torch


def _encode(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        # HF tokenizers return a list[int] or a BatchEncoding depending on args;
        # tokenizers.Tokenizer returns an Encoding object with .ids
        result = tokenizer.encode(text)
        return result.ids if hasattr(result, "ids") else result
    raise TypeError(f"unrecognized tokenizer type: {type(tokenizer)}")


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


class AttackerGenerator:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, seed_prompt: str, max_new_tokens: int = 128, temperature: float = 0.9) -> str:
        ids = _encode(self.tokenizer, seed_prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)
        new_ids = out[0, len(ids) :].tolist()
        text = _decode(self.tokenizer, new_ids)
        return text.strip()

    def generate_n(self, seed_prompt: str, n: int, max_new_tokens: int = 128, temperature: float = 0.9) -> list[str]:
        """Independent samples for best-of-N branch search in the multi-turn
        agent -- higher temperature sampling means these won't collapse to
        the same string."""
        return [self.generate(seed_prompt, max_new_tokens=max_new_tokens, temperature=temperature) for _ in range(n)]
