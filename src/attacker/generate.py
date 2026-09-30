import torch


def _encode(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{'role': 'user', 'content': text}]
        kwargs = {'tokenize': True, 'add_generation_prompt': True}
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except (TypeError, ValueError):
                pass
    if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
        result = tokenizer.encode(text)
        return result.ids if hasattr(result, 'ids') else result
    raise TypeError(f'unrecognized tokenizer type: {type(tokenizer)}')


def _decode(tokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(ids)


class AttackerGenerator:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, seed_prompt: str, max_new_tokens: int = 128, temperature: float = 0.9) -> str:
        ids = _encode(self.tokenizer, seed_prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(x)
        out = self.model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            attention_mask=attention_mask,
        )
        new_ids = out[0, len(ids):].tolist()
        text = _decode(self.tokenizer, new_ids)
        return text.strip()

    def generate_n(self, seed_prompt: str, n: int, max_new_tokens: int = 128, temperature: float = 0.9) -> list[str]:
        return [self.generate(seed_prompt, max_new_tokens=max_new_tokens, temperature=temperature) for _ in range(n)]
