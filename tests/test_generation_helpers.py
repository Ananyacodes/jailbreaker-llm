from src.attacker.generate import _decode, _encode


class _ChatTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return [10, 20, 30]

    def decode(self, ids, skip_special_tokens=False):
        return f'{ids}:{skip_special_tokens}'


def test_encode_uses_chat_template_for_instruction_tuned_models():
    tokenizer = _ChatTokenizer()
    ids = _encode(tokenizer, 'try this technique')
    assert ids == [10, 20, 30]
    assert tokenizer.kwargs['add_generation_prompt'] is True
    assert tokenizer.kwargs['enable_thinking'] is False


def test_decode_removes_special_tokens_when_supported():
    assert _decode(_ChatTokenizer(), [1, 2]) == '[1, 2]:True'
