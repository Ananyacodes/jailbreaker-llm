from src.targets.target_interface import TargetModel


class _SequentialTarget(TargetModel):
    def respond(self, conversation: list[dict]) -> str:
        return conversation[-1]['content'].upper()


def test_default_respond_many_is_a_compatible_sequential_fallback():
    target = _SequentialTarget()
    results = target.respond_many([
        [{'role': 'user', 'content': 'one'}],
        [{'role': 'user', 'content': 'two'}],
    ])
    assert results == ['ONE', 'TWO']
