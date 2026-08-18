"""
Regression coverage for the crash bug this file is built around fixing: an
earlier version of TurnRecord/ConversationResult had type annotations but
no @dataclass decorator, so constructing a TurnRecord with positional args
raised TypeError and ConversationResult().turns was a dataclasses.Field
object (no .append). Nothing in src/eval/run_campaign.py ever ran.

Also covers the actual multi-turn loop end-to-end with stub
generator/judge/target, so the branch-search and full-context-scoring
logic has real test coverage -- none of this pipeline's orchestration code
had any before.
"""
from src.agent.multi_turn_loop import ConversationResult, MultiTurnAgent, TurnRecord


def test_turn_record_is_a_real_dataclass():
    # this alone would have caught the original bug: TypeError on construction
    record = TurnRecord("attack", "response", {"full_violation": 0.1}, "escalate")
    assert record.attacker_prompt == "attack"
    assert record.technique == "escalate"


def test_conversation_result_turns_defaults_to_a_real_list():
    result = ConversationResult()
    assert result.turns == []
    # this alone would have caught the original bug: AttributeError on .append
    result.turns.append(TurnRecord("a", "b", {"full_violation": 0.0}, "opening"))
    assert len(result.turns) == 1


def test_conversation_result_instances_do_not_share_mutable_state():
    a = ConversationResult()
    b = ConversationResult()
    a.turns.append(TurnRecord("x", "y", {"full_violation": 0.0}, "opening"))
    assert b.turns == []  # would fail if turns defaulted to one shared list


class _StubGenerator:
    """Returns a distinct, deterministic string per call so branch search
    has something to actually differentiate between."""

    def __init__(self):
        self.calls = 0

    def generate(self, seed_prompt: str, max_new_tokens: int = 128, temperature: float = 0.9) -> str:
        self.calls += 1
        return f"generated attack #{self.calls} in a reasonably long and varied sentence"


class _StubJudge:
    """Scores based on a marker embedded in the response text, so the test
    can control exactly which candidate branch should win."""

    def score(self, prompt: str, response: str) -> dict:
        if "WINNER" in response:
            return {"refusal": 0.0, "partial_compliance": 0.1, "full_violation": 0.9}
        if "REFUSE" in response:
            return {"refusal": 0.9, "partial_compliance": 0.1, "full_violation": 0.0}
        return {"refusal": 0.2, "partial_compliance": 0.3, "full_violation": 0.2}


class _StubTarget:
    """Deterministic canned responses keyed off the attacker's prompt text
    so branch search has a real winner to find, and the loop terminates."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    def respond(self, conversation: list[dict]) -> str:
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return text


def test_run_stops_immediately_on_full_violation():
    generator = _StubGenerator()
    judge = _StubJudge()
    target = _StubTarget(["WINNER response on turn one"])
    agent = MultiTurnAgent(generator, judge, target, max_turns=6, branch_candidates=2)

    result = agent.run("opening seed prompt")

    assert result.final_label == "full_violation"
    assert len(result.turns) == 1


def test_run_exhausts_max_turns_and_falls_back_to_refusal():
    generator = _StubGenerator()
    judge = _StubJudge()
    target = _StubTarget(["REFUSE every time, nothing works here at all"])
    agent = MultiTurnAgent(generator, judge, target, max_turns=3, branch_candidates=2)

    result = agent.run("opening seed prompt")

    assert result.final_label == "refusal"
    assert len(result.turns) == 3


def test_score_with_context_includes_prior_turns_not_just_latest():
    generator = _StubGenerator()
    judge = _StubJudge()
    target = _StubTarget(["neutral response, not a refusal or a winner"])
    agent = MultiTurnAgent(generator, judge, target, max_turns=1, branch_candidates=1)

    prior = [TurnRecord("earlier prompt", "earlier response", {"full_violation": 0.0}, "opening")]
    scores = agent._score_with_context(prior, "current prompt", "current response")

    assert set(scores) == {"refusal", "partial_compliance", "full_violation"}
