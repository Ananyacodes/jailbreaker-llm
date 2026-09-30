from src.attacker.reward import DiversityTracker, compute_reward, is_degenerate
from src.utils.config import load_config

TEST_CFG = {'reward': {'success_reward': 1.0, 'partial_success_reward': 0.3, 'refusal_penalty': -0.1, 'degenerate_output_penalty': -1.0, 'diversity_coef': 0.4}}


def test_degenerate_repeated_char():
    assert is_degenerate('aaaaaaaaaaaaaaa')


def test_degenerate_low_diversity():
    assert is_degenerate('the the the the the the the the the the the')


def test_not_degenerate_normal_text():
    assert not is_degenerate('please explain how this system works in detail')


def test_reward_full_violation():
    scores = {'full_violation': 0.9, 'partial_compliance': 0.05, 'refusal': 0.05}
    r = compute_reward(scores, 'a reasonably long and varied attacker prompt here', TEST_CFG)
    assert r > 0.8


def test_reward_degenerate_output_penalized_regardless_of_score():
    scores = {'full_violation': 0.99, 'partial_compliance': 0.0, 'refusal': 0.0}
    r = compute_reward(scores, 'aaaaaaaaaaaaaaa', TEST_CFG)
    assert r == TEST_CFG['reward']['degenerate_output_penalty']


def test_diversity_tracker_empty_buffer_has_no_penalty():
    tracker = DiversityTracker()
    assert tracker.similarity_penalty('anything at all') == 0.0


def test_reward_with_empty_diversity_tracker_matches_base_reward():
    scores = {'full_violation': 0.9, 'partial_compliance': 0.05, 'refusal': 0.05}
    text = 'a reasonably long and varied attacker prompt here'
    base = compute_reward(scores, text, TEST_CFG)
    with_tracker = compute_reward(scores, text, TEST_CFG, diversity_tracker=DiversityTracker())
    assert base == with_tracker


def test_smoke_rl_section_has_the_shape_expected_by_compute_reward():
    rl_cfg = load_config('configs/smoke_test.yaml')['rl']
    scores = {'full_violation': 0.1, 'partial_compliance': 0.2, 'refusal': 0.7}
    result = compute_reward(scores, 'a sufficiently varied attacker prompt for testing', rl_cfg)
    assert isinstance(result, float)
