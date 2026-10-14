from unittest.mock import patch
import pytest
from verl.utils.reward_score.feedback_lenient import boxed_candidates, choice_candidates, compute_score


def strict_failure(*args, **kwargs):
    return dict(acc=0., score=0., pred='', incorrect_format=1, truncated=1)


def test_boxed_candidates_survive_repeated_and_unfinished_boxes():
    text = r'\boxed{(3,\frac{\pi}{2})}' * 10 + r'\boxed{(3,\frac'
    assert boxed_candidates(text) == [r'(3,\frac{\pi}{2})']


def test_recovers_earlier_math_answer_preserving_strict_metrics():
    with patch('verl.utils.reward_score.feedback_lenient.strict_score', strict_failure):
        result = compute_score('math', r'\boxed{204}' + ' repeat' * 100 + r'\boxed{', '204')
    assert result['acc'] == 1 and result['strict_acc'] == 0
    assert result['incorrect_format'] == 1 and result['lenient_rescued'] == 1


def test_repeated_answer_tags_and_conflicting_answers():
    assert choice_candidates('<answer>B</answer>' + '</answer>' * 20) == ['B']
    with patch('verl.utils.reward_score.feedback_lenient.strict_score', strict_failure):
        result = compute_score('sciknoweval', '<answer>A</answer><answer>B</answer>', 'B')
    assert result['acc'] == 1 and result['candidate_count'] == 2


def test_reasoning_numbers_or_options_are_not_answers():
    assert boxed_candidates('We use 204 in an intermediate calculation.') == []
    assert choice_candidates('A is wrong, B is plausible, C is wrong, D is wrong.') == []


def test_wrong_explicit_choice_is_not_rescued():
    with patch('verl.utils.reward_score.feedback_lenient.strict_score', strict_failure):
        result = compute_score('sciknoweval', '<answer>A</answer>' * 5, 'B')
    assert result['acc'] == 0 and result['lenient_rescued'] == 0


def test_training_is_rejected():
    with pytest.raises(ValueError, match='evaluation-only'):
        compute_score('math', r'\boxed{204}', '204', {'split':'train'})


def test_real_math_parser_recovers_complete_earlier_equivalent_box():
    text = r'\boxed{\frac{1}{2}}' + ' repetition' * 30 + r'\boxed{\frac'
    result = compute_score('math500', text, '0.5', {'split':'test','truncated':True})
    assert result['strict_acc'] == 0 and result['acc'] == 1


def test_real_mcq_repeated_closing_tags_recovered():
    result = compute_score('sciknoweval', '<answer>B</answer>' + '</answer>' * 20, 'B')
    assert result['strict_incorrect_format'] == 1 and result['acc'] == 1
    assert result['incorrect_format'] == 1
