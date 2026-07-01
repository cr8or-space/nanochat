"""
Tests for the MATH answer grader (tasks/math.py) — the sympy-based verifier used
for eval and RL rewards. Pure functions, no dataset/network required.
"""

import pytest

from tasks.math import extract_boxed, answers_equal, MATH


@pytest.mark.parametrize("text,expected", [
    (r"so the answer is $\boxed{42}$.", "42"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"first \boxed{1} then \boxed{2}", "2"),                 # last box wins
    (r"nested \boxed{\frac{a}{b} + \sqrt{2}}", r"\frac{a}{b} + \sqrt{2}"),
    ("The final answer is 7", "7"),                            # no-box fallback
    ("no answer here", None),
])
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


@pytest.mark.parametrize("pred,ref", [
    ("42", "42"),
    (r"\frac{1}{2}", "0.5"),
    (r"\frac{1}{2}", r"\dfrac{1}{2}"),                         # dfrac normalization
    (r"2\sqrt{2}", r"\sqrt{8}"),                               # sympy equivalence
    ("x^2 + 2x + 1", "(x+1)^2"),                               # symbolic identity
    (r"\frac{2}{4}", r"\frac{1}{2}"),                          # unreduced fraction
    ("1,000", "1000"),                                        # thousands separator
])
def test_answers_equal_true(pred, ref):
    assert answers_equal(pred, ref) is True


@pytest.mark.parametrize("pred,ref", [
    ("42", "43"),
    (r"\frac{1}{2}", r"\frac{1}{3}"),
    ("x^2", "x^3"),
    (None, "5"),
    ("5", None),
])
def test_answers_equal_false(pred, ref):
    assert answers_equal(pred, ref) is False


def test_evaluate_uses_boxed_reference():
    """evaluate() pulls the ref from the conversation's last assistant part and grades the pred."""
    conv = {"messages": [
        {"role": "user", "content": "compute 1/2 + 1/2"},
        {"role": "assistant", "content": [{"type": "text", "text": r"... The final answer is $\boxed{1}$."}]},
    ]}
    task = MATH.__new__(MATH)  # bypass __init__ (no dataset load)
    assert task.evaluate(conv, r"after work, \boxed{1}") == 1
    assert task.evaluate(conv, r"i think \boxed{2}") == 0
    assert task.reward(conv, r"\boxed{1.0}") == 1.0        # 1.0 == 1 via sympy
