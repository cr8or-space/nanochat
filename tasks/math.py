"""
MATH (competition mathematics) evaluation with a sympy-based answer checker.

Unlike GSM8K (grade-school, single integer after a #### marker), MATH answers are
symbolic (fractions, radicals, expressions) wrapped in \\boxed{...}. Exact string
match is too brittle, so we extract the boxed answer and compare pred vs ref by
symbolic equivalence (sympy), falling back to normalized string equality.

The grader (`extract_boxed`, `answers_equal`) is exposed at module level so it can
be reused as an RL verifier (see scripts/chat_rl.py) and unit-tested without the
dataset. Default dataset is HuggingFaceH4/MATH-500 (a 500-problem test split).
"""

import re
import signal
from contextlib import contextmanager

from tasks.common import Task


# -----------------------------------------------------------------------------
# Answer extraction + grading (pure functions, no dataset dependency)

def extract_boxed(text):
    """Return the content of the LAST \\boxed{...} in text (brace-balanced), or None."""
    if not text:
        return None
    idx = text.rfind(r"\boxed")
    if idx == -1:
        # some completions write "final answer is X" without \boxed
        m = re.search(r"final answer is[:\s]*\$?(.+?)\$?[.\s]*$", text, re.IGNORECASE)
        return m.group(1).strip() if m else None
    i = idx + len(r"\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i + 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None  # unbalanced


def _normalize(s):
    """Normalize a LaTeX-ish answer string for string comparison."""
    if s is None:
        return None
    s = s.strip()
    # strip surrounding $...$ and \[...\], \(...\)
    s = s.strip("$").strip()
    for a, b in [(r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\,", ""), (r"\ ", " "),
                 (r"\%", ""), ("%", ""), (r"^{\circ}", ""), (r"\circ", ""), (r"\$", ""),
                 ("\\text{", "{"), (" ", "")]:
        s = s.replace(a, b)
    s = s.rstrip(".")
    # strip thousands separators (comma between a digit and a 3-digit group), but keep commas in
    # tuples/coordinates like (1,2) — those aren't followed by exactly three digits.
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)
    # \dfrac -> \frac
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return s


def _latex_to_expr(s):
    """Best-effort convert a simple LaTeX answer to a sympy-parseable string."""
    if s is None:
        return None
    s = _normalize(s)
    # \frac{a}{b} -> ((a)/(b)), repeatedly for nested/multiple
    frac = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    prev = None
    while prev != s:
        prev = s
        s = frac.sub(r"((\1)/(\2))", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    s = s.replace(r"\pi", "pi").replace("^", "**")
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    return s


@contextmanager
def _time_limit(seconds):
    """Guard sympy calls, which can hang on adversarial input. SIGALRM => main thread only."""
    def _handler(signum, frame):
        raise TimeoutError
    try:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def answers_equal(pred, ref):
    """True if pred and ref denote the same answer (string-normalized OR sympy-equivalent)."""
    if pred is None or ref is None:
        return False
    if _normalize(pred) == _normalize(ref):
        return True
    try:
        import sympy
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application)
        tf = standard_transformations + (implicit_multiplication_application,)
        pe, re_ = _latex_to_expr(pred), _latex_to_expr(ref)
        if not pe or not re_:
            return False
        with _time_limit(3):
            a = parse_expr(pe, transformations=tf, evaluate=True)
            b = parse_expr(re_, transformations=tf, evaluate=True)
            return bool(sympy.simplify(a - b) == 0)
    except Exception:
        return False


# -----------------------------------------------------------------------------

class MATH(Task):

    def __init__(self, split="test", dataset="HuggingFaceH4/MATH-500", **kwargs):
        super().__init__(**kwargs)
        from datasets import load_dataset
        # MATH-500 exposes a single "test" split with problem/solution/answer columns.
        self.ds = load_dataset(dataset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row["problem"]
        solution = row.get("solution", "")
        answer = row.get("answer", "")  # MATH-500 provides the extracted final answer
        # Reference answer is carried in the final assistant text part so evaluate() can recover it.
        final = solution.rstrip()
        if answer:
            final = (final + "\n" if final else "") + f"The final answer is $\\boxed{{{answer}}}$."
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": [{"type": "text", "text": final}]},
        ]
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant"
        ref_text = assistant_message['content'][-1]['text']
        ref = extract_boxed(ref_text)
        pred = extract_boxed(assistant_response)
        return int(answers_equal(pred, ref))

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))
