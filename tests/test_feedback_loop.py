import ast
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from dotenv import load_dotenv
from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.generators.test_generator import TestGenerator
from test_adequacy_study.runners.pytest_runner import PytestRunner
from test_adequacy_study.test_generation.feedback_loop.feedback_runner import FeedbackRunner

load_dotenv()



@pytest.fixture(scope="module")
def humaneval_task():
    cache_dir = Path(os.environ["HF_DATASETS_CACHE"]) / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    loader = HumanEvalLoader()
    return next(iter(loader.load()))


@pytest.fixture(scope="module")
def builder():
    return PythonProgramBuilder()


@pytest.fixture(scope="module")
def cut_default(builder, humaneval_task):
    return builder.build_program(task=humaneval_task)


@pytest.fixture(scope="module")
def runner():
    return PytestRunner(timeout=5.0)


VALID_TESTS = """
from solution import has_close_elements

def test_positive():
    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False

def test_negative():
    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
"""

FAILING_TESTS = """
from solution import has_close_elements

def test_will_fail():
    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == True  # wrong expected value
"""

SYNTAX_ERROR_TESTS = """
from solution import has_close_elements

def test_broken(
    assert True  # missing closing paren → SyntaxError
"""

IMPORT_ERROR_TESTS = """

def test_import_error():
    assert non_existent_function([1.0, 2.0], 0.5) == False
"""


def _make_runner(runner, builder, generator):
    """Helper: build a FeedbackRunner with a real runner/builder but mocked generator."""
    return FeedbackRunner(
        generator=generator,
        runner=runner,
        builder=builder,
        max_iterations=3,
    )


def _mock_generator(responses: list[str]):
    """
    Returns a mock TestGenerator whose generate() cycles through `responses`.
    Each call returns (response, history) and appends to history.
    """
    generator = MagicMock(spec=TestGenerator)
    call_count = {"n": 0}

    def fake_generate(prompt_variables, samples, history):
        idx = min(call_count["n"], len(responses) - 1)
        code = responses[idx]
        call_count["n"] += 1
        new_history = (history or []) + [
            {"role": "user", "content": str(prompt_variables)},
            {"role": "assistant", "content": code},
        ]
        return [code], new_history

    generator.generate.side_effect = fake_generate
    return generator



class TestFeedbackRunnerLoop:

    def test_passes_on_first_iteration(self, runner, builder, humaneval_task, cut_default):
        """Valid tests pass immediately — loop exits after 1 iteration."""
        generator = _mock_generator([VALID_TESTS])
        fb_runner = _make_runner(runner, builder, generator)

        tests, result, api_calls, last_report = fb_runner.run(task=humaneval_task, cut=cut_default)

        assert tests == VALID_TESTS.strip() or VALID_TESTS in tests
        assert result is not None
        assert result.verdict == Verdict.PASSED
        assert generator.generate.call_count == 1

    def test_failing_tests_do_not_trigger_retry(self, runner, builder, humaneval_task, cut_default):
        """Failing tests trigger a second iteration; second attempt returns valid tests."""
        generator = _mock_generator([FAILING_TESTS, VALID_TESTS])
        fb_runner = _make_runner(runner, builder, generator)

        tests, result, api_calls, last_report = fb_runner.run(task=humaneval_task, cut=cut_default)

        assert result is not None
        assert result.verdict == Verdict.FAILED
        assert generator.generate.call_count == 1

    def test_syntax_error_triggers_retry_without_running(self, runner, builder, humaneval_task, cut_default):
        """Syntax error skips build+run and retries directly; second attempt is valid."""
        generator = _mock_generator([SYNTAX_ERROR_TESTS, VALID_TESTS])
        fb_runner = _make_runner(runner, builder,  generator)

        with patch.object(fb_runner.builder, "build_tests", wraps=fb_runner.builder.build_tests) as mock_build:
            tests, result, api_calls, last_report = fb_runner.run(task=humaneval_task, cut=cut_default)

        # build_tests should NOT have been called on the syntax-error iteration
        assert mock_build.call_count == 1
        assert result.verdict == Verdict.PASSED
        assert generator.generate.call_count == 2

    def test_max_iterations_respected(self, runner, builder, humaneval_task, cut_default):
        """Loop never exceeds max_iterations even if tests keep failing."""
        generator = _mock_generator([FAILING_TESTS] * 5)
        fb_runner = _make_runner(runner, builder, generator)
        fb_runner.max_iterations = 3

        tests, result, api_calls, last_report = fb_runner.run(task=humaneval_task, cut=cut_default)

        assert generator.generate.call_count <= 3

    def test_syntax_error_only_returns_none_result(self, runner, builder, humaneval_task, cut_default):
        """If every iteration has a syntax error, run_result stays None."""
        generator = _mock_generator([SYNTAX_ERROR_TESTS] * 3)
        fb_runner = _make_runner(runner, builder, generator)
        fb_runner.max_iterations = 3

        tests, result, api_calls, last_report = fb_runner.run(task=humaneval_task, cut=cut_default)

        assert result is None

    def test_error_verdict_on_non_assertion_failure(self, runner, builder, humaneval_task, cut_default):
        """A runtime error (not AssertionError) should produce Verdict.ERROR."""
        error_tests = """
from solution import has_close_elements

def test_runtime_error():
    raise RuntimeError("unexpected crash")
"""
        generator = _mock_generator([error_tests])
        fb_runner = _make_runner(runner, builder,  generator)
        fb_runner.max_iterations = 1

        tests, result, api_calls, last_report = fb_runner.run(task=humaneval_task, cut=cut_default)

        assert result is not None
        assert result.verdict == Verdict.ERROR

    def test_history_is_passed_on_followup(self, runner, builder, humaneval_task, cut_default):
        """On the second iteration, generate() receives a non-None history."""
        histories = []
        generator = _mock_generator([IMPORT_ERROR_TESTS, VALID_TESTS])

        original_side_effect = generator.generate.side_effect

        def tracking_generate(prompt_variables, samples, history):
            histories.append(history)
            return original_side_effect(prompt_variables, samples, history)

        generator.generate.side_effect = tracking_generate
        fb_runner = _make_runner(runner, builder, generator)
        fb_runner.run(task=humaneval_task, cut=cut_default)

        assert histories[0] is None  # first call: no history
        assert histories[1] is not None  # second call: history passed


    def test_import_error_triggers_retry(self, runner, builder, humaneval_task, cut_default):
        """ImportError triggers a retry; second attempt with valid imports passes."""
        generator = _mock_generator([IMPORT_ERROR_TESTS, VALID_TESTS])
        fb_runner = _make_runner(runner, builder, generator)

        tests, result, api_calls, last_report = fb_runner.run(task=humaneval_task, cut=cut_default)

        assert result is not None
        assert result.verdict == Verdict.PASSED
        assert generator.generate.call_count == 2