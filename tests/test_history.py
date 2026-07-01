import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from test_adequacy_study.generators.openai_model import OpenAIModel
from test_adequacy_study.generators.test_generator import TestGenerator


@pytest.fixture(scope="module")
def humaneval_task():
    cache_dir = Path(os.environ["HF_DATASETS_CACHE"]) / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    loader = HumanEvalLoader(cache_dir=str(cache_dir))
    return next(iter(loader.load()))


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=OpenAIModel)
    llm.call.return_value = (["def test_foo(): assert foo() == 42"], None)
    return llm


@pytest.fixture
def generator(mock_llm):
    gen = TestGenerator()
    gen.llm = mock_llm
    return gen


# ── OpenAIModel.call() ────────────────────────────────────────────────────────

class TestOpenAIModelHistory:

    def test_no_history_builds_system_and_user(self):

        #only adds response to history
        model = OpenAIModel('gpt-4o-mini')
        model.client = MagicMock()
        model.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))]
        )

        model.call(prompt="hello", system_prompt="you are helpful")

        messages = model.client.chat.completions.create.call_args[1]["messages"]
        assert messages[0] == {"role": "system", "content": "you are helpful"}
        assert messages[1] == {"role": "user", "content": "hello"}
        assert messages[2] == {"role": "assistant", "content": "response"}

    def test_history_inserted_between_system_user_and_new_response(self):
        model = OpenAIModel('gpt-4o-mini')
        model.client = MagicMock()
        model.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))]
        )
        history = [
            {"role": "system", "content": "you are system"},
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        response, messages = model.call(prompt="new question", system_prompt="you are helpful", history=history)

        assert messages[0]["role"] == "system"
        assert messages[1] == history[1]
        assert messages[2] == history[2]
        assert messages[-1]['role'] == "assistant"
        assert messages[-2] == {"role": "user", "content": "new question"}

    def test_caller_history_updated(self):
        model = OpenAIModel('gpt-4o-mini')
        model.client = MagicMock()
        model.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))]
        )
        history = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}]

        response, updated_history = model.call(prompt="new", history=history)

        assert history == updated_history


    def test_n_gt_1_rejects_history(self):
        model = OpenAIModel('gpt-4o-mini')
        with pytest.raises(AssertionError):
            model.call(prompt="hello", n=2, history=[{"role": "user", "content": "x"}])

    def test_returned_history_contains_assistant_message(self):
        model = OpenAIModel('gpt-4o-mini')
        model.client = MagicMock()
        model.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="my response"))]
        )

        _, history = model.call(prompt="hello", system_prompt="sys")

        assert history[-1] == {"role": "assistant", "content": "my response"}


# ── AbstractCodeGenerator.generate() ─────────────────────────────────────────

class TestGeneratorHistory:

    def test_no_history_returns_history_with_three_messages(self, generator, humaneval_task, mock_llm):
        mock_llm.call.return_value = (["def test_foo(): pass"], [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "def test_foo(): pass"},
        ])

        _, history = generator.generate({"code": humaneval_task.canonical_solution})

        assert len(history) == 3
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert history[2]["role"] == "assistant"

    def test_with_history_extends_it(self, generator, humaneval_task, mock_llm):
        existing_history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        mock_llm.call.return_value = (["def test_foo(): pass"], existing_history + [
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "def test_foo(): pass"},
        ])

        _, history = generator.generate(
            {"code": humaneval_task.canonical_solution},
            history=existing_history,
        )

        assert len(history) == 5

    def test_n_gt_1_returns_none_history(self, generator, humaneval_task, mock_llm):
        mock_llm.call.return_value = (["test_a", "test_b"], None)

        _, history = generator.generate(
            {"code": humaneval_task.canonical_solution},
            samples=2,
        )

        assert history is None

    def test_history_passed_to_llm(self, generator, humaneval_task, mock_llm):
        existing_history = [
            {"role": "user", "content": "prev"},
            {"role": "assistant", "content": "prev answer"},
        ]
        mock_llm.call.return_value = (["tests"], existing_history + [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "tests"},
        ])

        generator.generate(
            {"code": humaneval_task.canonical_solution},
            history=existing_history,
        )

        _, kwargs = mock_llm.call.call_args
        assert kwargs["history"] == existing_history