import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder

load_dotenv()



@pytest.fixture(scope="module")
def humaneval_task():
    cache_dir = Path(os.environ["HF_DATASETS_CACHE"]) / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    loader = HumanEvalLoader(cache_dir=str(cache_dir))

    task = next(iter(loader.load()))
    return task


@pytest.fixture(scope="module")
def builder():
    return PythonProgramBuilder()


@pytest.fixture(scope="module")
def cut_default(builder, humaneval_task):
    return builder.build_program(task=humaneval_task)


@pytest.fixture(scope="module")
def cut_custom(builder, humaneval_task):
    return builder.build_program(
        task=humaneval_task,
        code="def fake(): return 42"
    )


def test_cut_default_uses_canonical_solution(cut_default, humaneval_task):
    assert humaneval_task.canonical_solution in cut_default.content


def test_cut_custom_overrides_solution(cut_custom):
    assert "fake" in cut_custom.content
    assert "42" in cut_custom.content


def test_cut_metadata(cut_default, humaneval_task):
    assert cut_default.task_id == humaneval_task.task_id
    assert cut_default.language == "python"


def test_cut_to_file(cut_default):
    os.makedirs("tmp", exist_ok=True)
    file_path = Path("tmp/solution.py")

    returned = cut_default.to_file(str(file_path))

    assert file_path.exists()
    assert returned == str(file_path)
    assert file_path.read_text(encoding="utf-8") == cut_default.content


# -------------------------
# TestSuite tests
# -------------------------

@pytest.fixture(scope="module")
def testsuite_default(builder, humaneval_task):
    return builder.build_tests(task=humaneval_task)


@pytest.fixture(scope="module")
def testsuite_custom(builder, humaneval_task):
    return builder.build_tests(
        task=humaneval_task,
        test_suite="def test_dummy(): assert True"
    )


def test_testsuite_default_uses_dataset_tests(testsuite_default, humaneval_task):
    assert humaneval_task.tests in testsuite_default.source


def test_testsuite_custom_overrides(testsuite_custom):
    assert "test_dummy" in testsuite_custom.source


def test_testsuite_metadata(testsuite_default, humaneval_task):
    assert testsuite_default.task_id == humaneval_task.task_id
    assert testsuite_default.language == "python"

