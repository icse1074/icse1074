
import pytest
from dotenv import load_dotenv
from tqdm import tqdm
import os
from test_adequacy_study.benchmarks.naturalcodebench import NaturalCodeBenchLoader
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.data_models.test_suite import TestFramework, TestSuite
from test_adequacy_study.runners.pytest_runner import PytestRunner

load_dotenv()

EXPECTED_SIZE = 70


@pytest.fixture(scope="module")
def loader(tmp_path_factory):

    return NaturalCodeBenchLoader(data_file=os.environ.get("NCB_DATASET"))

def test_len(loader):
    loader.load()
    assert len(loader) == EXPECTED_SIZE
    for task in loader.load() :
        print(task)
        break


@pytest.fixture(scope="module")
def naturalcodebench_task(loader):
    task = next(iter(loader.load()))

    return task


@pytest.fixture(scope="module")
def builder():
    return PythonProgramBuilder()


@pytest.fixture(scope="module")
def cut_default(builder, naturalcodebench_task):
    return builder.build_program(task=naturalcodebench_task)


@pytest.fixture(scope="module")
def test_suite(naturalcodebench_task):
    return TestSuite(
        task_id=naturalcodebench_task.task_id,
        source=naturalcodebench_task.tests,
        framework=TestFramework.NATURALCODEBENCH,
        language="python",
    )


@pytest.fixture(scope="module")
def runner():
    return PytestRunner(timeout=120)


def test_naturalcodebench_canonical_solution(runner, naturalcodebench_task):
    cut = CUT(
        task=naturalcodebench_task,
        implementation=naturalcodebench_task.canonical_solution,
        language="python",
    )

    suite = TestSuite(
        task_id="small/test",
        source=naturalcodebench_task.tests,
        framework=TestFramework.PYTEST,
        language="python",
    )

    report = runner.run(cut, suite)
    assert report.verdict == Verdict.PASSED


def test_naturalcodebench_wrong_solution(runner, naturalcodebench_task):
    cut = CUT(
        task=naturalcodebench_task,
        implementation="""
import string
from collections import Counter


def word_count(file_path):
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 移除标点符号并转换为小写
        translator = str.maketrans("", "", string.punctuation)
        content = content.translate(translator).lower()

        # 使用 Counter 统计单词出现次数
        words = content.split()
        word_counter = Counter(words)

        # 按照出现次数降序排列
        sorted_word_count = sorted(word_counter.items(), key=lambda x: x[0], reverse=True)

        for word, count in sorted_word_count:
            print(f"'{word}': {count}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
""",

        language="python",
    )

    suite = TestSuite(
        task_id=naturalcodebench_task.task_id,
        source=naturalcodebench_task.tests,
        framework=TestFramework.PYTEST,
        language="python",
    )

    report = runner.run(cut, suite)
    assert report.verdict == Verdict.FAILED

def test_naturalcodebench_wrong_solution_only_implementation(runner, naturalcodebench_task):
    cut = CUT(
        task=naturalcodebench_task,
        implementation="""
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 移除标点符号并转换为小写
        translator = str.maketrans("", "", string.punctuation)
        content = content.translate(translator).lower()

        # 使用 Counter 统计单词出现次数
        words = content.split()
        word_counter = Counter(words)

        # 按照出现次数降序排列
        sorted_word_count = sorted(word_counter.items(), key=lambda x: x[0], reverse=True)

        for word, count in sorted_word_count:
            print(f"'{word}': {count}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
""",

        language="python",
    )

    suite = TestSuite(
        task_id=naturalcodebench_task.task_id,
        source=naturalcodebench_task.tests,
        framework=TestFramework.PYTEST,
        language="python",
    )

    report = runner.run(cut, suite)
    assert report.verdict == Verdict.FAILED

def test_all_canonical_solutions(runner, loader):
    tasks = list(loader.load())
    failures = {}

    for task in tqdm(tasks, total=len(tasks)):
        if task.task_id in ["NaturalCodeBench/154", "NaturalCodeBench/165",
                            "NaturalCodeBench/173", "NaturalCodeBench/175",
                            "NaturalCodeBench/170"] :
            continue
        cut = CUT(
            task=task,
            implementation=task.canonical_solution,
            language="python",
        )

        suite = TestSuite(
            task_id=task.task_id,
            source=task.tests,
            framework=TestFramework.PYTEST,
            language="python",
        )

        report = runner.run(cut, suite)
        if report.verdict != Verdict.PASSED:
            failures[task.task_id] = {
                "verdict": report.verdict.value,
                "stdout": report.stdout,
                "stderr": report.stderr,
                "tests": [
                    {"node_id": t.node_id, "outcome": t.outcome, "message": t.message}
                    for t in (report.detailed_test_results or [])
                ],
            }
            tqdm.write(f"[{report.verdict.value.upper()}] {task.task_id}")

    import json
    with open("failures_ncb.json", "w") as f:
        json.dump(failures, f, indent=2)

    #tqdm.write(f"\n{len(failures)}/{len(tasks)} failed — see failures_ncb.json")
    assert not failures, f"{len(failures)} canonical solutions failed"


def test_canonical_solution_192(runner, loader):
    task = next(t for t in loader.load() if t.task_id == "NaturalCodeBench/192")

    cut = CUT(
        task=task,
        implementation=task.canonical_solution,
        language="python",
    )

    suite = TestSuite(
        task_id=task.task_id,
        source=task.tests,
        framework=TestFramework.PYTEST,
        language="python",
    )

    report = runner.run(cut, suite)

    #tests that have hardcoded test files that dont exist : NaturalCodeBench/154, 165, 173, 175
    #task that fails for assertion error : 170
    #
    print(f"verdict: {report.verdict}")
    print(f"stdout:\n{report.stdout}")
    print(f"stderr:\n{report.stderr}")
    for t in (report.detailed_test_results or []):
        print(f"  [{t.outcome}] {t.node_id}: {t.message}")

    assert report.verdict == Verdict.PASSED
