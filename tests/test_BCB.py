import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from tqdm import tqdm
from test_adequacy_study.benchmarks.bigcodebench import BigCodeBenchLoader
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.data_models.test_suite import TestSuite, TestFramework
from test_adequacy_study.runners.bigcodebench_test_runner import BigCodeBenchTestRunner

load_dotenv()

EXPECTED_SIZE = 1140


@pytest.fixture(scope="module")
def loader(tmp_path_factory):

    return BigCodeBenchLoader()

def test_len(loader):
    loader.load()
    assert len(loader) == EXPECTED_SIZE
    for task in loader.load() :
        print(task)
        break

@pytest.fixture(scope="module")
def bigcodebench_task(loader):
    task = next(iter(loader.load()))

    return task


@pytest.fixture(scope="module")
def builder():
    return PythonProgramBuilder()


@pytest.fixture(scope="module")
def cut_default(builder, bigcodebench_task):
    return builder.build_program(task=bigcodebench_task)


@pytest.fixture(scope="module")
def test_suite(bigcodebench_task):
    return TestSuite(
        task_id=bigcodebench_task.task_id,
        source=bigcodebench_task.tests,
        framework=TestFramework.BIGCODEBENCH,
        language="python",
    )

@pytest.fixture(scope="module")
def runner():
    return BigCodeBenchTestRunner(timeout=120)



def test_bigcodebench_canonical_solution(runner, bigcodebench_task) :
    cut = CUT(
        task=bigcodebench_task,
        implementation=bigcodebench_task.canonical_solution,
        language="python",
    )

    suite = TestSuite(
        task_id=bigcodebench_task.task_id,
        source=bigcodebench_task.tests,
        framework=TestFramework.BIGCODEBENCH,
        language="python",
    )

    report = runner.run(cut, suite)
    assert report.verdict == Verdict.PASSED

to_exclude = ['BigCodeBench/219', 'BigCodeBench/1005', 'BigCodeBench/971', 'BigCodeBench/856', 'BigCodeBench/612',#assertion error
              'BigCodeBench/495', #overflow of value
              'BigCodeBench/1028', #commad returns 1
              'BigCodeBench/593', 'BigCodeBench/596', "BigCodeBench/823", "BigCodeBench/612" #assertion
              ]
def test_all_canonical_solutions(runner, loader):
    tasks = list(loader.load())
    failures = {}

    for task in tqdm(tasks, total=len(tasks)):
        if task.task_id in to_exclude :
            continue
        cut = CUT(
            task=task,
            implementation=task.canonical_solution,
            language="python",
        )

        suite = TestSuite(
            task_id=task.task_id,
            source=task.tests,
            framework=TestFramework.BIGCODEBENCH,
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

    # dump to file for investigation
    import json
    with open("failures.json", "w") as f:
        json.dump(failures, f, indent=2)

    tqdm.write(f"\n{len(failures)}/{len(tasks)} failed — see failures.json")
    assert not failures, f"{len(failures)} canonical solutions failed"




@pytest.mark.parametrize("task_id", [
    "BigCodeBench/1", "BigCodeBench/2", "BigCodeBench/28", "BigCodeBench/55",
    "BigCodeBench/60", "BigCodeBench/67", "BigCodeBench/70", "BigCodeBench/85",
    "BigCodeBench/86", "BigCodeBench/88", "BigCodeBench/131", "BigCodeBench/146",
    "BigCodeBench/149", "BigCodeBench/150", "BigCodeBench/152", "BigCodeBench/158",
    "BigCodeBench/190", "BigCodeBench/193", "BigCodeBench/201", "BigCodeBench/202",
    "BigCodeBench/206", "BigCodeBench/216", "BigCodeBench/229",
    "BigCodeBench/240", "BigCodeBench/243", "BigCodeBench/250", "BigCodeBench/254",
    "BigCodeBench/256", "BigCodeBench/258", "BigCodeBench/260", "BigCodeBench/263",
    "BigCodeBench/266", "BigCodeBench/273", "BigCodeBench/274", "BigCodeBench/281",
    "BigCodeBench/283", "BigCodeBench/288", "BigCodeBench/299", "BigCodeBench/306",
    "BigCodeBench/316", "BigCodeBench/320", "BigCodeBench/325", "BigCodeBench/327",
    "BigCodeBench/328", "BigCodeBench/329", "BigCodeBench/330", "BigCodeBench/335",
    "BigCodeBench/336", "BigCodeBench/339", "BigCodeBench/340", "BigCodeBench/344",
    "BigCodeBench/346", "BigCodeBench/350", "BigCodeBench/352", "BigCodeBench/361",
    "BigCodeBench/362", "BigCodeBench/365", "BigCodeBench/368", "BigCodeBench/370",
    "BigCodeBench/384", "BigCodeBench/386", "BigCodeBench/389", "BigCodeBench/391",
    "BigCodeBench/394", "BigCodeBench/398", "BigCodeBench/400", "BigCodeBench/402",
    "BigCodeBench/403", "BigCodeBench/409", "BigCodeBench/410", "BigCodeBench/412",
    "BigCodeBench/423", "BigCodeBench/426", "BigCodeBench/431", "BigCodeBench/454",
    "BigCodeBench/466", "BigCodeBench/483", "BigCodeBench/484",
    "BigCodeBench/510", "BigCodeBench/543", "BigCodeBench/545", "BigCodeBench/547",
    "BigCodeBench/548", "BigCodeBench/549", "BigCodeBench/550", "BigCodeBench/562",
    "BigCodeBench/563", "BigCodeBench/575", "BigCodeBench/595", "BigCodeBench/597",
    "BigCodeBench/602", "BigCodeBench/603", "BigCodeBench/609",
    "BigCodeBench/625", "BigCodeBench/626", "BigCodeBench/629", "BigCodeBench/630",
    "BigCodeBench/631", "BigCodeBench/632", "BigCodeBench/641", "BigCodeBench/642",
    "BigCodeBench/643", "BigCodeBench/644", "BigCodeBench/645", "BigCodeBench/647",
    "BigCodeBench/649", "BigCodeBench/651", "BigCodeBench/671", "BigCodeBench/673",
    "BigCodeBench/674", "BigCodeBench/675", "BigCodeBench/676", "BigCodeBench/678",
    "BigCodeBench/679", "BigCodeBench/681", "BigCodeBench/682", "BigCodeBench/683",
    "BigCodeBench/684", "BigCodeBench/685", "BigCodeBench/692", "BigCodeBench/700",
    "BigCodeBench/707", "BigCodeBench/711", "BigCodeBench/714", "BigCodeBench/717",
    "BigCodeBench/724", "BigCodeBench/730", "BigCodeBench/739", "BigCodeBench/743",
    "BigCodeBench/745", "BigCodeBench/747", "BigCodeBench/754", "BigCodeBench/755",
    "BigCodeBench/760", "BigCodeBench/771", "BigCodeBench/775", "BigCodeBench/779",
    "BigCodeBench/782", "BigCodeBench/784", "BigCodeBench/785", "BigCodeBench/793",
    "BigCodeBench/798", "BigCodeBench/804", "BigCodeBench/810", "BigCodeBench/812",
    "BigCodeBench/813", "BigCodeBench/814", "BigCodeBench/816", "BigCodeBench/823",
    "BigCodeBench/832", "BigCodeBench/843", "BigCodeBench/846", "BigCodeBench/855",
    "BigCodeBench/868", "BigCodeBench/892", "BigCodeBench/898",
    "BigCodeBench/899", "BigCodeBench/902", "BigCodeBench/905", "BigCodeBench/906",
    "BigCodeBench/907", "BigCodeBench/909", "BigCodeBench/918", "BigCodeBench/923",
    "BigCodeBench/926", "BigCodeBench/938", "BigCodeBench/947", "BigCodeBench/952",
    "BigCodeBench/962", "BigCodeBench/974", "BigCodeBench/992",
    "BigCodeBench/997", "BigCodeBench/998",  "BigCodeBench/1006",
    "BigCodeBench/1012", "BigCodeBench/1038", "BigCodeBench/1045",
    "BigCodeBench/1046", "BigCodeBench/1050", "BigCodeBench/1059", "BigCodeBench/1067",
    "BigCodeBench/1070", "BigCodeBench/1086", "BigCodeBench/1103", "BigCodeBench/1104",
    "BigCodeBench/1116", "BigCodeBench/1122", "BigCodeBench/1125", "BigCodeBench/1127",
])
def test_canonical_solution(runner, loader, task_id):
    task = next(t for t in loader.load() if t.task_id == task_id)

    cut = CUT(
        task=task,
        implementation=task.canonical_solution,
        language="python",
    )

    suite = TestSuite(
        task_id=task.task_id,
        source=task.tests,
        framework=TestFramework.BIGCODEBENCH,
        language="python",
    )

    report = runner.run(cut, suite)

    print(f"verdict: {report.verdict}")
    print(f"stdout:\n{report.stdout}")
    print(f"stderr:\n{report.stderr}")
    for t in (report.detailed_test_results or []):
        print(f"  [{t.outcome}] {t.node_id}: {t.message}")

    assert report.verdict == Verdict.PASSED

def test_canonical_solution_647(runner, loader):
    task = next(t for t in loader.load() if t.task_id == "BigCodeBench/1085")

    cut = CUT(
        task=task,
        implementation=task.canonical_solution,
        language="python",
    )

    suite = TestSuite(
        task_id=task.task_id,
        source=task.tests,
        framework=TestFramework.BIGCODEBENCH,
        language="python",
    )

    report = runner.run(cut, suite)

    print(f"verdict: {report.verdict}")
    print(f"stdout:\n{report.stdout}")
    print(f"stderr:\n{report.stderr}")
    for t in (report.detailed_test_results or []):
        print(f"  [{t.outcome}] {t.node_id}: {t.message}")

    assert report.verdict == Verdict.PASSED


def test_bigcodebench_wrong_solution(runner, bigcodebench_task) :
    cut = CUT(
        task=bigcodebench_task,
        implementation="""
    permutations = list(itertools.permutations(numbers))
    sum_diffs = 15

    for perm in permutations:
        perm = list(perm)
        shuffle(perm)
        diffs = [abs(perm[i] - perm[i+1]) for i in range(len(perm)-1)]
        sum_diffs += sum(diffs)

    avg_sum_diffs = sum_diffs / len(permutations)
    
    return avg_sum_diffs""",

        language="python",
    )

    suite = TestSuite(
        task_id="small/test",
        source=bigcodebench_task.tests,
        framework=TestFramework.BIGCODEBENCH,
        language="python",
    )

    report = runner.run(cut, suite)
    assert report.verdict == Verdict.FAILED