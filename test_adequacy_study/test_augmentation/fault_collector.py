"""
fault_collector.py

Runs code generations from a batch against augmented test suites,
collecting faults (generations that differ from canonical on at least one test).
"""

import logging

from tqdm import tqdm

from test_adequacy_study.file_utils import read_jsonl, write_jsonl
from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.runners.mutation_runner import _same_outcome

logger = logging.getLogger(__name__)


def load_augmented_tests(augmented_tests_path: str) -> dict[str, dict]:
    """
    Load clean augmented tests.
    Returns {task_id: {"suites": [...], "unique_test_names": set[str]}}
    """
    by_task = {}
    for record in read_jsonl(augmented_tests_path):
        task_id = str(record.get("task_id"))
        suites  = record.get("augmented_tests", [])
        unique_test_names = set(record.get("unique_test_names", []))
        if task_id and suites:
            by_task[task_id] = {"suites": suites, "unique_test_names": unique_test_names}
    return by_task


def run_generation_against_suites(
    generation: str,
    task_id: str,
    suites: list[dict],
    canonical_results_per_suite: list[dict],
    unique_test_names: set[str],
    runner,
    builder,
    task,
) -> dict:
    """
    Run a generation against all augmented suites, compare to canonical.
    Only considers tests whose base name is in unique_test_names.
    Returns {failures, score} if it's a fault, else None.
    """
    from test_adequacy_study.data_models.test_suite import TestSuite, TestFramework

    faulty_cut = builder.build_program(task=task, code=generation)
    if not faulty_cut.syntactically_valid:
        return None

    all_failures, total_tests = [], 0

    #runner.set_strict(False)
    for suite_info, canonical_results in zip(suites, canonical_results_per_suite):
        suite  = TestSuite(task_id=task_id, language="python",
                           source=suite_info["source"], framework=TestFramework.PYTEST)
        report = runner.run(faulty_cut, suite)

        if report.verdict == Verdict.TIMEOUT:
            logger.warning("[%s] generation timed out on %s", task_id, suite_info["suite_id"])
            continue
        if report.verdict == Verdict.ERROR:
            logger.warning("[%s] generation errored out on %s", task_id, suite_info["suite_id"])
            continue

        if not report.detailed_test_results:
            continue

        # Only consider tests in unique_test_names
        relevant_results = [
            t for t in report.detailed_test_results
            if t.node_id.split("::")[-1] in unique_test_names
        ]
        total_tests += len(relevant_results)

        faulty_by_id = {
            t.node_id.split("/", 1)[-1]: {"outcome": t.outcome, "message": t.message, "crash_path": t.crash_path}
            for t in relevant_results
        }

        for node_id, ref in canonical_results.items():
            # Only compare if this test is in unique_test_names
            base_name = node_id.split("::")[-1]
            if base_name not in unique_test_names:
                continue
            flt = faulty_by_id.get(node_id)
            if flt is None:
                continue
            if not _same_outcome(ref, flt):
                all_failures.append({
                    "node_id":    node_id,
                    "outcome":    flt["outcome"],
                    "message":    flt["message"],
                    "crash_path": flt["crash_path"],
                })

    if not all_failures:
        return None

    return {
        "failures": all_failures,
        "score":    len(all_failures) / total_tests if total_tests > 0 else 0.0,
    }


def run_canonical_on_suites(task_id: str, task, suites: list[dict], runner, builder) -> list[dict]:
    """Run canonical solution on each suite once. Returns list of {node_id: {outcome, message}}."""
    from test_adequacy_study.data_models.test_suite import TestSuite, TestFramework

    canonical_cut = builder.build_program(task=task, code=task.canonical_solution)
    results = []
    for suite_info in suites:
        suite  = TestSuite(task_id=task_id, language="python",
                           source=suite_info["source"], framework=TestFramework.PYTEST)
        report = runner.run(canonical_cut, suite)
        results.append({
            t.node_id.split("/", 1)[-1]: {"outcome": t.outcome, "message": t.message}
            for t in (report.detailed_test_results or [])
        })
    return results


def collect_faults(
    batch_lookup: dict[str, list[str]],
    augmented_tests_path: str,
    output_path: str,
    runner,
    builder,
    task_lookup: dict,
) -> int:
    """
    Run all generations against augmented suites, write faults to output_path.
    Returns number of faults written.
    """
    suites_by_task = load_augmented_tests(augmented_tests_path)
    logger.info("Loaded augmented tests for %d tasks", len(suites_by_task))

    filtered = {t: g for t, g in batch_lookup.items() if str(t) in suites_by_task}
    logger.info("Tasks in batch with augmented suites: %d", len(filtered))

    total_faults = 0

    for task_id, generations in tqdm(filtered.items()):
        if str(task_id) not in task_lookup:
            logger.warning("[%s] not found in benchmark — skipping", task_id)
            continue

        task              = task_lookup[str(task_id)]
        suites            = suites_by_task[str(task_id)]["suites"]
        unique_test_names = suites_by_task[str(task_id)]["unique_test_names"]

        canonical_results_per_suite = run_canonical_on_suites(task_id, task, suites, runner, builder)
        logger.info("[%s] running %d generations against %d suites (%d unique tests)",
                    task_id, len(generations), len(suites), len(unique_test_names))

        for gen_idx, generation in tqdm(enumerate(generations), total=len(generations), leave=False):
            fault = run_generation_against_suites(
                generation, task_id, suites, canonical_results_per_suite,
                unique_test_names, runner, builder, task
            )
            if fault is None:
                continue

            write_jsonl(output_path, [{
                "task_id":          task_id,
                "completion":       generation,
                "completion_index": gen_idx,
                "failures":         fault["failures"],
                "score":            fault["score"],
            }], append=True)
            total_faults += 1
            logger.info("[%s] gen %d — fault, score=%.3f (%d tests differ from canonical)",
                        task_id, gen_idx, fault["score"], len(fault["failures"]))

    logger.info("Done. %d faults written to %s", total_faults, output_path)
    return total_faults