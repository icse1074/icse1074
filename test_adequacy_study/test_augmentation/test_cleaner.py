"""
cleaner.py

Deduplicates and validates augmented test suites per task_id.
"""

import ast
import logging
from tqdm import tqdm

from test_adequacy_study.file_utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def deduplicate_suites(suites: list[str]) -> list[str]:
    def deduplicate_suites_str(suites: list[str]) -> list[str]:
        seen, unique = set(), []
        for suite in suites:
            normalized = suite.strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def deduplicate_suites_ast(suites: list[str]) -> list[str]:
        seen_ast, unique = set(), []
        for suite in suites:
            try:
                normalized = ast.dump(ast.parse(suite))
                if normalized not in seen_ast:
                    seen_ast.add(normalized)
                    unique.append(suite)
            except SyntaxError:
                unique.append(suite)
        return unique

    return deduplicate_suites_ast(deduplicate_suites_str(suites))



def extract_test_names(detailed_test_results) -> set[str]:
    """
    Extract base test names from detailed_test_results.
    node_id format: 'hash/tests/test_generated.py::TestCases::test_foo'
    Returns {'test_foo', ...}
    """
    names = set()
    for result in (detailed_test_results or []):
        names.add(result.node_id.split("::")[-1])
    return names


def load_raw_suites(input_path: str) -> dict[str, list[str]]:
    """
    Read augmented test file, return {task_id: suites} for the first
    successful record per task. Skips failed records.
    """
    suites_by_task: dict[str, list[str]] = {}
    skipped = 0

    for record in read_jsonl(input_path):
        task_id = record.get("task_id")
        suites  = record.get("augmented_tests", [])
        if not record.get("success", False):
            skipped += 1
            continue
        if not task_id or not suites or task_id in suites_by_task:
            continue
        suites_by_task[task_id] = suites

    logger.info("Found %d tasks (%d failed records skipped)", len(suites_by_task), skipped)
    return suites_by_task


def clean_task(task_id: str, suites: list[str], task, runner, builder) -> dict:
    """
    Deduplicate suites, run each against canonical to get test names,
    drop suites fully covered by larger ones.
    Returns a clean record dict or None if nothing survived.
    """
    from test_adequacy_study.data_models.test_suite import TestSuite, TestFramework

    unique_suites = deduplicate_suites(suites)
    logger.info("[%s] %d raw → %d after dedup", task_id, len(suites), len(unique_suites))

    canonical_cut = builder.build_program(task=task, code=task.canonical_solution)
    if len(suites) > 1 :
        print("hi")
    # Run canonical to get test names per suite
    validated_suites = []
    for suite_idx, suite_source in tqdm(enumerate(unique_suites), total=len(unique_suites), leave=False):
        suite_id = f"suite{suite_idx}"
        try:
            ast.parse(suite_source)
        except SyntaxError:
            logger.warning("[%s] %s has a syntax error — skipping", task_id, suite_id)
            continue

        suite  = TestSuite(task_id=task_id, language="python", source=suite_source, framework=TestFramework.PYTEST)
        report = runner.run(canonical_cut, suite)
        test_names = extract_test_names(report.detailed_test_results)

        if not test_names:
            logger.warning("[%s] %s produced no test names — skipping", task_id, suite_id)
            continue

        logger.info("[%s] %s — %d tests", task_id, suite_id, len(test_names))
        validated_suites.append({
            "suite_id":   suite_id,
            "source":     suite_source,
            "test_names": test_names,
        })

    if not validated_suites:
        return None

    # Drop suites fully covered by larger ones
    # Keep track of unique test names
    covered_names: set[str] = set()
    deduplicated  = []
    for suite_info in sorted(validated_suites, key=lambda s: len(s["test_names"]), reverse=True):
        new_names = suite_info["test_names"] - covered_names
        if not new_names:
            logger.info("[%s] %s fully covered — dropping", task_id, suite_info["suite_id"])
            continue
        covered_names.update(suite_info["test_names"])
        deduplicated.append(suite_info)

    if not deduplicated:
        return None

    all_test_names = sorted(covered_names)
    logger.info("[%s] %d unique test names across %d suites", task_id, len(all_test_names), len(deduplicated))

    return {
        "task_id":           task_id,
        "augmented_tests":   [{"suite_id": s["suite_id"], "source": s["source"], "test_names": sorted(s["test_names"])} for s in deduplicated],
        "unique_test_names": all_test_names,
        "n_suites":          len(deduplicated),
        "n_unique_tests":    len(all_test_names),
    }


def clean_augmented_tests(
    input_path: str,
    output_path: str,
    benchmark: str,
    runner,
    builder,
    task_lookup: dict,
) -> str:
    """Full cleaning pipeline. Returns output_path."""
    raw_suites_by_task = load_raw_suites(input_path)
    written = 0

    for task_id, suites in raw_suites_by_task.items():
        if str(task_id) not in task_lookup:
            logger.warning("[%s] not found in benchmark — skipping", task_id)
            continue

        record = clean_task(str(task_id), suites, task_lookup[str(task_id)], runner, builder)
        if record is None:
            continue

        write_jsonl(output_path, [record], append=True)
        written += 1
        logger.info("[%s] written (%d/%d tasks done)", task_id, written, len(raw_suites_by_task))

    # Summary
    suite_counts = [record["n_suites"] for record in read_jsonl(output_path)]
    if suite_counts:
        logger.info(
            "Cleaning complete. %d tasks written. Suites per task — min=%d  max=%d  avg=%.1f  total=%d",
            written,
            min(suite_counts),
            max(suite_counts),
            sum(suite_counts) / len(suite_counts),
            sum(suite_counts),
        )
    else:
        logger.info("Cleaning complete. No tasks written to %s", output_path)

    return output_path