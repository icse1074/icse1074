"""
RQ3: Update minimisation results FT/FD based on completed_tests_analysis.
Handles both OR and US variations.
"""

import json
import logging
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output/augmented_benchmarks")


def get_paths(fault_model: str, benchmark: str, variation: str) -> dict[str, Path]:
    """Get file paths for a fault model, benchmark, and variation (OR or US)."""
    var_path = "" if variation == "OR" else "us"

    return {
        "minimisation": OUTPUT_ROOT / "results" / fault_model / benchmark / var_path / "minimisation_results.jsonl",
        "completed_tests": OUTPUT_ROOT / "rq3" / benchmark / fault_model / var_path / "analysis_fd_ft.jsonl",
        "output": OUTPUT_ROOT / "rq3" / benchmark / fault_model / var_path / "rq3_minimisation_results.jsonl",
    }


def load_completed_tests_analysis(path: str) -> dict[str, dict]:
    """Load improved test FT/FD values."""
    improved = {}
    try:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                task_id = record.get("task_id")
                injected = record.get("injected", {})
                improved[task_id] = {
                    "per_test_ft": injected.get("per_test_ft", {}),
                    "per_test_fd": injected.get("per_test_fd", {}),
                }
    except FileNotFoundError:
        logger.warning("File not found: %s", path)
    return improved


def normalize_test_name(test_name: str) -> str:
    """Normalize test name for matching."""
    return test_name.replace(".py", "").replace("::", ".").replace("/", ".")


def get_test_ft_fd(test_name: str, improved_tests: dict) -> tuple[int, int]:
    """Get updated FT/FD from improved tests."""
    # test_name might include model prefix like "gpt-4.1-mini/tests/..."
    # Try to match both with and without prefix

    # Try exact match
    if test_name in improved_tests.get("per_test_ft", {}):
        ft = improved_tests["per_test_ft"].get(test_name, 0)
        fd = improved_tests["per_test_fd"].get(test_name, 0)
        return int(ft), int(fd)

    # Try without model prefix (remove everything before first "/tests/")
    if "/tests/" in test_name:
        test_path = test_name.split("/tests/", 1)[1]
        test_path = "tests/" + test_path
        if test_path in improved_tests.get("per_test_ft", {}):
            ft = improved_tests["per_test_ft"].get(test_path, 0)
            fd = improved_tests["per_test_fd"].get(test_path, 0)
            return int(ft), int(fd)

    # Try normalized
    normalized = normalize_test_name(test_name)
    if normalized in improved_tests.get("per_test_ft", {}):
        ft = improved_tests["per_test_ft"].get(normalized, 0)
        fd = improved_tests["per_test_fd"].get(normalized, 0)
        return int(ft), int(fd)

    return 0, 0


def update_selected_tests(selected_tests_list, improved_tests: dict) -> list:
    """Update FT/FD values in selected_tests list.

    Structure: [run][step] = [{"name": ..., "ft": ..., "fd": ...}]
    Each step has a list containing one test dict
    """
    updated = []
    for run_tests in selected_tests_list:
        updated_run = []
        for step_tests in run_tests:
            # step_tests is a list containing test dict(s)
            updated_step = []

            if isinstance(step_tests, list):
                for test_info in step_tests:
                    if isinstance(test_info, dict):
                        test_name = test_info.get("name")
                        if test_name:
                            improved_ft, improved_fd = get_test_ft_fd(test_name, improved_tests)
                            updated_step.append({
                                "name": test_name,
                                "ft": improved_ft,
                                "fd": improved_fd,
                            })
            elif isinstance(step_tests, dict):
                # Single test dict (not wrapped in list)
                test_name = step_tests.get("name")
                if test_name:
                    improved_ft, improved_fd = get_test_ft_fd(test_name, improved_tests)
                    updated_step.append({
                        "name": test_name,
                        "ft": improved_ft,
                        "fd": improved_fd,
                    })

            if updated_step:
                updated_run.append(updated_step)

        if updated_run:
            updated.append(updated_run)

    return updated


def reaggregate_metrics(selected_tests_list: list) -> dict[str, list]:
    """Re-aggregate trigger/detection from updated selected_tests."""
    metrics = {"trigger": [], "detection": []}

    for run_tests in selected_tests_list:
        found_trigger = False
        found_detection = False

        for step_tests in run_tests:
            # step_tests is list of test dicts
            if isinstance(step_tests, list):
                for test_info in step_tests:
                    if isinstance(test_info, dict):
                        if test_info.get("ft"):
                            found_trigger = True
                        if test_info.get("fd"):
                            found_detection = True
            elif isinstance(step_tests, dict):
                if step_tests.get("ft"):
                    found_trigger = True
                if step_tests.get("fd"):
                    found_detection = True

        metrics["trigger"].append(float(found_trigger))
        metrics["detection"].append(float(found_detection))

    return metrics


def process_rq3_variation(fault_model: str, benchmark: str, variation: str):
    """Update minimisation results for one variation (OR or US)."""

    paths = get_paths(fault_model, benchmark, variation)
    min_path = paths["minimisation"]
    completed_path = paths["completed_tests"]
    out_path = paths["output"]

    if not min_path.exists():
        logger.warning("[%s] Minimisation file not found: %s", variation, min_path)
        return 0

    if not completed_path.exists():
        logger.warning("[%s] Completed tests file not found: %s", variation, completed_path)
        return 0

    logger.info("[%s] Loading improved tests from %s", variation, completed_path)
    improved_tests = load_completed_tests_analysis(str(completed_path))
    logger.info("[%s] Loaded improved tests for %d tasks", variation, len(improved_tests))

    logger.info("[%s] Processing minimisation results from %s", variation, min_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_updated = 0
    with open(min_path) as infile, open(out_path, "w") as outfile:
        for line in infile:
            record = json.loads(line)
            task_id = str(record.get("task_id"))

            if task_id not in improved_tests:
                # No improved tests, keep original
                outfile.write(json.dumps(record) + "\n")
                continue

            improved = improved_tests[task_id]

            # Update each criterion's selected_tests
            keys_to_update = [k for k in record.keys() if "selected_tests" in k]
            for key in keys_to_update:
                selected_tests_list = record[key]
                # Update FT/FD in selected tests
                updated_list = update_selected_tests(selected_tests_list, improved)
                record[key] = updated_list

                # Re-aggregate trigger/detection metrics
                parts = key.split("_")
                version = parts[0]  # e.g. "truncated"
                criterion = parts[1]  # e.g. "random"

                metrics = reaggregate_metrics(updated_list)
                # Convert to lists of numpy arrays (same format as original minimisation)
                record[f"{version}_{criterion}_trigger"] = [np.array(metrics["trigger"])]
                record[f"{version}_{criterion}_detection"] = [np.array(metrics["detection"])]

            outfile.write(json.dumps(record, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")
            n_updated += 1

    logger.info("[%s] Saved RQ3 results to %s (%d tasks updated)", variation, out_path, n_updated)
    return n_updated


def process_rq3(fault_model: str, benchmark: str):
    """Update minimisation results for both OR and US variations."""

    logger.info("=" * 60)
    logger.info("RQ3: Processing %s / %s", fault_model, benchmark)
    logger.info("=" * 60)

    total_updated = 0

    for variation in ["OR", "US"]:
        n = process_rq3_variation(fault_model, benchmark, variation)
        total_updated += n

    logger.info("=" * 60)
    logger.info("Total tasks updated: %d", total_updated)
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    for benchmark in ["mbpp"] :
        for model in ["gpt-4.1-mini", "gpt-5-mini", "deepseek-v4-flash", "claude-haiku-4-5", "meta-llama_llama-3.3-70B-Instruct"] :

            process_rq3(model, benchmark)