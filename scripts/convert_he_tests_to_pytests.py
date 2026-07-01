import json
import logging
import math
import sys
from typing import Iterator

from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from test_adequacy_study.providers.loader_provider import LoaderProvider
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation

# Assuming your loader is importable from your project setup:
# from your_module import HumanEvalLoader, Task

logger = logging.getLogger(__name__)


def format_inputs(inp_val):
    """Mirrors the dynamic unpacking logic:

    candidate(*inp) if isinstance(inp, (list, tuple)) else candidate(inp)
    """
    if isinstance(inp_val, (list, tuple)):
        return ", ".join(repr(x) for x in inp_val)
    return str(inp_val)


def exceeds_digit_limit(val, limit=4300) -> bool:
    """Recursively checks if any integer within the given structure exceeds the digit limit

    WITHOUT triggering Python's internal string conversion ValueError.
    """
    if isinstance(val, int):
        if val == 0:
            return False
        # Calculate number of digits mathematically to avoid string conversion limits
        num_digits = int(math.log10(abs(val))) + 1
        return num_digits > limit

    elif isinstance(val, (list, tuple, set)):
        return any(exceeds_digit_limit(item, limit) for item in val)

    elif isinstance(val, dict):
        # Check both keys and values if they could contain huge integers
        return any(
            exceeds_digit_limit(k, limit) or exceeds_digit_limit(v, limit)
            for k, v in val.items()
        )

    return False


def generate_pytest_from_task(task) -> str:
    """Generates a unittest.TestCase class string from a Task object.

    Correctly interprets each test case structure as [inp, exp].
    """
    entry_point = task.entry_point
    if not entry_point:
        entry_point = f"unknown_function_{task.task_id.replace('/', '_')}"

    class_lines = [
        "import unittest",
        f"from solution import {entry_point}",
        "",
        "",
        "class TestSolution(unittest.TestCase):",
    ]

    if not task.tests or "test_inputs" not in task.tests:
        class_lines.append("    pass")
        return "\n".join(class_lines) + "\n"

    test_inputs = task.tests["test_inputs"]

    # Ensure test_inputs is a list of test cases
    test_cases = (
        test_inputs if isinstance(test_inputs, list) else [test_inputs]
    )

    test_counter = 0
    for case in test_cases:
        # Check if the case is a valid pair of [inp, exp]
        if isinstance(case, (list, tuple)) and len(case) == 2:
            inp_data = case[0]
            exp_data = case[1]
        elif isinstance(case, dict) and "inp" in case and "exp" in case:
            inp_data = case["inp"]
            exp_data = case["exp"]
        else:
            # Fallback if the structure doesn't match a pair
            inp_data = case
            exp_data = "None"

        # Check individual assertion data before string conversion
        if exceeds_digit_limit(inp_data) or exceeds_digit_limit(exp_data):
            print(
                f"Skipping assertion for task {task.task_id} exceeding digit limit"
            )
            continue

        try:
            formatted_inp = format_inputs(inp_data)
            exp_val = (
                repr(exp_data) if not isinstance(exp_data, str) else exp_data
            )
        except ValueError:
            continue

        # Generates: assert entry_point(<formatted_inp>) == <exp_val>
        assertion_str = f"assert {entry_point}({formatted_inp}) == {exp_val}"

        method_str = (
            f"    def test_existing_ground_truth_test{test_counter}(self):\n"
            f"        {assertion_str}"
        )
        class_lines.append(method_str)
        test_counter += 1

    if test_counter == 0:
        class_lines.append("    pass")

    return "\n".join(class_lines) + "\n"

def process_benchmark_loader(loader: HumanEvalLoader, output_jsonl_path: str):
    """Iterates through the loader, extracts Task instances, generates the 'pytest' string,

    and exports all attributes into a target JSONL file.
    """
    processed_count = 0

    with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
        # Load tasks using the iterator interface
        for task in loader.load(no_tests=False):
            if task is None:
                print(f"Skipping 1 task...")
                continue
            print(f"Iterating task {task.task_id}")
            try:
                # 1. Generate the test suite class string
                pytest_class_str = generate_pytest_from_task(task)

                # 2. Convert Task data into a dictionary payload to preserve original information
                record = {
                    "task_id": task.task_id,
                    "stub": task.stub,
                    "entry_point": task.entry_point,
                    "canonical_solution": task.canonical_solution,
                    "tests": task.tests,
                    "pytest": pytest_class_str,  # Appending your new payload key
                }

                # 3. Stream data row out securely
                outfile.write(json.dumps(record) + "\n")
                processed_count += 1

            except Exception as e:
                if "Exceeds the limit" in str(e):
                    logger.error(
                        f"Skipping Task {task.task_id} completely: contained digits exceeding string limits."
                    )
                    continue

    print(
        f"Success! Processed {processed_count} tasks from loader into: {output_jsonl_path}"
    )


# --- Execution Block ---
if __name__ == "__main__":
    #sys.set_int_max_str_digits(0)

    # Initialize your custom loader pipeline
    loader = LoaderProvider().get(benchmark_name="he", variation = BenchmarkVariation.UNDER_SPECIFIED)

    output_path = ""
    process_benchmark_loader(loader, output_path)