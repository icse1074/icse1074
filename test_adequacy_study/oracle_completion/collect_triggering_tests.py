"""
For each model & benchmark :
1. Read generated tests
2. Read analysis results
3. For each test in generated tests:
   - Check if it's triggering in analysis results
   - If yes, extract and store the test identifier (TestClass::test_function)
4. Write triggering tests to a new file
"""

import json
import os.path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict

from test_adequacy_study.file_utils import read_jsonl, write_jsonl


def extract_test_identifier(node_id: str) -> str:
    """
    Extract test identifier from full node ID.

    Handles both class methods and standalone functions.

    Examples:
        Class method:
            Input: "8d1d159b628247acb966bd232ad048ee/tests/test_generated.py::TestTaskFunc::test_all_keys_present"
            Output: "TestTaskFunc::test_all_keys_present"

        Standalone function:
            Input: "path/to/test_file.py::test_function"
            Output: "test_function"
    """
    parts = node_id.split("::")

    if len(parts) < 2:
        # No :: found, return as is
        return node_id

    # Everything after the file path (first ::) is the test identifier
    # This handles both:
    # - "file.py::TestClass::test_method" → "TestClass::test_method"
    # - "file.py::test_function" → "test_function"
    return "::".join(parts[1:])


def is_test_triggering(test_node_id: str, analysis_results: Dict) -> bool:
    """
    Check if a test is triggering based on analysis results.

    A test is triggering if it has a non-zero value in per_test_ft (fault triggering).
    """
    # Extract the test identifier from the full node ID
    test_identifier = extract_test_identifier(test_node_id)

    # Check in per_test_ft - if value > 0, it's triggering
    if "per_test_ft" in analysis_results:
        per_test_ft = analysis_results["per_test_ft"]

        # The key in per_test_ft might be the full node ID or just the identifier
        # Check both formats
        for key in per_test_ft:
            if test_identifier in key or key in test_node_id:
                if per_test_ft[key] > 0:
                    return True

    return False


def collect_triggering_tests(
        generated_tests_file: str,
        analysis_results_file: str,
        output_file: str,
) -> None:
    """
    Collect triggering tests from generated tests and analysis results.

    Args:
        generated_tests_file: Path to generated tests JSON file
        analysis_results_file: Path to analysis results JSON file
        output_file: Path to output file for triggering tests
    """

    # Read generated tests
    generated_tests_per_task = read_jsonl(generated_tests_file)
    # Read analysis results
    analysis_results_per_task = read_jsonl(analysis_results_file)
    analysis_results_dict = defaultdict(dict)
    for task in analysis_results_per_task:

        if str(task["task_id"]) not in analysis_results_dict :
            task_id =str(task["task_id"])
            analysis_results_dict[task_id] = task
    # Collect triggering tests


    for task in tqdm(generated_tests_per_task):
        task_id = str(task["task_id"])
        triggering_tests = []
        if task.get("execution_report") :
            if task.get("execution_report").get("detailed_test_results"):
                for test_result in task.get("execution_report").get("detailed_test_results"):
                    node_id = test_result.get("node_id", "")

                    # Check if this test is triggering
                    if is_test_triggering(node_id, analysis_results_dict[task_id]):
                        # Extract clean identifier
                        test_identifier = extract_test_identifier(node_id)
                        triggering_tests.append(test_identifier)

        # Prepare output structure (similar to generated tests)
        output_data = {
            "task_id": task_id,
            "code_under_test" : task.get("code_under_test"),
            "response" : task.get("response"),
            "model_id" : task.get("model_id"),
            "triggering_test_count": len(triggering_tests),
            "triggering_tests": triggering_tests,
        }
        write_jsonl(output_file, [output_data], append=True)

    print(f"Saved to {output_file}")



if __name__ == "__main__":

    benchmark = "bcb"
    for benchmark_variation in [""] :
        for model in ["gpt-5-mini"]:
            output_path = "output/augmented_benchmarks"
            os.makedirs(os.path.join(output_path, "triggering_tests"), exist_ok=True)
            collect_triggering_tests(
                generated_tests_file  = os.path.join(output_path, "generated_tests", benchmark, model, benchmark_variation, "tests.jsonl"),
                analysis_results_file =  os.path.join(output_path, "results", model, benchmark, benchmark_variation, "analysis_results.jsonl"),
                output_file = os.path.join(output_path, "triggering_tests", benchmark, model, benchmark_variation, "tests.jsonl"),
            )


