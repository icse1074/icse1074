import json
import sys
from pathlib import Path


def normalize_test_code(test_code):
    """
    Minimal normalization: only strip trailing whitespace per line.
    We want to match test code exactly, not lose structure.
    """
    lines = test_code.split('\n')
    return '\n'.join(line.rstrip() for line in lines)


def extract_test_name(test_code):
    """Extract the test function name from test code."""
    for line in test_code.split('\n'):
        if 'def test_' in line:
            # Extract function name
            start = line.index('def ') + 4
            end = line.index('(', start)
            return line[start:end]
    return None


def match_completed_to_tasks(tests_file, completed_file, output_file):
    """
    Match completed tests back to their task_ids.

    Args:
        tests_file: Path to tests.jsonl with task_id and test_suite
        completed_file: Path to completed tests JSONL (one test per line)
        output_file: Path to output JSONL with task_id added
    """

    # Load all tasks with their test suites
    tasks_by_test_name = {}
    all_tasks = {}

    print(f"Loading tasks from {tests_file}...")
    with open(tests_file, 'r') as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                task_id = task.get('task_id')
                all_tasks[task_id] = task

                # Index by test node names from processed_tests
                if 'processed_tests' in task:
                    for test_info in task['processed_tests']:
                        test_node = test_info.get('test_node')
                        if test_node:
                            if test_node not in tasks_by_test_name:
                                tasks_by_test_name[test_node] = []
                            tasks_by_test_name[test_node].append(task_id)

    print(f"Loaded {len(all_tasks)} tasks")
    print(f"Indexed {len(tasks_by_test_name)} test names")

    # Load completed tests and try to match them
    matched_count = 0
    unmatched_count = 0
    results = []

    print(f"\nProcessing completed tests from {completed_file}...")
    with open(completed_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                completed_test = json.loads(line)

                # Try to match by test_node if it exists
                test_node = completed_test.get('test_node')
                matched_task_id = None
                match_method = None

                if test_node and test_node in tasks_by_test_name:
                    matched_task_id = tasks_by_test_name[test_node][0]
                    match_method = "by_test_node"
                    matched_count += 1
                else:
                    # Try to match by exact test code - look for the completed test within test suite
                    completed_code = completed_test.get('completed_test', '')
                    if not completed_code:
                        unmatched_count += 1
                        match_method = "no_code"
                    else:
                        completed_normalized = normalize_test_code(completed_code)
                        found = False

                        # Look for exact match in any task's test suite
                        for task_id, task in all_tasks.items():
                            if 'test_suite' in task:
                                test_suite = task['test_suite']
                                # Check if completed test appears exactly in the suite
                                if completed_code in test_suite:
                                    matched_task_id = task_id
                                    match_method = "exact_code_match"
                                    matched_count += 1
                                    found = True
                                    break

                                # Also try normalized comparison
                                if not found:
                                    test_suite_normalized = normalize_test_code(test_suite)
                                    if completed_normalized in test_suite_normalized:
                                        matched_task_id = task_id
                                        match_method = "normalized_code_match"
                                        matched_count += 1
                                        found = True
                                        break

                        if not found:
                            unmatched_count += 1
                            match_method = "no_match"

                # Add task_id to the completed test record
                if matched_task_id:
                    completed_test['task_id'] = matched_task_id
                    completed_test['match_method'] = match_method
                else:
                    completed_test['task_id'] = None
                    completed_test['match_method'] = match_method

                results.append(completed_test)

                if line_num % 100 == 0:
                    print(f"  Processed {line_num} tests...")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                unmatched_count += 1
                continue

    # Write results
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total completed tests: {len(results)}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")
    print(f"Match rate: {100 * matched_count / len(results):.1f}%" if results else "N/A")

    return results


if __name__ == "__main__":
    tests_file = "output/augmented_benchmarks/processed_tests/bcb/gpt-5-mini/tests.jsonl"
    completed_file = "output/augmented_benchmarks/processed_tests/bcb/gpt-5-mini/completed_tests.jsonl"

    # Verify files exist
    if not Path(tests_file).exists():
        print(f"Error: {tests_file} not found")
        sys.exit(1)
    if not Path(completed_file).exists():
        print(f"Error: {completed_file} not found")
        sys.exit(1)

    match_completed_to_tasks(tests_file, completed_file, output_file=completed_file)