import ast
import json


def extract_function_name(assertion_str):
    """Safely parses an assertion string to extract the name of the function being called."""
    try:
        tree = ast.parse(assertion_str.strip())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, "id"):
                    return node.func.id
                elif hasattr(node.func, "attr"):
                    return node.func.attr
    except Exception:
        pass
    return None


def generate_test_class_string(assertions, line_num):
    """Generates the unittest.TestCase class string for a given list of assertions."""
    if not assertions:
        return ""

    # Ensure assertions are in a list format
    assertions_list = (
        assertions if isinstance(assertions, list) else [assertions]
    )

    # 1. Extract function name from the FIRST assertion
    first_assertion = str(assertions_list[0]).strip()
    extracted_function = extract_function_name(first_assertion)

    if not extracted_function:
        raise Exception("Invalid function")

    # 2. Build the class structure
    class_lines = [
        "import unittest",
        f"from solution import {extracted_function}",
        "",
        "class TestSolution(unittest.TestCase):",
    ]

    # 3. Convert all assertions into methods
    for test_counter, assertion in enumerate(assertions_list):
        assertion_clean = str(assertion).strip()
        if not assertion_clean:
            continue

        method_str = (
            f"    def test_existing_ground_truth_test{test_counter}(self):\n"
            f"        {assertion_clean}"
        )
        class_lines.append(method_str)

    return "\n".join(class_lines) + "\n"


def process_and_save_jsonl(input_path, output_path):
    """Reads input JSONL, injects the 'pytest' string class key into each record,

    and writes it to a new JSONL file.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as infile, open(
            output_path, "w", encoding="utf-8"
        ) as outfile:

            for line_num, line in enumerate(infile, 1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_num}")
                    continue

                test_suites = record.get("test_list")

                # Generate the class string if test_suites exists, otherwise leave empty
                if test_suites:
                    pytest_class_str = generate_test_class_string(
                        test_suites, line_num
                    )
                else:
                    pytest_class_str = ""

                # Add the new key to the existing record while keeping all original keys
                record["pytest"] = pytest_class_str

                # Write the updated record as a JSON line to the output file
                outfile.write(json.dumps(record) + "\n")

        print(f"Success! Processed file saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")


# --- Modified Execution Block ---
if __name__ == "__main__":
    file_path = ""
    output_file_path = (
        ""
    )

    # Process everything and save to the new file
    process_and_save_jsonl(file_path, output_file_path)