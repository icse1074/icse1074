import json
import re
from typing import Tuple, Dict, Any

from test_adequacy_study.helpers.parsers import parse_code_from_markdown


def parse_custom_id(custom_id: str) -> Tuple[str, int]:
    """
    Parses the custom_id formatted as 'taskId_completionIndex'.

    Example: 'HumanEval/0_0' -> ('HumanEval/0', 0)
    """
    # Uses regex to split from the rightmost underscore, handling potential underscores in the taskId
    match = re.match(r"^(.*)_(\d+)$", custom_id)
    if not match:
        raise ValueError(f"Invalid custom_id format: {custom_id}")

    task_id, completion_index = match.groups()
    return task_id, int(completion_index)


def transform_batch_output(input_file_path: str, output_file_path: str) -> None:
    """
    Reads an OpenAI batch output file and transforms it into the target JSONL structure.
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
            open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line_number, line in enumerate(infile, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                # 1. Parse the custom_id using the helper function
                custom_id = record.get("custom_id", "")
                task_id, completion_index = parse_custom_id(custom_id)

                # 2. Extract the completion text from OpenAI's response structure
                # response -> body -> choices -> [0] -> message -> content
                response_node = record.get("response", {})
                body_node = response_node.get("body", {}) if response_node else {}
                choices = body_node.get("choices", []) if body_node else []

                if not choices:
                    print(f"Warning: No choices found on line {line_number}. Skipping.")
                    continue


                completion_text = parse_code_from_markdown(choices[0].get("message", {}).get("content", ""))

                # 3. Construct the new record structure
                new_record = {
                    "task_id": task_id,
                    "completion": completion_text,
                    "completion_index": completion_index
                }

                # 4. Write to the new JSONL file
                outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error processing line {line_number}: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_batch_jsonl = "data/batch_mbpp_gpt41mini.jsonl"
    output_transformed_jsonl = "data/generations_mbpp_gpt41mini.jsonl"

    # Run the transformation
    transform_batch_output(input_batch_jsonl, output_transformed_jsonl)
    print(f"Transformation complete! Output saved to {output_transformed_jsonl}")