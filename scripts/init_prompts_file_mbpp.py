import ast
import json

from test_adequacy_study.types.prompt_input_holder import PromptInputDict, PromptInputHolder


def extract_first_signature(code_string):
    """Parses Python code and returns the signature of the first function found."""
    try:
        tree = ast.parse(code_string)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                lines = code_string.splitlines()
                # ast.FunctionDef's lineno points to the exact line where 'def' starts
                signature_line_idx = node.lineno - 1

                # If the function has a multi-line signature or decorators,
                # node.lineno points to the first line. For a standard signature,
                # returning just this line is exactly what you need:
                return lines[signature_line_idx].strip()
    except SyntaxError:
        print("Warning: Could not parse code due to a SyntaxError.")
        return None

    return None


def process_jsonl(input_file_path):
    """Reads a JSONL file and creates strings with text and the function signature."""
    combined_results = []

    with open(input_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                task_id = str(record.get('task_id'))
                text = record.get("text", "").strip()
                code_block = record.get("code", "")

                # Extract only the signature of the first function
                signature = extract_first_signature(code_block)

                if text and signature:
                    # Combine the text and the signature
                    combined_string = f"{text}\n\nFunction signature:\n{signature}"
                    combined_results.append((task_id, combined_string))
                else:
                    print(
                        f"Skipping line {line_num}: Missing text or no function signature found."
                    )

            except json.JSONDecodeError:
                print(f"Skipping line {line_num}: Invalid JSON format.")

    return combined_results


# --- Example Usage ---
if __name__ == "__main__":
    input_filename = "data/mbpp.jsonl"

    prompts_dict = PromptInputDict()

    # Process and print results
    results = process_jsonl(input_filename)
    for task_id, result in results:
        prompts_dict.set(task_id, PromptInputHolder(original_prompt=result, round_trip_prompt=""))

    prompts_dict.to_jsonl("output/artifact_mutating_prompts/prompts/mbpp_original_with_signature.jsonl")