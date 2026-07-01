import json
import sys


def print_all_task_ids(file_path: str):
    """Reads a JSONL file and prints every task_id found."""
    task_ids_found: list[str] = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            print("🔍 Finding and printing all Task IDs:\n" + "-" * 35)

            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    data = json.loads(line)

                    # Check if 'task_id' exists in the record
                    if "task_id" in data:
                        task_ids_found.append(str(data["task_id"]))
                    else:
                        print(
                            f"Row {line_num:03d} -> ⚠️ 'task_id' field missing"
                        )

                except json.JSONDecodeError:
                    print(f"Row {line_num:03d} -> ❌ Error: Invalid JSON syntax")

            print("-" * 35 + "\nProcessing complete.")

    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")

    print(f"# Task ids: {len(task_ids_found)}\nFound task ids: {task_ids_found}")
    print("Command friendly printed:", " ".join(task_ids_found))


# --- Execution ---
if __name__ == "__main__":
    # Update this with your actual file name
    file_path = "output/hpc_tests/mbpp_deepseek_augmented_tests.jsonl"
    print_all_task_ids(file_path)