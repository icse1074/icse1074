import re

# Define the input log file path
LOG_FILE_PATH = "output/hpc_he/he_log_errors_3.log"  # Replace with your actual log file name


def parse_log_file(file_path):
    # Regex pattern to match the specific line structure
    pattern = r"\[(HumanEval/\d+)\]\s+success=(\w+)"

    # Initialize an empty list to store the task IDs
    task_ids = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            print(f"{'TASK ID':<20} | {'SUCCESS':<10} | ORIGINAL LINE")
            print("-" * 80)

            for line in file:
                line = line.strip()
                match = re.search(pattern, line)

                if match:
                    task_id = match.group(1)
                    success = match.group(2)

                    # Append the task ID to our list
                    task_ids.append(task_id)

                    # Print the line-by-line breakdown
                    print(f"{task_id:<20} | {success:<10} | {line}")

            # Print the final space-separated string of task IDs
            print("\n" + "=" * 80)
            print("SPACE-SEPARATED TASK IDS:")
            print("=" * 80)
            print(" ".join(task_ids))

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")


if __name__ == "__main__":
    parse_log_file(LOG_FILE_PATH)