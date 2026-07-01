import os
import re
import shutil
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = "output/tests/"  # Change to your source directory
TARGET_DIR = "output/artifact/"  # Change to your desired output directory
TASK_NAME = "tests"  # The task name given to the function (e.g., "tests", "faults")

# Allowed values based on your specifications
BENCHMARKS = ["mbpp_US", "mbpp", "he_US", "he"]
MODELS = ["gpt-5-mini", "deepseek-v4-flash", "claude-haiku-4-5", "gpt-4.1-mini", "meta-llama_llama-3.3-70B-Instruct"]
GUIDANCES = ["none", "line", "branch", "mutation"]


def organize_jsonl_files(source_dir, target_dir, task):
    source_path = Path(source_dir)
    target_base = Path(target_dir)

    if not source_path.exists():
        print(f"Source directory '{source_dir}' does not exist.")
        return

    # Escape pieces and sort by length descending so longer strings (e.g., 'mbpp_US')
    # match before shorter substrings (e.g., 'mbpp')
    benchmarks_pat = "|".join(sorted(map(re.escape, BENCHMARKS), key=len, reverse=True))
    models_pat = "|".join(sorted(map(re.escape, MODELS), key=len, reverse=True))
    guidances_pat = "|".join(sorted(map(re.escape, GUIDANCES), key=len, reverse=True))

    # Dynamically build regex and folder logic based on the task type
    if task.lower() == "tests":
        # Full pattern including guidance
        pattern = re.compile(
            rf"(?P<benchmark>{benchmarks_pat}).*?"
            rf"(?P<guidance>{guidances_pat}).*?"
            rf"(?P<model>{models_pat})",
            re.IGNORECASE
        )
    else:
        # Simplified pattern ignoring guidance entirely
        pattern = re.compile(
            rf"(?P<benchmark>{benchmarks_pat}).*?"
            rf"(?P<model>{models_pat})",
            re.IGNORECASE
        )

    copied_count = 0

    # Iterate through all files in the source directory
    for file_path in source_path.glob("*.jsonl"):
        filename = file_path.name
        match = pattern.search(filename)

        if match:
            # Extract matched components
            benchmark = match.group("benchmark")
            model = match.group("model")

            # Determine destination folder structure based on the task
            if task.lower() == "tests":
                guidance = match.group("guidance")
                # Format: task / benchmark / model / guidance /
                dest_dir = target_base / task / benchmark / model / guidance
            else:
                # Format: task / benchmark / model /
                dest_dir = target_base / task / benchmark / model

            dest_dir.mkdir(parents=True, exist_ok=True)

            # Define the simplified new filename (e.g., faults.jsonl)
            dest_file_path = dest_dir / f"{task}.jsonl"

            # Skip if already exists
            if dest_file_path.exists():
                print(f"Skipped (Already exists in target): {filename}")
                continue

            # Copy the file
            shutil.copy2(file_path, dest_file_path)
            print(f"Copied: {filename} -> {dest_file_path}")
            copied_count += 1
        else:
            print(f"Skipped (No match found): {filename}")

    print(f"\nTask complete. Organized {copied_count} files.")

# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    organize_jsonl_files(SOURCE_DIR, TARGET_DIR, TASK_NAME)