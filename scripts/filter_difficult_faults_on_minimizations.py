import json
import os
from pathlib import Path

# --- Configuration ---
FAULT_MODELS = [
    "gpt-5-mini",
    "gpt-4.1-mini",
    "claude-haiku-4-5",
    "meta-llama_llama-3.3-70B-Instruct",
    # "deepseek-v4-flash",
]

# Set up your primary directory structures
RESULTS_FOLDER = Path("test_adequacy_study/evaluation/original_prompt/results")
FAULTS_FOLDER = Path("output/artifact/faults_below_05")
OUTPUT_FOLDER = Path("test_adequacy_study/evaluation/original_prompt/results/below_05")


def get_valid_task_ids(faults_file_path):
    """Reads faults.jsonl and extracts all unique task_id strings into a set."""
    valid_ids = set()
    if not faults_file_path.exists():
        print(f"Warning: Faults file missing -> {faults_file_path}")
        return valid_ids

    with open(faults_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Keep track of task_id if it exists
                if "task_id" in data:
                    valid_ids.add(str(data["task_id"]))
            except json.JSONDecodeError:
                print(
                    f"Malformed JSON skipped in {faults_file_path.name} at line {line_num}"
                )

    return valid_ids


def filter_minimization_files(benchmark, guidance):
    """Iterates over models, grabs the allowed task_ids, and creates

    a filtered version of the minimization results in the output folder.
    """
    for model in FAULT_MODELS:
        # 1. Paths as per your specified file structures
        # Note the structural difference: faults folder drops the "guidance" nesting level
        faults_path = FAULTS_FOLDER / benchmark / model / "faults_below_05.jsonl"
        minimization_path = (
            RESULTS_FOLDER
            / model
            / benchmark
            / guidance
            / "minimisation_results.jsonl"
        )

        # Destination structure replicates the original results path inside the new output folder
        output_dir = OUTPUT_FOLDER / model / benchmark / guidance
        output_path = output_dir / "minimisation_results.jsonl"

        # Check if the source minimization file exists before proceeding
        if not minimization_path.exists():
            print(
                f"Skipping {model}: Source minimization file not found ({minimization_path})"
            )
            continue

        # 2. Extract allowed task IDs
        allowed_ids = get_valid_task_ids(faults_path)
        if not allowed_ids:
            print(
                f"Skipping {model}: No valid task IDs found in faults file (or file missing)."
            )
            continue

        # Create output directories if they don't exist yet
        output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Filter line-by-line and stream out to the new file
        copied_count = 0
        total_count = 0

        with open(minimization_path, "r", encoding="utf-8") as src, open(
            output_path, "w", encoding="utf-8"
        ) as dest:

            for line in src:
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    record = json.loads(line_str)
                    total_count += 1

                    # Look up task_id (cast to string to guarantee type alignment)
                    task_id = str(record.get("task_id", ""))

                    if task_id in allowed_ids:
                        dest.write(line_str + "\n")
                        copied_count += 1

                except json.JSONDecodeError:
                    # Ignore lines that aren't raw data dictionaries (e.g., headers or metadata)
                    continue

        print(
            f"[{model}] Processed {total_count} records -> Kept {copied_count} matches. Output: {output_path}"
        )


# --- Execution Block ---
if __name__ == "__main__":
    # Provide your evaluation configurations here
    GIVEN_BENCHMARK = "mbpp"
    GIVEN_GUIDANCE = "none"

    print("Starting cross-reference filtering script...\n")
    filter_minimization_files(
        benchmark=GIVEN_BENCHMARK, guidance=GIVEN_GUIDANCE
    )
    print("\nFiltering workflow complete.")