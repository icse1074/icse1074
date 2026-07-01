import json
import random
from collections import defaultdict
from pathlib import Path

from test_adequacy_study.file_utils import read_jsonl


def merge_clean_tests_balanced(
    input_dir: str,
    output_path: str,
    file_pattern: str = "augmented_he_us_*_merged_clean.jsonl",
    seed: int = 42,
    selection: str = "random",  # "random" or "most_tests"
) -> None:
    """
    Merges multiple *_merged_clean.jsonl files (one per model) into a single file.
    If a task_id appears in multiple files, selection determines how the record is chosen:
        "random"      -> chosen randomly, balanced so no single source dominates
        "most_tests"  -> the record with the highest n_unique_tests is chosen
    """
    random.seed(seed)

    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(file_pattern))

    if not files:
        print(f"No files found matching {file_pattern} in {input_dir}")
        return

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f.name}")

    # task_id -> list of (source_file, record)
    task_to_records: dict[str, list[tuple[str, dict]]] = defaultdict(list)

    for file in files:
        records = read_jsonl(str(file))
        for r in records:
            task_id = r.get("task_id")
            if task_id is None:
                continue
            task_to_records[task_id].append((file.name, r))

    print(f"\nTotal unique task_ids across all files: {len(task_to_records)}")

    source_counts = defaultdict(int)
    final_records = []

    task_ids = list(task_to_records.keys())
    random.shuffle(task_ids)

    for task_id in task_ids:
        candidates = task_to_records[task_id]  # [(source_file, record), ...]

        if len(candidates) == 1:
            chosen_source, chosen_record = candidates[0]

        elif selection == "most_tests":
            # pick the record with the highest n_unique_tests, break ties randomly
            max_tests = max(r.get("n_unique_tests", 0) for _, r in candidates)
            best = [c for c in candidates if c[1].get("n_unique_tests", 0) == max_tests]
            chosen_source, chosen_record = random.choice(best)

        else:  # "random", balanced
            min_count = min(source_counts[src] for src, _ in candidates)
            least_used = [c for c in candidates if source_counts[c[0]] == min_count]
            chosen_source, chosen_record = random.choice(least_used)

        source_counts[chosen_source] += 1
        final_records.append(chosen_record)

    print(f"\nFinal distribution by source file (selection='{selection}'):")
    for src, count in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {src}: {count} ({count / len(final_records):.1%})")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in final_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(final_records)} merged records to {output_path}")

if __name__ == "__main__":
    merge_clean_tests_balanced(
        input_dir="test_augmentation/augmented_tests/he",
        output_path="augmented_tests/he/augmented_he_merged_clean.jsonl",
        selection = "most_tests",
    )