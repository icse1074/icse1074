import json

faults_filepath = "output/artifact_mutating_prompts/faults/mbpp/gpt-5-mini/faults.jsonl"
benchmark_filepath = "data/mbpp.jsonl"
output_path = "output/artifact_mutating_prompts/faults/mbpp/gpt-5-mini/faults_with_original_prompt.jsonl"

# Step 1: Read the second file and map task_id to its code
code_mapping = {}
with open(benchmark_filepath, "r", encoding="utf-8") as f2:
    for line in f2:
        if line.strip():
            record2 = json.loads(line)
            t_id = record2.get("task_id")
            code = record2.get("code", "")
            if t_id is not None:
                code_mapping[t_id] = code

# Step 2: Read the first file, swap the data, and write to the new file
with open(faults_filepath, "r", encoding="utf-8") as f1, \
        open(output_path, "w", encoding="utf-8") as out_f:
    for line in f1:
        if line.strip():
            record1 = json.loads(line)
            t_id = record1.get("task_id")

            # If a matching task_id is found in the second file, replace it
            if t_id in code_mapping:
                record1["completion"] = code_mapping[t_id]
                out_f.write(json.dumps(record1) + "\n")
            else:
                print(f"Warning: No matching code found for task_id {t_id}")

print(f"Success! Combined file created at: {output_path}")