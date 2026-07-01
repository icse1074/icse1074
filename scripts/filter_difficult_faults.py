import json
import os
import sys


def read_jsonl(file_path):
    """Reads a .jsonl file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def filter_jsonl_files(input_folder: str, output_folder: str, threshold: float):
    # 4. Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # 1. Iterate through the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 2. Use read_jsonl to get records
            records = read_jsonl(input_path)

            # 3. Filter records where score < 0.5
            filtered_records = [r for r in records if r.get('score', 0) < threshold]

            # Save the new file
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in filtered_records:
                    f.write(json.dumps(record) + '\n')

            print(f"Processed: {filename} ({len(records)} -> {len(filtered_records)} records)")


# Run the script
if __name__ == "__main__":
    sys.set_int_max_str_digits(0)
    input_folder = "output/faults"
    output_folder = "faults/below_0.5/"
    threshold = 0.5
    filter_jsonl_files(input_folder, output_folder, threshold)
