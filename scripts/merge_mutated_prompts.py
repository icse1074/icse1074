import os
import sys

from tqdm import tqdm

from test_adequacy_study.benchmarks.bigcodebench import BigCodeBenchLoader
from test_adequacy_study.file_utils import read_jsonl, write_jsonl

sys.set_int_max_str_digits(0)
BENCHMARK_PROMPT_FIELD = {
    "mbpp": "text",
    "he": "prompt",
    "ncb": "problem",
    "bcb": "complete_prompt",
}

def get_prompt_field(benchmark: str) -> str :
    return BENCHMARK_PROMPT_FIELD.get(benchmark)

def merge_records(source_records, mutation_records, benchmark):
    # Create a lookup map for the source data using task_id
    # We convert task_id to string to ensure they match even if types differ
    source_map = {str(r.get('task_id', r.get('_id'))): r for r in source_records}

    merged_output = []

    for mut in tqdm(mutation_records):
        tid = str(mut.get('task_id'))

        # Only proceed if we find a matching task_id in the source file
        if tid not in source_map:
            continue

        src = source_map[tid]
        if not get_prompt_field(benchmark=benchmark) :
            print("Unsupported benchmark : ", benchmark)
            return []

        #copy to the new record
        new_record = dict(src)

        #add the mutated prompt

        new_record[get_prompt_field(benchmark=benchmark)] = mut.get('mutated_prompt')
        merged_output.append(new_record)

    return merged_output

if __name__ == "__main__":
    source_data = read_jsonl('data/he.jsonl')
    mutation_data = read_jsonl('data/he_us_mutaed.jsonl')
    output = "data/he_us.jsonl"


    result = merge_records(source_data, mutation_data, benchmark="he")

    # Save output:
    write_jsonl(output, result)

