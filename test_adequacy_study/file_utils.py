import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime

from test_adequacy_study.types.benchmark_variation import BenchmarkVariation


def write_jsonl(filename: str, data: list, append: bool = False):
    """
    Writes or appends a list of dictionaries or dataclasses to a JSONL file.

    If the file or any of its parent directories do not exist, they will be
    automatically created.

    Dataclasses are automatically converted to
    dictionaries before serialization.

    :param filename:
    :param data:
    :param append:
    :return:
    """
    mode = "ab" if append else "wb"
    filename = os.path.expanduser(filename)

    # Extract the directory path and create it if it doesn't exist
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Write data to file
    with open(filename, mode) as fp:
        for x in data:
            # json.dumps does not support dataclass. Convert to json if this is the case
            data_to_export = asdict(x) if is_dataclass(x) else x
            fp.write((json.dumps(data_to_export) + "\n").encode("utf-8"))


def read_jsonl(file_path: str) -> list[dict]:
    dicts = []
    with open(file_path, "rb") as f:
        for line in f:
            string = line.decode("utf-8").strip()
            if string:
                try:
                    data = json.loads(string)
                    dicts.append(data)
                except json.JSONDecodeError as e:
                    print("Error decoding JSONL:", file_path)

    return dicts

def read_jsonl_pytests(file_path: str) -> dict:
    dicts = {}
    with open(file_path, "rb") as f:
        for line in f:
            string = line.decode("utf-8").strip()
            if string:
                try:
                    data = json.loads(string)
                    dicts[data['task_id']] = data['pytest']
                except json.JSONDecodeError as e:
                    print("Error decoding JSONL:", file_path)

    return dicts


def read_jsonl_line(file_path: str, task_id) -> list[dict]:
    with open(file_path, "rb") as f:
        for line in f:
            string = line.decode("utf-8").strip()
            if string:
                try:
                    data = json.loads(string)
                    if data["task_id"] == task_id:
                        return data
                except json.JSONDecodeError as e:
                    print("Error decoding JSONL:", file_path)

    return None


def get_output_filepath(suffix: str, model: str = "", with_timestamp: bool = True) -> str:
    """
    Generates the path to store the output based on the given arguments
    The role of this function is to help organize the output files when running multiple experiments

    :param suffix:
    :param model:
    :return:
    """
    model_sanitized = model.lower().replace('/', '_')
    timestamp_suffix = f"_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}" if with_timestamp else ""
    return f"output/{suffix}_{model_sanitized}{timestamp_suffix}.jsonl"


def get_dataset_filepath(name: str, variation: BenchmarkVariation = BenchmarkVariation.NONE) -> str:
    """
    Returns the DIR_DATA path specified in the .env file alongside the name + extension
    :param variation:
    :param name:
    :return:
    """
    if variation == BenchmarkVariation.NONE:
        filename = f"{name}.jsonl"
    else:
        filename = f"{name + '_' + variation.value}.jsonl"

    return os.getenv('DIR_DATA') + filename
