from typing import Iterator

from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.file_utils import read_jsonl, get_dataset_filepath
from test_adequacy_study.helpers.parsers import get_function_signature
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation


class MBPPLoader(BenchmarkLoader):
    def __init__(self, variation: BenchmarkVariation = None):
        self.__dataset = read_jsonl(get_dataset_filepath('mbpp', variation))
        print("Dataset loaded from file: ", get_dataset_filepath('mbpp', variation))

    def load(self, no_tests=False) -> Iterator[Task]:
        for record in self.__dataset:
            yield self.__parse_record_as_task(record)

    def __len__(self) -> int:
        return len(self.__dataset)

    def __parse_record_as_task(self, record: dict) -> Task:
        entry_point = get_function_signature(record['code'])
        prompt = record['text'] + "\n" + entry_point

        return Task(
            task_id=record['task_id'],
            stub=prompt,
            entry_point="",
            canonical_solution=record['code'],
            tests={'test_inputs': record['test_list']},
            libs=record['test_setup_code']
        )
