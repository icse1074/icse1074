from typing import Iterator

from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.file_utils import read_jsonl


class GeneratedCodeLoader(BenchmarkLoader):
    """
    Loads the records of jsonl files that contain generated code solutions
    It will generate new Task instances for each record.

    Ideally, this class should be used to generate tests for each generated code solution of this loader
    """
    def __init__(self, input_filepath: str, task_id_prefix: str = "generation_"):
        self.__dataset = read_jsonl(input_filepath)
        self.__task_id_prefix = task_id_prefix

    def load(self) -> Iterator[Task]:
        for record in self.__dataset:
            yield self.__parse_record_as_task(record)

    def __len__(self) -> int:
        return len(self.__dataset)

    def __parse_record_as_task(self, record: dict) -> Task:
        return Task(
            task_id=self.__task_id_prefix + str(record['task_id']),
            stub="",
            entry_point="",
            canonical_solution="",
            generated_solution=record["completion"],
            tests={},
        )
