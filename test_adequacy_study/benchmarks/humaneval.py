import logging
import os.path
from typing import Iterator, Optional, Dict

from evalplus.data import get_human_eval_plus
from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.data_models.task import Task
from dotenv import load_dotenv

from test_adequacy_study.file_utils import read_jsonl_line, get_dataset_filepath
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation

load_dotenv()
logger = logging.getLogger(__name__)

HF_REPO = "evalplus/humanevalplus"

class HumanEvalLoader(BenchmarkLoader):
    """Load HumanEval+ from the Hugging Face hub.

    The dataset is downloaded once and cached by the HF datasets library
    (in HF_DATASETS_CACHE env var).

    Args:
        cache_dir: Override the HF cache directory (optional).
                   Prefer setting HF_DATASETS_CACHE in the environment.
            """

    def __init__(self, variation: BenchmarkVariation = BenchmarkVariation.NONE) -> None:
        self._dataset:  Optional[dict[str, dict]] = None
        self.__dataset_path = get_dataset_filepath('he', variation)
        logger.info("HumanEval Benchmark will be load from: %s", self.__dataset_path)

    def _load_dataset(self) -> dict[str, dict]:
        self._dataset = get_human_eval_plus()

        return self._dataset

    def __parse_row(self, row: dict, task_id : str, no_tests : bool = False) -> Task:
        task_id: str = task_id           # "HumanEval/0"
        prompt: str = row["prompt"]              # signature + docstring
        entry_point: str = row["entry_point"]    # function name

        if not no_tests:
            if os.path.exists(self.__dataset_path):

                inputs = read_jsonl_line(self.__dataset_path, task_id)['tests']
            else:
                inputs = row["base_input"] + row["plus_input"]

            tests = dict(
                test_inputs=inputs,
                atol=row["atol"],
                test=row["test"],
            )
        else:
            tests = None
        canonical_solution: str = row["canonical_solution"] # reference solution (correct)

        task = Task(
            task_id=task_id,
            stub=prompt,
            entry_point=entry_point,
            canonical_solution=canonical_solution,
            tests=tests
        )
        return task

    # BenchmarkLoader interface
    def load(self, no_tests = True) -> Iterator[Task]:
        self._load_dataset()
        for task, row in self._dataset.items():
            yield self.__parse_row(row, task, no_tests)

    def __len__(self) -> int:
        if self._dataset is None:
            return len(self._load_dataset())

        return len(self._dataset)
