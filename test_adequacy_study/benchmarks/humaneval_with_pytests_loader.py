import logging
import os.path
from typing import Iterator, Optional, Dict

from evalplus.data import get_human_eval_plus
from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.data_models.task import Task
from dotenv import load_dotenv

from test_adequacy_study.file_utils import read_jsonl_line, get_dataset_filepath, read_jsonl_pytests
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation

load_dotenv()
logger = logging.getLogger(__name__)

HF_REPO = "evalplus/humanevalplus"


class HumanEvalWithPytestsLoader(BenchmarkLoader):
    """Load HumanEval+ from the Hugging Face hub.

    The dataset is downloaded once and cached by the HF datasets library
    (in HF_DATASETS_CACHE env var).

    Args:
        cache_dir: Override the HF cache directory (optional).
                   Prefer setting HF_DATASETS_CACHE in the environment.
            """

    def __init__(self, variation: BenchmarkVariation = BenchmarkVariation.NONE) -> None:
        self._dataset: Optional[dict[str, dict]] = None
        self.__dataset_path = get_dataset_filepath('he_with_pytests', variation)
        logger.info("HumanEval Benchmark will be load from: %s", self.__dataset_path)

    def _load_dataset(self) -> dict[str, dict]:
        self._dataset = get_human_eval_plus()
        self.__pytests = read_jsonl_pytests(self.__dataset_path)
        print(f"Loaded {len(self.__pytests)} pytests")
        return self._dataset

    def __parse_row(self, row: dict, task_id: str, no_tests: bool) -> Task:
        task_id: str = task_id  # "HumanEval/0"
        prompt: str = row["prompt"]  # signature + docstring
        entry_point: str = row["entry_point"]  # function name
        canonical_solution: str = row["canonical_solution"]  # reference solution (correct)

        task = Task(
            task_id=task_id,
            stub=prompt,
            entry_point=entry_point,
            canonical_solution=canonical_solution,
            tests=self.__pytests[task_id]
        )
        return task

    # BenchmarkLoader interface
    def load(self, no_tests=False) -> Iterator[Task]:
        self._load_dataset()
        for task, row in self._dataset.items():
            yield self.__parse_row(row, task, no_tests)

    def __len__(self) -> int:
        if self._dataset is None:
            return len(self._load_dataset())

        return len(self._dataset)
