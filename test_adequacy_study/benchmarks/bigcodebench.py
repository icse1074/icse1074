import ast
import logging
import re

from datasets import load_dataset
from datasets import Dataset
from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.file_utils import read_jsonl, write_jsonl, get_dataset_filepath

logger = logging.getLogger(__name__)
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation

#huggingface


class BigCodeBenchLoader(BenchmarkLoader):
    """
    Loads BigCodeBench tasks from HuggingFace.
    Option: filters to the hard subset only.
    """
    EXCLUDED = ['BigCodeBench/219', 'BigCodeBench/1005', 'BigCodeBench/971', 'BigCodeBench/856', 'BigCodeBench/612',#assertion error
              'BigCodeBench/495', #overflow of value
              'BigCodeBench/1028', #commad returns 1
              'BigCodeBench/593', 'BigCodeBench/596', "BigCodeBench/823", "BigCodeBench/612" #assertion
              ]
    DATASET_FULL = "bigcode/bigcodebench"
    SPLIT = "v0.1.4"

    def __init__(
        self,
        variation : BenchmarkVariation=BenchmarkVariation.NONE,
        from_datasets = False
    ):
        self._dataset : Dataset = None
        if from_datasets:
            self.__dataset_path = None
        else :
            self.__dataset_path = get_dataset_filepath('bcb', variation)

    def _load_dataset(self) :
        self._dataset = load_dataset(
            self.DATASET_FULL,
            split=self.SPLIT,
        )

    def _load_dataset_from_file(self):
        self._dataset = list(read_jsonl(self.__dataset_path))


    def save_to_file(self, jsonl_file: str, ):
        #Write all rows to a JSONL file
        if self._dataset is None:
            if self.__dataset_path:
                self._load_from_file()
            else:
                self._load_dataset()

        write_jsonl(jsonl_file, self._dataset)
        logger.info(f"Saved {len(self._dataset)} rows to {jsonl_file}")


    #some tasks need matplotlib but it is not specified in the dataset so we need to add them
    #some constants need to be imported from solution
    def _parse_row(self, row, no_tests):
        CONSTANT_PATTERN = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*=", re.MULTILINE)
        CLASS_PATTERN = re.compile(r"^class ([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]", re.MULTILINE)
        FUNCTION_PATTERN = re.compile(r"^def ([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
        IMPORT_PATTERN = re.compile(r"^(?:import .+|from .+ import .+)$", re.MULTILINE)

        def _extract_constants(stub: str) -> list[str]:
            return CONSTANT_PATTERN.findall(stub)

        def _extract_classes(stub: str) -> list[str]:
            return CLASS_PATTERN.findall(stub)

        def _extract_functions(stub: str) -> list[str]:
            # exclude task_func itself, it's always added explicitly
            return [f for f in FUNCTION_PATTERN.findall(stub) if f != "task_func"]

        def _extract_imports(stub: str) -> list[str]:
            return IMPORT_PATTERN.findall(stub)

        try:
            libs_list = ast.literal_eval(row["libs"]) if isinstance(row["libs"], str) else row["libs"]
        except (SyntaxError, ValueError):
            libs_list = []

        lib_imports = "\n".join(_extract_imports(row["complete_prompt"]))
        # if 'os' is used in the solution but not listed in libs, inject it
        if "os" not in libs_list and "os." in row["canonical_solution"]:
            row["complete_prompt"] = "import os\n" + row["complete_prompt"]
        # if 'datetime' is used in the solution but not listed in libs, inject it #647
        if "datetime" not in libs_list and "datetime." in row["canonical_solution"]:
            row["complete_prompt"] = "from datetime import datetime\n" + row["complete_prompt"]
        # if 'json' is used in the solution but not listed in libs, inject it #632
        if "json" not in libs_list and "json." in row["canonical_solution"]:
            row["complete_prompt"] = "import json\n" + row["complete_prompt"]
        # if 'punctuation' is used in the solution but not listed in libs, inject it #632
        if "punctuation" not in libs_list and "punctuation" in row["canonical_solution"]:
            row["complete_prompt"] = "from string import punctuation\n" + row["complete_prompt"]
        constants = _extract_constants(row["complete_prompt"])
        classes = _extract_classes(row["complete_prompt"])
        functions = _extract_functions(row["complete_prompt"])

        names_to_import = ["task_func"] + constants + classes + functions
        solution_import = f"from solution import {', '.join(names_to_import)}"

        tests = f"{solution_import}\n{lib_imports}\n\n{row['test']}" if not no_tests else {}
        if "US" in self.__dataset_path:
            row["complete_prompt"] = row["complete_prompt"].replace('\\"', '"')
        return Task(
            task_id=row["task_id"],
            stub=row["complete_prompt"],
            entry_point=row["entry_point"],
            canonical_solution=row["canonical_solution"],
            tests=tests,
            libs=libs_list,
        )
    def load(self, no_tests: bool = False):
        if self._dataset is None:
            if self.__dataset_path:
                #if the data is in a jsonl file, we read from here
                self._load_dataset_from_file()
            else :
                self._load_dataset()

        for row in self._dataset:
            if row["task_id"] in self.EXCLUDED:
                continue
            yield self._parse_row(row, no_tests)

    def __len__(self):
        if self._dataset is None:
            if self.__dataset_path:
                # if the data is in a jsonl file, we read from here
                self._load_dataset_from_file()
            else:
                self._load_dataset()

        return len(self._dataset)