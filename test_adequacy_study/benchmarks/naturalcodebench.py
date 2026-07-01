import ast
import logging
from typing import Iterator, List, Dict, Optional
import re
from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.file_utils import read_jsonl, get_dataset_filepath
from dotenv import load_dotenv
import os

from test_adequacy_study.types.benchmark_variation import BenchmarkVariation

load_dotenv()
logger = logging.getLogger(__name__)

#file from repository : https://github.com/THUDM/NaturalCodeBench

class NaturalCodeBenchLoader(BenchmarkLoader):
    EXCLUDED = [154, 165, 170, 173, 175] #dataset tests have hardcoded files that dont exist -> can't compare faulty to reference solution
    def __init__(
        self,
        variation : BenchmarkVariation = BenchmarkVariation.NONE
    ):
        self.__dataset_path = get_dataset_filepath('ncb', variation)
        self._dataset : Optional[List[Dict]] = []

    def _load_dataset(self) :
        lines = read_jsonl(
            self.__dataset_path
        )
        for line in lines:
            if line["_id"] not in self.EXCLUDED:
               self._dataset.append(line)
        return self._dataset
    def _safe_parse(self, code: str):
        """
        Attempts to parse code, stripping incomplete trailing lines until it succeeds.
        """
        lines = code.splitlines()
        while lines:
            try:
                return ast.parse("\n".join(lines))
            except SyntaxError:
                lines.pop()
        return None

    def _extract_metadata(self, reference_solution: str) -> dict:
        """
        Extracts imports, functions, classes, and signature from a reference solution.
        """
        tree = self._safe_parse(reference_solution)

        imports = [ast.unparse(n) for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom))]
        entry_points = [n.name for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and not n.name.startswith("__")]
        class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        args = ast.unparse(func.args)
        returns = f" -> {ast.unparse(func.returns)}" if func.returns else ""
        signature = f"def {func.name}({args}){returns}:"

        return {
            "imports": imports,
            "entry_points": entry_points,
            "class_names": class_names,
            "entry_point": entry_points[0],
            "signature": signature,
        }

    def _build_stub(self, metadata: dict, problem: str) -> str:
        """
        Builds a HumanEval style stub from metadata and problem description.
        """
        imports_str = "\n".join(metadata["imports"])
        safe_problem = problem.replace('"""', "'''").replace('"', '\\"')
        stub = f"{imports_str}\n\n{metadata['signature']}\n    \"\"\"{safe_problem}\"\"\"\n"
        return stub

    def _build_tests(self, metadata: dict, testcases: str) -> str:
        """
        Builds the test file with correct imports from solution and canonical imports.
        """
        imports_str = "\n".join(metadata["imports"])
        all_names = ", ".join(metadata["class_names"] + metadata["entry_points"])
        if "pytest" in testcases:
            imports_str = imports_str+"\nimport pytest"
        return f"{imports_str}\nfrom solution import {all_names}\n\n{testcases}"

    def _parse_row(self, row: dict, no_tests: bool) -> Task:
        if row["_id"] == 192:
            print("yay")
        match = re.search(r"```python\n(.*?)\n```", row['reference_solution'], re.DOTALL)
        reference_solution = match.group(1) if match else row['reference_solution']

        metadata = self._extract_metadata(reference_solution)

        return Task(
            task_id="NaturalCodeBench/" + str(row["_id"]),
            stub=self._build_stub(metadata, row['problem']),
            entry_point=metadata["entry_point"],
            canonical_solution=reference_solution,
            tests=self._build_tests(metadata, row['testcases']) if not no_tests else {},
            libs=metadata["imports"],
        )
    def load(self, no_tests = False) -> Iterator[Task]:
        if not len(self._dataset):
            self._load_dataset()

        for row in self._dataset:
            yield self._parse_row(row, no_tests)

    def __len__(self) -> int:
        if not len(self._dataset):
            self._load_dataset()
        return len(self._dataset)
