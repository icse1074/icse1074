import ast
import subprocess
import time
from pathlib import Path
from typing import Dict

from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.execution_report import ExecutionReport, Verdict
from test_adequacy_study.data_models.test_suite import TestSuite
from test_adequacy_study.runners.helpers.execution_sandbox import SandboxResult
from test_adequacy_study.runners.test_runner import TestRunner

# Tasks skipped entirely - Taken from dataset repository
EXCLUDED_TASK_IDS = {
    "HumanEval/32",
    "HumanEval/38",
    "HumanEval/50",
    "HumanEval/2",
    "HumanEval/4",
    "HumanEval/21",
    "HumanEval/45",
}
# correct only with integers — float comparison would produce false failures
SKIP_FLOAT_TASK_IDS = {
    "HumanEval/53",
    "HumanEval/42",
    "HumanEval/58",
    "HumanEval/62",
    "HumanEval/71",
}


# Upper bound for integer expected values to avoid huge-number edge cases
LARGE_INT_THRESHOLD = 2 ** 62

# Skip individual test cases whose string representation exceeds this length
MAX_TEST_REPR_LEN = 10_000


class HumanEvalTestRunner(TestRunner):
    run_template = """
import sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100000000)
    
failures = []

{implementation}

def run_tests():
    candidate = {entry_point}

    tests = {tests}

    for inp, exp in tests:
        try:
            out = candidate(*inp) if isinstance(inp, (list, tuple)) else candidate(inp)

            if isinstance(exp, float):
                atol = {atol}
                if abs(out - exp) >= atol:
                    failures.append({{"inp": inp, "exp": exp, "out": out}})

            else:
                if out != exp:
                    failures.append({{"inp": inp, "exp": exp, "out": out}})

        except Exception as e:
            failures.append({{"inp": inp, "exp": exp, "out": "EXCEPTION: " + str(e)}})

run_tests()

for failure in failures:
    print(failure)
    print()

if (len(failures) > 0):
    exit(1)
"""

    @staticmethod
    def _is_floats(x) -> bool:
        """Mirror of evalplus is_floats — check if value is float-typed."""
        import numpy as np
        if isinstance(x, float):
            return True
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return all(isinstance(i, float) for i in x)
        if isinstance(x, np.ndarray):
            return x.dtype in (np.float64, np.float32)
        return False

    @staticmethod
    def _should_skip_test(
        inp,
        exp,
        task_id: str,
        atol: float,
        large_num: bool = False,
    ) -> bool:
        """
        Return True when a single (inp, exp) pair should be excluded from the
        generated test suite, mirroring every filter applied in the owner's
        get_single_test / generate_tests.

        Filters applied (in order):
        1. Large integers (abs > 2^62) when large_num=False.
        2. skip_float tasks — float assertions skipped for integer-manipulation tasks.
        3. Test-case repr too long (>10 000 chars) — avoids memory / perf problems.
        """

        # 1. Large integer guard
        if not large_num and isinstance(exp, int) and abs(exp) > LARGE_INT_THRESHOLD:
            return True

        # 2. skip_float: for specific tasks, skip any float-valued expected output
        effective_atol = atol
        if atol == 0 and HumanEvalTestRunner._is_floats(exp):
            effective_atol = 1e-6
        if task_id in SKIP_FLOAT_TASK_IDS and effective_atol != 0:
            return True

        # 3. Repr-length guard — mirrors the owner's `if len(curr_test) < 10000`
        test_repr = repr((inp, exp))
        if len(test_repr) > MAX_TEST_REPR_LEN:
            return True

        return False

    def _prepare_hf(self, cut, suite, run_dir) -> Path:
        script = run_dir / "run.py"

        content = f"""
{cut.content}

{suite.source if isinstance(suite.source, str) else ''}

# IMPORTANT: execute evaluation
check({cut.entry_point})
"""
        script.write_text(content, encoding="utf-8")
        return script

    def _prepare(self, cut: CUT, suite: TestSuite, run_dir: Path) -> Path:
        """
        Build the run script, applying all filters that match the owner's
        generate_data.py behaviour.
        """
        if isinstance(suite.source, str):
            return self._prepare_hf(cut, suite, run_dir)

        task_id = suite.task_id

        # Gate 1: entirely excluded tasks — nothing to run
        if task_id in EXCLUDED_TASK_IDS:
            script = run_dir / "run.py"
            script.write_text(
                "# Task excluded from HumanEval test suite\nprint([])\nexit(0)\n"
            )
            return script

        script = run_dir / "run.py"

        code = cut.content
        raw_tests = suite.source["test_inputs"]
        atol = suite.source.get("atol", 0)
        large_num = suite.source.get("large_num", False)

        filtered_tests = [
            (inp, exp)
            for inp, exp in raw_tests
            if not self._should_skip_test(
                inp, exp, task_id, atol, large_num
            )
        ]

        content = self.run_template.format(
            implementation=code,
            entry_point=cut.entry_point,
            tests=repr(filtered_tests),
            atol=atol if atol != 0 else 1e-6,
        )

        script.write_text(content)
        return script

    def _parse_failures(self, stdout: str) -> list[dict]:
        results = []
        for line in stdout.splitlines():
            line = line.strip()
            if line:
                results.append(ast.literal_eval(line))
        return results

    def _build_report(self, task_id: str, result: SandboxResult) -> ExecutionReport:
        detailed_test_results = self._parse_failures(result.stdout)

        return ExecutionReport(
            task_id=task_id,
            verdict=Verdict.PASSED if result.returncode == 0 else Verdict.FAILED,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration,
            detailed_test_results=detailed_test_results
        )

    def _execute(self, entry_point: Path, task_id: str) -> ExecutionReport:
        result: SandboxResult = self.sandbox.run(
            cmd=["python3", str(entry_point)],
            cwd=entry_point.parent,
            timeout=self.timeout,
        )

        return self._build_report(task_id, result)