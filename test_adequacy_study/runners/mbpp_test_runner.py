import ast
import subprocess
import time
from pathlib import Path
from typing import Dict

from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.execution_report import ExecutionReport, Verdict
from test_adequacy_study.data_models.test_suite import TestSuite
from test_adequacy_study.helpers.parsers import convert_assertions
from test_adequacy_study.runners.helpers.execution_sandbox import SandboxResult
from test_adequacy_study.runners.test_runner import TestRunner


class MBPPTestRunner(TestRunner):
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
            if out != exp:
                failures.append({{"inp": inp, "exp": exp, "out": out}})
        except Exception as e:
            failures.append({{"inp": inp, "exp": exp, "out": "EXCEPTION: " + str(e)}})

run_tests()

for failure in failures:
    print(failure)
    print()

exit(len(failures))
"""
    def _prepare(self, cut: CUT, suite: TestSuite, run_dir: Path) -> Path:
        """
        Build the run script, applying all filters that match the owner's
        generate_data.py behaviour.
        """
        script = run_dir / "run.py"

        code = cut.implementation
        raw_tests = suite.source["test_inputs"]
        entry_point, converted_tests = convert_assertions(raw_tests)

        content = self.run_template.format(
            implementation=code,
            entry_point=entry_point,
            tests=converted_tests,
        )

        script.write_text(content)
        return script

    def _execute(self, entry_point: Path, task_id: str) -> ExecutionReport:
        result: SandboxResult = self.sandbox.run(
            cmd=["python3", str(entry_point)],
            cwd=entry_point.parent,
            timeout=self.timeout,
        )

        return ExecutionReport(
            task_id=task_id,
            verdict=Verdict.PASSED if result.returncode == 0 else Verdict.FAILED,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration,
            detailed_test_results=self._parse_failures(result.stdout)
        )

    def _parse_failures(self, stdout: str) -> list[dict]:
        results = []
        for line in stdout.splitlines():
            line = line.strip()
            if line:
                results.append(ast.literal_eval(line))
        return results