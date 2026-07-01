from __future__ import annotations

from pathlib import Path

import json

from test_adequacy_study.data_models.execution_report import ExecutionReport, Verdict, TestResult
from test_adequacy_study.data_models.test_materializer import materialize
from test_adequacy_study.runners.test_runner import TestRunner

class PytestRunner(TestRunner):
    """
    Two-file execution:
      - solution.py  : the CUT, importable as a module
      - tests.py : the pytest test file, imports from solution

    pytest is invoked on the test file directly.
    """
    strict = True
    def set_strict(self, strict: bool) -> None:
        self.strict = strict
    def _prepare(self, cut, suite, run_dir) -> Path:
        solution_file = run_dir / "solution.py"
        solution_file.write_text(cut.content, encoding="utf-8")

        #writing tests to test dir
        materialize(suite, run_dir)

        return run_dir

    def _determine_verdict(self, detailed_test_results) -> Verdict:

        if not detailed_test_results:
            return Verdict.ERROR

        passed = True
        failed_messages = []

        for test in detailed_test_results:
            if test.outcome == "failed":
                passed = False
                if self.strict:
                    #Only assertionErrors
                    is_legitimate = (
                            "AssertionError" in (test.message or "") or
                            "assert" in (test.message or "") or
                            "FileNotFoundError" in (test.message or "")
                    )
                    if not is_legitimate:
                        return Verdict.ERROR


        # non-strict: if all tests fail with the same message, mutation/fault unconditionally crashes
        if not self.strict and failed_messages and len(failed_messages) == len(detailed_test_results):
            unique_messages = set(failed_messages)
            if len(unique_messages) == 1:
                shared_message = next(iter(unique_messages))
                if "AssertionError" not in (shared_message or ""):
                    return Verdict.ERROR

        return Verdict.PASSED if passed else Verdict.FAILED

    def _build_report(self, task_id = None, result = None, report_path = None) :
        if "TIMEOUT after" in result.stderr :
            return ExecutionReport(
                task_id=task_id,
                verdict=Verdict.TIMEOUT,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=result.duration,
                detailed_test_results=None
            )
        detailed_test_results = self._parse_json_output(report_path)
        verdict = self._determine_verdict(detailed_test_results)

        # parsing of failed tests

        return ExecutionReport(
            task_id=task_id,
            verdict=verdict,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration,
            detailed_test_results=detailed_test_results
        )

    def _parse_json_output(self, json_report):

        try:
            with open(json_report) as f:
                report = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        results = []
        for test in report.get("tests", []):
            outcome = test.get("outcome", "unknown")

            message = None
            crash_path = None
            for phase in ("call", "setup", "teardown"):
                phase_data = test.get(phase, {})
                if phase_data.get("outcome") in ("failed", "error"):
                    #longrepr = phase_data.get("longrepr", {})
                    crash = phase_data.get("crash", {})
                    message = phase_data.get("crash", {}).get("message")
                    crash_path = crash.get("path")
                    assert message is not None
                    break

            results.append(TestResult(
                node_id=test["nodeid"],
                outcome=outcome,
                message=message,
                crash_path=crash_path,

            ))

        return results

    def _execute(self, entry_point: Path, task_id: str) -> ExecutionReport:

        report_path = str(entry_point.parent / ".pytest_report.json")
        result = self.sandbox.run(
            cmd=[
                "python", "-m", "pytest",
                str(entry_point.parent),
                "-q",
                "--json-report",
                f"--json-report-file={report_path}",
                "--json-report-indent=2",
            ],
            cwd=entry_point.parent,
            timeout=self.timeout,
        )

        return self._build_report(task_id, result, report_path)


