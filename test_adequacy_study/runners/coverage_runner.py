from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
import uuid
import shutil
from test_adequacy_study.data_models.coverage_report import CoverageReport
from test_adequacy_study.runners.pytest_runner import PytestRunner

logger = logging.getLogger(__name__)


class CoverageRunner(PytestRunner):
    """
    Extends PytestRunner to also measure line coverage using coverage.py.
    Reuses _prepare() to write solution.py and test files.
    """

    def run(self, cut, suite) -> CoverageReport:
        run_dir = Path(self.work_dir) / str(cut.task_id).replace("/", "_") / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            entry_point = self._prepare(cut, suite, run_dir)
            return self.__execute_coverage(entry_point, run_dir)
        finally:
            shutil.rmtree(run_dir.parent, ignore_errors=True)

    def __execute_coverage(self, entry_point: Path, run_dir: Path) -> CoverageReport:
        coverage_json_path = run_dir / ".coverage_report.json"
        coveragerc_path = run_dir / ".coveragerc" #for dynamic context : check which test covers which line

        # write coveragerc to enable dynamic context
        coveragerc_path.write_text(
            "[run]\n"
            "dynamic_context = test_function\n"
            "branch = True\n"
            "[json]\n"
            "show_contexts = true\n",
            encoding="utf-8"
        )

        result = self.sandbox.run(
            cmd=[
                "python", "-m", "pytest",
                str(entry_point),
                "-q",
                "--cov=solution",
                "--cov-config=" + str(coveragerc_path),
                "--cov-report=json:" + str(coverage_json_path),
            ],
            cwd=run_dir,
            timeout=self.timeout,
        )


        logger.debug("Coverage stdout: %s", result.stdout)
        logger.debug("Coverage stderr: %s", result.stderr)
        logger.debug("Coverage report exists: %s", coverage_json_path.exists())

        return self.__parse_coverage_report(coverage_json_path)

    def __parse_coverage_report(self, coverage_json_path: Path) -> CoverageReport:
        try:
            with open(coverage_json_path) as f:
                report = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Could not parse coverage report: %s", e)
            return CoverageReport(coverage_pct=0.0)

        files = report.get("files", {})
        solution_data = next(
            (v for k, v in files.items() if k.endswith("solution.py")),
            None,
        )

        if solution_data is None:
            logger.warning("solution.py not found in coverage report")
            return CoverageReport(coverage_pct=0.0)

        summary = solution_data.get("summary", {})

        # Line coverage
        missing_lines = solution_data.get("missing_lines", [])
        #covered_lines = solution_data.get("executed_lines", []) #includes lines run at import time (will not be covered by tests)
        covered_lines = [line for line in solution_data.get("contexts").keys() if solution_data.get("contexts")[line] != ['']]
        total_lines = len(covered_lines) + len(missing_lines)

        # Branch coverage
        missing_branches = solution_data.get("missing_branches", [])
        covered_branches = solution_data.get("executed_branches", [])
        total_branches = summary.get("num_branches", 0)

        # Unified coverage
        coverage_pct = summary.get("percent_covered", 0.0)

        # invert contexts: line -> [tests] becomes test -> set of lines
        contexts = solution_data.get("contexts", {})
        per_test_line_coverage: dict[str, list[int]] = defaultdict(list)
        for line_str, tests in contexts.items():
            for test in tests:
                if test and int(line_str) not in per_test_line_coverage[test]:
                    per_test_line_coverage[test].append(int(line_str))
        # get test -> set of branches
        executed_branches = solution_data.get("executed_branches", {})
        per_test_branch_coverage: dict[str, list[tuple[int, int]]] = defaultdict(list)

        # Map executed branches to the tests that took them.
        # A branch [from, to] is taken by a test if and only if that test executed both from_line AND to_line
        # example [7,9] and [7,8] are two branches, T executes the lines 7, 9 only so [7,8] is not executed by T
        for branch in executed_branches:
            from_line, to_line = branch
            tests_covering_from = set(contexts.get(str(from_line), []))
            tests_covering_to = set(contexts.get(str(to_line), []))
            tests_that_took_branch = tests_covering_from & tests_covering_to
            for test in tests_that_took_branch:
                if test and (from_line, to_line) not in per_test_branch_coverage[test]:
                    per_test_branch_coverage[test].append((from_line, to_line))

        return CoverageReport(

            # Unified/Total coverage -> not really used
            coverage_pct=coverage_pct,

            # Line coverage
            missing_lines=missing_lines,
            covered_lines=covered_lines,
            total_lines=total_lines,

            # Branch coverage
            missing_branches=missing_branches,
            covered_branches=covered_branches,
            total_branches=total_branches,

            per_test_line_coverage=dict(per_test_line_coverage),
            per_test_branch_coverage=dict(per_test_branch_coverage),

        )
