#analysis.py
from __future__ import annotations

import logging

from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.evaluation.helpers.run_version import run_tests_on_code
from test_adequacy_study.helpers.parsers import is_syntactically_valid
from test_adequacy_study.runners.coverage_runner import CoverageRunner
from test_adequacy_study.runners.mutation_runner import MutationRunner, MutantInfo, _same_outcome
from test_adequacy_study.runners.test_runner import TestRunner
from test_adequacy_study.evaluation.config import WORK_DIR
from test_adequacy_study.evaluation.helpers.data_models import AnalysisRecord, CoverageRecord, MutationRecord

logger = logging.getLogger(__name__)


def _compute_ft_fd(
    ref_results:   dict[str, dict],
    fault_results: dict[str, dict],
) -> tuple[int, int, dict[str, int], dict[str, int]]:
    """
    Compare per-test outcomes on the reference vs faulty version.
    Returns (fault_triggered, fault_detected, per_test_ft, per_test_fd).
    """
    found_trigger = found_detect = False
    per_test_ft: dict[str, int] = {}
    per_test_fd: dict[str, int] = {}

    for test in ref_results:
        if test not in fault_results:
            continue

        ref = ref_results[test]
        flt = fault_results[test]

        triggers = not _same_outcome(ref, flt)
        detects = ref["outcome"] == "passed" and flt["outcome"] == "failed"
        if detects:
            print("YAY FAULT DETECTED")

        per_test_ft[test] = int(triggers)
        per_test_fd[test] = int(detects)

    return per_test_ft, per_test_fd


class FaultAnalyzer:
    """
    Analyses a single fault against a single test file.

    1. Runs tests on reference and faulty code → FT/FD
    2. Measures line & branch coverage on faulty code
    3. Run mutation analysis on faulty code

    """

    def __init__(
        self,
        runner: TestRunner = None, #no optional because it depends on the benchmark (bcb needs env)
        builder: PythonProgramBuilder = None,
        coverage_runner: CoverageRunner = None, #both line and branch coverage
        mutation_runner: MutationRunner = None,
    ):
        self.builder = builder or PythonProgramBuilder()
        self.runner = runner
        self.coverage_runner  = coverage_runner or CoverageRunner(work_dir=WORK_DIR)
        self.mutation_runner  = mutation_runner or MutationRunner(work_dir=WORK_DIR)

    def assign_builder(self, builder : PythonProgramBuilder):
        self.builder = builder

    def assign_coverage_runner(self, coverage_runner: CoverageRunner):
        self.coverage_runner = coverage_runner

    def assign_mutation_runner(self, mutation_runner: MutationRunner):
        self.mutation_runner = mutation_runner
    def assign_runner(self, runner : TestRunner) :
        self.runner = runner
    def analyze(
        self,
        *,
        fault: dict,
        task: Task,
        test_file: str,
        test_model: str,
        mutants: list[MutantInfo],
        fault_model: str,
        benchmark: str,
    ) -> AnalysisRecord:
        task_id = task.task_id

        try:
            # build code under test for both versions
            faulty_cut = self.builder.build_program(task=task, code=fault["completion"])
            reference_cut  = self.builder.build_program(task=task, code=task.canonical_solution)

            # MBPP has a different structure of CUT
            if benchmark == 'mbpp':
                faulty_cut.content = faulty_cut.implementation
                reference_cut.content = reference_cut.implementation
                faulty_cut.syntactically_valid = is_syntactically_valid(faulty_cut.content)
                reference_cut.syntactically_valid = is_syntactically_valid(reference_cut.content)
                faulty_cut.task_id = str(faulty_cut.task_id)
                reference_cut.task_id = str(reference_cut.task_id)

            if not faulty_cut.syntactically_valid:
                logger.warning("[%s] faulty version has syntax error, skipping", task_id)
                return None
            if not reference_cut.syntactically_valid:
                logger.warning("[%s] reference version has syntax error, skipping", task_id)
                return None

            #build test suite
            #for bcb to surpress showing matplot lib figures
            test_file = "import matplotlib\nmatplotlib.use('Agg')\n" + test_file
            suite = self.builder.build_tests(task, test_suite=test_file)

            # trigger/detection
            ref_results   = run_tests_on_code(
                cut=reference_cut, task=task, tests=test_file,
                builder=self.builder, runner=self.runner,
            )
            fault_results = run_tests_on_code(
                faulty_cut, task=task, tests=test_file,
                builder=self.builder, runner=self.runner,
            )

            if not ref_results or not fault_results:
                return None

            per_test_ft, per_test_fd = _compute_ft_fd(ref_results, fault_results)

            cov_report = self.coverage_runner.run(cut=faulty_cut, suite=suite)

            # mutation testing
            mut_report = self.mutation_runner.run(
                cut=faulty_cut, suite=suite, mutants=mutants,
            )
            try :
                analysis_record =  AnalysisRecord(
                    task_id=task_id,
                    fault_model=fault_model,
                    benchmark=benchmark,
                    test_model=test_model,
                    per_test_ft=per_test_ft,
                    per_test_fd=per_test_fd,
                    line_coverage_record=CoverageRecord(
                        coverage_pct=len(cov_report.covered_lines)/cov_report.total_lines,
                        covered=cov_report.covered_lines,
                        missing=cov_report.missing_lines,
                        per_test_coverage=cov_report.per_test_line_coverage,
                    ),
                    branch_coverage_record=CoverageRecord(
                        coverage_pct=len(cov_report.covered_branches) / cov_report.total_branches if cov_report.total_branches else 1.0,
                        covered=cov_report.covered_branches,
                        missing=cov_report.missing_branches,
                        per_test_coverage=cov_report.per_test_branch_coverage,
                    ),
                    mutation_record=MutationRecord(
                        mutation_score=mut_report.mutation_score,
                        total_mutants=mut_report.total_mutants,
                        killed_mutants=mut_report.killed_mutants,
                        per_test_kills=mut_report.per_test_kills)
                )

                return analysis_record
            except Exception as e:
                logger.warning("[%s] could not run tests on code due to error: %s", task_id, e)
                return None

        except Exception as exc:
            logger.error(
                "[%s] test_model=%s error: %s", task_id, test_model, exc,
                exc_info=True,
            )
            return None