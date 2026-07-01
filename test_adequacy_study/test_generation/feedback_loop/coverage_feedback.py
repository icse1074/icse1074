from __future__ import annotations
from typing import Optional
import logging

from test_adequacy_study.data_models.coverage_report import CoverageReport
from test_adequacy_study.helpers.coverage_messages import get_missing_line_messages, get_missing_branch_messages
from test_adequacy_study.runners.coverage_runner import CoverageRunner
from test_adequacy_study.test_generation.feedback_loop.feedback_loop import FeedbackLoop, FeedbackResult
from test_adequacy_study.types.refinement_mode import RefinementMode

logger = logging.getLogger(__name__)


class CoverageFeedback(FeedbackLoop):

    def __init__(self, coverage_runner: CoverageRunner, refinement: RefinementMode):
        self.coverage_runner = coverage_runner
        self.refinement = refinement

    def augment_initial_prompt_variables(self, initial_prompt_variables, task_id) -> dict:
        return {**initial_prompt_variables, "coverage": self.refinement}

    def evaluate(self, task, tests, run_result, cut=None, suite=None, test_id = None, **kwargs) -> FeedbackResult:
        #TODO : lacks implementation of evaluating based on one test only
        from test_adequacy_study.runners.test_runner import Verdict
        if run_result.verdict in [Verdict.ERROR, Verdict.SYNTAX_ERROR]:
            return FeedbackResult(retry=False, prompt_variables={})

        coverage_report: CoverageReport = self.coverage_runner.run(cut=cut, suite=suite)
        self.last_report = coverage_report

        logger.info("[%s] %s", task.task_id, coverage_report)

        if self.refinement == RefinementMode.LINE_COVERAGE:
            if coverage_report.has_full_line_coverage:
                logger.info("[%s] 100%% line coverage reached", task.task_id)
                return FeedbackResult(retry=False, prompt_variables={})

            missing_lines_with_code = get_missing_line_messages(cut, coverage_report)
            return FeedbackResult(
                retry=True,
                prompt_variables={
                    "tests": tests,
                    "coverage_pct": f"{coverage_report.coverage_pct:.1f}",
                    "missing_lines": missing_lines_with_code,
                },
            )
        elif self.refinement == RefinementMode.BRANCH_COVERAGE:
            if coverage_report.has_full_branch_coverage:
                logger.info("[%s] 100%% branch coverage reached", task.task_id)
                return FeedbackResult(retry=False, prompt_variables={})

            missing_branches_with_code = get_missing_branch_messages(cut, coverage_report)
            return FeedbackResult(
                retry=True,
                prompt_variables={
                    "tests": tests,
                    "nr_missing_branches": f"{len(coverage_report.missing_branches)}",
                    "missing_branches": missing_branches_with_code,
                },
            )
        else:
            raise Exception("CoverageFeedback does not support the following refinement: ", str(self.refinement))