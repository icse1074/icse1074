from __future__ import annotations
from typing import Optional
import logging

from test_adequacy_study.data_models.mutation_report import MutationReport, MutantInfo
from test_adequacy_study.helpers.coverage_messages import get_missing_line_messages, get_missing_branch_messages
from test_adequacy_study.runners.mutation_runner import MutationRunner
from test_adequacy_study.test_generation.feedback_loop.feedback_loop import FeedbackLoop, FeedbackResult
from test_adequacy_study.types.refinement_mode import RefinementMode

logger = logging.getLogger(__name__)


class MutationFeedback(FeedbackLoop):

    def __init__(self, mutation_runner: MutationRunner, refinement: RefinementMode, mutants):
        self.mutation_runner = mutation_runner
        self.refinement = refinement
        if not mutants :
            raise Exception('Mutation Guided Test Generation requires mutants')
        self.mutants = []
        for mutant in mutants:
            self.mutants.append(MutantInfo(
                mutant_id=mutant["mutant_id"],
                operator=mutant["operator"],
                line=mutant["line"],
                original=mutant["original"],
                mutated=mutant["mutated"],
                task_id=mutant["task_id"],
            ))

    def _format_mutant(self, mutant : MutantInfo):
        return (
            f"Mutant #{mutant.mutant_id} [{mutant.operator}] at line {mutant.line}:\n"
            f"  Original: {mutant.original}\n"
            f"  Mutated:  {mutant.mutated}"
        )
    def augment_initial_prompt_variables(self, initial_prompt_variables, task_id) -> dict:
        if not self.mutants:
            return {}
        task_mutants = [mutant for mutant in self.mutants if mutant.task_id == task_id]
        initial_mutants_variable = "\n\n".join(self._format_mutant(m) for m in task_mutants)
        return {**initial_prompt_variables, "mutants": initial_mutants_variable}


    def evaluate(self, task, tests, run_result, cut=None, suite=None, test_id = None, **kwargs) -> FeedbackResult:
        #TODO : lacks implementation of evaluating based on one test only
        from test_adequacy_study.runners.test_runner import Verdict
        if run_result.verdict in [Verdict.ERROR, Verdict.SYNTAX_ERROR]:
            return FeedbackResult(retry=False, prompt_variables={})



        mutation_report: MutationReport = self.mutation_runner.run(
            cut=cut,
            suite=suite,
            mutants=[mutant for mutant in self.mutants if mutant.task_id == task.task_id],
        )
        self.last_report = mutation_report

        logger.info("[%s] %s", task.task_id, mutation_report)

        if mutation_report.max_mutation_score_reached:
            logger.info("[%s] 100%% mutation score reached", task.task_id)
            return FeedbackResult(retry=False, prompt_variables={})

        survived_ids = set(mutation_report.total_mutants) - set(mutation_report.killed_mutants) - set(mutation_report.incompetent_mutants)
        surviving_mutants_text = "\n\n".join(
            self._format_mutant(mutation_report.per_mutant[mid])
            for mid in sorted(survived_ids)
            if mid in mutation_report.per_mutant
        )

        return FeedbackResult(
            retry=True,
            prompt_variables={
                "tests": tests,
                "mutation_score": f"{mutation_report.mutation_score:}",
                "n_surviving": str(len(survived_ids)),
                "surviving_mutants": surviving_mutants_text,
            },

        )
