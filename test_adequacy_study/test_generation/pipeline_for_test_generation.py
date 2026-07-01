import logging
import os
from typing import List, Optional

from tqdm import tqdm

from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.TestGenerationResponse import TestGenerationResponse
from test_adequacy_study.file_utils import write_jsonl
from test_adequacy_study.generators.test_generator import TestGenerator
from test_adequacy_study.runners.coverage_runner import CoverageRunner
from test_adequacy_study.runners.mutation_runner import MutationRunner
from test_adequacy_study.runners.test_runner import TestRunner
from test_adequacy_study.test_generation.feedback_loop.coverage_feedback import CoverageFeedback
from test_adequacy_study.test_generation.feedback_loop.feedback_loop import FeedbackLoop
from test_adequacy_study.test_generation.feedback_loop.feedback_runner import FeedbackRunner
from test_adequacy_study.test_generation.feedback_loop.mutation_feedback import MutationFeedback
from test_adequacy_study.test_generation.pipeline import PipelineConfig
from test_adequacy_study.types.prompt_generation_type import PromptGenerationType
from test_adequacy_study.types.prompt_input_holder import PromptInputDict
from test_adequacy_study.types.refinement_mode import RefinementMode
from test_adequacy_study.types.test_generation_type import TestGenerationType

logger = logging.getLogger(__name__)


class TestGenerationPipeline:
    """
    Orchestrates the full test generation loop for code under test:

    BenchmarkLoader → TestGenerator → ProgramBuilder → TestRunner → faults

    The pipeline is fully agnostic of benchmark format, test framework,
    and model backend. All format-specific logic lives in the injected
    components.

    Refinement modes:
      - NONE     : generate tests + error fix only
      - COVERAGE : generate tests + error fix + coverage refinement
    """

    def __init__(
            self,
            loader: BenchmarkLoader,
            runner: TestRunner,
            generator: TestGenerator,
            builder: ProgramBuilder,
            mode: RefinementMode = RefinementMode.NONE,
            fault: bool = True,
            refinement_iterations: int = 5,
            config: PipelineConfig = None,
            input_tests_per_id: dict[str, dict[str]] = None,
            input_mutants : list[str] = None,
            prompt_inputs: PromptInputDict = None,
    ):
        self.loader = loader
        self.builder = builder
        self.fault = fault
        self.config = config or PipelineConfig()
        self.input_tests_per_id = input_tests_per_id
        self.input_mutants = input_mutants
        refinement = self._build_refinement(mode, runner, input_mutants)
        self.prompt_inputs = prompt_inputs

        self.feedback_runner = FeedbackRunner(
            generator=generator,
            builder=builder,
            runner=runner,
            refinement=refinement,
            max_iterations=refinement_iterations,
        )

    def _build_refinement(self, mode: RefinementMode, runner: TestRunner, input_mutants : list[dict] = None) -> Optional[FeedbackLoop]:
        if mode == RefinementMode.NONE:
            return None
        elif mode == RefinementMode.LINE_COVERAGE or mode == RefinementMode.BRANCH_COVERAGE:
            coverage_runner = CoverageRunner(timeout=runner.timeout)
            return CoverageFeedback(coverage_runner, mode)
        elif mode == RefinementMode.MUTATION_COVERAGE :
            mutation_runner = MutationRunner(timeout=runner.timeout)
            return MutationFeedback(mutation_runner, mode, mutants=input_mutants)
        else:
            raise ValueError(f"Unknown refinement mode: {mode}")



    def run_task(self, task) -> Optional[TestGenerationResponse]:
        code_under_test = task.generated_solution if self.fault else task.canonical_solution
        cut = self.builder.build_program(task=task, code=code_under_test)

        if not cut.syntactically_valid:
            logger.debug("[%s] has syntax error, skipping", task.task_id)
            return None

        # FIXME: In case a task has no tests, this will throw an Exception
        tests = None if self.input_tests_per_id is None else self.input_tests_per_id[task.task_id]['response']

        initial_prompt_variables = self.__get_initial_input_variables(str(task.task_id), cut)
        test_suite, results, n_api_calls, report = self.feedback_runner.run(task, cut, tests, initial_prompt_variables)
        return TestGenerationResponse(
            task_id=task.task_id,
            model_id=self.feedback_runner.generator.model_id,
            code_under_test=code_under_test,
            response=test_suite,
            execution_report=results,
            api_calls=n_api_calls,
            refinement_report=report,
        )

    def run(self) -> List[TestGenerationResponse]:
        responses = []

        logger.info("Pipeline executing for [%d] records", len(self.loader))
        all_tasks = list(self.loader.load())
        #option to start execution from a certain task
        if self.config.start_task_id is not None:
            ids = [t.task_id for t in all_tasks]
            if self.config.start_task_id in ids:
                start_idx = ids.index(self.config.start_task_id)
                all_tasks = all_tasks[start_idx:]

        #option to only run a part of the benchmark
        if self.config.slice is not None and self.config.slice.strip() != "":
            idx, total = self.config.slice.split("/")
            idx, total = int(idx), int(total)
            all_tasks = [t for i, t in enumerate(all_tasks) if i % total == idx]

        for task in tqdm(all_tasks, total=len(all_tasks)):
            if self.config.tasks and task.task_id not in self.config.tasks:
                continue

            # Execute this task X times in case of failure
            test_generation_response = None
            for repetition in range(0, int(os.getenv("REPEAT_FAILED_REQUEST_ITERATIONS", 1))):
                try:
                    test_generation_response = self.run_task(task)
                    break
                except Exception as e:
                    logger.error("An error has been thrown when generating tests for [%s] on iteration [%d]", task.task_id, repetition)
                    test_generation_response = None

            if test_generation_response is None:
                continue

            responses.append(test_generation_response)
            write_jsonl(self.config.output_file, [test_generation_response], append=True)

        logger.info("Pipeline complete. %d test suites generated", len(responses))

        return responses

    def __get_initial_input_variables(self, task_id: str, cut) -> dict[str, str]:
        """
        The function is used to add initial input variables to the test generation feedback runner

        At the moment is used to append the prompts to the test generation process

        :return:
        """
        if self.prompt_inputs is None:
            return {
                'code': cut.content
            }

        if self.feedback_runner.generator.generation_type == TestGenerationType.FROM_ORIGINAL_PROMPT:
            return {
                "prompt": self.prompt_inputs.holders.get(task_id).original_prompt,
            }
        elif self.feedback_runner.generator.generation_type == TestGenerationType.FROM_DIFF_PROMPT:
            return {
                "prompt": self.prompt_inputs.holders.get(task_id).diff_or_combined_prompt,
            }
        elif self.feedback_runner.generator.generation_type == TestGenerationType.PROMPTS_DELTA:
            return {
                "prompt1": self.prompt_inputs.holders.get(task_id).original_prompt,
                "prompt2": self.prompt_inputs.holders.get(task_id).round_trip_prompt,
            }
        elif self.feedback_runner.generator.generation_type == TestGenerationType.FROM_PROMPT_AND_CODE:
            return {
                "prompt": self.prompt_inputs.holders.get(task_id).original_prompt,
                'code': cut.content
            }



