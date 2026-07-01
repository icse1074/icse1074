import json
import logging
from collections import defaultdict
from typing import List, Optional, Dict, Any

from tqdm import tqdm

from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.builders.mbpp_python_program_builder import MbppPythonProgramBuilder
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.file_utils import write_jsonl, read_jsonl
from test_adequacy_study.generators.test_generator import TestGenerator
from test_adequacy_study.providers.program_builder_provider import ProgramBuilderProvider
from test_adequacy_study.runners.test_runner import TestRunner
from test_adequacy_study.test_generation.feedback_loop.feedback_runner import FeedbackRunner
from test_adequacy_study.test_generation.feedback_loop.error_fixer_feedback import ErrorFixerFeedback
from test_adequacy_study.test_generation.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


class OracleCompletionPipeline:
    """
    Oracle completion pipeline: completes missing oracle in tests.
    two modes :
        1. prefix-only : llm is provided with the test prefix alone and it generates assertions
        2. masked-oracle : llm is provided with the test with oracles being masked
        -> in both cases we have already removed indications to oracle/comments etc to avoid contamination

    Input:
    - Benchmark + variation → loads tasks with NL prompts
    - extracted_tests_file → JSONL with test_node, test_parts.prefix, test_with_masked_assertions

    Process:
    - For each task and its extracted tests
    - For each test, choose mode (prefix-only or masked-assertions)
    - Run error-fixing feedback loop to complete the oracle
    - Output: original record + "completed_test" field
    """

    def __init__(
            self,
            loader: BenchmarkLoader,
            runner: TestRunner,
            generator: TestGenerator,
            builder: ProgramBuilder,
            extracted_tests_file: str,
            mode : str,
            refinement_iterations: int = 5,
            config: PipelineConfig = None,
    ):
        self.loader = loader
        self.runner = runner
        self.generator = generator
        self.builder = builder
        self.extracted_tests_file = extracted_tests_file
        self.use_masked_mode = True if mode == "masked-oracle" else False
        self.config = config or PipelineConfig()

        self.feedback_runner = FeedbackRunner(
            generator=generator,
            builder=builder,
            runner=runner,
            refinement=None,  # simple error fixer loop
            max_iterations=refinement_iterations,
        )

        # Load extracted tests file into memory: {task_id: [test_record, ...]}
        self.extracted_tests = self._load_extracted_tests()

    def _load_extracted_tests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load JSONL file and group by task_id"""
        extracted_tests = defaultdict(dict)
        try:
            tasks = read_jsonl(self.extracted_tests_file)
            for task in tasks:
                if task["task_id"] not in extracted_tests:
                    extracted_tests[task["task_id"]] = task
            logger.info("Loaded %d tasks with extracted tests", len(extracted_tests))
        except FileNotFoundError:
            logger.error("Extracted tests file not found: %s", self.extracted_tests_file)
            raise

        return extracted_tests

    def run(self) -> List[Dict[str, Any]]:
        """
        Main execution loop: for each task, complete oracles for all its tests
        """
        responses = []

        logger.info("Oracle completion pipeline starting")
        all_tasks = list(self.loader.load())

        # Apply config filters
        if self.config.start_task_id is not None:
            ids = [t.task_id for t in all_tasks]
            if self.config.start_task_id in ids:
                start_idx = ids.index(self.config.start_task_id)
                all_tasks = all_tasks[start_idx:]

        if self.config.slice is not None and self.config.slice.strip() != "":
            idx, total = self.config.slice.split("/")
            idx, total = int(idx), int(total)
            all_tasks = [t for i, t in enumerate(all_tasks) if i % total == idx]


        for task in tqdm(all_tasks, total=len(all_tasks)):
            '''if task.task_id in finished_tasks :
                logger.info("task {} already finished locally -- skipping ".format(task.task_id))
                continue'''
            if self.config.tasks and task.task_id not in self.config.tasks:
                continue

            # Get extracted tests for this task
            if str(task.task_id) not in self.extracted_tests:
                logger.warning("[%s] No extracted tests found, skipping", task.task_id)
                continue

            task_tests = self.extracted_tests[str(task.task_id)]

            # Build code under test

            if isinstance(self.builder, MbppPythonProgramBuilder):
                cut = self.builder.build_program(task=task, code=task.canonical_solution)
            else:
                cut = self.builder.build_program(task=task, code=task.canonical_solution)
                if not cut.syntactically_valid:
                    logger.warning("[%s] has syntax error, skipping", task.task_id)
                    continue

            # Complete oracles for each test in this task
            for test_record in task_tests["processed_tests"]:
                try:
                    completed_record = self.complete_test_oracle(
                        task=task,
                        cut=cut,
                        test_record=test_record,
                        test_suite = task_tests["test_suite"]
                    )
                    if completed_record:
                        responses.append(completed_record)
                        write_jsonl(self.config.output_file, [completed_record], append=True)

                except Exception as e:
                    logger.error(
                        "[%s] Error completing oracle for test %s: %s",
                        task.task_id,
                        test_record.get("test_node"),
                        e
                    )

        logger.info("Oracle completion pipeline finished. %d tests completed", len(responses))
        return responses

    def complete_test_oracle(
            self,
            task,
            cut: CUT,
            test_record: Dict[str, Any],
            test_suite : str,
    ) -> Optional[Dict[str, Any]]:
        """
        Complete oracle values for a single test via error-fixing feedback loop

        Returns: test_record with added "completed_test" field, or None if failed
        """

        #assign generated suite to task
        task.tests = test_record
        task.tests["complete_test_suite"] = test_suite
        test_node = test_record.get("test_node")
        logger.info("[%s] Completing oracle for %s", task.task_id, test_node)

        # Choose mode: prefix-only or masked-assertions
        if self.use_masked_mode:
            test_to_complete = test_record.get("test_with_masked_assertions")
            mode = "masked-oracle"
        else:
            # Extract test prefix from test_parts
            test_parts = test_record.get("test_parts", {})
            test_to_complete = test_parts.get("prefix")
            mode = "prefix-only"

        if not test_to_complete:
            logger.warning("[%s] No test content for mode %s", task.task_id, mode)
            return None

        # Build initial prompt variables for LLM
        initial_prompt_variables = self._build_prompt_variables(
            task=task,
            test_to_complete=test_to_complete,
            mode=mode,
        )

        # Run error-fixing feedback loop
        try:
            completed_test, run_result, api_calls, history = self.feedback_runner.run(
                task=task,
                cut=cut,
                initial_prompt_variables=initial_prompt_variables,
            )
        except Exception as e:
            logger.error("[%s] Feedback loop failed: %s", task.task_id, e)
            return None

        if completed_test is None:
            logger.warning("[%s] No completed test generated", task.task_id)
            return None

        # Add completed test to record and return
        test_record["completed_test"] = completed_test
        test_record["completion_mode"] = mode
        test_record["api_calls"] = api_calls

        return test_record

    def _build_prompt_variables(
            self,
            task,
            test_to_complete: str,
            mode: str,
    ) -> Dict[str, str]:
        """
        Build prompt variables for LLM based on mode

        Mode "prefix-o": LLM writes entire assertions from scratch
        Mode "masked": LLM fills in <MASK> values
        """
        prompt_variables = {
            "prompt": task.stub or "",
        }

        if mode == "masked-oracle":
            prompt_variables["test_with_masks"] = test_to_complete
        else:  # prefix mode
            prompt_variables["test_prefix"] = test_to_complete

        return prompt_variables