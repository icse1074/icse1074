import ast
import logging
from typing import Optional, List, Tuple, Dict, Union
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.execution_report import ExecutionReport, Verdict
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.test_generation.feedback_loop.feedback_loop import FeedbackLoop
from test_adequacy_study.test_generation.feedback_loop.error_fixer_feedback import ErrorFixerFeedback
from test_adequacy_study.generators.test_generator import TestGenerator
from test_adequacy_study.runners.test_runner import TestRunner
from test_adequacy_study.types.test_generation_type import TestGenerationType

logger = logging.getLogger(__name__)


class FeedbackRunner:
    def __init__(
        self,
        generator: TestGenerator,
        builder: ProgramBuilder,
        runner: TestRunner,
        refinement: Optional[FeedbackLoop] = None,
        max_iterations: int = 3,
    ):
        self.generator = generator
        self.builder = builder
        self.runner = runner
        self.error_loop = ErrorFixerFeedback()
        self.refinement = refinement
        self.max_iterations = max_iterations

    def run(self, task: Task, cut: CUT, input_tests: str = None, initial_prompt_variables: dict = None):

        # Stage 01: generate initial tests + fix errors OR use existing test cases
        if input_tests is None:
            tests, results, api_calls, history = self.generate_initial_tests(task, cut, initial_prompt_variables)
        else:
            logger.info("Using existing test cases instead of re-generating")
            tests, results, api_calls, history = self.load_existing_initial_tests(task, cut, input_tests, initial_prompt_variables)

        if self.refinement is None:
            return tests, results, api_calls, None

        # stage 02: coverage/mutation refinement loop
        # each iteration: refine → fix errors → run → check coverage
        for iteration in range(self.max_iterations):
            logger.info("[%s] CoverageRefinement — iteration %d", task.task_id, iteration + 1)

            # evaluate coverage/mutation on current tests
            suite = self.builder.build_tests(task, tests)
            feedback = self.refinement.evaluate(task, tests, results, cut=cut, suite=suite)

            if not feedback.retry:
                logger.info("[%s] refinement done after %d iteration(s)",
                            task.task_id, iteration + 1)
                break

            # refine tests based on coverage/mutation feedback, then fix any errors
            tests, results, n_calls, history = self._run_loop(
                loop=self.error_loop,
                task=task,
                cut=cut,
                history = history,
                initial_prompt_variables=feedback.prompt_variables,
            )
            api_calls += n_calls

        last_report = self.refinement.last_report

        return tests, results, api_calls, last_report
    #ONLY ERROR FIXER LOOP
    def _run_loop(
            self,
            loop: FeedbackLoop,
            task: Task,
            cut: CUT,
            history: List[Dict],
            initial_prompt_variables: dict,
    ) -> Tuple[Union[str, None], Union[ExecutionReport, None], int, Union[list, None]]:

        prompt_variables = initial_prompt_variables
        tests = None
        run_result = None
        api_calls = 0

        for iteration in range(self.max_iterations):
            logger.info("[%s] %s — iteration %d", task.task_id, type(loop).__name__, iteration + 1)
            try :
                test_suites, history = self.generator.generate(
                    prompt_variables=prompt_variables,
                    samples=1,
                    history=history,
                )
            except Exception as e :
                if "431" in str(e) or "headers too large" in str(e).lower():
                    logger.warning("[%s] Headers too large, truncating history and retrying", task.task_id)
                    history = self._truncate_history(history, keep_last_n_turns=1)
                    try:
                        test_suites, history = self.generator.generate(
                            prompt_variables=prompt_variables,
                            samples=1,
                            history=history,
                        )
                    except Exception as e2:
                        if "431" in str(e2) or "headers too large" in str(e2).lower():
                            logger.warning("[%s] Still too large after truncation, skipping", task.task_id)
                            return tests, run_result, api_calls, history
                        raise
                else:
                    raise
            api_calls += 1

            if not test_suites:
                logger.error("[%s] No test suites generated, stopping", task.task_id)
                break

            tests = test_suites[0]

            #we need to check if just one method -> we need to put in original test suite and run the test method

            try:
                ast.parse(tests)
            except SyntaxError as e:
                logger.warning("[%s] Syntax error in generated tests: %s", task.task_id, e)
                prompt_variables = {"errors": str(e), "tests": tests}
                continue


            one_test_method, tests = self._ensure_full_test_suite(task.task_id, tests, task.tests)
            suite = self.builder.build_tests(task, tests)
            run_result = self.runner.run(cut=cut, suite=suite)

            if task.tests.get("test_node") :
                feedback = loop.evaluate(task, tests, run_result, test_id=task.tests['test_node'])
            else :
                feedback = loop.evaluate(task, tests, run_result)
            if not feedback.retry:
                logger.info("[%s] %s — done after %d iteration(s)",
                            task.task_id, type(loop).__name__, iteration + 1)
                break
            else :
                pass
            prompt_variables = feedback.prompt_variables
        #in case of oracle completion we only return one test
        if one_test_method :
            return one_test_method, run_result, api_calls, history
        return tests, run_result, api_calls, history

    def generate_initial_tests(self, task: Task, cut: CUT, initial_prompt_variables: dict = None) -> tuple[Union[str,None], Union[ExecutionReport,None], int, Union[list[dict], None]]:
        """
        Uses the model to generate tests for the given CUT, for the first time (meaning, first query)

        :param task:
        :param cut:
        :return:
        """
        history = None

        if initial_prompt_variables is None:
            initial_prompt_variables = {}


        initial_prompt_variables["code"] = cut.content

        if self.refinement is not None:
            initial_prompt_variables = self.refinement.augment_initial_prompt_variables(
                initial_prompt_variables=initial_prompt_variables,
                task_id=task.task_id
            )
        return self._run_loop(
            loop=self.error_loop,
            task=task,
            cut=cut,
            history=history,
            initial_prompt_variables=initial_prompt_variables,
        )

    def load_existing_initial_tests(self, task: Task, cut: CUT, initial_tests: str, initial_prompt_variables: dict = None) -> tuple[Union[str, None], Union[ExecutionReport, None], int, Union[list[dict],None]]:
        """
        Simulates a model interaction. Uses the provided initial_tests as the model's response
        No actual model invocations take place

        :param task:
        :param cut:
        :param initial_tests:
        :return:
        """
        if initial_prompt_variables is None:
            initial_prompt_variables = {}


        initial_prompt_variables["code"] = cut.content

        #todo : check if this still works after adding mutants
        history = self.generator.get_history_with_hardcoded_response(
            prompt_variables=initial_prompt_variables,
            response=initial_tests
        )

        run_result = ExecutionReport(
            task_id=task.task_id,
            verdict=Verdict.PASSED,
        )
        return initial_tests, run_result, 1, history

    def _inject_method_into_class(self,
              complete_test_suite: str,
              generated_method: str,
              start_line: int,
              end_line: int):
        """
        Replace the method originally at [start_line, end_line] (1-indexed,
        inclusive) in complete_test_suite with generated_method, re-indented
        to match the original method's indentation level.
        """
        suite_lines = complete_test_suite.split('\n')
        # Determine original indent from the method's first line (the `def` line)
        original_first_line = suite_lines[start_line - 1]
        base_indent = original_first_line[: len(original_first_line) - len(original_first_line.lstrip())]

        # generated_method may come back unindented (just `def test_x(self): ...`
        # at column 0) -- re-indent every line to base_indent, preserving any
        gen_lines = generated_method.split('\n')
        gen_first_indent = len(gen_lines[0]) - len(gen_lines[0].lstrip())

        reindented = []
        for line in gen_lines:
            if not line.strip():
                reindented.append('')
                continue
            current_indent = len(line) - len(line.lstrip())
            relative = current_indent - gen_first_indent
            reindented.append(' ' * (len(base_indent) + relative) + line.lstrip())

        #suite_lines = suite_lines[:start_line - 1] + reindented + suite_lines[end_line:]

        #handles size mismatch
        suite_lines[start_line - 1:end_line] = reindented
        return '\n'.join(suite_lines)

    def _ensure_full_test_suite(self, task_id, generated_tests: str, test_suite : dict) -> str:
        """
        If `tests` is just test method (not a full class/module),
        wrap it into the original class skeleton so it can be built and run.
        """
        try:
            tree = ast.parse(generated_tests)
        except SyntaxError:
            return generated_tests  # let the existing syntax-error retry path handle it

        has_class = any(isinstance(n, ast.ClassDef) for n in tree.body)
        if has_class :
            return None, generated_tests  # already a full suite

        # inject just this method, keeping everything else (imports, other tests) intact.
        if self.generator.generation_type in [TestGenerationType.ORACLE_COMPLETION, TestGenerationType.ASSERTION_GENERATION]:
            new_suite =  self._inject_method_into_class(
                complete_test_suite=test_suite['complete_test_suite'],
                generated_method=generated_tests,
                start_line=test_suite['start_line'],
                end_line=test_suite['end_line'],
            )
            #check if ast parses it correctly, if not we throw error that propagates so I can read it in logs
            try:
                ast.parse(new_suite)
                logger.info(
                    "[%s] Method injected correctly into suite (node=%s)",
                    task_id, test_suite.get('test_node'),
                )
            except SyntaxError as e:
                logger.error(
                    "[%s] Injected suite failed to parse (node=%s): %s",
                    task_id, test_suite.get('test_node'), e,
                )
                raise #will be caught in pipeline_for_oracle_completion

            return generated_tests, new_suite

        return None, generated_tests