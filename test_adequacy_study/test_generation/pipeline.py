import ast
import logging
from dataclasses import asdict
from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.file_utils import write_jsonl
from test_adequacy_study.generators.code_generator import CodeGenerator
from test_adequacy_study.runners.test_runner import TestRunner
from tqdm import tqdm
from test_adequacy_study.helpers.parsers import align_code_to_list, is_syntactically_valid
from test_adequacy_study.types.fault_collection_result import FaultCollectionResult
from test_adequacy_study.types.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class FaultCollectionPipeline:
    """
    Orchestrates the full fault collection loop:

        BenchmarkLoader → CodeGenerator → ProgramBuilder → TestRunner → faults

    The pipeline is fully agnostic of benchmark format, test framework,
    and model backend. All format-specific logic lives in the injected
    components.
    """

    def __init__(
            self,
            loader: BenchmarkLoader,
            runner: TestRunner,
            generator: CodeGenerator,
            builder: ProgramBuilder,
            config: PipelineConfig = None,
            target : str = "faults" #either collect generations (generations) or additionally run tests to collect faults using benchmark tests (faults)
    ):
        self.loader = loader
        self.runner = runner
        self.generator = generator
        self.builder = builder
        self.config = config or PipelineConfig()
        self.target = target

    def run(self) -> FaultCollectionResult:
        faults_list = FaultCollectionResult()
        all_tasks = list(self.loader.load())

        generations_file = self.config.output_file #saves all the generations

        #option to start execution from a certain task
        if self.config.start_task_id is not None:
            all_task_ids = [t.task_id for t in all_tasks]
            if self.config.start_task_id in all_task_ids:
                start_idx = all_task_ids.index(self.config.start_task_id)
                all_tasks = all_tasks[start_idx:]

        #option to only run a part of the benchmark
        if self.config.slice is not None and self.config.slice.strip() != "":
            idx, total = self.config.slice.split("/")
            idx, total = int(idx), int(total)
            all_tasks = [t for i, t in enumerate(all_tasks) if i % total == idx]

        total_number_of_tasks = len(all_tasks) - (self.config.skip_n_tasks or 0)

        for idx, task in enumerate(tqdm(all_tasks, total=total_number_of_tasks)):

            # Exclude n tasks (if specified under configuration)
            if self.config.skip_n_tasks and idx < self.config.skip_n_tasks:
                continue

            # Exclude tasks outside the ones specified (if any, otherwise iterate all)
            if self.config.tasks and task.task_id not in self.config.tasks:
                continue

            if self.config.exclude_ids is not None and task.task_id in self.config.exclude_ids:
                logger.info(f"Skipping task {task.task_id} because it is part of the exclude list")
                continue

            faults_list.total_tasks += 1
            logger.info("[%s] Generating %d completions", task.task_id, self.config.n_generations)

            # Step 1: build test suite
            test_suite = self.builder.build_tests(task)

            # Step 2: generate n completions
            raw_codes, history = self.generator.generate(
                prompt_variables={"prompt": task.stub},
                samples=self.config.n_generations)

            faults_list.total_generated += len(raw_codes)


            # Step 3: Loop through generated code (above n completions)
            for i, code in enumerate(raw_codes):
                if not code.strip():
                    logger.debug("[%s] completion %d is empty, skipping", task.task_id, i)
                    continue

                # Rename generated code to match the function name of the assertions
                if self.config.match_function_names is True:
                    code = align_code_to_list(code, test_suite.source['test_inputs'])

                #save generations to file
                generation_dict = {
                    "task_id": task.task_id,
                    "completion": code,
                    "completion_index": i,
                }
                write_jsonl(generations_file, [generation_dict], append=True)

                if self.target == "generations":
                    continue
                # Build code under test
                cut = self.builder.build_program(
                    task=task,
                    code=code)

                if not is_syntactically_valid(code):
                    logger.debug("[%s] completion %d has syntax error, skipping", task.task_id, i)
                    continue

                faults_list.total_run += 1


                # Run tests. In case of failures, save the generated code in the output_file specified in config
                try:
                    # MBPP compatibility
                    cut.syntactically_valid = cut.syntactically_valid or is_syntactically_valid(cut.implementation)

                    run_result = self.runner.run(
                        cut=cut,
                        suite=test_suite)
                    logger.debug("[%s] completion %d → %s", task.task_id, i, run_result.verdict)
                    if run_result.verdict != Verdict.PASSED :
                        print("check me")
                        print(run_result.verdict)
                    # Step 5: collect faults only
                    if run_result.verdict == Verdict.FAILED:
                        # parse the failed assertions -> parsing done in test runner
                        #failures = self._parse_failures(run_result.stdout)

                        if run_result.detailed_test_results and isinstance(run_result.detailed_test_results[0], str):
                            #HE and MBPP
                            failures = run_result.detailed_test_results
                        elif run_result.detailed_test_results:
                            failures = [asdict(t) for t in run_result.detailed_test_results if t.outcome == "failed"]
                        else:
                            failures = self._parse_failures(run_result.stdout)
                        fault_dict = {
                            "task_id": task.task_id,
                            "completion": code,
                            "completion_index": i,
                            "failures": failures,
                        }
                        faults_list.faults.append(fault_dict)
                        write_jsonl(self.config.output_file, [fault_dict], append=True)
                except Exception as e:
                    logging.error("An exception was thrown when running tests for task {}: {}".format(task.task_id, e))

        logger.info("Pipeline complete. %s", faults_list.summary())
        return faults_list

    def _parse_failures(self, stdout: str) -> list[dict]:
        results = []
        for line in stdout.splitlines():
            line = line.strip()
            if line:
                results.append(ast.literal_eval(line))
        return results
