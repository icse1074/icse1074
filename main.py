import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from test_adequacy_study.benchmarks.generated_code_loader import GeneratedCodeLoader
from test_adequacy_study.benchmarks.humaneval_with_pytests_loader import HumanEvalWithPytestsLoader
from test_adequacy_study.benchmarks.mbpp_with_pytests_loader import MBPPWithPytestsLoader
from test_adequacy_study.builders.mbpp_python_program_builder import MbppPythonProgramBuilder
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.file_utils import get_output_filepath, read_jsonl
from test_adequacy_study.generators.code_generator import CodeGenerator
from test_adequacy_study.generators.test_generator import TestGenerator
from test_adequacy_study.providers.loader_provider import LoaderProvider
from test_adequacy_study.providers.program_builder_provider import ProgramBuilderProvider
from test_adequacy_study.providers.runner_provider import RunnerProvider
from test_adequacy_study.runners.mutation_runner import MutationRunner
from test_adequacy_study.test_generation.ground_truth_test_augmentation import TestAugmentationPipeline
from test_adequacy_study.test_generation.pipeline import FaultCollectionPipeline
from test_adequacy_study.test_generation.pipeline_for_oracle_completion import OracleCompletionPipeline
from test_adequacy_study.test_generation.pipeline_for_test_generation import TestGenerationPipeline
from test_adequacy_study.test_generation.pipeline_for_test_generation import RefinementMode
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation
from test_adequacy_study.types.pipeline_config import PipelineConfig
from test_adequacy_study.types.prompt_input_holder import PromptInputDict
from test_adequacy_study.types.test_generation_type import TestGenerationType

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.set_int_max_str_digits(100000000)


def execute_fault_collection(model_name: str, config: PipelineConfig, benchmark_name: str, variation: BenchmarkVariation, target : str = "faults"):
    """
    Executes the Fault Collection Pipeline that will use the HumanEval benchmark to generate code.
    The Fault Collection Pipeline will execute the HumanEval tests and collect the generated codes that are faulty

    :param config:
    :param model_name:
    :return:
    """
    loader = LoaderProvider.get(benchmark_name, variation)
    runner = RunnerProvider.get(benchmark_name, os.getenv("WORK_DIR"))

    # Perhaps this can become universal
    if benchmark_name == 'mbpp':
        config.match_function_names = True

    pipeline = FaultCollectionPipeline(
        loader=loader,
        generator=CodeGenerator(model=model_name),
        runner=runner,
        builder=PythonProgramBuilder(),
        config=config,
        target = target,
    )

    result = pipeline.run()
    print(result.summary())



def execute_test_generation_on_faults(
        model_name: str,
        config: PipelineConfig,
        input_file: str,
        benchmark_name :str,
        mode: RefinementMode = RefinementMode.NONE,
        input_tests = None,
        input_mutants = None,
        work_dir: str = None,
        prompt_inputs: PromptInputDict = None,
):
    if input_tests is not None:

        # Map input tests so they are accessible by their task_id
        input_tests = {item["task_id"]: item for item in input_tests}

    loader = GeneratedCodeLoader(input_file, task_id_prefix="")
    runner = RunnerProvider.get_for_test_generation(benchmark_name, work_dir or os.getenv("WORK_DIR"))
    runner.timeout = 180
    pipeline = TestGenerationPipeline(
        loader=loader,
        generator=TestGenerator(model=model_name, temperature=0.1, refinement_mode=mode),
        runner=runner,
        builder=PythonProgramBuilder(),
        mode=mode,
        fault=True,
        refinement_iterations=5,
        config=config,
        input_tests_per_id=input_tests,
        input_mutants=input_mutants,
        prompt_inputs=prompt_inputs
    )

    results = pipeline.run()

def execute_mutant_generation(input_file: str, config: PipelineConfig):
    loader = GeneratedCodeLoader(input_file, task_id_prefix="")
    builder = PythonProgramBuilder()
    mutation_runner = MutationRunner()

    for task in tqdm(loader.load(), total=len(loader)):
        cut = builder.build_program(task=task, code=task.generated_solution)

        mutation_runner.generate(
            cut=cut,
            save_path=Path(config.output_file)
        )


def execute_benchmark_tests_augmentation(
    model_name: str,
    config: PipelineConfig,
    benchmark_name: str,
    benchmark_variation: BenchmarkVariation,
    batch_id: str =None,
    generations_path : str = None,
):
    tests_runner    = RunnerProvider.get_for_test_generation(benchmark_name, os.getenv("WORK_DIR"))
    tests_runner.timeout = 180
    generator = TestGenerator(model=model_name, generation_type=TestGenerationType.AUGMENTATION)

    if benchmark_name == 'mbpp':
        builder = MbppPythonProgramBuilder()
        loader = MBPPWithPytestsLoader(benchmark_variation)
    elif benchmark_name == 'he':
        loader = HumanEvalWithPytestsLoader(benchmark_variation)
        builder = PythonProgramBuilder()
    else:
        builder = PythonProgramBuilder()
        loader = LoaderProvider.get(benchmark_name, benchmark_variation)

    pipeline = TestAugmentationPipeline(
        loader=loader,
        tests_runner=tests_runner,
        generator=generator,
        builder=builder,
        batch_job_id=batch_id,
        generations_path = generations_path,
        config=config,
    )
    pipeline.run()

def execute_oracle_completion(
        model_name: str,
        config: PipelineConfig,
        input_file: str,
        benchmark_name :str,
        benchmark_variation: BenchmarkVariation,
        mode : str = "prefix-only" #other option : masked-oracle
):
    builder = ProgramBuilderProvider.get(benchmark_name)
    loader = LoaderProvider.get(benchmark_name, benchmark_variation)
    tests_runner    = RunnerProvider.get_for_test_generation(benchmark_name, os.getenv("WORK_DIR"))

    generator = TestGenerator(model=model_name, generation_type=TestGenerationType.ORACLE_COMPLETION if mode == "masked-oracle" else TestGenerationType.ASSERTION_GENERATION)

    pipeline = OracleCompletionPipeline(
        loader=loader,
        builder=builder,
        runner=tests_runner,
        generator=generator,
        extracted_tests_file = input_file,
        mode=mode,
        config=config,
    )
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fault revelation with LLMs')
    parser.add_argument('--task', type=str, choices=['fault_collection', 'test_generation', 'mutant_generation', 'test_augmentation', 'oracle_completion'], required=True)
    parser.add_argument('--model', type=str, default=None, required=True, help='Model name to use under given task')
    parser.add_argument('--output', '-o', type=str, default=None, required=False, help='(Optional) The suffix of the output filepath where the resuls will be saved (don not add exception)')
    parser.add_argument('-n', '--n-generations', type=int, default=10, help='Specify the number of times to generate code/test for the same task')
    parser.add_argument('-b', '--benchmark', type=str, choices=['he', 'bcb', 'ncb', 'mbpp'], required=False, help='Specify the benchmark that is being used')
    parser.add_argument('-bv', '--benchmark-variation', type=BenchmarkVariation, choices=list(BenchmarkVariation), default=BenchmarkVariation.NONE, required=False, help='Specify whether a benchmark variation should be used')
    parser.add_argument('-tc', '--test-criterion', type=RefinementMode, choices=list(RefinementMode), default=RefinementMode.NONE, required=False, help='Specify refinement mode based on test criterion. Test generation flag only')
    parser.add_argument('--skip-n-tasks', type=int, required=False, help="Specify whether to skip any number of tasks, starting from the beginning")
    parser.add_argument('--input-faults', type=str, default=None, required=False, help='(Test Generation) The input faults (full jsonl filepath) to generate tests')
    parser.add_argument('--input-prompts', type=str, default=None, required=False,
                        help='(Test Generation) The input prompts (full jsonl filepath) to generate tests')
    parser.add_argument('--input-tests', type=str, default=None, required=False, help='(Test Generation) The input tests (full jsonl filepath) to augment coverage')
    parser.add_argument('--input-mutants', type=str, default=None, required=False, help='(Test Generation) The input mutants (full jsonl filepath) to generate tests')
    parser.add_argument('--input-generations', type=str, default=None, required=False, help='(Test Augmentation) The input generations (full jsonl filepath) to augment benchmark tests')
    parser.add_argument('--batch-id', type=str, default=None, required=False, help='(Test Augmentation) Completed batch job ID to load generations from')
    parser.add_argument('--slice', type=str, default=None, required=False,
                        help='Slice of tasks to process in format INDEX/TOTAL (e.g. 0/40, 1/40). Used for HPC parallelism.')
    parser.add_argument('--start-task-id', type=str, default=None, required=False,
                        help='Skip all tasks before this task_id')
    parser.add_argument('--work-dir', type=str, default=None, required=False, help='Working directory for running tests. By default the value in .env is used')
    parser.add_argument('--collection-target', type=str, default="faults", required=False, help='Target of collection, either generations only or faults, By default the target is faults')
    parser.add_argument("--oracle-completion-mode", type=str, choices=['prefix-only', 'masked-oracle'])
    parser.add_argument(
        "--exclude-ids",
        nargs="+",
        type=str,
        help="A list of task ids to exclude separated by spaces"
    )

    args = parser.parse_args()

    if args.task == 'fault_collection':
        if args.benchmark is None:
            raise Exception("For Fault Collection tasks that benchmark argument is required")

        output_file = args.output
        config = PipelineConfig(
            n_generations=args.n_generations,
            output_file=output_file,
            slice=args.slice,
            start_task_id=args.start_task_id,
            exclude_ids=args.exclude_ids
        )

        if args.skip_n_tasks is not None:
            print("Configured to skip {} tasks".format(args.skip_n_tasks))
            config.skip_n_tasks = args.skip_n_tasks

        execute_fault_collection(args.model, config, args.benchmark, args.benchmark_variation, target=args.collection_target)

    elif args.task == 'test_generation':
        if args.input_faults is None:
            raise Exception("For Test Generation tasks the input faults is required")

        if args.test_criterion == "mutation" and not args.input_mutants:
            raise Exception("For Mutation guided Test Generation tasks the input mutants is required")

        output_file = args.output
        config = PipelineConfig(
            n_generations=args.n_generations,
            output_file=output_file,
            slice=args.slice,
            start_task_id=args.start_task_id,

        )

        input_tests = None if args.input_tests is None else read_jsonl(args.input_tests)
        input_mutants = None if args.input_mutants is None else read_jsonl(args.input_mutants)

        if args.input_prompts is not None:
            prompt_inputs = PromptInputDict.from_jsonl(args.input_prompts)
        else:
            prompt_inputs = None

        execute_test_generation_on_faults(
            model_name=args.model,
            config=config,
            input_file=args.input_faults,
            mode=args.test_criterion,
            input_tests=input_tests,
            input_mutants=input_mutants,
            benchmark_name=args.benchmark,
            work_dir=args.work_dir,
            prompt_inputs=prompt_inputs
        )

    elif args.task == 'mutant_generation':
        if args.input_faults is None:
            raise Exception("For Mutant Generation the --input-faults path is required")

        output_file = args.output
        config = PipelineConfig(
            output_file=output_file
        )

        execute_mutant_generation(input_file=args.input_faults, config=config)

    elif args.task == 'test_augmentation':
        if args.benchmark is None:
            raise Exception("For Test Augmentation the --benchmark argument is required")
        if args.batch_id is None:
            if args.input_generations is None:
                raise Exception("For Test Augmentation the --batch-id or input-generations argument is required")

        output_file = args.output
        config = PipelineConfig(
            n_generations=1,
            output_file=output_file,
            slice=args.slice,
            start_task_id=args.start_task_id,
            exclude_ids=args.exclude_ids
        )
        execute_benchmark_tests_augmentation(
            model_name=args.model,
            batch_id=args.batch_id,
            generations_path=args.input_generations,
            config=config,
            benchmark_name=args.benchmark,
            benchmark_variation=args.benchmark_variation,
        )

    elif args.task == 'oracle_completion':
        if args.input_tests is None:
            raise Exception("For Oracle Completion, you must provide test files that have tests processed into prefix, assertions, masked assertions")

        output_file = args.output
        config = PipelineConfig(
            n_generations=1,
            output_file=output_file,
            slice=args.slice,
            start_task_id=args.start_task_id,

        )

        execute_oracle_completion(
            model_name=args.model,
            config=config,
            input_file=args.input_tests,
            mode=args.oracle_completion_mode,
            benchmark_name=args.benchmark,
            benchmark_variation=args.benchmark_variation,
        )


    else:
        raise Exception("Not implemented")

    exit(0)
