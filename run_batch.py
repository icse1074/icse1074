import argparse
import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.file_utils import get_output_filepath
from test_adequacy_study.generators.code_generator import CodeGenerator
from test_adequacy_study.providers.loader_provider import LoaderProvider
from test_adequacy_study.providers.runner_provider import RunnerProvider
from test_adequacy_study.runners.humaneval_test_runner import HumanEvalTestRunner
from test_adequacy_study.runners.mbpp_test_runner import MBPPTestRunner
from test_adequacy_study.test_generation.batch_fault_collection_pipeline import BatchFaultCollectionPipeline
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation
from test_adequacy_study.types.pipeline_config import PipelineConfig

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(100000000)

def execute_fault_collection_batch(model_name: str, config: PipelineConfig, benchmark_name: str, batch_id: str = None, exclude_indexes: list[int] = None, temperature: float = 0.8, variation: BenchmarkVariation = BenchmarkVariation.NONE, work_dir: str = None):
    """
    Executes the Fault Collection Pipeline that will use the HumanEval benchmark to generate code.
    The Fault Collection Pipeline will execute the HumanEval tests and collect the generated codes that are faulty

    :param config:
    :param model_name:
    :return:
    """
    loader = LoaderProvider.get(benchmark_name, variation)
    runner = RunnerProvider.get(benchmark_name, work_dir=work_dir)

    # Perhaps this can become universal
    if benchmark_name == 'mbpp' and batch_id is not None:
        config.match_function_names = True

    pipeline = BatchFaultCollectionPipeline(
        loader=loader,
        generator=CodeGenerator(model=model_name, temperature=temperature),
        runner=runner,
        builder=PythonProgramBuilder(),
        config=config,
        exclude_indexes=exclude_indexes
    )

    if batch_id is None:
        result = pipeline.run_create_and_upload()
    else:
        result = pipeline.run_collect_faults(batch_id)
        print("The following script might be of your interest")
        print(f"python run_hard_to_detect_faults.py --model {model_name} -i {config.output_file} -b {benchmark_name} -bv {variation.value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fault revelation with LLMs - Batch API')
    parser.add_argument('--task', type=str, choices=['upload', 'collect'])
    parser.add_argument('--batch-id', type=str, default=None, required=False)
    parser.add_argument('--model', type=str, default=None, required=True, help='Model name to use under given task')
    parser.add_argument('--output', '-o', type=str, default=None, required=False, help='(Optional) The suffix of the output filepath where the resuls will be saved (don not add exception)')
    parser.add_argument('-n', '--n-generations', type=int, default=10, help='Specify the number of times to generate code/test for the same task')
    parser.add_argument('-b', '--benchmark', type=str, choices=['he', 'bcb', 'mbpp', 'ncb'], required=True, help='Specify the benchmark that is being used')
    parser.add_argument('-bv', '--benchmark-variation', type=BenchmarkVariation, choices=list(BenchmarkVariation), default=BenchmarkVariation.NONE, required=False, help='Specify whether a benchmark variation should be used')
    parser.add_argument('--skip-n-tasks', type=int, required=False, help="Specify whether to skip any number of tasks, starting from the beginning")
    parser.add_argument('--exclude-indexes', type=int, nargs='+', required=False, help="Specify whether to skip any number of tasks, starting from the beginning")
    parser.add_argument('--temperature', type=float, required=False, default=0.8, help='Specify the temperature of the model')
    parser.add_argument('--slice', type=str, default=None, required=False,
                        help='Slice of tasks to process in format INDEX/TOTAL (e.g. 0/40, 1/40). Used for HPC parallelism.')
    parser.add_argument('--work-dir', type=str, default=None, required=False, help='Working directory for running tests. By default the value in .env is used')

    args = parser.parse_args()

    output_file = get_output_filepath(args.output or f"batch_{args.task}_{args.benchmark}_{args.benchmark_variation.value}", args.model)
    config = PipelineConfig(
        n_generations=args.n_generations,
        output_file=output_file,
        slice=args.slice,
    )

    if args.skip_n_tasks is not None:
        print("Configured to skip {} tasks".format(args.skip_n_tasks))
        config.skip_n_tasks = args.skip_n_tasks

    # Batch processing
    if args.task == 'upload':
        execute_fault_collection_batch(args.model,
                                       config,
                                       args.benchmark,
                                       temperature=args.temperature,
                                       variation=args.benchmark_variation,
                                       work_dir=args.work_dir)
    elif args.task == 'collect':
        if args.batch_id is None:
            raise Exception("--batch-id is required for collect")
        execute_fault_collection_batch(args.model,
                                       config,
                                       args.benchmark,
                                       args.batch_id,
                                       args.exclude_indexes,
                                       variation=args.benchmark_variation,
                                       work_dir=args.work_dir)
        print("Output file:", output_file)
    else:
        raise Exception("--task must be either 'upload' or 'collect'")
