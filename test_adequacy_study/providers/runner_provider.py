import os

from dotenv import load_dotenv

from test_adequacy_study.runners.bigcodebench_test_runner import BigCodeBenchTestRunner
from test_adequacy_study.runners.humaneval_test_runner import HumanEvalTestRunner
from test_adequacy_study.runners.mbpp_test_runner import MBPPTestRunner
from test_adequacy_study.runners.pytest_runner import PytestRunner
from test_adequacy_study.runners.test_runner import TestRunner
load_dotenv()


class RunnerProvider:

    @staticmethod
    def get(benchmark_name: str, work_dir: str = None) -> TestRunner:

        if benchmark_name == 'mbpp':
            return MBPPTestRunner(work_dir=work_dir or os.getenv('WORK_DIR'))
        elif benchmark_name == 'he':
            return HumanEvalTestRunner()
        elif benchmark_name == 'bcb':
            return BigCodeBenchTestRunner()
        elif benchmark_name == 'ncb':
            return PytestRunner()
        else:
            raise Exception("Benchmark is not supported")

    @staticmethod
    def get_for_test_generation(benchmark_name: str, work_dir: str = None) -> TestRunner:

        if benchmark_name == 'mbpp':
            return PytestRunner(work_dir=work_dir or os.getenv('WORK_DIR'))
        elif benchmark_name == 'he':
            return PytestRunner(work_dir=work_dir or os.getenv('WORK_DIR'))
        elif benchmark_name == 'bcb':
            return BigCodeBenchTestRunner()
        elif benchmark_name == 'ncb':
            return PytestRunner()
        else:
            raise Exception("Benchmark is not supported")
