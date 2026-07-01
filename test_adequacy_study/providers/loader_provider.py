import os

from dotenv import load_dotenv

from test_adequacy_study.benchmarks.bigcodebench import BigCodeBenchLoader
from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from test_adequacy_study.benchmarks.humaneval_with_pytests_loader import HumanEvalWithPytestsLoader
from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.benchmarks.mbpp_loader import MBPPLoader
from test_adequacy_study.benchmarks.mbpp_with_pytests_loader import MBPPWithPytestsLoader
from test_adequacy_study.benchmarks.naturalcodebench import NaturalCodeBenchLoader
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation

load_dotenv()


class LoaderProvider:

    @staticmethod
    def get(benchmark_name: str, variation: BenchmarkVariation) -> BenchmarkLoader:

        if benchmark_name == 'mbpp':
            return MBPPLoader(variation)
        elif benchmark_name == 'he':
            return HumanEvalLoader(variation)
        elif benchmark_name == 'bcb':
            return BigCodeBenchLoader(variation)
        elif benchmark_name == 'ncb':
            return NaturalCodeBenchLoader(variation)
        else:
            raise Exception("Benchmark is not supported")

    @staticmethod
    def get_with_pytests(benchmark_name: str, variation: BenchmarkVariation) -> BenchmarkLoader:
        if benchmark_name == 'mbpp':
            return MBPPWithPytestsLoader(variation)
        elif benchmark_name == 'he':
            return HumanEvalWithPytestsLoader(variation)
        elif benchmark_name == 'bcb':
            return BigCodeBenchLoader(variation)
        elif benchmark_name == 'ncb':
            return NaturalCodeBenchLoader(variation)
        else:
            raise Exception("Benchmark is not supported")