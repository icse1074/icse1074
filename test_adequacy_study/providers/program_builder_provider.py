from test_adequacy_study.builders.mbpp_python_program_builder import MbppPythonProgramBuilder
from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder


class ProgramBuilderProvider:

    @staticmethod
    def get(benchmark_name: str):
        if benchmark_name == 'mbpp':
            return MbppPythonProgramBuilder()
        return PythonProgramBuilder()