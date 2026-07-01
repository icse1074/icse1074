from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from test_adequacy_study.data_models.execution_report import ExecutionReport
from test_adequacy_study.data_models.refinement_report import RefinementReport


@dataclass
class TestGenerationResponse:
    task_id: str
    code_under_test: str
    response: str | None
    model_id: str = None
    execution_report: Optional[ExecutionReport] = None
    api_calls: int = 0
    refinement_report: Optional[RefinementReport] = None

    def to_test_class(self):
        """
        Should generate a Python test class based on the response

        :return:
        """
        pass
