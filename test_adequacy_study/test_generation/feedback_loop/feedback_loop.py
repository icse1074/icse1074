from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from test_adequacy_study.data_models.refinement_report import RefinementReport
from test_adequacy_study.data_models.task import Task

@dataclass
class FeedbackResult:
    retry: bool
    prompt_variables: dict
    updated_tests: Optional[str] = None

class FeedbackLoop(ABC):
    """
    A single feedback concern. Implementations look at a RunResult (and
    optionally the task/tests), decide whether a retry is warranted, and
    return the extra prompt variables to inject into the next generation.
    """
    last_report: Optional[RefinementReport] = None


    def augment_initial_prompt_variables(self, initial_prompt_variables, task_id) -> dict:
        #depending on the criteria, the initial prompt variables change
        return {}

    @abstractmethod
    def evaluate(self,
                 task : Task,
                 tests : str,
                 run_result,
                 test_id = None,
                 **kwargs) -> FeedbackResult:
        #option to evaluate based on a single test node and return only the failures on that test
        ...