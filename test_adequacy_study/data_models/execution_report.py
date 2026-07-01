from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict


class Verdict(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"
    ERROR = "error"          # unexpected crash in the runner itself



@dataclass
class TestResult:
    node_id: str       # e.g. "test_foo.py::TestClass::test_bar"
    outcome: str       # "passed" | "failed" | "error" | "skipped"
    message: Optional[str] = None  # failure/error reason, None if passed
    crash_path: Optional[str] = None


@dataclass(frozen=True)
class ExecutionReport:
    task_id: str
    verdict: Verdict
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    detailed_test_results : Optional[List[TestResult]] = None


    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASSED