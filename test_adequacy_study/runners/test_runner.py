from __future__ import annotations

import ast
import os
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv
from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.execution_report import ExecutionReport, Verdict
from test_adequacy_study.data_models.test_suite import TestSuite
from test_adequacy_study.runners.helpers.execution_sandbox import ExecutionSandbox

load_dotenv()

class TestRunner(ABC):
    """
    Base runner: manages the run lifecycle (isolation, execution, cleanup).
    Subclasses implement _prepare() and optionally override _execute().
    """

    def __init__(self, timeout: float = 120, work_dir: str = os.environ.get("WORK_DIR"), sandbox: ExecutionSandbox | None = None):
        self.timeout = timeout
        self.work_dir = work_dir
        self.sandbox = sandbox or ExecutionSandbox()

    def run(self, cut: CUT, suite: TestSuite) -> ExecutionReport:
        # print("running tests from WORK_DIR : ", self.work_dir )
        if not cut.syntactically_valid:
            return ExecutionReport(
                task_id=cut.task_id,
                verdict=Verdict.SYNTAX_ERROR,
                stderr="SyntaxError detected before execution.",
            )

        run_dir = Path(self.work_dir) / str(cut.task_id).replace("/", "_") / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)

        try :
            if isinstance(suite.source, str) :
                ast.parse(suite.source)
        except SyntaxError :
            return ExecutionReport(
            task_id=cut.task_id,
            verdict=Verdict.SYNTAX_ERROR,
            stdout=None,
            stderr=None,
            duration_seconds=0,
        )

        try:
            entry_point = self._prepare(cut, suite, run_dir)
            return self._execute(entry_point, cut.task_id)
        finally:
            import shutil
            shutil.rmtree(run_dir.parent, ignore_errors=True)

    @abstractmethod
    def _prepare(self, cut: CUT, suite: TestSuite, run_dir: Path) -> Path:
        ...

    @abstractmethod
    def _execute(self, entry_point: Path, task_id: str) -> ExecutionReport:
        ...