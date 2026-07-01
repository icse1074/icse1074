from dataclasses import dataclass, field
from typing import List

from test_adequacy_study.data_models.execution_report import ExecutionReport


@dataclass
class FaultCollectionResult:
    total_tasks: int = 0
    total_generated: int = 0
    total_run: int = 0
    faults: List[ExecutionReport] = field(default_factory=list)

    @property
    def total_faults(self) -> int:
        return len(self.faults)

    @property
    def fault_rate(self) -> float:
        return self.total_faults / self.total_run if self.total_run else 0.0

    def summary(self) -> str:
        return (
            f"Tasks: {self.total_tasks} | "
            f"Generated: {self.total_generated} | "
            f"Executed: {self.total_run} | "
            f"Faults: {self.total_faults} | "
            f"Fault rate: {self.fault_rate:.1%}"
        )
