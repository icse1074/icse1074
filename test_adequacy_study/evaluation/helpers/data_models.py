from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class CoverageRecord:
    coverage_pct: float = 0.0
    covered : list[int]= field(default_factory=list)
    missing : list[int]= field(default_factory=list)
    per_test_coverage: dict[str, list[int]] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_pct":      self.coverage_pct,
            "total": len(self.covered) + len(self.missing),
            "missing": self.missing,
            "covered": self.covered,
            "per_test_coverage": self.per_test_coverage,
        }

    def from_dict(cls, data: dict[str, Any]):
        return cls(
            coverage_pct = data.get("coverage_pct", 0.0),
            total = data.get("total", 0),
            missing=data.get("missing", []),
            covered=data.get("covered", []),
            per_test_coverage = data.get("per_test_coverage", {})
        )

@dataclass
class MutationRecord:
    mutation_score: float = 0.0
    total_mutants: list[int] = field(default_factory=list)
    killed_mutants: list[int] = field(default_factory=list)
    per_test_kills: dict[str, list[str]] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_score": self.mutation_score,
            "total_mutants": self.total_mutants,
            "killed_mutants": self.killed_mutants,
            "per_test_kills": self.per_test_kills,

        }
    def from_dict(cls, data: dict[str, Any]):
        return cls(
            mutation_score=data.get("mutation_score", 0.0),
            total_mutants=data.get("total_mutants", []),
            killed_mutants=data.get("killed_mutants", []),
            per_test_kills=data.get("per_test_kills", {}),

        )
@dataclass
class AnalysisRecord:
    task_id: str
    fault_model: str
    benchmark: str
    test_model: int

    # FT / FD
    per_test_ft: dict[str, int] = field(default_factory=dict)
    per_test_fd: dict[str, int] = field(default_factory=dict)


    # Line Coverage
    line_coverage_record : CoverageRecord = None

    # Branch Coverage
    branch_coverage_record : CoverageRecord = None


    # Mutation
    mutation_record : MutationRecord = None


    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id":           self.task_id,
            "fault_model":             self.fault_model,
            "benchmark":         self.benchmark,
            "test_model":     self.test_model,
            "per_test_ft":       self.per_test_ft,
            "per_test_fd":       self.per_test_fd,
            "line_coverage": self.line_coverage_record.to_dict(),
            "branch_coverage": self.branch_coverage_record.to_dict(),
            "mutation": self.mutation_record.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisRecord":
        return cls(
            task_id=d["task_id"],
            model=d.get("model", "unknown"),
            benchmark=d.get("benchmark", "unknown"),
            test_model=d["test_model"],
            per_test_ft=d.get("per_test_ft", {}),
            per_test_fd=d.get("per_test_fd", {}),
            line_coverage_record=CoverageRecord.from_dict(d["line_coverage"]),
            branch_coverage_record=CoverageRecord.from_dict(d["branch_coverage"]),
            mutation_record=MutationRecord.from_dict(d["mutation_record"]),
                    )



@dataclass
class StrategyRuns:
    """All shuffles for one strategy × one metric combination."""
    strategy: str
    metric: str
    runs: list[np.ndarray] = field(default_factory=list)


@dataclass
class SelectionResult:
    """Full selection simulation output for one (task_id, model, benchmark)."""
    task_id: str
    model: str
    benchmark: str
    pool_size: int

    # {version: {strategy: {metric: [array_per_shuffle]}}}
    # version ∈ {"truncated", "full"}
    data: dict[str, dict[str, dict[str, list[np.ndarray]]]] = field(default_factory=dict)