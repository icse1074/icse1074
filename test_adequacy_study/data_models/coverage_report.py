from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple

from test_adequacy_study.data_models.refinement_report import RefinementReport


@dataclass
class CoverageReport(RefinementReport):
    coverage_pct: float
    missing_lines: List[int] = field(default_factory=list)
    covered_lines: List[int] = field(default_factory=list)
    total_lines: int = 0
    missing_branches: List[List[int]] = field(default_factory=list)
    covered_branches: List[List[int]] = field(default_factory=list)
    total_branches: int = 0
    per_test_line_coverage: Dict[str, List[int]] = field(default_factory=dict)  # test_name → lines covered
    per_test_branch_coverage: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict) # test_name → branches covered

    @property
    def is_full_coverage(self) -> bool:
        return self.coverage_pct >= 100.0

    @property
    def has_full_branch_coverage(self) -> bool:
        return len(self.missing_branches) <= 0

    @property
    def has_full_line_coverage(self) -> bool:
        return len(self.missing_lines) <= 0

    def __str__(self):
        return (
            f"Coverage: {self.coverage_pct:.1f}% "
            f"({len(self.covered_lines)}/{self.total_lines} lines) "
            f"({len(self.covered_branches)}/{self.total_branches} branches) "
            f"Missing lines: {self.missing_lines} "
            f"Missing branches: {self.missing_branches} "
        )