from dataclasses import dataclass, field

from test_adequacy_study.data_models.refinement_report import RefinementReport


@dataclass
class MutantInfo:
    mutant_id: int
    operator: str
    line: int
    original: str
    mutated: str
    task_id : str = None


@dataclass
class MutationReport(RefinementReport):
    mutation_score: float
    total_mutants: list[int] = field(default_factory=list)
    killed_mutants: list[int] = field(default_factory=list)
    incompetent_mutants: list[int] = field(default_factory=list)
    per_test_kills: dict[str, list[int]] = field(default_factory=dict)
    per_mutant: dict[int, MutantInfo] = field(default_factory=dict)

    @property
    def max_mutation_score_reached(self) -> bool:
        return self.mutation_score == 1