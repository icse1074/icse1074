from abc import ABC, abstractmethod

from test_adequacy_study.evaluation.config import METRICS

from dataclasses import dataclass, field

import random


@dataclass
class TestEntry:
    # one test with all the metrics
    name: str
    covered_lines: set[int] = field(default_factory=set)
    covered_branches: set[tuple] = field(default_factory=set)
    kills: set[int] = field(default_factory=set)
    trigger: int = 0
    detection: int = 0


@dataclass
class MinimizedSuiteState:
    """Accumulated state as tests are selected one by one."""
    found_trigger: bool = False
    found_detection: bool = False
    covered_lines: set[int] = field(default_factory=set)
    covered_branches: set[tuple] = field(default_factory=set)
    killed: set[int] = field(default_factory=set)
    selected_tests: list[dict] = field(default_factory=list)

    def update(self, test: TestEntry) -> None:
        if test.trigger:
            self.found_trigger = True
        if test.detection:
            self.found_detection = True
        self.covered_lines |= test.covered_lines
        self.covered_branches |= test.covered_branches
        self.killed |= test.kills
        self.selected_tests.append({
            "name": test.name,
            "ft": test.trigger,
            "fd": test.detection,
        })

    def metrics(self, total_lines, total_branches, total_mutants) -> dict[str, float]:
        return {
            "trigger": float(self.found_trigger),
            "detection": float(self.found_detection),
            "line_cov": len(self.covered_lines) / len(total_lines) if total_lines else 0.0,
            "branch_cov": len(self.covered_branches) / len(total_branches) if total_branches else 0.0,
            "ms": len(self.killed) / len(total_mutants) if total_mutants else 0.0,
        }


class MinimisationStrategy(ABC):
    """Base class for test minimisation CRITERIA."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(
            self,
            pool: list[TestEntry],
            totals: dict,
            n_shuffles: int,
    ) -> dict[str, list]:
        """
        Returns {
            metric: [[run_0_step_0, ...], [run_1_step_0, ...], ...],
            ...
            "selected_tests": [
                [
                    [{"name": "test_0", "ft": 1, "fd": 0}, ...],  # run 0
                    [{"name": "test_2", "ft": 0, "fd": 1}, ...]   # run 1
                ]
            ]
        }
        """
        ...

    def _empty_runs(self) -> dict[str, list]:
        return {m: [] for m in METRICS}

    def _record_step(
            self,
            test: TestEntry,
            state: MinimizedSuiteState,
            totals: dict,
            vecs: dict[str, list],
            selected: list[list[dict]]  = None,
    ) -> None:
        state.update(test)
        for m, v in state.metrics(total_lines=totals["total_lines"], total_mutants=totals["total_mutants"],
                                  total_branches=totals["total_branches"]).items():
            vecs[m].append(v)

        # Track selected tests at each step
        if selected is not None:
            selected.append(state.selected_tests.copy())


class RandomSelection(MinimisationStrategy):

    @property
    def name(self) -> str:
        return "random"

    def run(self, pool, totals, n_shuffles):
        runs = self._empty_runs()
        all_selected = []  # Track selected tests across all runs

        for _ in range(n_shuffles):
            shuffled = pool.copy()
            random.shuffle(shuffled)

            state = MinimizedSuiteState()
            vecs = self._empty_runs()
            selected = []  # Selected tests for this run

            for test in shuffled:
                self._record_step(test, state, totals, vecs, selected)

            for m in METRICS:
                runs[m].append(vecs[m])

            all_selected.append(selected)

        runs["selected_tests"] = all_selected
        return runs


class CoverageSelection(MinimisationStrategy):
    coverage_key: str  # "covered_lines" or "covered_branches"

    def run(self, pool, totals, n_shuffles):
        runs = self._empty_runs()
        all_selected = []  # Track selected tests across all runs

        for _ in range(n_shuffles):
            remaining = pool.copy()
            random.shuffle(remaining)  # break ties randomly

            state = MinimizedSuiteState()
            vecs = self._empty_runs()
            selected = []  # Selected tests for this run

            coverage_total_key = "total_lines" if self.coverage_key == "covered_lines" else "total_branches"
            while remaining:
                current = getattr(state, self.coverage_key)
                if len(current) == totals[coverage_total_key]:
                    break
                test = remaining.pop(0)  # already shuffled, just take next
                if not (getattr(test, self.coverage_key) - current):
                    continue  # discard
                self._record_step(test, state, totals, vecs, selected)

            for m in METRICS:
                runs[m].append(vecs[m])

            all_selected.append(selected)

        runs["selected_tests"] = all_selected
        return runs


class LineCoverageSelection(CoverageSelection):
    coverage_key = "covered_lines"

    @property
    def name(self) -> str:
        return "line_coverage"


class BranchCoverageSelection(CoverageSelection):
    coverage_key = "covered_branches"

    @property
    def name(self) -> str:
        return "branch_coverage"


class MutationSelection(MinimisationStrategy):
    """Greedy: iterate over mutants in random order, pick the test that kills each."""

    @property
    def name(self) -> str:
        return "mutation"

    def run(self, pool, totals, n_shuffles):
        runs = self._empty_runs()
        all_selected = []  # Track selected tests across all runs

        for _ in range(n_shuffles):
            all_mutants = totals["total_mutants"]
            if not all_mutants:
                continue

            remaining_mutants = all_mutants.copy()
            random.shuffle(remaining_mutants)

            selected_tests = set()  # track which tests have been selected
            state = MinimizedSuiteState()
            vecs = self._empty_runs()
            selected = []  # Selected tests for this run

            while remaining_mutants and len(selected_tests) < len(pool):
                m_id = remaining_mutants.pop(0)

                if m_id in state.killed:
                    continue

                # candidate tests that haven't been selected yet and can kill this mutant
                candidates = [t for t in pool if t.name not in selected_tests]
                random.shuffle(candidates)

                for test in candidates:
                    if m_id in test.kills:
                        selected_tests.add(test.name)
                        # remove m_id and all mutants killed by test
                        remaining_mutants = [mutant for mutant in remaining_mutants if mutant not in test.kills]
                        self._record_step(test, state, totals, vecs, selected)
                        break

            for m in METRICS:
                runs[m].append(vecs[m])

            all_selected.append(selected)

        runs["selected_tests"] = all_selected
        return runs


_ALL_CRITERIA: list[MinimisationStrategy] = [
    RandomSelection(),
    LineCoverageSelection(),
    BranchCoverageSelection(),
    MutationSelection(),
]

CRITERION_REGISTRY: dict[str, MinimisationStrategy] = {
    s.name: s for s in _ALL_CRITERIA
}