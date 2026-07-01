from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from test_adequacy_study.evaluation.config import (
    METRICS,
    N_SHUFFLES,
    TEST_MODELS, CRITERIA,
)
from test_adequacy_study.evaluation.test_criteria import TestEntry, MinimisationStrategy, \
    CRITERION_REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPoolBuilder:
    """Builds a flat list of TestEntry objects from analysis records."""

    def build(
            self,
            records: list[dict],
            test_models: list[str] | None = None,
    ) -> list[TestEntry]:
        """
        test_models:
            None         → include all models
            ["gpt-4.1-mini", "gpt-5-mini"]   → include only these models
        """
        pool = []
        for record in records:
            if test_models is not None and record["test_model"] not in test_models:
                continue
            pool.extend(self._entries_from_record(record))
        return pool

    def _entries_from_record(self, record: dict) -> list[TestEntry]:
        test_model = record["test_model"]

        per_test_line_cov = record.get("line_coverage", {}).get("per_test_coverage", {})
        per_test_branch_cov = record.get("branch_coverage", {}).get("per_test_coverage", {})
        per_test_kills = record.get("mutation", {}).get("per_test_kills", {})
        per_test_ft = record.get("per_test_ft", {})
        per_test_fd = record.get("per_test_fd", {})

        entries = []

        def _normalize(test_name: str) -> str:
            name = test_name.replace(".py", "").replace("::", ".").replace("/", ".")
            return name

        for test_name in per_test_ft.keys():
            entries.append(TestEntry(
                name=test_model + "/" + test_name,
                covered_lines=set(per_test_line_cov.get(_normalize(test_name), [])),
                covered_branches=set(map(tuple, per_test_branch_cov.get(_normalize(test_name), []))),
                kills=set(per_test_kills.get(_normalize(test_name), [])),
                trigger=per_test_ft.get(test_name, 0),  # check if test triggers fault
                detection=per_test_fd.get(test_name, 0),  # check if
            ))
        return entries

    @staticmethod
    def get_totals(records: list[dict]) -> dict:
        """Totals are the same per fault — take from the first record."""
        first = records[0]
        return {
            "total_lines": first.get("line_coverage", {}).get("covered", []) + first.get("line_coverage", {}).get(
                "missing", []),
            "total_branches": first.get("branch_coverage", {}).get("covered", []) + first.get("branch_coverage",
                                                                                              {}).get("missing", []),
            "total_mutants": first.get("mutation", {}).get("total_mutants", []),
        }


class MinimisationSimulator:
    """
    runs the full minimisation simulation for a set of criteria.

    Usage
    -----
    # All models, all criteria, resolved output path from config
    simulator = MinimisationSimulator()
    results = simulator.run(
        analysis_file="results/gpt-5-mini/ncb/analysis_results.jsonl",
        fault_model="gpt-5-mini",
        benchmark="ncb",
    )

    # One model, two criteria
    simulator = MinimisationSimulator(criteria=["random", "line_coverage"])
    results = simulator.run(
        analysis_file="...",
        fault_model="gpt-5-mini",
        benchmark="ncb",
        test_models="gpt-5-mini",
    )
    """

    def __init__(
            self,
            criteria: list[str] | None = None,
            n_shuffles: int = N_SHUFFLES,
    ):
        criteria_names = criteria or CRITERIA
        self.criteria: list[MinimisationStrategy] = [
            CRITERION_REGISTRY[s] for s in criteria_names
        ]
        self.n_shuffles = n_shuffles
        self.pool_builder = TestPoolBuilder()

    def run(
            self,
            analysis_file: str,
    ) -> dict[str, dict]:
        """
        test_models:
            None         → pool from all models
                → pool from this model only
            ["gpt-4.1-mini" , "gpt-5-mini"]   → pool from these models only
        """

        analysis = self._load(analysis_file)
        logger.info("Loaded %d faults", len(analysis))

        results = {}
        for task_id, records in tqdm(analysis.items()):
            pool = self.pool_builder.build(records, test_models=TEST_MODELS)
            totals = self.pool_builder.get_totals(records)

            if not pool:
                logger.warning("[%s] empty pool, skipping", task_id)
                continue
            if len(totals["total_mutants"]) == 0:
                logger.warning("[%s] no mutants, skipping", task_id)
                continue

            first = records[0]
            line_cov = first.get("line_coverage", {}).get("coverage_pct", 0.0)
            mutation_score = first.get("mutation", {}).get("mutation_score", 0.0)
            if line_cov == 0.0 or mutation_score == 0.0:
                logger.warning("[%s] zero line coverage or mutation score, skipping", task_id)
                continue
            results[task_id] = self._run_fault(pool, totals)

        output_path = analysis_file.replace("analysis", "minimisation")
        self.save(results, output_path)
        return results

    def save(self, results: dict[str, dict], output_file: str) -> None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for task_id, result in results.items():
                record = {
                    "task_id": task_id,
                    "pool_size": result["pool_size"],
                    "has_branches": result["has_branches"],
                    "criteria": list(result["truncated"].keys()),
                }
                for version in ["truncated", "full"]:
                    for criterion, metrics in result[version].items():
                        for metric, runs in metrics.items():
                            if metric == "selected_tests":
                                # Store selected tests as-is (already a list of dicts)
                                record[f"{version}_{criterion}_{metric}"] = runs
                            else:
                                # Convert numpy arrays to lists
                                record[f"{version}_{criterion}_{metric}"] = [
                                    run.tolist() if isinstance(run, (list, np.ndarray)) else run
                                    for run in runs
                                ]
                f.write(json.dumps(record) + "\n")
        logger.info("Saved to %s", output_file)

    def _run_fault(self, pool: list[TestEntry], totals: dict) -> dict:
        active_criteria = self.criteria
        if len(totals["total_branches"]) == 0:
            active_criteria = [s for s in self.criteria if s.name != "branch_coverage"]

        all_runs = {}
        for criterion in active_criteria:
            all_runs[criterion.name] = criterion.run(pool, totals, self.n_shuffles)

        result = self._aggregate(all_runs, active_criteria)
        result["pool_size"] = len(pool)
        result["has_branches"] = True if len(totals["total_branches"]) > 0 else False
        return result

    def _aggregate(self, all_runs: dict[str, dict], active_criteria: list) -> dict:
        criterion_names = [s.name for s in active_criteria]

        truncated = {s: {m: [] for m in METRICS} for s in criterion_names}
        full = {s: {m: [] for m in METRICS} for s in criterion_names}

        # Add selected_tests to both truncated and full
        truncated_selected = {s: [] for s in criterion_names}
        full_selected = {s: [] for s in criterion_names}

        for i in range(self.n_shuffles):
            # min effort across criteria for this run
            min_len = min(len(all_runs[s]["trigger"][i]) for s in criterion_names)
            if min_len == 0:
                continue

            for s in criterion_names:
                for m in METRICS:
                    run = all_runs[s][m][i]
                    truncated[s][m].append(np.array(run[:min_len]))
                    full[s][m].append(np.array(run))

                # Truncate selected tests to min_len
                selected = all_runs[s]["selected_tests"][i]
                truncated_selected[s].append(selected[:min_len])
                full_selected[s].append(selected)

        # Add selected_tests to results
        for s in criterion_names:
            truncated[s]["selected_tests"] = truncated_selected[s]
            full[s]["selected_tests"] = full_selected[s]

        return {"truncated": truncated, "full": full}

    @staticmethod
    def _load(path: str) -> dict[str, list[dict]]:
        results = defaultdict(list)
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                results[record["task_id"]].append(record)
        return results

