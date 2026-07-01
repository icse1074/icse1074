from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from test_adequacy_study.runners.mutation_runner import MutantInfo
from test_adequacy_study.file_utils import read_jsonl
from test_adequacy_study.evaluation.config import FAULTS_FILE_PATTERN, MUTANTS_FILE_PATTERN

logger = logging.getLogger(__name__)


class FaultLoader:
    """
    Loads faults and mutants for a given (model, benchmark) combination.

    """

    def load_faults(self, model: str, benchmark: str) -> list[dict]:
        """Return all fault records for this (model, benchmark) pair."""
        faults_file = Path(FAULTS_FILE_PATTERN.format(fault_model=model, benchmark=benchmark))
        try:
            faults_file = next(faults_file.parent.glob(faults_file.name))
        except StopIteration:
            logger.warning("FaultLoader: faults file not found — %s", faults_file)
            return []
        faults = read_jsonl(str(faults_file))
        logger.info("FaultLoader: loaded %d faults from %s", len(faults), faults_file)
        return faults

    def load_mutants(self, model: str, benchmark: str) -> dict[str, list[MutantInfo]]:
        """Return {task_id: [MutantInfo]} for this (model, benchmark) pair."""
        mutants_file = Path(MUTANTS_FILE_PATTERN.format(fault_model=model, benchmark=benchmark))
        try:
            mutants_file = next(mutants_file.parent.glob(mutants_file.name))
        except StopIteration:
            logger.warning("FaultLoader: mutants file not found — %s", mutants_file)
            return []

        mutants: dict[str, list[MutantInfo]] = defaultdict(list)
        for record in read_jsonl(str(mutants_file)):
            mutants[record["task_id"]].append(MutantInfo(
                mutant_id=record["mutant_id"],
                operator=record["operator"],
                line=record["line"],
                original=record["original"],
                mutated=record["mutated"],
            ))

        n_tasks   = len(mutants)
        n_mutants = sum(len(v) for v in mutants.values())
        logger.info(
            "FaultLoader: loaded %d mutants across %d tasks from %s",
            n_mutants, n_tasks, mutants_file.name,
        )
        return dict(mutants)