from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from tqdm import tqdm
from pathlib import Path

from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.data_models.mutation_report import MutantInfo, MutationReport
from test_adequacy_study.file_utils import write_jsonl
from test_adequacy_study.runners.pytest_runner import PytestRunner

logger = logging.getLogger(__name__)

CODE_MUTATE_BIN = os.getenv("CODE_MUTATE")


# --- Mutant Parsing ---

def _parse_mutants(output: str) -> list[MutantInfo]:
    import re
    import whatthepatch

    mutants = []
    skipped_empty = []
    skipped_malformed = []

    pattern = r"\[#(\d+)\] Mutation(.*?)(?=\n\[#|\Z)"
    matches = re.findall(pattern, output, re.DOTALL)

    for m_id, block in matches:
        try:
            op_match = re.search(r"- \[(\w+)\] ([\w\.]+)", block)
            if not op_match:
                skipped_malformed.append(m_id)
                continue

            mut_operator = op_match.group(1)

            diff_start = block.find("---")
            if diff_start == -1: #empty mutants
                logger.debug("Skipping empty-diff mutant #%s (operator: %s)", m_id, mut_operator)
                skipped_empty.append(m_id)
                continue

            diff_text = block[diff_start:]
            patches = list(whatthepatch.parse_patch(diff_text))

            original_lines = []
            mutated_lines = []
            changed_line_numbers = []

            for patch in patches:
                for change in patch.changes:
                    if change.old is None:
                        mutated_lines.append(change.line)
                        changed_line_numbers.append(change.new)
                    elif change.new is None:
                        original_lines.append(change.line)
                        changed_line_numbers.append(change.old)

            if len(changed_line_numbers) != 2 or changed_line_numbers[0] != changed_line_numbers[1]:
                logger.warning("Skipping mutant #%s: unexpected line number structure %s", m_id, changed_line_numbers)
                skipped_malformed.append(m_id)
                continue

            mutants.append(MutantInfo(
                mutant_id=int(m_id),
                operator=mut_operator,
                line=changed_line_numbers[0],
                original="\n".join(original_lines).strip(),
                mutated="\n".join(mutated_lines).strip(),
            ))

        except Exception as e:
            logger.warning("Failed to parse mutant #%s: %s", m_id, e)
            skipped_malformed.append(m_id)
            continue


    assert len(mutants) + len(skipped_empty) + len(skipped_malformed) == len(matches), (
        f"Parsing mismatch: {len(mutants)} parsed + {len(skipped_empty)} empty + "
        f"{len(skipped_malformed)} malformed != {len(matches)} total blocks"
    )

    if skipped_malformed:
        logger.warning("Malformed mutants skipped: %s", skipped_malformed)

    return mutants

def _load_mutants_from_file(path: Path) -> list[MutantInfo]:
    mutants = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            mutants.append(MutantInfo(
                mutant_id=record["mutant_id"],
                operator=record["operator"],
                line=record["line"],
                original=record["original"],
                mutated=record["mutated"],
            ))
    return mutants



# --- Saving Mutants ---

def _save_mutants_to_file(mutants: list[MutantInfo], path: Path, task_id : str) -> None:
    for mutant in mutants:
        record = {
            "task_id": task_id,
            "mutant_id": mutant.mutant_id,
            "operator": mutant.operator,
            "line": mutant.line,
            "original": mutant.original,
            "mutated": mutant.mutated,
        }
        write_jsonl(filename=path, data=[record], append=True)


# --- Mutant Application ---

def _apply_mutant(solution_path: Path, mutant: MutantInfo) -> bool:
    lines = solution_path.read_text(encoding="utf-8").splitlines()
    line_idx = mutant.line - 1  # 1-indexed to 0-indexed

    if 0 <= line_idx < len(lines) and lines[line_idx].strip() == mutant.original:
        # happy path: line matches exactly
        indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
        lines[line_idx] = (" " * indent) + mutant.mutated
    else:
        # fallback: search all lines
        matching = [i for i, line in enumerate(lines) if line.strip() == mutant.original]
        if not matching:
            logger.warning("Failed to apply mutant #%d: original line not found: %s",
                           mutant.mutant_id, mutant.original)
            return False
        line_idx = matching[0]
        indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
        lines[line_idx] = (" " * indent) + mutant.mutated

    solution_path.write_text("\n".join(lines), encoding="utf-8")
    return True

# --- Outcome Comparison ---

def _same_outcome(ref: dict, flt: dict) -> bool:
    if ref["outcome"] != flt["outcome"]:
        return False
    if ref["outcome"] == flt["outcome"] == "failed":
        return ref["message"] == flt["message"]
    return True


# --- Mutation Runner ---

class MutationRunner(PytestRunner):
    """
    Two separate concerns:
      - generate(): run cm on solution code, parse and optionally save mutants
      - run():      run tests against each mutant, return MutationReport
                    optionally loads mutants from a pre-saved file
    """

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        cut,
        save_path: Path | None = None,
    ) -> list[MutantInfo]:
        """
        Generate mutants for the given CUT using code_mutate.
        Optionally saves mutants to a JSONL file at save_path.
        Returns the list of MutantInfo objects.
        """
        run_dir = Path(self.work_dir) / cut.task_id.replace("/", "_") / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            solution_path = run_dir / "solution.py"
            solution_path.write_text(cut.content, encoding="utf-8")

            mutants = self._generate_mutants(solution_path, run_dir)
            logger.info("[%s] Generated %d mutants", cut.task_id, len(mutants))

            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                _save_mutants_to_file(mutants, save_path, task_id=cut.task_id)
                logger.info("[%s] Saved mutants to %s", cut.task_id, save_path)

            return mutants

        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run(
        self,
        cut,
        suite,
        mutants_file: Path | None = None,
        mutants: list[MutantInfo] | None = None,
    ) -> MutationReport:
        """
        Run mutation testing for the given CUT and test suite.

        Mutants are loaded from (in priority order):
          1. mutants argument (already in memory)
          2. mutants_file (pre-saved JSONL file)
          3. generated fresh using code_mutate

        Returns a MutationReport.
        """
        run_dir = Path(self.work_dir) / str(cut.task_id).replace("/", "_") / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._execute_mutation(cut, suite, run_dir, mutants_file, mutants)
        finally:
            shutil.rmtree(run_dir.parent, ignore_errors=True)

    def _execute_mutation(
        self,
        cut,
        suite,
        run_dir: Path,
        mutants_file: Path | None,
        mutants: list[MutantInfo] | None,
    ) -> MutationReport:

        # write solution and tests
        self._prepare(cut, suite, run_dir)
        solution_path = run_dir / "solution.py"
        original_source = solution_path.read_text(encoding="utf-8")
        incompetent_mutants = []

        # resolve mutants
        if mutants is not None:
            logger.info("Using %d mutants from memory", len(mutants))
        elif mutants_file is not None and Path(mutants_file).exists():
            mutants = _load_mutants_from_file(Path(mutants_file))
            logger.info("Loaded %d mutants from %s", len(mutants), mutants_file)
        else:
            mutants = self._generate_mutants(solution_path, run_dir)
            logger.info("Generated %d mutants fresh", len(mutants))

        if not mutants:
            logger.warning("No mutants available for %s", cut.task_id)
            return MutationReport(
                mutation_score=0.0,
                total_mutants=[],
                killed_mutants=[],
            )

        # reference run
        _, ref_results = self._run_and_get_results(run_dir)
        if not ref_results:
            logger.warning("No reference results for %s", cut.task_id)
            return MutationReport(
                mutation_score=0.0,
                total_mutants=[mutant.mutant_id for mutant in mutants],
                killed_mutants=[],
            )

        # run each mutant
        per_test_kills: dict[str, list[int]] = {}
        per_mutant: dict[int, MutantInfo] = {}
        killed_mutants = []

        for mutant in tqdm(mutants, desc="running mutants"):
            per_mutant[mutant.mutant_id] = mutant

            applied = _apply_mutant(solution_path, mutant)
            if not applied:
                solution_path.write_text(original_source, encoding="utf-8")
                continue

            incompetent, mutant_results = self._run_and_get_results(run_dir)
            solution_path.write_text(original_source, encoding="utf-8")

            if incompetent:
                #mutant has syntax/execution error
                incompetent_mutants.append(mutant.mutant_id)
                continue

            killed = False
            def _normalize(test_name: str) -> str:
                name = test_name.replace(".py", "").replace("::", ".").replace("/", ".")
                return name
            for test_name in ref_results:
                if test_name not in mutant_results:
                    continue
                if not _same_outcome(ref_results[test_name], mutant_results[test_name]):
                    killed = True
                    per_test_kills.setdefault(_normalize(test_name), []).append(mutant.mutant_id)

            if killed:
                killed_mutants.append(mutant.mutant_id)
            if not killed :
                print("alive mutant ")

        competent_mutants = len(mutants) - len(incompetent_mutants)
        mutation_score = len(killed_mutants) / competent_mutants if competent_mutants > 0 else 0.0

        return MutationReport(
            mutation_score=mutation_score,
            incompetent_mutants=incompetent_mutants,
            total_mutants = [mutant.mutant_id for mutant in mutants],
            killed_mutants=killed_mutants,
            per_test_kills=per_test_kills,
            per_mutant=per_mutant,
        )

    def _generate_mutants(self, solution_path: Path, run_dir: Path) -> list[MutantInfo]:
        if not CODE_MUTATE_BIN:
            raise EnvironmentError("CODE_MUTATE_BIN environment variable not set")

        result = self.sandbox.run(
            cmd=[CODE_MUTATE_BIN, "-t", str(solution_path)],
            cwd=run_dir,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            logger.warning("cm failed: %s", result.stderr)
            return []

        return _parse_mutants(result.stdout)

    def _run_and_get_results(self, run_dir: Path) -> dict[str, dict] | None:
        report_path = run_dir / ".pytest_report.json"

        result = self.sandbox.run(
            cmd=[
                "python", "-m", "pytest",
                str(run_dir),
                "-q",
                "--json-report",
                f"--json-report-file={str(report_path)}",
                "--json-report-indent=2",
            ],
            cwd=run_dir,
            timeout=self.timeout,
        )

        self.set_strict(False)
        test_results =  self._build_report(result=result, report_path=report_path)
        #_parse_json_output(report_path)

        incompetent = False
        if not test_results or not test_results.detailed_test_results :
            return True, {}


        if test_results.verdict not in [Verdict.PASSED, Verdict.FAILED] :
            incompetent = True
        return incompetent, {
            result.node_id: {
                "outcome": result.outcome,
                "message": result.message,
            }
            for result in test_results.detailed_test_results
        }
