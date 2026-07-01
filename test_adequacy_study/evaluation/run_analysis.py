"""
Full evaluation pipeline: analysis → minimisation → results.

This is the entry point for the entire evaluation.

Usage
-----
# Run everything defined in config.py
python run_analysis.py

# Run a specific subset
python run_analysis.py --fault-models gpt-5-mini --test-models gpt-5-mini gpt-4.1-mini --benchmarks ncb

"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from tqdm import tqdm

from test_adequacy_study.evaluation.helpers.data_models import AnalysisRecord
from test_adequacy_study.evaluation.helpers.read_faults import FaultLoader
from test_adequacy_study.evaluation.helpers.read_test_pool import TestPoolLoader
from test_adequacy_study.evaluation.analysis import FaultAnalyzer
from test_adequacy_study.file_utils import write_jsonl
from test_adequacy_study.providers.loader_provider import LoaderProvider
from test_adequacy_study.providers.program_builder_provider import ProgramBuilderProvider
from test_adequacy_study.providers.runner_provider import RunnerProvider
from test_adequacy_study.evaluation.config import (
    FAULT_MODELS,
    TEST_MODELS,
    BENCHMARKS,
    ANALYSIS_FILE_PATTERN,
    TESTS_FILE_PATTERN,
    BENCHMARK_VARIATION
)
from test_adequacy_study.runners.bigcodebench_coverage_runner import BCBCoverageRunner
from test_adequacy_study.runners.bigcodebench_mutation_runner import BCBMutationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)




def _test_models_tag(test_models: list[str] | None) -> str:
    """combines model ids"""
    if not test_models:
        return "all"
    return "+".join(sorted(test_models))


class Runner:
    """
    For each (fault_model, benchmark) pair, loads tests from all requested
    test_models merged into a single pool and runs the full analysis.

    To run a subset, modify FAULT_MODELS / TEST_MODELS / BENCHMARKS in config.py
    or pass them as CLI args.
    """

    def __init__(self, slice: str | None = None):
        self.analyzer = FaultAnalyzer()
        self.test_pool_loader = TestPoolLoader()
        self.fault_loader = FaultLoader()
        self.slice = slice

    def run(
        self,
        fault_models: list[str] | None = None,
        test_models:  list[str] | None = None,
        benchmarks:   list[str] | None = None,
    ) -> None:
        fault_models = fault_models or FAULT_MODELS
        test_models  = test_models  or TEST_MODELS
        benchmarks   = benchmarks   or BENCHMARKS

        for benchmark in benchmarks:

            #choose runner to run tests
            self.analyzer.assign_runner(RunnerProvider.get_for_test_generation(benchmark))
            self.analyzer.assign_builder(ProgramBuilderProvider().get(benchmark))
            if benchmark == "bcb" :
                self.analyzer.assign_coverage_runner(BCBCoverageRunner())
                self.analyzer.assign_mutation_runner(BCBMutationRunner())

            for fault_model in fault_models:
                logger.info("=" * 60)
                logger.info(
                    "Starting: benchmark=%s  fault_model=%s  test_models=%s",
                    benchmark, fault_model, _test_models_tag(test_models),
                )
                try:
                    self._run_one(benchmark, fault_model, test_models)
                except Exception as exc:
                    logger.error(
                        "Failed: benchmark=%s fault_model=%s test_models=%s — %s",
                        benchmark, fault_model, _test_models_tag(test_models), exc,
                        exc_info=True,
                    )

    def _run_one(self, benchmark: str, fault_model: str, test_models: list[str] | None) -> None:
        faults  = self.fault_loader.load_faults(model=fault_model, benchmark=benchmark)
        mutants = self.fault_loader.load_mutants(model=fault_model, benchmark=benchmark)

        if self.slice:
            idx, total = self.slice.split("/")
            idx, total = int(idx), int(total)
            faults = [f for i, f in enumerate(faults) if i % total == idx]

        #todo : change
        tasks = {str(t.task_id): t for t in LoaderProvider.get(benchmark, variation=BENCHMARK_VARIATION).load(no_tests=True)}

        tests_file = Path(TESTS_FILE_PATTERN.format(benchmark=benchmark, fault_model=fault_model))
        try :
            tests_file = next(tests_file.parent.glob(tests_file.name))
        except Exception as exc:
            logger.error("TestPoolLoader: file not found <UNK> %s", tests_file)
            return
        test_pool  = self.test_pool_loader.load_from_file(tests_file, test_models=test_models)

        tag = _test_models_tag(test_models)
        logger.info(
            "Loaded: %d faults | %d tasks | %d test pools | %d tasks with mutants",
            len(faults), len(tasks), len(test_pool), len(mutants),
        )

        if not faults:
            logger.warning(
                "No faults for fault_model=%s benchmark=%s, skipping",
                fault_model, benchmark,
            )
            return

        output_path = ANALYSIS_FILE_PATTERN.format(
            fault_model=fault_model, test_models=tag, benchmark=benchmark,
        )
        if self.slice:
            output_path = output_path.replace(".jsonl", f"_slice{self.slice.replace('/', '_')}.jsonl")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for fault in tqdm(faults, desc=f"{fault_model}/{tag}/{benchmark}"):
            task_id = str(fault["task_id"])


            task = tasks.get(task_id)
            if task is None:
                logger.warning("[%s] task not found, skipping", task_id)
                continue

            test_suites = test_pool.get(str(task_id))
            if not test_suites:
                logger.warning("[%s] no test pool, skipping", task_id)
                continue

            fault_mutants = mutants.get(str(task_id), [])
            if not fault_mutants:
                logger.warning("[%s] no mutants, skipping", task_id)
                continue

            for test_suite in test_suites:
                record: AnalysisRecord | None = self.analyzer.analyze(
                    fault=fault,
                    task=task,
                    test_file=test_suite['test_suite'],
                    test_model=test_suite["model_id"],
                    mutants=fault_mutants,
                    fault_model=fault_model,
                    benchmark=benchmark,
                )
                if record is not None:
                    write_jsonl(output_path, [record.to_dict()], append=True)
                    n_written += 1

        logger.info(
            "Done: benchmark=%s fault_model=%s test_models=%s — wrote %d records to %s",
            benchmark, fault_model, tag, n_written, output_path,
        )





def _banner(msg: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(msg)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Step 1 — Analysis
# ---------------------------------------------------------------------------

def run_analysis_step(fault_model: str, test_models: list[str], benchmark: str, slice: str | None = None) -> bool:

    _banner(f"STEP 1 — ANALYSIS | {fault_model} / {benchmark}")
    try:
        Runner(slice=slice).run(
            fault_models=[fault_model],
            test_models=test_models,
            benchmarks=[benchmark],
        )
        return True
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Step 2 — Minimisation
# ---------------------------------------------------------------------------

def run_minimisation_step(fault_model: str, benchmark: str) -> bool:
    from test_adequacy_study.evaluation.minimisation import MinimisationSimulator

    _banner(f"STEP 2 — MINIMISATION | {fault_model} / {benchmark}")

    analysis_file = ANALYSIS_FILE_PATTERN.format(
        fault_model=fault_model,
        benchmark=benchmark,
    )
    if not Path(analysis_file).exists():
        logger.error("Analysis file not found, skipping minimisation: %s", analysis_file)
        return False

    try:
        MinimisationSimulator().run(analysis_file=analysis_file)
        return True
    except Exception as exc:
        logger.error("Minimisation failed: %s", exc, exc_info=True)
        return False




def main():
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--fault-models",      nargs="*", default=None,
                        help="Override FAULT_MODELS from config.py")
    parser.add_argument("--test-models",       nargs="*", default=None,
                        help="Override TEST_MODELS from config.py. Omit to use all models.")
    parser.add_argument("--benchmarks",        nargs="*", default=None,
                        help="Override BENCHMARKS from config.py")
    parser.add_argument("--skip-analysis",     action="store_true",
                        help="Skip step 1 (analysis already done)")
    parser.add_argument("--skip-minimisation", action="store_true",
                        help="Skip step 2 (minimisation already done)")
    parser.add_argument("--slice",             type=str, default=None,
                        help="HPC slicing, e.g. '0/5' for job 0 of 5")
    args = parser.parse_args()

    fault_models = args.fault_models or FAULT_MODELS
    test_models  = args.test_models  or TEST_MODELS
    benchmarks   = args.benchmarks   or BENCHMARKS

    _banner("PIPELINE START")
    logger.info("fault_models      : %s", fault_models)
    logger.info("test_models       : %s", _test_models_tag(test_models))
    logger.info("benchmarks        : %s", benchmarks)
    logger.info("skip_analysis     : %s", args.skip_analysis)
    logger.info("skip_minimisation : %s", args.skip_minimisation)
    logger.info("skip_results      : %s", args.skip_results)

    failed  = []
    t_start = time.time()

    # steps 1 and 2 run per (fault_model, benchmark) pair
    for benchmark in benchmarks:
        for fault_model in fault_models:

            if not args.skip_analysis:
                ok = run_analysis_step(fault_model, test_models, benchmark, slice=args.slice or None)
                if not ok:
                    failed.append(f"analysis     | {fault_model}/{benchmark}")
                    continue  # no point minimising without analysis output

            if not args.skip_minimisation:
                ok = run_minimisation_step(fault_model, benchmark)
                if not ok:
                    failed.append(f"minimisation | {fault_model}/{benchmark}")


    _banner("PIPELINE COMPLETE")
    logger.info("Total time: %.1fs", time.time() - t_start)

    if failed:
        logger.warning("Failed steps:")
        for f in failed:
            logger.warning("  - %s", f)
    else:
        logger.info("All steps completed successfully.")


if __name__ == "__main__":
    main()