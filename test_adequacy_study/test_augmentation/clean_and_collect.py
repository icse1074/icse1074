"""
clean_and_collect.py

Chains the cleaner and fault collector into a single pipeline:

  1. Clean augmented test suites (dedup, validate against canonical)
  2. Run generations from batch against clean suites
  3. Write faults to output

Modes (--step):
  all      : run clean then collect (default)
  clean    : run clean only, writes clean file
  collect  : run collect only, requires --clean-input

Slicing (--slice INDEX/TOTAL):
  Partitions the augmented tests so independent HPC jobs each process a
  disjoint subset. Tasks are assigned by (enumeration_index % TOTAL == INDEX),
  matching the same convention used in FaultCollectionPipeline.
  The slice is applied during collect (and during clean when running 'all'),
  so each job reads the full clean file but only executes its own slice.

Usage:
    # Full pipeline
    python clean_and_collect.py \
        --augmented-tests output/tests/augmented/augmented_bcb.jsonl \
        --batch-id <BATCH_ID> --model gpt-4.1-mini --benchmark bcb \
        -o output/faults/faults_bcb_augmented.jsonl

    # Clean only (run once, share the output across slice jobs)
    python clean_and_collect.py --step clean \
        --augmented-tests output/tests/augmented/augmented_bcb.jsonl \
        --benchmark bcb --clean-output output/tests/augmented/augmented_bcb_clean.jsonl

    # Collect only — slice 3 of 40 parallel jobs
    python clean_and_collect.py --step collect \
        --clean-input output/tests/augmented/augmented_bcb_clean.jsonl \
        --batch-id <BATCH_ID> --model gpt-4.1-mini --benchmark bcb \
        --slice 3/40 \
        -o output/faults/faults_bcb_augmented_3_40.jsonl

    # Full pipeline with slicing (clean + collect in one shot, per-slice output)
    python clean_and_collect.py \
        --augmented-tests output/tests/augmented/augmented_bcb.jsonl \
        --batch-id <BATCH_ID> --model gpt-4.1-mini --benchmark bcb \
        --slice 3/40 \
        -o output/faults/faults_bcb_augmented_3_40.jsonl
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from test_adequacy_study.providers.program_builder_provider import ProgramBuilderProvider

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_slice(slice_str: str) -> tuple[int, int]:
    """Parse 'INDEX/TOTAL' and validate bounds."""
    try:
        idx, total = slice_str.split("/")
        idx, total = int(idx), int(total)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--slice must be in INDEX/TOTAL format (e.g. 3/40), got: {slice_str!r}"
        )
    if total < 1:
        raise ValueError(f"TOTAL must be >= 1, got {total}")
    if not (0 <= idx < total):
        raise ValueError(f"INDEX must be in [0, TOTAL), got {idx}/{total}")
    return idx, total


def _apply_slice(records: list, slice_str: str) -> list:
    """Return the subset of records belonging to this slice."""
    idx, total = _parse_slice(slice_str)
    subset = [r for i, r in enumerate(records) if i % total == idx]
    logger.info("Slice %d/%d: %d / %d records selected", idx, total, len(subset), len(records))
    return subset

from test_adequacy_study.types.benchmark_variation import BenchmarkVariation

def build_infrastructure(benchmark: str, benchmark_variation = BenchmarkVariation.NONE):
    from test_adequacy_study.providers.loader_provider import LoaderProvider
    from test_adequacy_study.providers.runner_provider import RunnerProvider

    loader      = LoaderProvider.get_with_pytests(benchmark, benchmark_variation)
    task_lookup = {str(task.task_id): task for task in loader.load()}
    runner      = RunnerProvider.get_for_test_generation(benchmark, os.getenv("WORK_DIR"))
    runner.timeout = 180
    builder     = ProgramBuilderProvider.get(benchmark)

    return task_lookup, runner, builder


def load_batch(model: str, batch_id: str) -> dict[str, list[str]]:
    from test_adequacy_study.generators.test_generator import TestGenerator
    from test_adequacy_study.services.batch_processing_service import BatchProcessingService

    print("check model here")
    generator = TestGenerator(model=model)
    if generator.llm.model.startswith("claude"):
        return BatchProcessingService.get_batch_results_lookup_from_anthropic_api(generator.llm, batch_id)
    else:
        return BatchProcessingService.get_batch_results_lookup_from_openai_api(generator.llm, batch_id)


def load_generations_file(generations_path: str) -> dict[str, list[str]]:
    """
    Load generations from a JSONL file produced by FaultCollectionPipeline.
    Each record has {task_id, completion, completion_index}.
    Returns {task_id: [completion, ...]} ordered by completion_index,
    matching the dict[str, list[str]] format expected by collect_faults.
    """
    from test_adequacy_study.file_utils import read_jsonl

    records: dict[str, list[tuple[int, str]]] = {}
    for rec in read_jsonl(generations_path):
        task_id = rec.get("task_id")
        completion = rec.get("completion")
        completion_index = rec.get("completion_index", 0)
        if task_id and completion is not None:
            records.setdefault(task_id, []).append((completion_index, completion))

    # Sort by completion_index to preserve original generation order
    return {
        task_id: [c for _, c in sorted(completions)]
        for task_id, completions in records.items()
    }


def run(
    step: str,
    benchmark: str,
    augmented_tests_path: str  = None,
    batch_id: str  = None,
    model: str  = None,
    generations_path: str = None,
    output_path: str  = None,
    keep_clean: bool = False,
    clean_output_path: str  = None,
    clean_input_path: str  = None,
    slice_str: str = None,
    benchmark_variation = BenchmarkVariation.NONE,
):
    from test_cleaner import clean_augmented_tests
    from fault_collector import collect_faults

    task_lookup, runner, builder = build_infrastructure(benchmark, benchmark_variation=benchmark_variation)

    if step in ("all", "clean"):
        assert augmented_tests_path, "--augmented-tests is required for clean step"

        if clean_output_path is None:
            if step == "clean" or keep_clean:
                p = Path(augmented_tests_path)
                clean_output_path = str(p.parent / f"{p.stem}_clean{p.suffix}")
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
                clean_output_path = tmp.name
                tmp.close()

        logger.info("Step: Cleaning augmented tests → %s", clean_output_path)
        clean_augmented_tests(
            input_path  = augmented_tests_path,
            output_path = clean_output_path,
            benchmark   = benchmark,
            runner      = runner,
            builder     = builder,
            task_lookup = task_lookup,
        )

        if step == "clean":
            logger.info("Clean complete. Output: %s", clean_output_path)
            return

    if step in ("all", "collect"):
        assert batch_id or generations_path, \
            "One of --batch-id or --generations-path is required for collect step"
        assert not (batch_id and generations_path), \
            "--batch-id and --generations-path are mutually exclusive"
        if batch_id:
            assert model, "--model is required when using --batch-id"
        assert output_path, "--output is required for collect step"

        # For collect-only, use the provided clean input
        if step == "collect":
            assert clean_input_path, "--clean-input is required for collect-only step"
            clean_output_path = clean_input_path

        if batch_id:
            logger.info("Step: Loading batch %s", batch_id)
            batch_lookup = load_batch(model, batch_id)
        else:
            logger.info("Step: Loading generations from file %s", generations_path)
            batch_lookup = load_generations_file(generations_path)
            logger.info("Loaded generations for %d tasks", len(batch_lookup))

        # Apply slice to the augmented tests so each HPC job handles a
        # disjoint subset. We read, filter, write to a temp file, then pass
        # that through to collect_faults — keeping collect_faults unaware of
        # slicing and avoiding any changes to its interface.
        collect_input_path = clean_output_path
        slice_tmp_path = None
        if slice_str:
            from test_adequacy_study.file_utils import read_jsonl, write_jsonl

            all_records = read_jsonl(clean_output_path)
            sliced_records = _apply_slice(all_records, slice_str)

            slice_tmp = tempfile.NamedTemporaryFile(
                suffix=".jsonl", delete=False,
                prefix=f"slice_{slice_str.replace('/', '_')}_"
            )
            slice_tmp_path = slice_tmp.name
            slice_tmp.close()

            write_jsonl(slice_tmp_path, sliced_records)
            collect_input_path = slice_tmp_path
            logger.info("Slice temp file: %s", slice_tmp_path)

        logger.info("Step: Collecting faults → %s", output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        n_faults = collect_faults(
            batch_lookup         = batch_lookup,
            augmented_tests_path = collect_input_path,
            output_path          = output_path,
            runner               = runner,
            builder              = builder,
            task_lookup          = task_lookup,
        )

        # Clean up temp files
        if slice_tmp_path:
            Path(slice_tmp_path).unlink(missing_ok=True)
            logger.info("Removed slice temp file")

        if step == "all" and not keep_clean:
            Path(clean_output_path).unlink(missing_ok=True)
            logger.info("Removed temp clean file")

        logger.info("Collect complete. %d faults written to %s", n_faults, output_path)


if __name__ == "__main__":
    #sys.set_int_max_str_digits(0)
    sys.set_int_max_str_digits(100000000)
    parser = argparse.ArgumentParser(description="Clean augmented tests and/or collect faults.")
    parser.add_argument("--step",            default="all", choices=["all", "clean", "collect"],
                        help="'all' runs clean then collect (default). 'clean' runs clean only. 'collect' runs collect only.")
    parser.add_argument("--augmented-tests", default=None, help="Path to augmented tests .jsonl (required for clean/all).")
    parser.add_argument("--clean-input",     default=None, help="Path to pre-cleaned tests .jsonl (required for collect-only).")
    parser.add_argument("--clean-output",    default=None, help="Explicit path for the clean file.")
    parser.add_argument("--batch-id",          default=None, help="Completed batch job ID (mutually exclusive with --generations-path).")
    parser.add_argument("--generations-path",  default=None, help="Path to generations .jsonl file (mutually exclusive with --batch-id).")
    parser.add_argument("--model",             default=None, help="Model name used for the batch (required when using --batch-id).")
    parser.add_argument("--benchmark",       required=True, choices=["he", "bcb", "ncb", "mbpp"])
    parser.add_argument("--output", "-o",    default=None, help="Output faults .jsonl path (required for collect/all).")
    parser.add_argument("--keep-clean",      action="store_true",
                        help="Keep the intermediate clean file when running 'all'.")
    parser.add_argument('-bv', '--benchmark-variation', type=BenchmarkVariation, choices=list(BenchmarkVariation), default=BenchmarkVariation.NONE, required=False, help='Specify whether a benchmark variation should be used')
    parser.add_argument("--slice",           default=None,
                        help="Slice of tasks in INDEX/TOTAL format (e.g. 3/40). "
                             "Each job processes every TOTAL-th record starting at INDEX. "
                             "Tip: run '--step clean' once first and share the clean file across jobs.")
    args = parser.parse_args()

    if args.augmented_tests:
        assert os.path.exists(args.augmented_tests), f"File not found: {args.augmented_tests}"
    if args.clean_input:
        assert os.path.exists(args.clean_input), f"File not found: {args.clean_input}"
    if args.generations_path:
        assert os.path.exists(args.generations_path), f"File not found: {args.generations_path}"
    if args.slice:
        _parse_slice(args.slice)  # validate early, before any work starts

    run(
        step                 = args.step,
        benchmark            = args.benchmark,
        augmented_tests_path = args.augmented_tests,
        batch_id             = args.batch_id,
        model                = args.model,
        generations_path     = args.generations_path,
        output_path          = args.output,
        keep_clean           = args.keep_clean,
        clean_output_path    = args.clean_output,
        clean_input_path     = args.clean_input,
        slice_str            = args.slice,
        benchmark_variation=args.benchmark_variation
    )
