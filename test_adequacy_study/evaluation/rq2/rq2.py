"""
RQ2 — Budget / cost comparison (mutation vs branch_coverage vs line_coverage
vs random).

Compares test-selection criteria on Fault Trigger (FT) as a function of a
fair shared effort level, with results separated by variation (OR, US) and
a merged summary -- mirrors RQ1's variation handling.

See rq2_curves.py for the computation/plotting logic and a detailed
description of the fair-budget algorithm.

Outputs:
  - curves_<model>_OR.pdf       OR faults only
  - curves_<model>_US.pdf       US faults only
  - curves_<model>_MERGED.pdf   OR + US combined
"""
from __future__ import annotations

import logging
from pathlib import Path

from test_adequacy_study.evaluation.config import (
    SELECTION_FILE_PATTERN,
    FAULTS_FILE_PATTERN,
    FAULT_MODELS,
    OUTPUT_ROOT,
)

from rq2_curves import process_model_rq2, plot_all_variations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK = "he"
REPORT_DIR = Path(OUTPUT_ROOT) / "report" / BENCHMARK / "rq2"


def main(
    fault_models: list[str] | None = None,
    benchmark: str = BENCHMARK,
    n_shuffles: int = 100,
    show_variability_band: bool = False,
) -> None:
    """
    Main RQ2 pipeline: load models, compute fair-budget curves, plot.

    Args:
        fault_models: list of fault model names (defaults to FAULT_MODELS from config)
        benchmark: benchmark name (default: BENCHMARK)
        n_shuffles: must match how many runs each task/criterion actually
            has. Tasks/criteria with fewer than n_shuffles runs are skipped
            for that criterion rather than silently truncated.
        show_variability_band: whether to shade +/- 1 std across runs.
    """
    fault_models = fault_models or FAULT_MODELS

    # Get templates from config
    selection_template = SELECTION_FILE_PATTERN.replace("{fault_model}", "{fault_model}")
    faults_template = FAULTS_FILE_PATTERN.replace("{fault_model}", "{fault_model}")

    model_results_list = []

    for fault_model in fault_models:
        logger.info("=" * 60)
        logger.info("RQ2 for fault_model=%s benchmark=%s", fault_model, benchmark)

        mr = process_model_rq2(
            fault_model,
            benchmark,
            selection_template,
            faults_template,
            n_shuffles,
        )
        if mr is None:
            continue

        model_results_list.append(mr)

    if not model_results_list:
        logger.warning("No RQ2 results produced — check file paths in config.")
        return

    # Generate plots
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating curve plots...")
    plot_all_variations(model_results_list, REPORT_DIR, show_variability_band=show_variability_band)

    logger.info("Done. Outputs in %s", REPORT_DIR)


if __name__ == "__main__":
    print(520+ 3925+ 1530+ 91 )
    #main()