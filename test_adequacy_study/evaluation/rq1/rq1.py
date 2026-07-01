"""
RQ1 — Criteria comparison (mutation vs branch_coverage vs line_coverage).

Compares three test-selection criteria on Fault Trigger (FT) and Fault Detection (FD),
with results separated by variation (OR, US) and a merged summary.

Outputs:
  - Boxplots per model/variation: boxplots_{model}_{variation}.pdf
  - Three summary CSVs:
    - rq1_summary_OR.csv     (OR faults only)
    - rq1_summary_US.csv     (US faults only)
    - rq1_summary_MERGED.csv (OR + US combined)
"""
from __future__ import annotations

import logging
from pathlib import Path

from test_adequacy_study.evaluation.config import (
    SELECTION_FILE_PATTERN,
    FAULTS_FILE_PATTERN,
    FAULT_MODELS,
    LABELS,
    OUTPUT_ROOT,
)

from core import process_model
from rq1_distribution import plot_all_variations
from rq1_summary import write_all_summary_csvs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK = "bcb"
REPORT_DIR = Path(OUTPUT_ROOT) / "report"/ BENCHMARK / "rq1"


def main(fault_models: list[str] | None = None, benchmark: str = BENCHMARK) -> None:
    """
    Main RQ1 pipeline: load models, generate plots and summaries.

    Args:
        fault_models: list of fault model names (defaults to FAULT_MODELS from config)
        benchmark: benchmark name (default: BENCHMARK)
    """
    fault_models = fault_models or FAULT_MODELS

    # Get templates from config
    selection_template = SELECTION_FILE_PATTERN.replace("{fault_model}", "{fault_model}")
    faults_template = FAULTS_FILE_PATTERN.replace("{fault_model}", "{fault_model}")

    model_results_list = []

    for fault_model in fault_models:
        logger.info("=" * 60)
        logger.info("RQ1 for fault_model=%s benchmark=%s", fault_model, benchmark)

        mr = process_model(
            fault_model,
            benchmark,
            selection_template,
            faults_template
        )
        if mr is None:
            continue

        model_results_list.append(mr)

    if not model_results_list:
        logger.warning("No RQ1 results produced — check file paths in config.")
        return

    # Generate plots and summaries
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating boxplots...")
    plot_all_variations(model_results_list, REPORT_DIR, LABELS)

    logger.info("Generating summary CSVs...")
    write_all_summary_csvs(model_results_list, REPORT_DIR)

    logger.info("Done. Outputs in %s", REPORT_DIR)


if __name__ == "__main__":
    main()