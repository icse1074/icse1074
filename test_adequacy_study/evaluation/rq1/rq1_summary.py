from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from core import CRITERIA, METRICS

logger = logging.getLogger(__name__)


def write_summary_csv_by_variation(
        all_model_results: list[dict],
        output_dir: Path,
        variation: str
) -> None:
    """
    Write summary CSV for a specific variation (OR, US, or MERGED).

    Args:
        all_model_results: list of dicts with OR/US/MERGED keys
        output_dir: directory to save CSV
        variation: "OR", "US", or "MERGED"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rq1_summary_{variation}.csv"

    columns = [f"{c}_{m}" for c in CRITERIA for m in METRICS]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fault_model", "n_difficult"] + columns)

        for mr_dict in all_model_results:
            if variation not in mr_dict:
                continue

            mr = mr_dict[variation]
            row = [mr["fault_model"], mr["n_difficult"]]
            for criterion in CRITERIA:
                for metric in METRICS:
                    vals = mr["data"][criterion][metric]
                    row.append(round(float(np.mean(vals)), 4) if vals else "N/A")
            writer.writerow(row)

    logger.info("Saved RQ1 summary CSV for %s to %s", variation, path)


def write_stats_csv(
        all_model_results: list[dict],
        output_dir: Path
) -> None:
    """
    Write pairwise statistical tests (Wilcoxon + Vargha-Delaney) for all variations.

    One row per (variation, model, metric, pair).
    Output columns: variation, fault_model, metric, pair, p_value, a12, avg_diff, significant, n_faults

    Args:
        all_model_results: list of dicts with OR/US/MERGED keys
        output_dir: directory to save CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rq1_stats.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variation", "fault_model", "metric", "pair",
            "p_value", "a12", "avg_diff", "significant", "n_faults"
        ])

        for mr_dict in all_model_results:
            for variation, mr in mr_dict.items():
                for metric in METRICS:
                    if metric not in mr["stats"]:
                        continue
                    for pair, result in mr["stats"][metric].items():
                        writer.writerow([
                            variation,
                            mr["fault_model"],
                            metric,
                            pair,
                            result["p_value"],
                            result["a12"],
                            result["avg_diff"],
                            result["significant"],
                            result["n_faults"],
                        ])

    logger.info("Saved RQ1 statistical tests CSV to %s", path)


def write_all_summary_csvs(
        all_model_results: list[dict],
        output_dir: Path
) -> None:
    """
    Generate all summary CSVs: three variation summaries + pairwise stats.

    Outputs:
        - rq1_summary_OR.csv      (means for OR faults)
        - rq1_summary_US.csv      (means for US faults)
        - rq1_summary_MERGED.csv  (means for all faults)
        - rq1_stats.csv           (pairwise Wilcoxon + Vargha-Delaney)

    Args:
        all_model_results: list of dicts with OR/US/MERGED keys
        output_dir: directory to save CSVs
    """
    for variation in ["OR", "US", "MERGED"]:
        write_summary_csv_by_variation(all_model_results, output_dir, variation)

    write_stats_csv(all_model_results, output_dir)
