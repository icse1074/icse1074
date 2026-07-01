"""
rq2_curves.py

Core computation and plotting logic for RQ2 (budget / cost comparison:
mutation vs branch_coverage vs line_coverage vs random).

For each fault model, restricted to faults with score <= DIFFICULTY_THRESHOLD
(score from faults.jsonl), computes and plots FT as a function of a *fair
shared effort* level, comparing the criteria using the SAME number of tests
at every effort point. Results are separated by variation (OR, US) and a
merged (OR+US) version, mirroring RQ1's variation handling in core.py.

Algorithm
---------
1. Per task, min_pool_size = min(pool_size_mutation, pool_size_branch_coverage,
   pool_size_line_coverage, pool_size_random) -- pool_size taken as the
   mean length of that criterion's truncated_ runs for that task.
2. Effort grid e = 0, 1, ..., 100 (percent). For each task,
   n_tests(e) = round(e/100 * min_pool_size), the SAME n_tests applied to
   all criteria for that task -- this is what makes it a fair,
   equal-test-count comparison rather than an equal-%-of-own-pool comparison.
3. truncated_{criterion}_{metric} runs are CUMULATIVE (run[i] already
   reflects the outcome using tests 0..i combined). So at effort e with
   n_tests(e) tests selected:
     - if n_tests(e) == 0: value = 0
     - else: value = run[n_tests(e) - 1]
4. For each run index r (1..n_shuffles) and effort e: compute the GLOBAL
   ratio across all difficult tasks simultaneously --
     ratio(criterion, metric, e, r) = (# tasks where lookup == 1 at run r)
                                       / (total # difficult tasks)
5. The plotted curve value at effort e = mean over r of ratio(..., e, r).

Outputs (under OUTPUT_ROOT/report/<benchmark>/rq2/):
  - curves_<model>_OR.pdf       OR faults only
  - curves_<model>_US.pdf       US faults only
  - curves_<model>_MERGED.pdf   OR + US combined
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from test_adequacy_study.evaluation.config import LABELS

from test_adequacy_study.evaluation.rq1.core import load_with_variations, filter_by_difficulty, DIFFICULTY_THRESHOLD

logger = logging.getLogger(__name__)

EFFORT_GRID = list(range(0, 101))  # 0%, 1%, ..., 100%

CRITERIA = ["mutation", "branch_coverage", "line_coverage", "random"]
METRICS = ["trigger"]  # detection (FD) dropped: near-zero everywhere, no signal


# ---------------------------------------------------------------------------
# Per-task fair budget setup
# ---------------------------------------------------------------------------

def get_pool_size(fault_result: dict, criterion: str, metric: str) -> float | None:
    """Mean length of this criterion's truncated_ runs for this task."""
    runs = fault_result.get(f"truncated_{criterion}_{metric}", [])
    runs = [r for r in runs if len(r) > 0]
    if not runs:
        return None
    return float(np.mean([len(r) for r in runs]))


def get_min_pool_size(fault_result: dict, metric: str) -> float | None:
    """min pool size across all criteria for this task. Returns None if any
    criterion is missing (so this task can't be fairly compared across all
    of them) -- consistent with the 'include everything that has data'
    policy, criteria with no data simply exclude the task from the
    fair-budget comparison rather than guessing a value for them."""
    sizes = []
    for criterion in CRITERIA:
        size = get_pool_size(fault_result, criterion, metric)
        if size is None:
            return None
        sizes.append(size)
    return min(sizes)


# ---------------------------------------------------------------------------
# Per-run, per-effort, global ratio
# ---------------------------------------------------------------------------

def compute_curves_for_model(
    results: dict[str, dict],
    n_shuffles: int,
) -> dict[str, dict[str, np.ndarray]]:
    """
    criterion -> metric -> array of shape (len(EFFORT_GRID),) with the
    averaged-across-runs ratio at each effort point. Also tracks per-run
    std (for an optional variability band) and eligible task counts.
    """
    eligible_tasks: dict[str, dict[str, float]] = {}
    for task_id, fault_result in results.items():
        per_metric = {}
        for metric in METRICS:
            min_pool = get_min_pool_size(fault_result, metric)
            if min_pool is not None and min_pool > 0:
                per_metric[metric] = min_pool
        if per_metric:
            eligible_tasks[task_id] = per_metric

    n_total_tasks = {
        metric: sum(1 for t in eligible_tasks.values() if metric in t)
        for metric in METRICS
    }

    ratios = {
        c: {m: np.zeros((n_shuffles, len(EFFORT_GRID))) for m in METRICS}
        for c in CRITERIA
    }

    for metric in METRICS:
        total = n_total_tasks[metric]
        if total == 0:
            continue

        for criterion in CRITERIA:
            hits = np.zeros((n_shuffles, len(EFFORT_GRID)))

            for task_id, per_metric in eligible_tasks.items():
                if metric not in per_metric:
                    continue
                min_pool_size = per_metric[metric]
                fault_result = results[task_id]
                runs = fault_result.get(f"truncated_{criterion}_{metric}", [])
                runs = [r for r in runs if len(r) > 0]
                if len(runs) < n_shuffles:
                    continue

                for r in range(n_shuffles):
                    run = runs[r]
                    for e_idx, e in enumerate(EFFORT_GRID):
                        n_tests = round(e / 100 * min_pool_size)
                        if n_tests == 0:
                            value = 0.0
                        else:
                            pos = min(n_tests - 1, len(run) - 1)
                            value = float(run[pos])
                        if value > 0:
                            hits[r, e_idx] += 1

            ratios[criterion][metric] = hits / total

    curves = {c: {m: ratios[c][m].mean(axis=0) for m in METRICS} for c in CRITERIA}
    curves["_std"] = {c: {m: ratios[c][m].std(axis=0) for m in METRICS} for c in CRITERIA}
    curves["_n_tasks"] = n_total_tasks

    return curves


# ---------------------------------------------------------------------------
# Per-model pipeline (OR / US / MERGED, mirrors core.process_model)
# ---------------------------------------------------------------------------

def process_model_rq2(
    fault_model: str,
    benchmark: str,
    selection_template: str,
    faults_template: str,
    n_shuffles: int,
) -> dict | None:
    """
    Load and process OR, US, and MERGED variations for a fault model.

    Returns dict with keys "OR", "US", "MERGED", each containing:
        {
            "fault_model": str,
            "variation": str,
            "benchmark": str,
            "curves": {criterion: {metric: np.ndarray}, "_std": ..., "_n_tasks": ...},
            "n_difficult": int,
        }
    """
    difficulty_scores_data = load_with_variations(
        faults_template, fault_model, benchmark, "faults"
    )
    results_data = load_with_variations(
        selection_template, fault_model, benchmark, "selection"
    )

    if not results_data or not difficulty_scores_data:
        logger.warning("Missing files for %s", fault_model)
        return None

    model_results = {}
    all_combined_results = {}
    all_combined_difficulties = {}

    for variation in ["OR", "US"]:
        difficulty_scores = difficulty_scores_data[variation]
        results = results_data[variation]

        if not results:
            logger.warning("No results for %s (%s)", fault_model, variation)
            continue

        results = filter_by_difficulty(results, difficulty_scores, DIFFICULTY_THRESHOLD)

        logger.info(
            "[%s] (%s) %d/%d minimisation tasks meet score <= %s threshold",
            fault_model, variation, len(results), len(difficulty_scores), DIFFICULTY_THRESHOLD,
        )

        curves = compute_curves_for_model(results, n_shuffles)
        n_tasks_str = ", ".join(f"{m}={curves['_n_tasks'][m]}" for m in METRICS)
        logger.info(
            "[%s] (%s) eligible tasks for fair-budget comparison: %s",
            fault_model, variation, n_tasks_str,
        )

        model_results[variation] = {
            "fault_model": fault_model,
            "variation": variation,
            "benchmark": benchmark,
            "curves": curves,
            "n_difficult": len(results),
        }

        all_combined_results.update(results)
        all_combined_difficulties.update(difficulty_scores)

    if all_combined_results:
        results_merged = filter_by_difficulty(
            all_combined_results, all_combined_difficulties, DIFFICULTY_THRESHOLD
        )

        logger.info(
            "[%s] (MERGED) %d/%d minimisation tasks meet score <= %s threshold",
            fault_model, len(results_merged), len(all_combined_difficulties), DIFFICULTY_THRESHOLD,
        )

        curves_merged = compute_curves_for_model(results_merged, n_shuffles)
        n_tasks_str = ", ".join(f"{m}={curves_merged['_n_tasks'][m]}" for m in METRICS)
        logger.info(
            "[%s] (MERGED) eligible tasks for fair-budget comparison: %s",
            fault_model, n_tasks_str,
        )

        model_results["MERGED"] = {
            "fault_model": fault_model,
            "variation": "MERGED",
            "benchmark": benchmark,
            "curves": curves_merged,
            "n_difficult": len(results_merged),
        }

    return model_results


# ---------------------------------------------------------------------------
# Output: plots
# ---------------------------------------------------------------------------

def plot_model_curves(
    model_result: dict,
    output_dir: Path,
    show_variability_band: bool = False,
) -> None:
    curves = model_result["curves"]
    fault_model = model_result["fault_model"]
    variation = model_result["variation"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(CRITERIA)))
    x = np.array(EFFORT_GRID)
    linestyles = ['-', '-', '--', '-']

    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 2.4))
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        for i, (criterion, color) in enumerate(zip(CRITERIA, colors)):
            mean_curve = curves[criterion][metric]
            label = f"{LABELS.get(criterion, criterion)}"

            ax.plot(
                x, mean_curve, label=label, color=color, linewidth=1.5,
                linestyle=linestyles[i % len(linestyles)],
                markersize=2.5, markevery=10,
            )

            if show_variability_band:
                std_curve = curves["_std"][criterion][metric]
                ax.fill_between(
                    x, mean_curve - std_curve, mean_curve + std_curve,
                    color=color, alpha=0.15,
                )

        ax.set_xlabel("Effort in % of analyzed tests", fontsize=9)
        ax.set_ylabel(f"{metric} ratio", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.8, handlelength=1.5,
                   borderpad=0.3, labelspacing=0.3)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"curves_{fault_model.replace('/', '_')}_{variation}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved RQ2 curves for %s (%s) to %s", fault_model, variation, path)


def plot_all_variations(
    all_model_results: list[dict],
    output_dir: Path,
    show_variability_band: bool = False,
) -> None:
    for mr_dict in all_model_results:
        for variation, mr in mr_dict.items():
            plot_model_curves(mr, output_dir, show_variability_band=show_variability_band)