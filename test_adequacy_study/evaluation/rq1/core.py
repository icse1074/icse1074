from __future__ import annotations

import json
import logging
import os.path

import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)

DIFFICULTY_THRESHOLD = 0.25
ALPHA = 0.05
CRITERIA = ["mutation", "branch_coverage", "line_coverage"]
METRICS = ["trigger", "detection"]


def load_difficulty_scores(faults_file: str) -> dict[str, float | None]:
    """Load difficulty scores from faults.jsonl."""
    scores: dict[str, float | None] = {}
    try:
        with open(faults_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                scores[r.get("task_id")] = r.get("score")
    except FileNotFoundError:
        logger.debug("File not found: %s", faults_file)
    return scores


def load_selection_results(path: str) -> dict[str, dict]:
    """Load selection results from minimisation_results.jsonl."""
    results = {}
    try:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                task_id = record.pop("task_id")
                record.pop("pool_size", None)
                record.pop("has_branches", None)
                record.pop("criteria", None)
                keys_to_skip = [k for k in record.keys() if "selected_tests" in k]
                for k in keys_to_skip:
                    record.pop(k, None)

                results[task_id] = {
                    k: [np.array(run) for run in v] for k, v in record.items()
                }
    except FileNotFoundError:
        logger.debug("File not found: %s", path)
    return results


def load_with_variations(
        template: str,
        fault_model: str,
        benchmark: str,
        file_type: str
) -> dict[str, dict]:
    """
    Load both OR and US variations.
    US task_ids get "_us" suffix to keep them separate.

    Returns: {"OR": or_data, "US": us_data_with_suffix}
    """
    variations_data = {}

    # Load OR
    or_path = template.format(fault_model=fault_model)
    if file_type == "faults":
        or_data = load_difficulty_scores(or_path)
    else:
        or_data = load_selection_results(or_path)
    variations_data["OR"] = or_data
    logger.debug("[%s] Loaded OR: %d items", fault_model, len(or_data))

    # Load US with "_us" suffix

    if file_type == "faults":
        us_path = or_path.replace(fault_model, f"{fault_model}/us")
        us_data = load_difficulty_scores(us_path)
    else:
        us_path = or_path.replace(benchmark, f"{benchmark}/us")
        if not os.path.exists(us_path):
            us_path = or_path.replace(fault_model, f"{fault_model}/us")
        us_data = load_selection_results(us_path)

    # Append "_us" suffix to US task_ids to keep them separate from OR
    us_data_suffixed = {f"{k}_us": v for k, v in us_data.items()}
    variations_data["US"] = us_data_suffixed
    logger.debug("[%s] Loaded US: %d items (with _us suffix)", fault_model, len(us_data_suffixed))

    return variations_data



def filter_by_difficulty(
        results: dict[str, dict],
        difficulty_scores: dict[str, float | None],
        threshold: float = DIFFICULTY_THRESHOLD
) -> dict[str, dict]:
    """Filter results to only include faults with difficulty score <= threshold."""
    difficult_ids = [
        tid for tid, s in difficulty_scores.items()
        if s is not None and s <= threshold
    ]
    return {tid: r for tid, r in results.items() if str(tid) in difficult_ids}


def collect_final_values(results: dict[str, dict]) -> dict[str, dict[str, list[float]]]:
    """
    For each criterion/metric, collect final values (mean of run[-1] per task).
    Returns: {criterion: {metric: [values]}}
    """
    data: dict[str, dict[str, list[float]]] = {c: {m: [] for m in METRICS} for c in CRITERIA}

    for task_id, fault_result in results.items():
        for criterion in CRITERIA:
            for metric in METRICS:
                runs = fault_result.get(f"full_{criterion}_{metric}", [])
                final_values = [float(run[-1]) for run in runs if len(run) > 0]
                if final_values:
                    data[criterion][metric].append(float(np.mean(final_values)))

    return data


def collect_criterion_values(
        results: dict[str, dict]
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Collect per-task values: {task_id: {criterion: {metric: value}}}
    """
    per_task: dict[str, dict[str, dict[str, float]]] = {}

    for task_id, fault_result in results.items():
        entry: dict[str, dict[str, float]] = {c: {} for c in CRITERIA}
        for criterion in CRITERIA:
            for metric in METRICS:
                runs = fault_result.get(f"full_{criterion}_{metric}", [])
                final_values = [float(run[-1]) for run in runs if len(run) > 0]
                if final_values:
                    entry[criterion][metric] = float(np.mean(final_values))
        per_task[task_id] = entry

    return per_task



def vargha_delaney(vals1: np.ndarray, vals2: np.ndarray) -> float:
    """Compute Vargha-Delaney A12 effect size."""
    n = len(vals1)
    if n == 0:
        return 0.5
    return sum(
        1.0 if v1 > v2 else 0.5 if v1 == v2 else 0.0
        for v1, v2 in zip(vals1, vals2)
    ) / n


def pairwise_stats(per_task: dict[str, dict[str, dict[str, float]]]) -> dict:
    """Compute pairwise Wilcoxon + Vargha-Delaney stats for each metric."""
    pairs = [
        (c1, c2)
        for i, c1 in enumerate(CRITERIA)
        for c2 in CRITERIA[i + 1:]
    ]

    stats = {m: {} for m in METRICS}
    for metric in METRICS:
        for c1, c2 in pairs:
            paired = [
                (entry[c1][metric], entry[c2][metric])
                for entry in per_task.values()
                if metric in entry[c1] and metric in entry[c2]
            ]
            if not paired:
                continue

            vals1 = np.array([p[0] for p in paired])
            vals2 = np.array([p[1] for p in paired])

            try:
                _, p_val = wilcoxon(vals1, vals2)
            except ValueError:
                p_val = 1.0

            stats[metric][f"{c1}_vs_{c2}"] = {
                "p_value": round(float(p_val), 6),
                "a12": round(vargha_delaney(vals1, vals2), 4),
                "avg_diff": round(float(np.mean(vals1 - vals2)), 4),
                "significant": p_val < ALPHA,
                "n_faults": len(paired),
            }

    return stats


def process_model(
        fault_model: str,
        benchmark: str,
        selection_template: str,
        faults_template: str
) -> dict | None:
    """
    Load and process both OR and US variations for a fault model.

    Returns:
        dict with keys "OR", "US", "MERGED", each containing:
        {
            "fault_model": str,
            "variation": str,
            "benchmark": str,
            "data": {criterion: {metric: [values]}},
            "per_task": {task_id: {criterion: {metric: value}}},
            "stats": {metric: {pair: stats}},
            "n_difficult": int,
        }
    """
    # Load both OR and US
    difficulty_scores_data = load_with_variations(
        faults_template, fault_model, benchmark, "faults"
    )
    results_data = load_with_variations(
        selection_template, fault_model, benchmark, "selection"
    )

    if not results_data or not difficulty_scores_data:
        logger.warning("Missing files for %s", fault_model)
        return None

    # Process each variation separately
    model_results = {}
    all_combined_results = {}
    all_combined_difficulties = {}

    for variation in ["OR", "US"]:
        if variation == "US":
            print("yay")
        difficulty_scores = difficulty_scores_data[variation]
        results = results_data[variation]

        if not results:
            logger.warning("No results for %s (%s)", fault_model, variation)
            continue

        # Filter by difficulty
        results = filter_by_difficulty(results, difficulty_scores)

        logger.info(
            "[%s] (%s) %d/%d minimisation tasks meet score <= %s threshold",
            fault_model, variation, len(results), len(difficulty_scores), DIFFICULTY_THRESHOLD,
        )

        data = collect_final_values(results)
        per_task = collect_criterion_values(results)
        stats = pairwise_stats(per_task)

        model_results[variation] = {
            "fault_model": fault_model,
            "variation": variation,
            "benchmark": benchmark,
            "data": data,
            "per_task": per_task,
            "stats": stats,
            "n_difficult": len(results),
        }

        # Accumulate for merged results
        all_combined_results.update(results)
        all_combined_difficulties.update(difficulty_scores)

    # Create merged results (OR + US combined)
    if all_combined_results:
        results_merged = filter_by_difficulty(all_combined_results, all_combined_difficulties)

        logger.info(
            "[%s] (MERGED) %d/%d minimisation tasks meet score <= %s threshold",
            fault_model, len(results_merged), len(all_combined_difficulties), DIFFICULTY_THRESHOLD,
        )

        data_merged = collect_final_values(results_merged)
        per_task_merged = collect_criterion_values(results_merged)
        stats_merged = pairwise_stats(per_task_merged)

        model_results["MERGED"] = {
            "fault_model": fault_model,
            "variation": "MERGED",
            "benchmark": benchmark,
            "data": data_merged,
            "per_task": per_task_merged,
            "stats": stats_merged,
            "n_difficult": len(results_merged),
        }

    return model_results