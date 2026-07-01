"""
Report: FD/FT comparison between base and injected suites, plus test quality
breakdown (correct / incorrect / insufficient), per benchmark and per model.

Reads {output_folder}/rq3/{benchmark}/{model}/{variation}/analysis_fd_ft.jsonl
  - us: .../rq3/{benchmark}/{model}/us/analysis_fd_ft.jsonl
  - or: .../rq3/{benchmark}/{model}/analysis_fd_ft.jsonl   (no subfolder)

Each record is one fault with "base" and "injected" sub-dicts, each containing
per_test_ft, per_test_fd, per_test_quality (already aligned per fault — no
before/after merge needed, since both suites' results live in the same record).

A fault is "detected" (FD) / "triggered" (FT) if ANY test in its per_test_fd /
per_test_ft fired (value == 1) — same convention as the rest of the pipeline.

Quality counts (correct / incorrect / insufficient) are pooled per test,
summed across all faults in scope.

Three views, per benchmark per model:
  1. Combined  — us + or pooled together
  2. US alone
  3. OR alone

Usage
-----
Edit OUTPUT_FOLDER / MODELS / BENCHMARKS below, then:
    python report_base_vs_injected.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

OUTPUT_FOLDER = Path("output/augmented_benchmarks")

MODELS = [
    "gpt-4.1-mini",
    "gpt-5-mini",
    "deepseek-v4-flash",
    "meta-llama_llama-3.3-70B-Instruct",
    "claude-haiku-4-5"]

BENCHMARKS = ["bcb"]
VARIATIONS = ["us", "or"]

ANALYSIS_FILENAME = "analysis_fd_ft.jsonl"

# If set, also writes CSVs: {CSV_PREFIX}_combined.csv, {CSV_PREFIX}_us.csv, {CSV_PREFIX}_or.csv
CSV_PREFIX = None


def read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_analysis_records(output_folder: Path, benchmark: str, model: str, variation: str) -> list[dict]:
    """
    .../rq3/{benchmark}/{model}/{variation}/analysis_fd_ft.jsonl      (us)
    .../rq3/{benchmark}/{model}/analysis_fd_ft.jsonl                  (or — no subfolder)
    """
    if variation == "or":
        file_path = output_folder / "rq3" / benchmark / model / ANALYSIS_FILENAME
    else:
        file_path = output_folder / "rq3" / benchmark / model / variation / ANALYSIS_FILENAME

    if not file_path.exists():
        return []
    return list(read_jsonl(file_path))


def is_detected(per_test_fd: dict[str, int]) -> bool:
    return any(v == 1 for v in per_test_fd.values())


def is_triggered(per_test_ft: dict[str, int]) -> bool:
    return any(v == 1 for v in per_test_ft.values())


def aggregate_suite(records: list[dict], suite_key: str) -> dict:
    """
    Aggregate one suite's ("base" or "injected") stats across all fault records.

    total_faults      : faults that have data for this suite
    detected_faults    : faults where any test in per_test_fd fired
    detection_rate     : detected_faults / total_faults
    triggered_faults   : faults where any test in per_test_ft fired
    trigger_rate       : triggered_faults / total_faults
    correct/incorrect/insufficient_tests : pooled per test, summed across all faults
    """
    total_faults = 0
    detected_faults = 0
    triggered_faults = 0
    quality_counts: Counter = Counter()

    for record in records:
        suite_data = record.get(suite_key)
        if suite_data is None:
            continue

        per_test_fd = suite_data.get("per_test_fd", {})
        per_test_ft = suite_data.get("per_test_ft", {})
        per_test_quality = suite_data.get("per_test_quality", {})

        total_faults += 1
        if is_detected(per_test_fd):
            detected_faults += 1
        if is_triggered(per_test_ft):
            triggered_faults += 1

        for quality_label in per_test_quality.values():
            quality_counts[quality_label] += 1

    return {
        "total_faults": total_faults,
        "detected_faults": detected_faults,
        "detection_rate": detected_faults / total_faults if total_faults else 0.0,
        "triggered_faults": triggered_faults,
        "trigger_rate": triggered_faults / total_faults if total_faults else 0.0,
        "correct_tests": quality_counts.get("correct", 0),
        "incorrect_tests": quality_counts.get("incorrect", 0),
        "insufficient_tests": quality_counts.get("insufficient", 0),
    }


# ---------------------------------------------------------------------------
# Three views: combined (us+or pooled), us alone, or alone
# ---------------------------------------------------------------------------

def compute_combined(output_folder: Path, models: list[str], benchmarks: list[str]) -> dict[str, dict[str, dict[str, dict]]]:
    """Returns results[benchmark][model][suite] = stats, pooling us+or together."""
    results: dict[str, dict[str, dict[str, dict]]] = {}

    for benchmark in benchmarks:
        results[benchmark] = {}
        for model in models:
            records = []
            for variation in VARIATIONS:
                records.extend(load_analysis_records(output_folder, benchmark, model, variation))
            results[benchmark][model] = {
                "base": aggregate_suite(records, "base"),
                "injected": aggregate_suite(records, "injected"),
            }

    return results


def compute_per_variation(
    output_folder: Path, models: list[str], benchmarks: list[str], variation: str
) -> dict[str, dict[str, dict[str, dict]]]:
    """Returns results[benchmark][model][suite] = stats, for ONE variation alone."""
    results: dict[str, dict[str, dict[str, dict]]] = {}

    for benchmark in benchmarks:
        results[benchmark] = {}
        for model in models:
            records = load_analysis_records(output_folder, benchmark, model, variation)
            results[benchmark][model] = {
                "base": aggregate_suite(records, "base"),
                "injected": aggregate_suite(records, "injected"),
            }

    return results


def compute_total_stats(results: dict[str, dict[str, dict]], suite_key: str) -> dict:
    """
    Sum counts across all models for one suite ("base" or "injected"), then
    derive detection_rate/trigger_rate fresh from those summed counts
    (not an average of per-model rates).
    """
    total_faults = 0
    detected_faults = 0
    triggered_faults = 0
    correct_tests = 0
    incorrect_tests = 0
    insufficient_tests = 0

    for model_stats in results.values():
        s = model_stats[suite_key]
        total_faults += s["total_faults"]
        detected_faults += s["detected_faults"]
        triggered_faults += s["triggered_faults"]
        correct_tests += s["correct_tests"]
        incorrect_tests += s["incorrect_tests"]
        insufficient_tests += s["insufficient_tests"]

    return {
        "total_faults": total_faults,
        "detected_faults": detected_faults,
        "detection_rate": detected_faults / total_faults if total_faults else 0.0,
        "triggered_faults": triggered_faults,
        "trigger_rate": triggered_faults / total_faults if total_faults else 0.0,
        "correct_tests": correct_tests,
        "incorrect_tests": incorrect_tests,
        "insufficient_tests": insufficient_tests,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x:.2%}"


def _fmt_delta(after: float, before: float) -> str:
    diff = after - before
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2%}"


def print_report(results: dict[str, dict[str, dict[str, dict]]], title: str) -> None:
    print()
    print("=" * 150)
    print(title)
    print("=" * 150)
    header = (f"{'benchmark':<10} {'model':<20} {'suite':<10} "
              f"{'faults':>8} {'detected':>10} {'det_rate':>10} "
              f"{'triggered':>10} {'trig_rate':>10} "
              f"{'correct':>9} {'incorrect':>10} {'insufficient':>13}")
    print(header)
    print("-" * len(header))

    for benchmark, models in results.items():
        for model, suites in models.items():
            base = suites["base"]
            injected = suites["injected"]

            print(f"{benchmark:<10} {model:<20} {'base':<10} "
                  f"{base['total_faults']:>8} {base['detected_faults']:>10} "
                  f"{_fmt_pct(base['detection_rate']):>10} "
                  f"{base['triggered_faults']:>10} {_fmt_pct(base['trigger_rate']):>10} "
                  f"{base['correct_tests']:>9} {base['incorrect_tests']:>10} {base['insufficient_tests']:>13}")
            print(f"{benchmark:<10} {model:<20} {'injected':<10} "
                  f"{injected['total_faults']:>8} {injected['detected_faults']:>10} "
                  f"{_fmt_pct(injected['detection_rate']):>10} "
                  f"{injected['triggered_faults']:>10} {_fmt_pct(injected['trigger_rate']):>10} "
                  f"{injected['correct_tests']:>9} {injected['incorrect_tests']:>10} {injected['insufficient_tests']:>13}")
            print(f"{benchmark:<10} {model:<20} {'Δ (i-b)':<10} "
                  f"{'':>8} {'':>10} "
                  f"{_fmt_delta(injected['detection_rate'], base['detection_rate']):>10} "
                  f"{'':>10} {_fmt_delta(injected['trigger_rate'], base['trigger_rate']):>10} "
                  f"{'':>9} {'':>10} {'':>13}")
            print()

        print("-" * len(header))
        total_base = compute_total_stats(models, "base")
        total_injected = compute_total_stats(models, "injected")

        print(f"{benchmark:<10} {'TOTAL':<20} {'base':<10} "
              f"{total_base['total_faults']:>8} {total_base['detected_faults']:>10} "
              f"{_fmt_pct(total_base['detection_rate']):>10} "
              f"{total_base['triggered_faults']:>10} {_fmt_pct(total_base['trigger_rate']):>10} "
              f"{total_base['correct_tests']:>9} {total_base['incorrect_tests']:>10} {total_base['insufficient_tests']:>13}")
        print(f"{benchmark:<10} {'TOTAL':<20} {'injected':<10} "
              f"{total_injected['total_faults']:>8} {total_injected['detected_faults']:>10} "
              f"{_fmt_pct(total_injected['detection_rate']):>10} "
              f"{total_injected['triggered_faults']:>10} {_fmt_pct(total_injected['trigger_rate']):>10} "
              f"{total_injected['correct_tests']:>9} {total_injected['incorrect_tests']:>10} {total_injected['insufficient_tests']:>13}")
        print(f"{benchmark:<10} {'TOTAL':<20} {'Δ (i-b)':<10} "
              f"{'':>8} {'':>10} "
              f"{_fmt_delta(total_injected['detection_rate'], total_base['detection_rate']):>10} "
              f"{'':>10} {_fmt_delta(total_injected['trigger_rate'], total_base['trigger_rate']):>10} "
              f"{'':>9} {'':>10} {'':>13}")
        print()


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_to_rows(results: dict[str, dict[str, dict[str, dict]]]) -> list[dict]:
    rows = []
    for benchmark, models in results.items():
        for model, suites in models.items():
            for suite in ["base", "injected"]:
                s = suites[suite]
                rows.append({
                    "benchmark": benchmark,
                    "model": model,
                    "suite": suite,
                    "total_faults": s["total_faults"],
                    "detected_faults": s["detected_faults"],
                    "detection_rate": s["detection_rate"],
                    "triggered_faults": s["triggered_faults"],
                    "trigger_rate": s["trigger_rate"],
                    "correct_tests": s["correct_tests"],
                    "incorrect_tests": s["incorrect_tests"],
                    "insufficient_tests": s["insufficient_tests"],
                })

        for suite in ["base", "injected"]:
            s = compute_total_stats(models, suite)
            rows.append({
                "benchmark": benchmark,
                "model": "TOTAL",
                "suite": suite,
                "total_faults": s["total_faults"],
                "detected_faults": s["detected_faults"],
                "detection_rate": s["detection_rate"],
                "triggered_faults": s["triggered_faults"],
                "trigger_rate": s["trigger_rate"],
                "correct_tests": s["correct_tests"],
                "incorrect_tests": s["incorrect_tests"],
                "insufficient_tests": s["insufficient_tests"],
            })
    return rows


def main():
    print(f"Output folder : {OUTPUT_FOLDER}")
    print(f"Models        : {MODELS}")
    print(f"Benchmarks    : {BENCHMARKS}")
    print(f"Variations    : {VARIATIONS}")

    combined_results = compute_combined(OUTPUT_FOLDER, MODELS, BENCHMARKS)
    print_report(combined_results, "BASE vs INJECTED — COMBINED (us + or pooled)")

    us_results = compute_per_variation(OUTPUT_FOLDER, MODELS, BENCHMARKS, "us")
    print_report(us_results, "BASE vs INJECTED — US ALONE")

    or_results = compute_per_variation(OUTPUT_FOLDER, MODELS, BENCHMARKS, "or")
    print_report(or_results, "BASE vs INJECTED — OR ALONE")

    if CSV_PREFIX:
        fieldnames = ["benchmark", "model", "suite", "total_faults", "detected_faults",
                      "detection_rate", "triggered_faults", "trigger_rate",
                      "correct_tests", "incorrect_tests", "insufficient_tests"]

        write_csv(f"{CSV_PREFIX}_combined.csv", report_to_rows(combined_results), fieldnames)
        write_csv(f"{CSV_PREFIX}_us.csv", report_to_rows(us_results), fieldnames)
        write_csv(f"{CSV_PREFIX}_or.csv", report_to_rows(or_results), fieldnames)

        print(f"\nWrote CSVs: {CSV_PREFIX}_combined.csv, {CSV_PREFIX}_us.csv, {CSV_PREFIX}_or.csv")

    return 0


if __name__ == "__main__":
    exit(main())