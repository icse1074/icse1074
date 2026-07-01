from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

from test_adequacy_study.types.benchmark_variation import BenchmarkVariation

load_dotenv()

OUTPUT_FOLDER = Path("output/augmented_benchmarks")
FAULTS_ROOT = OUTPUT_FOLDER / "faults"
MUTANTS_ROOT = OUTPUT_FOLDER / "mutants"
TESTS_ROOT = OUTPUT_FOLDER / "generated_tests"
PROCESSED_TESTS_ROOT = OUTPUT_FOLDER / "processed_tests"

WORK_DIR = os.environ.get("WORK_DIR")
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR, exist_ok=True)

FAULT_MODELS = [
    "gpt-5-mini",
    "gpt-4.1-mini",
   "claude-haiku-4-5",
    "meta-llama_llama-3.3-70B-Instruct",
    "deepseek-v4-flash",
]
TEST_MODELS = [
   # "gpt-5-mini",
    #"gpt-4.1-mini",
    #"claude-haiku-4-5",
   #"meta-llama_llama-3.3-70B-Instruct",
   #"deepseek-v4-flash",
]

BENCHMARKS = [
 #"he",
"bcb",
    #"bcb",
    #"ncb",
]

MODEL_NAME = FAULT_MODELS[0]
BENCHMARK_NAME = BENCHMARKS[0]
BENCHMARK_VARIATION = BenchmarkVariation.NONE
CRITERION_SETUP = "mutation"

FAULTS_FILE_PATTERN = str(FAULTS_ROOT / BENCHMARK_NAME / "{fault_model}" / "faults.jsonl")

MUTANTS_FILE_PATTERN = str(MUTANTS_ROOT / BENCHMARK_NAME / MODEL_NAME / "mutants.jsonl")

TESTS_FILE_PATTERN = str(TESTS_ROOT / BENCHMARK_NAME / MODEL_NAME / "tests.jsonl")
COMPLETED_TESTS_FILE_PATTERN = str(PROCESSED_TESTS_ROOT / BENCHMARK_NAME / MODEL_NAME / "completed_tests.jsonl")

# Output layout — each (model, benchmark) pair writes to its own subdirectory
OUTPUT_ROOT = Path(os.path.join(OUTPUT_FOLDER, "results"))
ANALYSIS_FILE_PATTERN = str(OUTPUT_ROOT / "{fault_model}" / BENCHMARK_NAME / "analysis_results.jsonl")
COMPLETED_TESTS_ANALYSIS_FILE_PATTERN = str(OUTPUT_ROOT / "{fault_model}" / BENCHMARK_NAME / "completed_tests_analysis_results.jsonl")



SELECTION_FILE_PATTERN = str(OUTPUT_ROOT / "{fault_model}" / BENCHMARK_NAME / "minimisation_results.jsonl")

#uncomment for rQ3
#OUTPUT_ROOT = Path(os.path.join(OUTPUT_FOLDER, "rq3"))
#SELECTION_FILE_PATTERN = str(OUTPUT_ROOT / BENCHMARK_NAME / "{fault_model}" / "rq3_minimisation_results.jsonl")
SKIP_ANALYSIS = False
SKIP_MINIMIZATION = False

N_SHUFFLES = 100
GRID_SIZE = 100
ALPHA = 0.05  # p value for wilcoxon test
BUDGET_POINTS_FAIR = [25, 50, 75, 100]

METRICS = ["trigger",
           "detection",
           "line_cov",
           "ms",
           "branch_cov"]

CRITERIA = ["random",
            "line_coverage",
            "branch_coverage",
            "mutation"]

LABELS = {
    "random": "Random",
    "line_coverage": "Line Coverage",
    "branch_coverage": "Branch Coverage",
    "mutation": "Mutation Score",
}


