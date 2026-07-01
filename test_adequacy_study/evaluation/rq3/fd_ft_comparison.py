import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from test_adequacy_study.builders.python_program_builder import PythonProgramBuilder
from test_adequacy_study.evaluation.helpers.run_version import run_tests_on_code
from test_adequacy_study.evaluation.rq3.inject_tests import inject_all_tests
from test_adequacy_study.file_utils import read_jsonl, write_jsonl
from test_adequacy_study.providers.loader_provider import LoaderProvider
from test_adequacy_study.providers.program_builder_provider import ProgramBuilderProvider
from test_adequacy_study.providers.runner_provider import RunnerProvider
from test_adequacy_study.runners.mutation_runner import _same_outcome
from test_adequacy_study.types.benchmark_variation import BenchmarkVariation
from dotenv import load_dotenv
load_dotenv()
BENCHMARKS = ["bcb"]
BENCHMARK_VARIATION = BenchmarkVariation.UNDER_SPECIFIED
MODELS = [
    "gpt-5-mini",
]

OUTPUT_FOLDER = Path("output/augmented_benchmarks")
if not os.path.exists(OUTPUT_FOLDER):
    OUTPUT_FOLDER = "llm_faults"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Created output folder: {OUTPUT_FOLDER}")
    FAULTS_ROOT = Path("output/artifacts/faults")
    TESTS_ROOT = Path("output/artifacts/generated_tests")
    PROCESSED_TESTS_ROOT = Path("output/artifacts/processed_tests")
else :
    FAULTS_ROOT = OUTPUT_FOLDER / "faults"
    TESTS_ROOT = OUTPUT_FOLDER / "generated_tests"
    PROCESSED_TESTS_ROOT = OUTPUT_FOLDER / "processed_tests"

def analyze_tests(
    ref_results:   dict[str, dict],
    fault_results: dict[str, dict],
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:

    per_test_ft: dict[str, int] = {}
    per_test_fd: dict[str, int] = {}
    per_test_quality: dict[str, str] = {}

    for test in ref_results:
        if test not in fault_results:
            continue

        ref = ref_results[test]
        flt = fault_results[test]

        triggers = not _same_outcome(ref, flt)
        detects = ref["outcome"] == "passed" and flt["outcome"] == "failed"
        if detects:
            print("YAY FAULT DETECTED")

        per_test_ft[test] = int(triggers)
        per_test_fd[test] = int(detects)

        ref_passed = ref["outcome"] == "passed"
        flt_passed = flt["outcome"] == "passed"

        if not ref_passed:
            # Fails on reference -- broken test, regardless of fault outcome
            # (covers both "fails on both" and "fails on reference, passes on fault")
            per_test_quality[test] = "incorrect"
        elif ref_passed and not flt_passed:
            # Passes on reference, fails on fault -- correctly detects the fault
            per_test_quality[test] = "correct"
        else:
            # Passes on both -- doesn't distinguish fault from reference
            per_test_quality[test] = "insufficient"

    return per_test_ft, per_test_fd, per_test_quality


def load_faults(model: str, benchmark: str) -> list[dict]:
    faults_file = Path(os.path.join(FAULTS_ROOT, benchmark, model, "us/faults.jsonl"))
    faults = read_jsonl(str(faults_file))
    print("FaultLoader: loaded faults from ", len(faults), faults_file)
    return faults

def main():

    for model in MODELS:
        for benchmark in BENCHMARKS:


            faults = load_faults(model=model, benchmark=benchmark)
            tasks = {str(t.task_id): t for t in
                     LoaderProvider.get(benchmark, variation=BENCHMARK_VARIATION).load(no_tests=True)}

            base_test_file = Path(os.path.join(TESTS_ROOT, benchmark, model, "us/tests.jsonl"))

            base_tests_by_task: dict[str, dict] = {}
            for record in read_jsonl(base_test_file):
                base_tests_by_task[str(record["task_id"])] = record

            completed_tests_file = Path(os.path.join(PROCESSED_TESTS_ROOT, benchmark, model, "us/completed_tests.jsonl"))


            completed_by_task: defaultdict[list] = {}
            for record in read_jsonl(completed_tests_file):
                task_id = str(record.get("task_id"))
                if task_id in completed_by_task:
                    completed_by_task[task_id].append(record)
                else:
                    completed_by_task[task_id] = [record]


            output_path = Path(os.path.join(OUTPUT_FOLDER, "rq3", benchmark, model, "us/analysis_fd_ft.jsonl"))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            #faults to keep :
            faults_to_evaluate = []
            for fault in faults:
                if str(fault["task_id"]) in completed_by_task:
                    faults_to_evaluate.append(fault)

            n_written = 0
            print("number of faults to evaluate: {}".format(len(faults_to_evaluate)))
            faults_to_evaluate = [fault for fault in faults_to_evaluate if fault["score"] <= 0.25]
            for fault in tqdm(faults_to_evaluate):
                #Prepare tests
                task = tasks.get(str(fault["task_id"]))
                if task is None:
                    print(f"skipping, task not found: {fault['task_id']}")
                    continue
                task_id = str(task.task_id)

                base_test_record = base_tests_by_task.get(task_id)
                if base_test_record is None:
                    print(f"skipping, no base test record: {task_id}")
                    continue
                base_test_suite = base_test_record.get("response", "")
                task_completed_tests = completed_by_task.get(task_id)

                injected_suite, injection_results = inject_all_tests(base_test_suite, task_completed_tests)

                faulty_cut = ProgramBuilderProvider().get(benchmark).build_program(task=task, code=fault["completion"])
                reference_cut = ProgramBuilderProvider().get(benchmark).build_program(task=task, code=task.canonical_solution)

                if not faulty_cut.syntactically_valid:
                    print("skipping, incorrect fault", task_id)
                    continue
                if not reference_cut.syntactically_valid:
                    print("skipping, incorrect reference", task_id)
                    continue

                # --- Build both suites: the original (un-injected) base suite,
                #     and the injected suite with completed tests substituted in ---
                base_test_file_content = "import matplotlib\nmatplotlib.use('Agg')\n" + base_test_suite

                injected_test_file_content = "import matplotlib\nmatplotlib.use('Agg')\n" + injected_suite

                # --- Run base suite on both reference and faulty versions ---
                base_ref_results = run_tests_on_code(
                    cut=reference_cut, task=task, tests=base_test_file_content,
                    builder=PythonProgramBuilder(), runner=RunnerProvider.get_for_test_generation(benchmark_name=benchmark, work_dir=os.getenv("WORK_DIR")),
                )
                base_fault_results = run_tests_on_code(
                    faulty_cut, task=task, tests=base_test_file_content,
                    builder=PythonProgramBuilder(), runner=RunnerProvider.get_for_test_generation(benchmark_name=benchmark, work_dir=os.getenv("WORK_DIR")),
                )

                # --- Run injected suite on both reference and faulty versions ---
                try :
                    injected_ref_results = run_tests_on_code(
                        cut=reference_cut, task=task, tests=injected_test_file_content,
                        builder=PythonProgramBuilder(), runner=RunnerProvider.get_for_test_generation(benchmark_name=benchmark, work_dir=os.getenv("WORK_DIR")),
                    )
                    injected_fault_results = run_tests_on_code(
                        faulty_cut, task=task, tests=injected_test_file_content,
                        builder=PythonProgramBuilder(), runner=RunnerProvider.get_for_test_generation(benchmark_name=benchmark, work_dir=os.getenv("WORK_DIR")),
                    )
                except Exception as e:
                    print(e)
                    continue
                try :
                    if not base_ref_results or not base_fault_results:
                        print(f"skipping, missing base test results: {task_id}")
                        continue
                    if not injected_ref_results or not injected_fault_results:
                        print(f"skipping, missing injected test results: {task_id}")
                        continue
                except Exception as e:
                    print(e)
                    continue
                base_per_test_ft, base_per_test_fd, base_per_test_quality = analyze_tests(
                    base_ref_results, base_fault_results
                )
                injected_per_test_ft, injected_per_test_fd, injected_per_test_quality = analyze_tests(
                    injected_ref_results, injected_fault_results
                )

                record = {
                    "task_id": task_id,
                    "fault_model": model,
                    "test_model": model,
                    "benchmark": benchmark,
                    "base": {
                        "per_test_ft": base_per_test_ft,
                        "per_test_fd": base_per_test_fd,
                        "per_test_quality": base_per_test_quality,
                    },
                    "injected": {
                        "per_test_ft": injected_per_test_ft,
                        "per_test_fd": injected_per_test_fd,
                        "per_test_quality": injected_per_test_quality,
                    },
                    "injection_results": injection_results,
                }

                write_jsonl(output_path, [record], append=True)
                n_written += 1

            print(f"Done: benchmark={benchmark} model={model} — wrote {n_written} records to {output_path}")


if __name__ == "__main__":
    main()