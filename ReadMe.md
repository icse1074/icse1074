# [Anonymous ICSE 2027 Submission #1074]

This repository contains the code, data, and artifacts accompanying our ICSE submission. It has been anonymized.

## Requirements

- Python >= 3.10
- Two separate virtual environments are required (see below).

## Setup

Two virtual environments are needed:

- **Main environment**: created via `venv_setup.sh`. Used for generation, test augmentation, runners, and evaluation.
- **BigCodeBench environment**: created via `bcb_venv_setup.sh`. Used specifically for BigCodeBench-related execution.

```bash
source venv_setup.sh
source bcb_venv_setup.sh
```

Additionally, this project depends on an external repository, **code_mutate**, used to generate mutants for mutation analysis. It must be cloned/available alongside this repository as expected by the mutation runners.

- Repo: https://github.com/cedricrupb/code_mutate

## Repository Structure

```
data/                   Benchmark data (JSONL)
scripts/                Helping scripts
main.py                 Main entry point
test_adequacy_study/
├── benchmarks/          Benchmark loaders
├── builders/            Python program & test builders
├── generators/          Code generation logic (prompts/ contains all prompts used)
├── oracle_completion/   Preprocessing for NL specification-guided oracle generation
├── runners/             Test runners, mutation runners, coverage runners
├── test_augmentation/   Augments benchmark tests (entry point: clean_and_collect)
│   └── augmented_tests/ Holds the augmented tests used to collect faults
├── test_generation/     Test generation with feedback loops
└── evaluation/          Evaluation scripts (entry point: run_analysis; config.py holds
                          analysis configuration)
output/                 All generated artifacts and results
```

## Data & Artifacts

Due to size constraints, the following are not included directly in this
repository and are instead hosted externally:

- **`data.zip`** — the full benchmark data archive, containing the benchmark
  records as JSONL files, along with underspecified variants of each
  benchmark, suffixed `_US` (e.g. `<benchmark_name>_US.jsonl`).
- **`output/`** — all generated artifacts and results (generations, faults,
  tests, mutants, analysis results, etc.), as described below.
- **`test_adequacy_study/test_augmentation/augmented_tests`** — augmented test suites 

All are hosted at: `<https://zenodo.org/records/21098028>`

Download and extract them before running the pipeline:
- `data.zip` → extract into `data/`
- the `output/` archive → extract into `output/` at the repository root

## Artifacts & Results (`output/`)

```
output/
├── generations/                 All raw model generations.
└── artifacts/
    ├── faults/                  Collected faults
    │   ├── all_faults.jsonl     All generated faulty implementations.
    │   └── faults.jsonl         Faulty implementations after filtering.
    ├── generated_tests/         Tests generated against the filtered faults.
    ├── mutants/                 Mutants produced on the filtered faults.
    ├── triggering_tests/        Triggering tests extracted after analysis.
    ├── processed_tests/         Tests decomposed into prefix/assertions,
    │                            plus fully completed tests after oracle generation.
    └── results/
        ├── <model>/<benchmark>/analysis_results.jsonl
        │                            Results of running all faults against
        │                            generated tests: fault-triggering and
        │                            fault-detection per test, coverage
        │                            reports, and mutation reports.
        ├── <model>/<benchmark>/minimisation_results.jsonl
        │                            Results of running the simulation 100 times.
        ├── rq3/                     Additional fault-detection (FD) plus a
        │                            corresponding minimisation analysis
        │                            computed over the generated oracles.
        └── report/                  Plots and tables summarizing the
                                     results above.
```

## Reproducing the study

1. Set up both virtual environments and the `code_mutate` dependency (see Setup).
2. Download and extract `data.zip` into `data/` (see Data & Artifacts).
3. Run `main.py` for code generation, test augmentation, test generation, mutant generation, or oracle_completion.
4. Run `test_adequacy_study/test_augmentation/clean_and_collect.py` to clean the augmented tests and collect the faults.
5. Run `test_adequacy_study/evaluation/analysis.py`, configured via
   `test_adequacy_study/evaluation/config.py`, to reproduce the analysis and minimisation results in
   `output/artifacts/results/`.
6. Run `test_adequacy_study/evaluation/{rq1, rq2, rq3}/{rq1, rq2, rq3}.py` for the results.