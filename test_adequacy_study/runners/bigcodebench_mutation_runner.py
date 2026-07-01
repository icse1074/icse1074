from pathlib import Path

from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.runners.mutation_runner import MutationRunner
from test_adequacy_study.runners.bigcodebench_test_runner import VENV_PYTHON, _clean_env, ensure_venv


class BCBMutationRunner(MutationRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ensure_venv()

    def _run_and_get_results(self, run_dir: Path):
        report_path = run_dir / ".pytest_report.json"

        result = self.sandbox.run(
            cmd=[
                str(VENV_PYTHON), "-m", "pytest",  # <-- venv python
                str(run_dir),
                "-q",
                "--json-report",
                f"--json-report-file={str(report_path)}",
                "--json-report-indent=2",
            ],
            cwd=run_dir,
            timeout=self.timeout,
            env=_clean_env(),                       # <-- clean env
        )
        self.set_strict(False)
        test_results = self._build_report(result=result, report_path=report_path)

        incompetent = False
        if not test_results or not test_results.detailed_test_results:
            return True, {}

        if test_results.verdict not in [Verdict.PASSED, Verdict.FAILED]:
            incompetent = True

        return incompetent, {
            r.node_id: {"outcome": r.outcome, "message": r.message}
            for r in test_results.detailed_test_results
        }