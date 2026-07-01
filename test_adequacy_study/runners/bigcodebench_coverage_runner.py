import os
from pathlib import Path
from test_adequacy_study.runners.coverage_runner import CoverageRunner
from test_adequacy_study.runners.bigcodebench_test_runner import (
    VENV_PYTHON, _clean_env, ensure_venv
)
from test_adequacy_study.data_models.coverage_report import CoverageReport
import logging
logger = logging.getLogger(__name__)

class BCBCoverageRunner(CoverageRunner):
    """CoverageRunner that executes inside the BigCodeBench venv."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ensure_venv()

    def _CoverageRunner__execute_coverage(
        self, entry_point: Path, run_dir: Path
    ) -> CoverageReport:

        coverage_json_path = run_dir / ".coverage_report.json"
        coveragerc_path = run_dir / ".coveragerc"

        logger.info("WORK DIR BCB %s", self.work_dir)
        exists = os.path.exists(run_dir)
        logger.info("COVERAGE DIR BCB %s", run_dir)
        logger.info("Folder exists: %s", exists)

        coveragerc_path.write_text(
            "[run]\n"
            "dynamic_context = test_function\n"
            "branch = True\n"
            "[json]\n"
            "show_contexts = true\n",
            encoding="utf-8",
        )

        result = self.sandbox.run(
            cmd=[

                str(VENV_PYTHON), "-m", "pytest",   # <-- venv python
                str(entry_point),
                "-q",
                "--cov=solution",
                "--cov-config=" + str(coveragerc_path),
                "--cov-report=json:" + str(coverage_json_path),
            ],
            cwd=run_dir,
            timeout=self.timeout,
            env=_clean_env(),                        # <-- clean env
        )

        logger.warning(result)

        return self._CoverageRunner__parse_coverage_report(coverage_json_path)