from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from test_adequacy_study.data_models.execution_report import ExecutionReport
from test_adequacy_study.runners.pytest_runner import PytestRunner

load_dotenv()

VENV_DIR = Path(os.path.expanduser(os.environ.get("BCB_VENV_DIR", "~/.bigcodebench_venv")))
VENV_PIP = VENV_DIR / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


#for debugging
def ensure_matplotlib() -> None:
    """
    Ensures matplotlib is installed in the BigCodeBench venv.
    Installs it if not present.
    """
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import matplotlib"],
        capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(
            [str(VENV_PIP), "install", "matplotlib"],
            check=True,
        )

def ensure_venv() -> None:
    """
    Verifies the BigCodeBench virtual environment exists and is ready.

    The venv must be created once before running using the setup script:
        bash setup_bigcodebench_env.sh

    Or manually:
        python -m venv ~/.bigcodebench_venv
        ~/.bigcodebench_venv/bin/pip install bigcodebench[evaluate] pytest-json-report pytest-timeout
    """
    if not VENV_PYTHON.exists():
        raise RuntimeError(
            f"BigCodeBench venv not found at {VENV_DIR}.\n"
            f"Run the setup script first:\n"
            f"  bash setup_bigcodebench_env.sh\n"
            f"Or create it manually:\n"
            f"  /opt/homebrew/bin/python3.10 -m venv {VENV_DIR}\n"
            f"  {VENV_PIP} install bigcodebench[evaluate] pytest-json-report pytest-timeout"
        )
    ensure_matplotlib()


def _clean_env() -> dict:
    """
    Returns a sanitized copy of the host environment safe for use inside the venv.

    Strips PYTHONPATH and PYTHONHOME to prevent the host Python  site.py
    from being loaded inside the venv, which causes UnicodeDecodeError on non-ASCII
    mount points (e.g. external SSDs).
    """
    env = os.environ.copy()

    for key in list(env):
        if (
                key.startswith("PYCHARM")
                or key.startswith("PYDEV")
        ):
            del env[key]

    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["TZ"] = "UTC"
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    return env


class BigCodeBenchTestRunner(PytestRunner):
    """
    Pytest-based runner for BigCodeBench tasks.

    Extends PytestRunner with an isolated virtual environment that has all
    BigCodeBench library dependencies pre-installed. The host project can remain
    on Python 3.9 — only the venv needs 3.10+.

    The venv must be created once before running using the setup script:
        bash setup_bigcodebench_env.sh

    Execution layout (inherited from PytestRunner):
        run_dir/
            solution.py           # CUT content, importable as a module
            generated_tests.py    # materialized test suite, imports from solution
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Forwarded to TestRunner (timeout, work_dir, sandbox).
        """
        super().__init__(**kwargs)
        ensure_venv()

    def _extract_missing_modules(self, stdout: str) -> list[str]:
        """Extract missing module names from ModuleNotFoundError in stdout."""
        return re.findall(r"ModuleNotFoundError: No module named '([^']+)'", stdout)


    def _execute(self, entry_point: Path, task_id: str) -> ExecutionReport:
        """
        Runs pytest inside the BigCodeBench venv and parses the JSON report.

        Overrides PytestRunner._execute to swap the Python executable
        from the host python to the venv's Python, and to pass a clean environment
        that strips host Python path variables.

        Args:
            entry_point: The run directory containing solution.py and test files.
            task_id: Identifier for the task, propagated into the ExecutionReport.

        Returns:
            ExecutionReport with verdict, stdout/stderr, duration, and per-test results.
        """
        report_path = str(entry_point.parent / ".pytest_report.json")
        result = self.sandbox.run(
            cmd=[
                str(VENV_PYTHON), "-m", "pytest",
                str(entry_point.parent),
                "-q",
                "--json-report",
                f"--json-report-file={report_path}",
                "--json-report-indent=2",
            ],
            cwd=entry_point.parent,
            timeout=180,
            env=_clean_env(),
        )
        missing = self._extract_missing_modules(result.stdout)
        if missing:
            with open("missing_modules.txt", "a") as f:
                for module in missing:
                    f.write(module + "\n")


        return self._build_report(task_id, result, report_path)