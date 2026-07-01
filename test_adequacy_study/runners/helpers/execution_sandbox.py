from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    duration: float


class ExecutionSandbox:
    """
    Lightweight sandbox using subprocess.

    Features:
    - timeout control
    - working directory isolation
    - optional environment isolation
    """
    @staticmethod
    def run(
        cmd: list[str],
        cwd: Path,
        timeout: float,
        env: dict | None = None,
    ) -> SandboxResult:
        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )

            return SandboxResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration=time.monotonic() - start,
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=f"TIMEOUT after {timeout}s",
                duration=timeout,
            )