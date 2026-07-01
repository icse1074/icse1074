from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict


TestSource = Union[str, Path, List[Path], Dict]

class TestFramework(str, Enum):
    HUMANEVAL = "humaneval"
    PYTEST = "pytest"
    BIGCODEBENCH = "bigcodebench"

@dataclass
class TestSuite:
    """
    Unified test representation.

    Supports:
    - inline code (str)
    - single file (Path)
    - multiple files (List[Path])
    """

    task_id : str
    source: TestSource
    framework: TestFramework  # "humaneval" | "pytest"
    language : str