import shutil
from pathlib import Path
from typing import List

from test_adequacy_study.data_models.test_suite import TestSuite


def materialize(suite: TestSuite, run_dir: Path) -> List[Path]:
    if isinstance(suite.source, str):
        return materialize_from_string(suite.source, run_dir)

    if isinstance(suite.source, Path):
        return materialize_from_path(suite.source, run_dir)

    return materialize_from_many(suite.source, run_dir)

def materialize_from_string(content: str, run_dir: Path) -> List[Path]:
    test_dir = run_dir / "tests"
    test_dir.mkdir(exist_ok=True)

    # make tests a package
    init_file = test_dir / "__init__.py"
    init_file.write_text("", encoding="utf-8")

    # write generated test file
    file = test_dir / "test_generated.py"
    file.write_text(content, encoding="utf-8")

    return [file]

def materialize_from_path(path: Path, run_dir: Path) -> List[Path]:
    test_dir = run_dir / "tests"
    test_dir.mkdir(exist_ok=True)

    # make tests a package
    init_file = test_dir / "__init__.py"
    init_file.write_text("", encoding="utf-8")

    target = test_dir / path.name
    shutil.copy(path, target)

    return [target]
def materialize_from_many(paths: List[Path], run_dir: Path) -> List[Path]:
    test_dir = run_dir / "tests"
    test_dir.mkdir(exist_ok=True)

    # make tests a package
    init_file = test_dir / "__init__.py"
    init_file.write_text("", encoding="utf-8")

    result: List[Path] = []

    for p in paths:
        if not isinstance(p, Path):
            raise TypeError(f"Expected Path, got {type(p)}")

        result.append(materialize_from_path(p, run_dir)[0])

    return result