from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TestPoolLoader:
    """
    Loads a test pool from JSONL files and returns {task_id: [test_file_string]}.

    Currently supports one mode :
      • load_from_file(path, test_model) — loads a single tests_{benchmark}_{fault_model}.jsonl
                                           and filters rows to a (or man) specific test_model.

    JSONL structure:
        {"task_id": "HumanEval/0", "model_id": "gpt-5-mini", "response": "...."}
    """


    def load_from_file(
        self,
        path: str | Path,
        test_models: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        Load a single tests_{benchmark}_{fault_model}.jsonl.

        *test_models* controls which rows are included:
          - None              → all models merged per task_id
          - list[str]         → only the listed models, merged per task_id

        Expected Line structure:
            {"task_id": "HumanEval/0", "model_id": "gpt-5-mini", "response": "...."}
        """
        path = Path(path)
        if not path.exists():
            logger.warning("TestPoolLoader: file not found — %s", path)
            return {}

        pool: dict[str, list[str]] = {}
        n_loaded = 0

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # todo: Sometimes model_id is not needed or not existant
                # if test_models is not None and record.get("model_id") not in test_models:
                #     continue
                task_id = record["task_id"]
                if task_id not in pool:
                    pool[task_id] = []
                pool[task_id].append({"model_id" : record.get("model_id"), "test_suite" : record.get("response", [])})
                n_loaded += 1

        logger.info(
            "TestPoolLoader: loaded %d rows%s from %s → %d unique tasks",
            n_loaded,
            f" for test_models={sorted(test_models)}" if test_models else " (all models)",
            path.name,
            len(pool),
        )
        return pool