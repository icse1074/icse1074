from dataclasses import dataclass
from typing import List

from test_adequacy_study.data_models.task import Task


@dataclass
class PipelineConfig:
    n_generations: int = 10
    timeout: float = 10.0
    tasks: List[Task] = None
    output_file: str = None
    skip_n_tasks: int = None
    match_function_names: bool = False
    slice : str = None
    start_task_id: str = None
    exclude_ids: List[str] = None
