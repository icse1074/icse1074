from test_adequacy_study.benchmarks.humaneval import HumanEvalLoader
from tqdm import tqdm

from test_adequacy_study.data_models.task import Task
from test_adequacy_study.file_utils import write_jsonl


def get_ground_truth(task : Task):
    exec_globals = {}

    code = task.stub + task.canonical_solution
    exec(code, exec_globals)

    fn = exec_globals[task.entry_point]

    full_tests = [
        (
            inp,
            fn(*inp) if isinstance(inp, (tuple, list)) else fn(inp)
        )
        for inp in task.tests["test_inputs"]
    ]

    return dict(
        task_id=task.task_id,
        tests=full_tests,
    )


if __name__ == "__main__":

    output_file = "plus_inputs.jsonl"
    #load dataset
    loader = HumanEvalLoader()
    tasks = loader.load()

    #get ground truth for each task
    for task in tqdm(tasks, desc="humaneval tasks"):
        print("hello")
        plus_inputs = get_ground_truth(task)
        write_jsonl(output_file, [plus_inputs], append=True)



