class Task:
    task_id: str
    stub: str
    entry_point: str
    canonical_solution: str
    tests : dict #could be dict or str
    generated_solution: str = None
    libs: list = None  # required libraries for BigCodeBench tasks

    def __init__(
        self,
        task_id,
        stub,
        entry_point,
        canonical_solution,
        tests,
        generated_solution=None,
        libs=None,
    ):
        self.task_id = task_id
        self.stub = stub
        self.entry_point = entry_point
        self.canonical_solution = canonical_solution
        self.generated_solution = generated_solution
        self.tests = tests
        self.libs = libs or []

    def __str__(self):
        return (f"Task(task_id={self.task_id}, "
                f"stub={self.stub}, "
                f"entry_point={self.entry_point}, "
                f"generated_solution={self.generated_solution}, "
                f"tests={self.tests})")