from pathlib import Path

from test_adequacy_study.generators.abstract_code_generator import AbstractCodeGenerator


class CodeGenerator(AbstractCodeGenerator):
    system_prompt_file = "system"
    task_prompt_file = "completion"

    default_prompts_dir = Path(__file__).resolve().parent / "prompts"
    def __init__(self, model: str = "gpt-4o-mini", prompt_dir: str = default_prompts_dir, temperature: float = 0.8):
        super().__init__(model, prompt_dir, temperature)