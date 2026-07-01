from pathlib import Path

from test_adequacy_study.generators.openai_model import OpenAIModel
from test_adequacy_study.types.prompt_generation_type import PromptGenerationType


class PromptGenerator:
    default_prompts_dir = Path(__file__).resolve().parent / "prompts"

    def __init__(
            self,
            model: str = "gpt-4o-mini",
            prompt_dir: str = default_prompts_dir,
            temperature: float = 0.8,
            prompt_generation_type: PromptGenerationType = PromptGenerationType.FROM_CODE_MINIMAL
    ):
        self.prompt_dir = Path(prompt_dir)
        self.call_count = 0
        self.model_id = model
        self.all_prompt_templates = self.__load_all_prompt_templates()
        self.prompt_template = self.all_prompt_templates[prompt_generation_type.value]
        self.llm = OpenAIModel(model=model, temperature=temperature)

    def get_prompt(self, prompt_variables: dict[str, str]):
        return self.prompt_template.format(**prompt_variables)

    def generate(self, generation_type: PromptGenerationType, prompt_variables: dict[str, str]):
        user_message = self.all_prompt_templates[generation_type.value].format(**prompt_variables)

        responses, history = self.llm.call(
            prompt=user_message,
            system_prompt=None,
            n=1,
            history=None,
        )

        return responses[0]

    def __load_prompt(self, name: str) -> str:
        path = self.prompt_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def __load_all_prompt_templates(self) -> dict[str, str]:
        templates = {}
        for prompt_type in list(PromptGenerationType):
            templates[prompt_type.value] = self.__load_prompt(prompt_type.get_prompt_filename())

        return templates