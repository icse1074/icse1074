from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from test_adequacy_study.generators.openai_model import OpenAIModel
from test_adequacy_study.helpers.parsers import parse_code_from_markdown


class AbstractCodeGenerator(ABC):
    system_prompt_file: str = ""
    task_prompt_file: str = ""

    default_prompts_dir = Path(__file__).resolve().parent / "prompts"

    def __init__(
            self,
            model: str = "gpt-4o-mini",
            prompt_dir: str = default_prompts_dir,
            temperature: float = 0.8,
    ):
        self.prompt_dir = Path(prompt_dir)
        self.call_count = 0
        self.model_id = model
        self.llm = OpenAIModel(model=model, temperature=temperature)
        self._read_prompt_templates()

    def generate(
            self,
            prompt_variables: dict[str, str],
            samples: int = 1,
            history: Optional[List[dict]] = None,
    ) -> Tuple[List[str], Optional[List[dict]]]:

        self.call_count += 1

        if history is None:
            prompts = self._build_prompt(prompt_variables)
            user_message = prompts["user"]
            system_prompt = prompts["system"]
        else:
            user_message = self._build_followup_message(prompt_variables)
            system_prompt = None

        responses, history = self.llm.call(
            prompt=user_message,
            system_prompt=system_prompt,
            n=samples,
            history=history,
        )

        return [parse_code_from_markdown(r) for r in responses], history

    def get_history_with_hardcoded_response(self, prompt_variables, response: str) -> List[dict]:
        """
        Simulates a code generation task but with hardcoded the model's response
        It is useful when you have the response and want to re-execute from that point
        This is an API-ish replication of "Branch into new chat" feature of ChatGPT and Claude

        :param prompt_variables:
        :param response:
        :return:
        """
        prompts = self._build_prompt(prompt_variables)
        user_message = prompts["user"]
        system_prompt = prompts["system"]

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ]

    def _build_followup_message(self, prompt_variables: dict[str, str]) -> str:
        prompt_file = self._get_followup_prompt_file(prompt_variables)
        template = self._load_prompt(prompt_file)
        return template.format(**prompt_variables)

    def _get_followup_prompt_file(self, prompt_variables: dict[str, str]) -> str:
        """Override in subclasses to route followup prompts based on variables."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support followup prompts. "
            f"Override _get_followup_prompt_file() to add support."
        )

    def _read_prompt_templates(self):
        self.system_prompt = self._load_prompt(self.system_prompt_file)
        self.task_prompt = self._load_prompt(self.task_prompt_file)

    def _load_prompt(self, name: str) -> str:
        path = self.prompt_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def _build_prompt(self, prompt_variables: dict[str, str]) -> Dict[str, str]:
        return {
            "system": self.system_prompt,
            "user": self.task_prompt.format(**prompt_variables),
        }