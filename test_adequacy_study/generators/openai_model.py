import os

from openai import OpenAI
from typing import List, Optional, Tuple
import time
from dotenv import load_dotenv
load_dotenv()


class OpenAIModel:
    FIXED_TEMPERATURE_MODELS = ["gpt-5-mini"]
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
    ):
        self.model = model
        self.temperature = temperature
        self.__init_model_client()

    @property
    def _temperature_kwargs(self) -> dict:
        """Returns temperature kwarg only for models that support it."""
        if any(self.model.startswith(m) for m in self.FIXED_TEMPERATURE_MODELS):
            return {}
        return {"temperature": self.temperature}


    def call(
        self,
        prompt: str,
        n: int = 1,
        system_prompt: Optional[str] = None,
        history: Optional[List[dict]] = None,
        delay: float = 0.0,
    ) -> Tuple[List[str], Optional[List[dict]]]:

        responses = []

        if n > 1:
            assert history is None
            for _ in range(n):
                messages = []

                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self._temperature_kwargs,
                )

                responses.append(response.choices[0].message.content)

                if delay > 0:
                    time.sleep(delay)

            return responses, messages

        else:
            if history is None:
                history = []

            if system_prompt:
                history.append({"role": "system", "content": system_prompt})

            history.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=history,
                **self._temperature_kwargs,
            )

            text = response.choices[0].message.content
            history.append({"role": "assistant", "content": text})
            responses.append(text)

            if delay > 0:
                time.sleep(delay)

            return responses, history

    def __init_model_client(self):
        """
        Depending on the model name used to instantiate this Object, the method will
        generate the appropriate OpenAI client instance with the related API base_url and key

        At the moment DeepSeek and OpenAI are supported
        :return:
        """
        if self.model.startswith("deepseek"):
            api_key = os.environ["DEEPSEEK_API_KEY"]
            base_url = os.environ["DEEPSEEK_BASE_URL"]
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        elif self.model.startswith("nebius-"):
            self.model = self.model.split("nebius-")[-1]
            if self.model in ["meta-llama_llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct"]:
                self.model = "meta-llama/Llama-3.3-70B-Instruct"
            api_key = os.environ["NEBIUS_API_KEY"]
            base_url = os.environ["NEBIUS_BASE_URL"]
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        elif self.model.startswith("claude"):

            # TODO: Caching is not supported for Anthropic + OpenAI sdk
            self.client = OpenAI(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.environ["ANTHROPIC_BASE_URL"]
            )
        else:
            self.client = OpenAI()
