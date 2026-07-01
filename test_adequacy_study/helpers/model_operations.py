def convert_system_prompt_to_anthropic(system_prompt: dict) -> list[dict]:
    return [
        {
            "type": "text",
            "text": system_prompt["content"]
        }
    ]