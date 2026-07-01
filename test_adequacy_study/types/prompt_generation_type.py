from enum import Enum


class PromptGenerationType(str, Enum):
    FROM_CODE_MINIMAL = "from_code"
    FROM_CODE_DETAILED = "from_code_detailed"
    FROM_ORIGINAL_CODE_DETAILED = "from_original_code_detailed"
    FROM_DIFF = "from_diff"
    FROM_COMBINED = "from_combined"

    def get_prompt_filename(self):
        if self == PromptGenerationType.FROM_ORIGINAL_CODE_DETAILED:
            return "generate_prompt_from_code_detailed"

        return f"generate_prompt_{self.value}"
