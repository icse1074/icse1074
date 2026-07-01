from enum import Enum


class TestGenerationType(Enum):
    REGULAR = "regular"
    AUGMENTATION = "augmentation"
    FROM_ORIGINAL_PROMPT = "from_original_prompt"
    FROM_DIFF_PROMPT = "from_diff_prompt"
    PROMPTS_DELTA = "prompts_delta"
    PROMPTS_COMBINED_DELTA = "prompts_combined_delta"
    FROM_PROMPT_AND_CODE = "from_prompt_and_code"
    ORACLE_COMPLETION = "oracle_completion"
    ASSERTION_GENERATION = "assertion_generation"
