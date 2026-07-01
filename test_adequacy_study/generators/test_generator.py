from pathlib import Path

from test_adequacy_study.generators.abstract_code_generator import AbstractCodeGenerator
from test_adequacy_study.types.refinement_mode import RefinementMode
from test_adequacy_study.types.test_generation_type import TestGenerationType


class TestGenerator(AbstractCodeGenerator):
    system_prompt_file = "test_generation_system"
    default_prompts_dir = Path(__file__).resolve().parent / "prompts"

    def __init__(
            self,
            model: str = "gpt-4o-mini",
            prompt_dir: str = default_prompts_dir,
            temperature: float = 0.1,
            refinement_mode: RefinementMode = RefinementMode.NONE,
            generation_type: TestGenerationType = TestGenerationType.REGULAR
    ):
        self.refinement_mode = refinement_mode
        self.generation_type = generation_type
        self.task_prompt_file = self._get_initial_prompt_file()
        super().__init__(model, prompt_dir, temperature)


    def _get_initial_prompt_file(self) -> str:
        if self.generation_type == TestGenerationType.REGULAR:
            if self.refinement_mode in [RefinementMode.LINE_COVERAGE, RefinementMode.BRANCH_COVERAGE]:
                return "test_generation_task_coverage"
            if self.refinement_mode == RefinementMode.MUTATION_COVERAGE:
                return "test_generation_task_mutation"
            if self.refinement_mode == RefinementMode.NONE:  # plain LLM
                return "test_generation_task"

        elif self.generation_type == TestGenerationType.AUGMENTATION:
            return "ground_truth_tests_augmentation"
        elif self.generation_type == TestGenerationType.ORACLE_COMPLETION:
            return "oracle_completion"
        elif self.generation_type == TestGenerationType.ASSERTION_GENERATION:
            return "assertion_generation"
        elif self.generation_type == TestGenerationType.FROM_ORIGINAL_PROMPT or self.generation_type == TestGenerationType.FROM_DIFF_PROMPT:
            return "test_generation_from_prompt"
        elif self.generation_type == TestGenerationType.PROMPTS_DELTA:
            return "test_generation_from_prompts_delta"
        elif self.generation_type == TestGenerationType.FROM_PROMPT_AND_CODE:
            return "test_generation_from_prompt_and_code"
        else:
            ValueError(f"No followup prompt for criteria : {str(self.refinement_mode)}")
            return None

    def _get_followup_prompt_file(self, prompt_variables: dict[str, str]) -> str:
        if "test_id" in prompt_variables:
            return "one_test_error_fix"
        if "errors" in prompt_variables:
            return "error_fix"
        if "tests" in prompt_variables:
            if self.refinement_mode == RefinementMode.LINE_COVERAGE:
                return "line_coverage_augmentation"
            elif self.refinement_mode == RefinementMode.BRANCH_COVERAGE:
                return "branch_coverage_augmentation"
            elif self.refinement_mode == RefinementMode.MUTATION_COVERAGE:
                return "mutation_coverage_augmentation"
            else:
                ValueError(f"No followup prompt for refinement mode: {str(self.refinement_mode)}")
        raise ValueError(f"No followup prompt for variables: {list(prompt_variables.keys())}")