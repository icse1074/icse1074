from enum import Enum


class RefinementMode(str, Enum):
    NONE = "none"
    LINE_COVERAGE = "line"
    BRANCH_COVERAGE = "branch"
    MUTATION_COVERAGE = 'mutation'