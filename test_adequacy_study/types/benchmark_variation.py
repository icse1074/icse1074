from enum import Enum


class BenchmarkVariation(str, Enum):
    NONE = ""
    UNDER_SPECIFIED = "US"
    LEXICAL_VAGUENESS = "LV"
    SYNTAX_FORMATTING = 'SF'
