import ast
import copy
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

from test_adequacy_study.oracle_completion.extract_test_prefix import AssertionProcessor
from test_adequacy_study.oracle_completion.visitors.oracle_indication_remover import OracleIndicationRemover
from test_adequacy_study.oracle_completion.visitors.oracle_masker import OracleMasker


ORACLE_MASK_CASES = [
    # simple comparisons
    ("a == 5", "a == '<MASK>'"),
    ("a == result[1]", "a == '<MASK>'"),
    ("a == result[x]", "a == '<MASK>'"),

    # subscripts & slices
    ("result[0] == 1", "result[0] == '<MASK>'"),
    ("result[1:3] == [2, 3]", "result[1:3] == '<MASK>'"),

    # literals
    ("result == {'a': 1}", "result == '<MASK>'"),
    ("result == {1, 2, 3}", "result == '<MASK>'"),
    ("result == (1, 2, 3)", "result == '<MASK>'"),

    # function calls
    ("len(result) == 5", "len(result) == '<MASK>'"),
    ("a == get_value()", "a == '<MASK>'"),
    ("len(str(result)) == 10", "len(str(result)) == '<MASK>'"),

    # chained comparisons
    ("a < 5 < c", "a < '<MASK>' < '<MASK>'"),

    # boolean logic
    ("(a == 5) and (b == 10)", "a == '<MASK>' and b == '<MASK>'"),
    ("(a == 5) or (b == 10)", "a == '<MASK>' or b == '<MASK>'"),

    # unittest style
    ("self.assertEqual(a, 5)", "self.assertEqual(a, '<MASK>')"),
    ("assertEqual(a, 5)", "assertEqual(a, '<MASK>')"),
    ("self.assertTrue(result)", "self.assertTrue('<MASK>')"),
    ("self.assertRaises(ValueError)", "self.assertRaises('<MASK>')"),
    ("self.assertRaises((ValueError, TypeError))", "self.assertRaises('<MASK>')"),

    # weird / external
    ("np.isclose(result, expected)", "np.isclose(result, '<MASK>')"),
    ("pd.testing.assert_frame_equal(df, expected_df)", "pd.testing.assert_frame_equal(df, '<MASK>')"),
    ("self.assertIs(ax, self.mock_ax)", "self.assertIs(ax, '<MASK>')"),

    # edge cases
    ("assert np.isclose(result, expected)", "assert np.isclose(result, '<MASK>')"),
    ("assert isinstance(result, (float, np.floating))", "assert isinstance(result, '<MASK>')"),
]

class TestOracleMasker(unittest.TestCase):

    def setUp(self):
        self.masker = OracleMasker()

    def mask_assertion(self, assertion: str) -> str:
        masker = OracleMasker()

        is_assert = assertion.strip().startswith('assert ')
        test_expr = assertion.strip()[7:].strip() if is_assert else assertion

        try:
            tree = ast.parse(test_expr, mode='eval')
            masked_tree = masker.visit(tree.body)
            result = ast.unparse(masked_tree)

            if result == test_expr and isinstance(tree.body, (ast.Subscript, ast.Name)):
                result = "<MASK>"

            if is_assert:
                result = f"assert {result}"

            return result
        except Exception:
            return assertion

    def test_oracle_cases(self):
        for inp, expected in ORACLE_MASK_CASES:
            with self.subTest(inp=inp):
                self.assertEqual(self.mask_assertion(inp), expected)


class TestAssertionProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = AssertionProcessor()

    def test_mask_values_matches_oracle(self):
        inputs = [c[0] for c in ORACLE_MASK_CASES]
        expected = [c[1] for c in ORACLE_MASK_CASES]

        result = self.processor.mask_values(inputs)

        for i, (r, e) in enumerate(zip(result, expected)):
            with self.subTest(case=inputs[i]):
                self.assertEqual(r, e)

    def _make_method(self, code: str):
        tree = ast.parse(code)
        return tree.body[0]

    def test_remove_all_oracle_assertions(self):
        """
        Ensure ALL oracle-style assertions are removed from code,
        including unittest, numpy, and function-call assertions.
        """

        # Build a synthetic test function using real oracle inputs
        oracles = [
            f"    {case[0]}"
            for case in ORACLE_MASK_CASES
        ]

        assertion_lines = []
        for oracle in oracles:
            if "assert" in oracle:
                assertion_lines.append(oracle)
            else :
                assertion_lines.append("    assert " + oracle)


        code = "def test():\n" + "\n".join(assertion_lines)

        method = self._make_method(code)
        method_copy = copy.deepcopy(method)

        self.processor.remove_assertions(method_copy)
        out = ast.unparse(method_copy)

        for inp, _ in ORACLE_MASK_CASES:
            # strip "assert " prefix variations
            stripped = inp.replace("assert ", "").strip()

            with self.subTest(assertion=inp):
                self.assertNotIn(stripped, out)

        # sanity check: function still exists and is valid
        self.assertIn("def test", out)

class TestOracleIndicationRemover(unittest.TestCase):
    """Test OracleIndicationRemover for removing oracle-indicating variables."""

    def setUp(self):
        self.remover = OracleIndicationRemover()

    def _make_method(self, code: str) -> ast.FunctionDef:
        """Helper to create a function from code string."""
        tree = ast.parse(code)
        return tree.body[0]

    def test_remove_expected_variable(self):
        """Test removing 'expected' variable"""
        code = """
def test_add():
    result = 2 + 3
    expected = 5
    oracle = 67
    assert result == expected
    assert result < oracle
"""
        expected = """def test_add():
    result = 2 + 3
    assert result == expected
    assert result < oracle"""
        method = self._make_method(code)
        method_copy = copy.deepcopy(method)
        method_copy = self.remover.visit(method_copy)
        unparsed = ast.unparse(method_copy)

        self.assertEqual(expected, unparsed)

    def test_case_insensitive_removal(self):
        """Test that removal is case-insensitive"""
        code = """
def test_case():
    Expected = 5
    ORACLE = 10
    x = 1
"""
        expected = """def test_case():\n    x = 1"""
        method = self._make_method(code)
        method_copy = copy.deepcopy(method)
        method_copy = self.remover.visit(method_copy)
        unparsed = ast.unparse(method_copy)

        assert expected == unparsed

    def test_case_np(self):
        code = """def test_case():\n    assert np.isclose(result, expected)"""

        method = self._make_method(code)
        method_copy = copy.deepcopy(method)
        method_copy = self.remover.visit(method_copy)
        unparsed = ast.unparse(method_copy)

        assert code == unparsed




if __name__ == '__main__':
    unittest.main()