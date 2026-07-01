import ast
from typing import Optional

ORACLE_INDICATOR_NAMES = {
    'expected', 'expected_value', 'expected_result', 'expected_output',
    'oracle', 'gold', 'target', 'gold_standard', 'ground_truth',
    'ref', 'reference', 'baseline',
    'true_value', 'correct', 'correct_value', 'valid',
    'should_be', 'supposed', 'check', 'verify',
}


class OracleIndicationRemover(ast.NodeTransformer):
    """Remove statements that assign to oracle/expected value variables."""

    def __init__(self):
        self.removed_statements = []

    @staticmethod
    def _is_oracle_name(name: str) -> bool:
        name = name.lower()
        return name in ORACLE_INDICATOR_NAMES or name.startswith("expected")

    def _targets_oracle_name(self, target: ast.expr) -> bool:
        """Check if a single assignment target is (or contains) an oracle-indicating name."""
        if isinstance(target, ast.Name):
            return self._is_oracle_name(target.id)

        if isinstance(target, ast.Attribute):
            # e.g. self.expected_mean = ...
            return self._is_oracle_name(target.attr)

        if isinstance(target, ast.Subscript):
            # e.g. results['expected'] = ... -- check string-literal keys
            sl = target.slice
            # Py3.9+: slice is the value directly; older versions wrap it in ast.Index
            if isinstance(sl, ast.Index):
                sl = sl.value
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                return self._is_oracle_name(sl.value)
            return False

        if isinstance(target, (ast.Tuple, ast.List)):
            # e.g. expected_mean, expected_median = ...
            return any(self._targets_oracle_name(elt) for elt in target.elts)

        if isinstance(target, ast.Starred):
            return self._targets_oracle_name(target.value)

        return False

    def visit_Assign(self, node: ast.Assign) -> Optional[ast.AST]:
        """Remove assignment statements to oracle-indicating variables."""
        for target in node.targets:
            if self._targets_oracle_name(target):
                self.removed_statements.append(ast.unparse(target))
                return None  # Remove this statement

        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Optional[ast.AST]:
        """Remove annotated assignments, e.g. `expected: float = 2.0`."""
        if self._targets_oracle_name(node.target):
            self.removed_statements.append(ast.unparse(node.target))
            return None

        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> Optional[ast.AST]:
        """Remove augmented assignments, e.g. `expected += 1`."""
        if self._targets_oracle_name(node.target):
            self.removed_statements.append(ast.unparse(node.target))
            return None

        return self.generic_visit(node)