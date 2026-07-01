import ast


class OracleMasker(ast.NodeTransformer):
    """Masks oracle values in assertions while preserving test structure."""

    def visit_Compare(self, node: ast.Compare):
        """Mask comparison values: a == 5 -> a == '<MASK>'"""
        node.left = self.visit(node.left)
        node.comparators = [ast.Constant(value="<MASK>")] * len(node.comparators)
        return node

    def visit_Subscript(self, node: ast.Subscript):
        """Preserve subscript structure: result[0] stays as is"""
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        """Preserve attribute access: obj.attr stays as is"""
        node.value = self.visit(node.value)
        return node

    def visit_Call(self, node: ast.Call):
        """Handle different assertion and comparison function calls."""
        func_name = self._get_func_name(node.func)

        # ========================
        # isinstance - SPECIAL CASE
        # ========================
        if func_name == "isinstance":
            # isinstance(x, SomeType) -> isinstance(x, '<MASK>')
            node.args[0] = self.visit(node.args[0])
            if len(node.args) >= 2:
                node.args[1] = ast.Constant(value="<MASK>")
            return node

        # ========================
        # Two-argument assertion methods
        # ========================
        two_arg_assertions = {
            'assertEqual', 'assertNotEqual', 'assertIn', 'assertNotIn',
            'assertLess', 'assertLessEqual', 'assertGreater', 'assertGreaterEqual',
            'assertIsInstance', 'assertNotIsInstance', 'assertIs', 'assertIsNot',
        }

        if func_name in two_arg_assertions:
            if len(node.args) >= 2:
                node.args[0] = self.visit(node.args[0])
                node.args[1:] = [ast.Constant(value="<MASK>")] * (len(node.args) - 1)
            return node

        # ========================
        # Single-argument assertion methods
        # ========================
        single_arg_assertions = {
            'assertTrue', 'assertFalse', 'assertIsNone', 'assertIsNotNone',
            'assertRaises', 'raises'
        }

        if func_name in single_arg_assertions:
            node.args = [ast.Constant(value="<MASK>")] * len(node.args)
            return node

        # ========================
        # isclose/assert_almost_equal
        # ========================
        if func_name.endswith("isclose"):
            if len(node.args) >= 2:
                node.args[0] = self.visit(node.args[0])
                node.args[1] = ast.Constant(value="<MASK>")
            return node

        # ========================
        # Default: mask all args except first
        # ========================
        node.func = self.visit(node.func)
        if len(node.args) > 1:
            node.args[0] = self.visit(node.args[0])
            node.args[1:] = [ast.Constant(value="<MASK>")] * (len(node.args) - 1)
        elif len(node.args) == 1:
            node.args[0] = self.visit(node.args[0])

        node.keywords = [
            ast.keyword(kw.arg, ast.Constant(value="<MASK>"))
            for kw in node.keywords
        ]

        return node

    def _get_func_name(self, func):
        """Extract function name from various node types."""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""



