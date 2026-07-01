import ast


class DocstringRemover(ast.NodeTransformer):

    @staticmethod
    def _strip_docstring(body: list) -> list:
        if not body:
            return body

        first = body[0]
        is_docstring = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )

        if is_docstring:
            return body[1:]

        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._strip_docstring(node.body)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._strip_docstring(node.body)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._strip_docstring(node.body)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = self._strip_docstring(node.body)
        return self.generic_visit(node)