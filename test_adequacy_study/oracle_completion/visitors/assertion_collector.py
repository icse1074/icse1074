import ast
from dataclasses import dataclass


@dataclass
class AssertionInfo:
    """Complete information about an assertion."""
    text: str  # Original assertion
    masked_text: str  # Masked version
    start_line: int
    end_line: int
    indent: str

class AssertionCollector(ast.NodeVisitor):
    def __init__(self, source_lines: list[str]):
        self.assertion_infos = []
        self.source_lines = source_lines

    def visit_Assert(self, node: ast.Assert):
        condition = ast.unparse(node.test)
        indent = self._get_indent(node.lineno)
        self.assertion_infos.append(AssertionInfo(
            text=condition,
            masked_text="",  # Will be filled later
            start_line=node.lineno,
            end_line=node.end_lineno,
            indent=indent
        ))
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            call = node.value
            if self._is_assertion_call(call):
                assertion_text = ast.unparse(call)
                indent = self._get_indent(node.lineno)
                self.assertion_infos.append(AssertionInfo(
                    text=assertion_text,
                    masked_text="",  # Will be filled later
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    indent=indent
                ))
        self.generic_visit(node)

    def _get_indent(self, line_num: int) -> str:
        """Get indentation from original source line."""
        if line_num <= len(self.source_lines):
            line = self.source_lines[line_num - 1]
            return line[:len(line) - len(line.lstrip())] if line.strip() else ""
        return ""

    def _is_assertion_call(self, call: ast.Call) -> bool:
        """Check if a call is an assertion method."""
        if isinstance(call.func, ast.Attribute):
            attr_name = call.func.attr.lower()

            if attr_name.startswith('assert') :
                return True

        elif isinstance(call.func, ast.Name):
            func_name = call.func.id.lower()

            # Direct assertion function calls
            if func_name.startswith('assert'):
                return True

        return False


