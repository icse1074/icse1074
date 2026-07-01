from __future__ import annotations

import ast
from pathlib import Path

from test_adequacy_study.data_models.task import Task
from test_adequacy_study.helpers.parsers import is_syntactically_valid


class CUT:
    def __init__(self, task : Task, implementation: str, language: str):
        self.task_id = task.task_id
        self.language = language
        self.entry_point = task.entry_point
        self.content = self._assemble_content(task.stub, implementation)
        self.implementation = implementation
        self.file_path = None
        self.syntactically_valid = is_syntactically_valid(self.content)

    def _extract_signature(self, stub: str) -> str | None :
        """
        Extract the bare function signature from a HumanEval stub.
        e.g. "def has_close_elements(numbers: List[float], threshold: float) -> bool:"
        """
        try:
            tree = ast.parse(stub)
        except SyntaxError:
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = ast.unparse(node.args)
                returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                return f"def {node.name}({args}){returns}:"

        return None

    def _signatures_match(self, stub: str, implementation: str) -> bool:
        """
        Returns True if the implementation already contains a definition of the
        same function as the stub, tolerating differences in default argument values.

        Comparison is done at the AST level:
          - function name must match
          - argument names and annotations must match
          - return annotation must match
          - default *values* are intentionally ignored
        """
        try:
            stub_tree = ast.parse(stub)
            impl_tree = ast.parse(implementation)
        except SyntaxError:
            return False

        def _sig_key(func: ast.FunctionDef) -> tuple:
            """Stable key that ignores default values."""
            args = func.args
            arg_names = [a.arg for a in args.args]
            arg_annots = [ast.unparse(a.annotation) if a.annotation else "" for a in args.args]
            returns = ast.unparse(func.returns) if func.returns else ""
            return (func.name, tuple(zip(arg_names, arg_annots)), returns)

        stub_keys = {
            _sig_key(node)
            for node in ast.walk(stub_tree)
            if isinstance(node, ast.FunctionDef)
        }
        impl_keys = {
            _sig_key(node)
            for node in ast.walk(impl_tree)
            if isinstance(node, ast.FunctionDef)
        }

        return bool(stub_keys & impl_keys)

    def _assemble_content(self, stub: str, implementation: str) -> str:
        """
        Assemble a full executable program by combining imports + signature + implementation.
        Ensures proper ordering and clean separation.
        """
        if self._signatures_match(stub, implementation):
            return implementation

        # Add indent to function body
        lines = implementation.splitlines()
        already_indented = any(
            line.startswith("    ") for line in lines if line.strip()
        )

        if not already_indented:
            implementation = "\n".join(
                ("    " + line if line.strip() else line)
                for line in lines
            )

        return f"{stub}\n{implementation}\n"

    def to_file(self, file_path: str) -> str:
        """
        Creates a temporary file and writes content to it.
        """

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        #write content to file
        path.write_text(self.content, encoding="utf-8")

        self.file_path = str(path)

        return self.file_path

    def __str__(self):
        return f"CUT(task_id={self.task_id}, content={self.content}, syntactically_valid={self.syntactically_valid})"