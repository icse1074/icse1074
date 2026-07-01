"""

For each triggering test:
1. Find the lcoation of the test method (first and last line)
2. Extract the test code
3. Two options :
    a. Remove all assertions
    b. remove oracles
4. Return prefix, assertions, assertions without oracles
"""

import ast
import argparse
import copy
import textwrap
from typing import List, Optional, Dict
from dataclasses import dataclass

from tqdm import tqdm

from test_adequacy_study.file_utils import write_jsonl, read_jsonl
from test_adequacy_study.oracle_completion.visitors.assertion_collector import AssertionCollector, AssertionInfo
from test_adequacy_study.oracle_completion.visitors.docstring_remover import DocstringRemover
from test_adequacy_study.oracle_completion.visitors.oracle_indication_remover import OracleIndicationRemover
from test_adequacy_study.oracle_completion.visitors.oracle_masker import OracleMasker



@dataclass
class TestParts:
    prefix: str
    assertion_infos: list[AssertionInfo]

    # Convenience properties
    @property
    def assertions(self) -> list[str]:
        return [info.text for info in self.assertion_infos]

    @property
    def assertions_without_oracles(self) -> list[str]:
        return [info.masked_text for info in self.assertion_infos]

@dataclass
class ExtractedTest:
    test_node: str
    start_line: int
    end_line: int
    original_code: str
    test_parts: TestParts

    def reconstruct_with_masked_assertions(self) -> str:
        lines = self.original_code.split('\n')

        # Process bottom-up: slicing/replacing a line span changes list length,
        # which would invalidate the start/end indices of any assertion above
        # it if we went top-down. Descending order avoids that entirely.
        sorted_infos = sorted(
            self.test_parts.assertion_infos,
            key=lambda info: info.start_line,
            reverse=True,
        )

        for info in sorted_infos:
            relative_start = info.start_line - self.start_line
            relative_end = info.end_line - self.start_line

            lines[relative_start:relative_end + 1] = [info.indent + info.masked_text]

        code = '\n'.join(lines)

        leading_indent = code[: len(code) - len(code.lstrip(' '))]
        #dedented = textwrap.dedent(code)
        indent_len = len(leading_indent)  # e.g. len("    ")

        dedented_lines = []
        for line in code.split('\n'):
            if line.startswith(leading_indent):
                dedented_lines.append(line[indent_len:])
            else:
                # line has less leading whitespace than expected —
                # this happens inside multi-line string literals; leave as-is
                dedented_lines.append(line)
        dedented = '\n'.join(dedented_lines)

        try:
            tree = ast.parse(dedented)
        except SyntaxError as e:
            print(f"Failed to parse reconstructed masked test for {self.test_node}: {e}")
            return None

        tree = OracleIndicationRemover().visit(tree)
        tree = DocstringRemover().visit(tree)
        unparsed = ast.unparse(tree)

        if leading_indent:
            unparsed = '\n'.join(
                leading_indent + line if line.strip() else line
                for line in unparsed.split('\n')
            )

        return unparsed

class AssertionProcessor:
    """Handles all assertion-related operations"""

    def collect(self, method_node: ast.FunctionDef, source_lines: list[str]) -> list[AssertionInfo]:
        """Collect all assertions with info."""
        collector = AssertionCollector(source_lines)
        collector.visit(method_node)
        return collector.assertion_infos

    def mask_assertion_infos(self, assertion_infos: list[AssertionInfo]) -> list[AssertionInfo]:
        """Mask the text in assertion infos."""
        for info in assertion_infos:
            info.masked_text = self._mask_assertion(info.text)
        return assertion_infos

    def _mask_assertion(self, assertion: str) -> str:
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

    def mask_values(self, assertions: List[str]) -> List[str]:
        masked = []
        for assertion in assertions:
            try:
                masked.append(self._mask_assertion(assertion))
            except Exception:
                masked.append(assertion)
        return masked

    def remove_assertions(self, node: ast.AST) -> None:
        def filter_statements(stmts: List[ast.stmt]) -> List[ast.stmt]:
            return [stmt for stmt in stmts if not self._is_assertion(stmt)]

        for child in ast.iter_child_nodes(node):
            self.remove_assertions(child)

        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = filter_statements(node.body)

        elif isinstance(node, ast.If):
            node.body = filter_statements(node.body)
            node.orelse = filter_statements(node.orelse)

        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            node.body = filter_statements(node.body)
            node.orelse = filter_statements(node.orelse)

        elif isinstance(node, (ast.With, ast.AsyncWith)):
            node.body = filter_statements(node.body)

        elif isinstance(node, ast.Try):
            node.body = filter_statements(node.body)
            node.orelse = filter_statements(node.orelse)
            node.finalbody = filter_statements(node.finalbody)
            for handler in node.handlers:
                handler.body = filter_statements(handler.body)

    def _is_assertion(self, stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Assert):
            return True

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func

            if isinstance(func, ast.Attribute):
                name = func.attr.lower()
                return (
                        name.startswith("assert")
                        or "assert" in name
                )

            if isinstance(func, ast.Name):
                name = func.id.lower()
                return name.startswith("assert")

        return False


class TestExtractor:
    """Extracts and processes test methods from source files."""

    def __init__(self, test_file_code: str):
        self.source_lines = test_file_code.splitlines()
        self.tree = ast.parse(test_file_code)
        self.classes = self._build_class_map()
        self.functions = self._build_function_map()
        self.assertion_processor = AssertionProcessor()

    def extract_test(self, test_identifier: str) -> Optional[ExtractedTest]:
        """Extract a specific test method.

        Supports formats:
        - "TestClass::test_method" (class method)
        - "test_function" (standalone function)
        """
        # Parse identifier
        if "::" in test_identifier:
            # Class method
            test_class, test_name = test_identifier.split("::", 1)
            return self._extract_from_class(test_class, test_name)
        else:
            # Standalone function
            return self._extract_from_function(test_identifier)

    def _extract_from_class(self, test_class: str, test_name: str) -> Optional[ExtractedTest]:
        """Extract test from a class method."""
        method_node = self._get_method(test_class, test_name)
        if not method_node:
            return None

        return self._process_test_node(method_node, f"{test_class}::{test_name}")

    def _extract_from_function(self, test_name: str) -> Optional[ExtractedTest]:
        """Extract test from a standalone function."""
        func_node = self.functions.get(test_name)
        if not func_node:
            return None

        return self._process_test_node(func_node, test_name)

    def _process_test_node(self, node: ast.FunctionDef, test_node_id: str) -> ExtractedTest:
        """Process a test node (class method or standalone function)."""
        # Collect assertions with all info
        assertion_infos = self.assertion_processor.collect(node, self.source_lines)

        # Mask them
        assertion_infos = self.assertion_processor.mask_assertion_infos(assertion_infos)

        # Process the method
        node_copy = copy.deepcopy(node)
        node_copy = OracleIndicationRemover().visit(node_copy)
        self.assertion_processor.remove_assertions(node_copy)

        DocstringRemover().visit(node_copy)
        prefix = ast.unparse(node_copy)

        try :
            return ExtractedTest(
                test_node=test_node_id,
                start_line=node.lineno,
                end_line=node.end_lineno,
                original_code='\n'.join(self.source_lines[node.lineno - 1:node.end_lineno]),
                test_parts=TestParts(
                    prefix=prefix,
                    assertion_infos=assertion_infos,
                )
            )
        except Exception as e:
            print(f"Failed to extract {test_node_id}: {e}")

    def _build_class_map(self) -> Dict[str, Dict]:
        """Build map of test classes and their methods."""
        classes = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {
                    'node': node,
                    'methods': {
                        item.name: item
                        for item in node.body
                        if isinstance(item, ast.FunctionDef)
                    }
                }
        return classes

    def _build_function_map(self) -> Dict[str, ast.FunctionDef]:
        """Build map of standalone test functions (not in classes)."""
        functions = {}
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                functions[node.name] = node
        return functions

    def _get_method(self, test_class: str, test_name: str) -> Optional[ast.FunctionDef]:
        """Get a specific test method from a class."""
        if test_class not in self.classes:
            return None
        return self.classes[test_class]['methods'].get(test_name)


def extract_tests(
        test_file_code: str,
        test_identifiers: List[str]
) -> List[ExtractedTest]:
    """Extract multiple tests from code.

    Supports:
    - "TestClass::test_method" (class methods)
    - "test_function" (standalone functions)
    """
    extractor = TestExtractor(test_file_code)
    results = []

    for test_id in tqdm(test_identifiers, desc="Extracting tests"):
        extracted = extractor.extract_test(test_id)
        if extracted:
            results.append(extracted)
        else:
            print(f"Test not found: {test_id}")


    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tests from JSONL file")
    parser.add_argument("--triggering-tests-path", required=True, help="Input JSONL file with triggering tests")
    parser.add_argument("--output", required=False, help="Output JSON file with extracted tests")
    args = parser.parse_args()


    # Read JSONL file and extract tests
    benchmark = "bcb"
    for variation in [""] :
        for model in ["gpt-5-mini"]:

            triggering_path = f"output/augmented_benchmarks/triggering_tests/{benchmark}/{model}/tests.jsonl"
            triggering_tests = read_jsonl(triggering_path)
            extracted_results = []

            for task in tqdm(triggering_tests, desc="Tasks"):
                test_file_code = task.get("response", None)
                triggering_test_list = task.get("triggering_tests", [])
                if not len(triggering_test_list) :
                    continue
                # Extract all tests for this task
                extracted = extract_tests(test_file_code, triggering_test_list)
                if len(extracted) == 0:
                    print(f"No triggering tests found in {task['task_id']} : \n{test_file_code}")
                # Convert to serializable format
                extracted_results = []
                for test in extracted:
                    extracted_results.append({
                        "test_node": test.test_node,
                        "start_line": test.start_line,
                        "end_line": test.end_line,
                        "original_code": test.original_code,
                        "test_with_masked_assertions": test.reconstruct_with_masked_assertions(),
                        "test_parts": {
                            "prefix": test.test_parts.prefix,
                            "assertions": test.test_parts.assertions,  # Property
                            "assertions_without_oracles": test.test_parts.assertions_without_oracles,  # Property
                            "assertion_infos": [
                                {
                                    "text": info.text,
                                    "masked_text": info.masked_text,
                                    "start_line" : info.start_line,
                                    "end_line" : info.end_line,
                                    "indent": info.indent,
                                }
                                for info in test.test_parts.assertion_infos
                            ]
                        }
                    })

                task_info = dict(
                    task_id=task["task_id"],
                    code_under_test=task["code_under_test"],
                    test_suite=task["response"],
                    processed_tests=extracted_results
                )
                write_jsonl(triggering_path.replace("triggering_tests", "processed_tests"), [task_info], append=True)


