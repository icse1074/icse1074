import re
import ast
import ast


def parse_code_from_markdown(raw: str) -> str:
    """
    Parses a markdown response typically returned by an LLM, that contains code implementation

    :param raw:
    :return:
    """

    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    return exclude_think_output(raw.strip())


def exclude_think_output(raw_content: str) -> str:
    """
    Removes <think>...</think> blocks from the LLM output.
    """
    clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)

    # Return stripped to remove any leading/trailing whitespace left behind
    return clean_content.strip()



def is_syntactically_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(e)
        return False


def convert_assertions(assertion_list: list[str]) -> tuple[str, list[tuple[any, any]]]:
    converted_outputs = []
    entry_point = "unknown_function"

    for assert_str in assertion_list:
        try:
            clean_str = assert_str.strip()
            if clean_str.startswith("assert "):
                clean_str = clean_str[7:]

            tree = ast.parse(clean_str, mode="eval")

            if isinstance(tree.body, ast.Compare) and isinstance(tree.body.ops[0], ast.Eq):
                left_side = tree.body.left
                right_side = tree.body.comparators[0]

                if isinstance(left_side, ast.Call):
                    entry_point = (
                        left_side.func.id
                        if isinstance(left_side.func, ast.Name)
                        else "unknown_function"
                    )

                    # Evaluate all arguments into a list
                    args_list = [ast.literal_eval(arg) for arg in left_side.args]
                    expected_val = ast.literal_eval(right_side)

                    # KEY FIX FOR YOUR LOOP:
                    # If there's only 1 argument, extract it from the list.
                    # If there are multiple, keep them as a tuple so `*inp` can unpack them.
                    inputs = args_list[0] if len(args_list) == 1 else tuple(args_list)

                    # This guarantees a strict 2-element tuple: (inp, exp)
                    converted_outputs.append((inputs, expected_val))

        except Exception as e:
            # Note: This will break your loop unpacking if an error occurs.
            # Consider logging this or handling it outside.
            pass

    return entry_point, converted_outputs

class FunctionRenamer(ast.NodeTransformer):
    """Safely renames function definitions and internal recursive calls."""

    def __init__(self, name_map):
        self.name_map = name_map

    def visit_FunctionDef(self, node):
        if node.name in self.name_map:
            node.name = self.name_map[node.name]
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.name_map:
            node.id = self.name_map[node.id]
        return node


def align_code_to_list(source_code, test_list):
    if not test_list:
        return source_code

    # 1. Grab the first assertion string (e.g., "assert target_func(1) == 2")
    first_assert = test_list[0]

    # 2. Extract the function name using basic string splitting
    # Split by 'assert ', then split by the opening parenthesis '('
    try:
        expected_name = first_assert.split("assert")[1].split("(")[0].strip()
    except IndexError:
        return source_code  # Fallback if the string structure is unexpected

    # 3. Get the current function name from the code
    tree = ast.parse(source_code)
    current_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]

    if not current_names or current_names[0] == expected_name:
        return source_code

    # 4. Rename it safely
    transformer = FunctionRenamer({current_names[0]: expected_name})
    modified_tree = transformer.visit(tree)
    ast.fix_missing_locations(modified_tree)

    return ast.unparse(modified_tree)


import ast


def get_function_signature(code_str: str) -> str:
    """
    Parses a string of Python code, finds the first function definition,
    and returns its complete signature line.
    """
    try:
        # Parse the code into an Abstract Syntax Tree
        tree = ast.parse(code_str)

        # Iterate through the nodes to find the first FunctionDef
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Split the original code into lines
                lines = code_str.splitlines()

                # Get the starting line of the function (ast lines are 1-indexed)
                start_line = node.lineno - 1

                # Find where the signature ends (at the colon)
                # We loop in case the signature spans multiple lines
                signature_lines = []
                for i in range(start_line, len(lines)):
                    signature_lines.append(lines[i])
                    # If we find the closing ':' for the def statement, we stop
                    if ':' in lines[i]:
                        break

                # Join the lines and strip any trailing code/whitespace past the colon
                full_signature = "\n".join(signature_lines)
                return full_signature.split(':', 1)[0].strip() + ":"

    except SyntaxError:
        return "Error: Invalid Python code provided."

    return "No function definition found."
