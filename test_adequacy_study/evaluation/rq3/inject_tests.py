"""
Multi-test injection helper that tracks line number changes.

When injecting multiple tests into a suite, each injection changes line numbers.
This module handles the sequential injection with automatic line number adjustment.
"""

from typing import List, Dict, Tuple


def inject_method_into_class(complete_test_suite: str,
                             generated_method: str,
                             start_line: int,
                             end_line: int) -> str:
    """
    Replace the method originally at [start_line, end_line] (1-indexed,
    inclusive) in complete_test_suite with generated_method, re-indented
    to match the original method's indentation level.

    Args:
        complete_test_suite: The test suite source code
        generated_method: The method code to inject (may be unindented)
        start_line: 1-indexed start line of method to replace (inclusive)
        end_line: 1-indexed end line of method to replace (inclusive)

    Returns:
        Modified test suite with method replaced
    """
    suite_lines = complete_test_suite.split('\n')

    # Determine original indent from the method's first line (the `def` line)
    original_first_line = suite_lines[start_line - 1]
    base_indent = original_first_line[: len(original_first_line) - len(original_first_line.lstrip())]

    # Re-indent generated_method to match base_indent
    gen_lines = generated_method.split('\n')

    # Find the indent of the first non-empty line in generated_method
    gen_first_indent = 0
    for first_line in gen_lines:
        if first_line.strip():
            gen_first_indent = len(first_line) - len(first_line.lstrip())
            break

    reindented = []
    for line in gen_lines:
        if not line.strip():
            reindented.append('')
            continue
        current_indent = len(line) - len(line.lstrip())
        relative = current_indent - gen_first_indent
        reindented.append(base_indent + ' ' * relative + line.lstrip())

    # Replace the old method with the new one (end_line is inclusive, so we need end_line not end_line-1)
    suite_lines[start_line - 1:end_line] = reindented
    return '\n'.join(suite_lines)


class LineNumberTracker:
    """
    Tracks line number changes as tests are injected sequentially.

    Problem: When you inject test 1 at lines 22-29, the original lines 30+ shift.
    If test 2 was originally at lines 35-42, it's now at 35-42 + delta.

    Solution: Keep track of delta for each injection and adjust subsequent line numbers.
    """

    def __init__(self, base_test_suite: str):
        """
        Initialize with the base test suite.

        Args:
            base_test_suite: The original, unmodified test suite
        """
        self.base_test_suite = base_test_suite
        self.current_suite = base_test_suite
        self.injections = []  # List of (test_node, original_start, original_end, actual_start, actual_end, delta)

    def _calculate_delta(self, original_start: int, original_end: int) -> int:
        """
        Calculate how many lines have been added/removed before this position
        due to previous injections.

        When we inject at lines 1-2 and replace it with 3 lines:
        - Old lines: 1-2 = 2 lines
        - New lines: 3 lines
        - Delta: +1 line

        Subsequent tests that were at lines 4+ will shift by +1
        """
        delta = 0
        for inj_original_start, inj_original_end, num_lines_new in self.injections:
            # If this injection happened BEFORE our target position (ended before we start)
            if inj_original_end < original_start:
                # How many lines were removed and how many added?
                num_lines_old = inj_original_end - inj_original_start + 1
                delta += (num_lines_new - num_lines_old)

        return delta

    def inject(self, test_node: str,
               generated_method: str,
               original_start_line: int,
               original_end_line: int) -> Tuple[bool, str]:
        """
        Inject a test, automatically adjusting for previous injections.

        Args:
            test_node: Name of the test for logging
            generated_method: The test method code to inject
            original_start_line: Start line in the ORIGINAL (base) suite (1-indexed, inclusive)
            original_end_line: End line in the ORIGINAL (base) suite (1-indexed, inclusive)

        Returns:
            Tuple of (success: bool, error_message: str or "")
            On success, self.current_suite is updated.
        """
        try:
            # Calculate where this method actually is now, accounting for previous injections
            delta = self._calculate_delta(original_start_line, original_end_line)
            actual_start = original_start_line + delta
            actual_end = original_end_line + delta

            # Inject into the current (modified) suite
            self.current_suite = inject_method_into_class(
                self.current_suite,
                generated_method,
                actual_start,
                actual_end
            )

            # Track this injection
            num_lines_new = len(generated_method.split('\n'))
            self.injections.append((
                original_start_line,
                original_end_line,
                num_lines_new
            ))

            return True, ""

        except Exception as e:
            return False, str(e)

    def get_current_suite(self) -> str:
        """Get the current test suite with all injections applied."""
        return self.current_suite

    def get_injection_summary(self) -> List[Dict]:
        """Get summary of all injections performed."""
        return [
            {
                'test_node': self.injections[i][0] if i < len(self.injections) else f'unknown_{i}',
                'original_range': f"{orig_start}-{orig_end}",
                'num_lines_new': num_lines_new,
            }
            for i, (orig_start, orig_end, num_lines_new) in enumerate(self.injections)
        ]


def inject_all_tests(base_test_suite: str,
                     completed_tests: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Convenience function: inject all completed tests into base suite.

    Args:
        base_test_suite: Original test suite
        completed_tests: List of dicts with keys:
            - test_node: test name
            - completed_test: test code to inject
            - start_line: original start line (1-indexed, inclusive)
            - end_line: original end line (1-indexed, inclusive)

    Returns:
        Tuple of (modified_suite: str, injection_results: List[Dict])
        injection_results contains:
            - test_node
            - success: bool
            - error: str or ""
    """
    tracker = LineNumberTracker(base_test_suite)
    results = []

    for test_info in completed_tests:
        test_node = test_info.get('test_node')
        completed_test = test_info.get('completed_test')
        start_line = test_info.get('start_line')
        end_line = test_info.get('end_line')

        if not all([completed_test, start_line, end_line]):
            results.append({
                'test_node': test_node,
                'success': False,
                'error': 'Missing required fields'
            })
            continue

        success, error = tracker.inject(test_node, completed_test, start_line, end_line)
        results.append({
            'test_node': test_node,
            'success': success,
            'error': error
        })

    return tracker.get_current_suite(), results

