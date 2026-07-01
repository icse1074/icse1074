from test_adequacy_study.data_models.code_under_test import CUT
from test_adequacy_study.data_models.coverage_report import CoverageReport


def get_missing_line_messages(cut: CUT, coverage_report: CoverageReport) -> str:
    """
    Returns a string that shall be fed to the model regarding the lines that are not covered by the test suite

    :param cut:
    :param coverage_report:
    :return:
    """
    code_lines = cut.content.splitlines()
    missing_lines_with_code = "\n".join(
        f"  line {ln}: {code_lines[ln - 1]}"
        for ln in coverage_report.missing_lines
        if 0 < ln <= len(code_lines)
    )

    return missing_lines_with_code


def get_missing_branch_messages(cut: CUT, coverage_report: CoverageReport) -> str:
    """
    Returns a string that shall be fed to the model regarding the branches that are not covered by the test suite

    :param cut:
    :param coverage_report:
    :return:
    """
    code_lines = cut.content.splitlines()
    messages = []

    #print(code_lines)
    for source, dest in coverage_report.missing_branches:
        if dest == -1:
            msg = f"In line {source} the condition was never False (the path exiting this block/function was missed): {code_lines[source - 1]}"
        elif source == dest:
            msg = f"In line {source} the implicit loop exit or internal branch path was never covered: {code_lines[source - 1]}"
        else:
            numbered = "\n".join(
                f"{i + 1}: {line}" for i, line in enumerate(code_lines[source - 1:dest], start=source - 1))
            msg = f"Line {source} branch to line {dest} was not covered:\n{numbered}"
            #msg = f"In line {source}, the branch that jumps to line {dest} was not covered: {code_lines[source - 1]}"

        messages.append(msg)

    return "\n".join(messages)

