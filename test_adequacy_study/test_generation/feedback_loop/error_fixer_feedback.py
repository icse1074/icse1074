# test_adequacy_study/feedback/error_fixer.py
from test_adequacy_study.test_generation.feedback_loop.feedback_loop import FeedbackLoop, FeedbackResult
#from test_adequacy_study.runners.test_runner import RunResult, Verdict
from test_adequacy_study.runners.test_runner import Verdict

class ErrorFixerFeedback(FeedbackLoop):
    """
    Retry when tests crash with execution errors (import errors, syntax
    errors at runtime, etc.). Assertion failures are intentional — skip.
    """

    def evaluate(self, task, tests, run_result, test_id = None, **kwargs) -> FeedbackResult:
        #TODO : check when there is verdictError
        if run_result.verdict in [Verdict.ERROR, Verdict.SYNTAX_ERROR]:
            print("oopsie")
        if test_id :
            detailed_test_results = run_result.detailed_test_results
            for detailed_test_result in detailed_test_results:
                if test_id in detailed_test_result.node_id:
                    if detailed_test_result.outcome == "passed" :
                        return FeedbackResult(retry=False, prompt_variables={})
                    if detailed_test_result.outcome == "failed" :
                        if any(pattern in detailed_test_result.message for pattern in {"assert", "AssertionError", "FileNotFoundError"}) :
                            print("yay fault failed on test")
                            return FeedbackResult(retry=False, prompt_variables={})
                        else :
                            return FeedbackResult(
                                retry=True,
                                prompt_variables={
                                    "tests": tests,
                                    "test_id": test_id,
                                    "errors": detailed_test_result.crash_path + "\n" + detailed_test_result.message
                                },
                            )


        else :
            if run_result.verdict not in [Verdict.ERROR, Verdict.SYNTAX_ERROR]:
                return FeedbackResult(retry=False, prompt_variables={})

            return FeedbackResult(
                retry=True,
                prompt_variables={
                    "tests": tests,
                    "errors": run_result.stderr or run_result.stdout,
                },
            )