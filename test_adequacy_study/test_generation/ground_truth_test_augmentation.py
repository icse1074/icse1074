import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Any
from tqdm import tqdm

from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.execution_report import ExecutionReport, Verdict, TestResult
from test_adequacy_study.data_models.test_suite import TestSuite, TestFramework
from test_adequacy_study.file_utils import write_jsonl, read_jsonl
from test_adequacy_study.generators.test_generator import TestGenerator
from test_adequacy_study.helpers.parsers import is_syntactically_valid
from test_adequacy_study.runners.mutation_runner import _same_outcome
from test_adequacy_study.services.batch_processing_service import BatchProcessingService
from test_adequacy_study.runners.test_runner import TestRunner
from test_adequacy_study.types.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

MAX_AUGMENTATION_ATTEMPTS = 5


def __convert_test_results_to_dict(detailed_test_results: list[TestResult]) -> dict[str, dict[str, Any]]:
    """Parses detailed test results into a formatted dictionary keyed by processed node IDs.

    Converts node_ids to strings, stripping leading paths if a slash is present.
    """
    results_dict = {}

    for t in (detailed_test_results):
        str_id = str(t.node_id)
        key = str_id.split("/", 1)[1] if "/" in str_id else str_id

        # 2. Build the dictionary entry
        results_dict[key] = {
            "outcome": t.outcome,
            "message": t.message,
        }

    return results_dict


def _suite_finds_difference(
        test_suite: TestSuite,
        canonical_results: ExecutionReport,
        faulty_cut,
        runner: TestRunner,
) -> bool:
    faulty_results = runner.run(faulty_cut, test_suite)

    if faulty_results.verdict is Verdict.TIMEOUT:
        return "timeout"
    if faulty_results.verdict is Verdict.ERROR:
        print([test for test in faulty_results.detailed_test_results if test.outcome != Verdict.PASSED])
        return "error"

    canonical_results_dict = __convert_test_results_to_dict(canonical_results.detailed_test_results or [])
    faulty_results_dict = __convert_test_results_to_dict(faulty_results.detailed_test_results or [])

    for test in canonical_results_dict:
        if test not in faulty_results_dict:
            continue

        cn = canonical_results_dict[test]
        flt = faulty_results_dict[test]
        if not _same_outcome(cn, flt):
            logger.info(f"{test} -> {cn} -> {flt}")
            return True

    print("same outcome :", )
    print(canonical_results_dict)
    print(faulty_results_dict)
    return False


def _load_batch(generator: TestGenerator, batch_job_id: str) -> dict[str, list[str]]:
    if generator.llm.model.startswith("claude"):
        return BatchProcessingService.get_batch_results_lookup_from_anthropic_api(batch_job_id)
    else:
        return BatchProcessingService.get_batch_results_lookup_from_openai_api(generator.llm, batch_job_id)



@dataclass
class AugmentationResult:
    task_id: str
    model_id: str
    already_detected: list[str] = field(default_factory=list)
    undetected: list[str] = field(default_factory=list)
    augmented_tests: Optional[str] = None
    api_calls: int = 0
    success: bool = False


class TestAugmentationPipeline:

    def __init__(
        self,
        loader: BenchmarkLoader,
        tests_runner: TestRunner,
        generator: TestGenerator,
        builder: ProgramBuilder,
        config: PipelineConfig,
        batch_job_id: str = None,
        generations_path : str = None
    ):
        self.loader = loader
        self.tests_runner = tests_runner
        self.generator = generator
        self.builder = builder
        self.batch_job_id = batch_job_id
        self.config       = config
        self.generations_path = generations_path

        base = config.output_file.removesuffix(".jsonl")
        self.augmented_output_file = f"{base}_augmented.jsonl"

    def _load_from_jsonl(self) -> dict[str, list[str]]:
        """Load generations from a jsonl file into the same format as _load_batch."""
        lookup: dict[str, list[str]] = {}
        for entry in read_jsonl(self.generations_path):
            task_id = str(entry["task_id"])
            completion = entry["completion"]
            lookup.setdefault(task_id, []).append(completion)
        return lookup

    def _augment_task(self, task, generations: list[str]) -> AugmentationResult:
        result = AugmentationResult(
            task_id=task.task_id,
            model_id=self.generator.model_id,
        )

        canonical_cut = self.builder.build_program(task=task, code=task.canonical_solution)

        if not canonical_cut.syntactically_valid:
            logger.debug("[%s] canonical solution has a syntax error — skipping", task.task_id)
            return result

        # Track all suites and their canonical results in sync
        all_test_sources: list[str] = [task.tests]
        initial_suite = TestSuite(task_id=task.task_id, language="python", source=task.tests, framework=TestFramework.PYTEST)
        all_canonical_results: list = [self.tests_runner.run(canonical_cut, initial_suite)]

        current_suite = copy.copy(initial_suite)
        #cut when humaneval because too big
        if "HumanEval" in task.task_id :
            lines = initial_suite.source.splitlines()
            trimmed = lines[:50]
            current_suite.source = "\n".join(trimmed)

        G: list[str] = []
        for gen_code in tqdm(generations, desc="Syntax check"):
            faulty_cut = self.builder.build_program(task=task, code=gen_code)
            if faulty_cut.syntactically_valid:
                G.append(gen_code)

        G_detected: list[str] = []
        G_undetected: list[str] = []

        i = 0
        while G:
            g = G.pop(0)
            logger.info("Currently working on generation [%s]", str(i))
            faulty_cut = self.builder.build_program(task=task, code=g)

            # Check against ALL accumulated suites before augmenting
            caught_by_existing = any(
                _suite_finds_difference(
                    TestSuite(task_id=task.task_id, language="python", source=src, framework=TestFramework.PYTEST),
                    canon_res,
                    faulty_cut,
                    self.tests_runner,
                )
                for src, canon_res in zip(all_test_sources, all_canonical_results)
            )

            if caught_by_existing:
                print("caught by existing suite")
                G_detected.append(g)
                i += 1
                continue

            logger.info("[%s] generation is a potential fault — augmenting (up to %d attempts)", task.task_id,
                        MAX_AUGMENTATION_ATTEMPTS)

            caught = False
            for attempt in range(1, MAX_AUGMENTATION_ATTEMPTS + 1):
                prompt_variables = {
                    "canonical_solution": canonical_cut.content,
                    "existing_tests": current_suite.source,
                    "faulty_solution": faulty_cut.content,
                }

                suites, _ = self.generator.generate(
                    prompt_variables=prompt_variables,
                    samples=1,
                )
                result.api_calls += 1

                if not suites or not suites[0]:
                    continue

                candidate = suites[0]
                logger.info("Currently running candidate suite [%s]", str(i))
                candidate_suite = TestSuite(task_id=task.task_id, language="python", source=candidate,
                                            framework=TestFramework.PYTEST)
                candidate_canonical_results = self.tests_runner.run(canonical_cut, candidate_suite)
                diff = _suite_finds_difference(candidate_suite, candidate_canonical_results, faulty_cut, self.tests_runner)

                if diff == "timeout":
                    logger.warning("[%s] can't be augmented due to timeout", task.task_id)
                    continue
                if diff == "error":
                    logger.warning("[%s] couldn't augment due to error", task.task_id)
                    continue
                elif diff:
                    logger.info("Candidate test suite caught fault")
                    all_test_sources.append(candidate)
                    all_canonical_results.append(candidate_canonical_results)
                    current_suite = candidate_suite
                    G_detected.append(g)
                    caught = True
                    logger.info("[%s] augmentation succeeded on attempt %d — TS updated", task.task_id, attempt)
                    break
                else:
                    logger.info("Candidate test suite could not catch fault")
                    all_test_sources.append(candidate)
                    all_canonical_results.append(candidate_canonical_results)
                    current_suite = candidate_suite
                    logger.info("[%s] augmentation did not succeed on attempt %d — TS updated", task.task_id, attempt)

            if not caught:
                G_undetected.append(g)
                logger.info("[%s] augmentation failed after %d attempts", task.task_id, MAX_AUGMENTATION_ATTEMPTS)
            i += 1

        result.already_detected = G_detected
        result.undetected = G_undetected
        result.augmented_tests = all_test_sources
        result.success = len(G_detected) > 0

        return result

    def run(self) -> list[AugmentationResult]:
        if self.generations_path is not None:
            batch_lookup = self._load_from_jsonl()
        elif self.batch_job_id is not None:
        #lookup generations from batch_id
            batch_lookup: dict[str, list[str]] = _load_batch(self.generator, self.batch_job_id)

        task_lookup:  dict[str, any]       = {str(t.task_id): t for t in self.loader.load()}

        task_ids = sorted(set(batch_lookup) & set(task_lookup))

        # Apply slice filter
        if self.config.slice is not None and self.config.slice.strip() != "":
            idx, total = self.config.slice.split("/")
            idx, total = int(idx), int(total)
            task_ids = [t for i, t in enumerate(task_ids) if i % total == idx]

        # Apply start_task_id
        if self.config.start_task_id is not None:
            ids = [t for t in task_ids]
            if self.config.start_task_id in ids:
                start_idx = ids.index(self.config.start_task_id)
                task_ids = task_ids[start_idx:]

        logger.info(
            "Augmenting %d task(s) for slice %s (%d in batch, %d in benchmark)",
            len(task_ids), self.config.slice, len(batch_lookup), len(task_lookup),
        )

        results: list[AugmentationResult] = []

        for task_id in tqdm(task_ids, total=len(task_ids)):

            # Exclude specified ids
            if self.config.exclude_ids is not None and task_id in self.config.exclude_ids:
                logger.info(f"Skipping task {task_id} because it is part of the exclude list")
                continue

            generations = batch_lookup[task_id]
            if not generations:
                logger.debug("[%s] no generations in batch — skipping", task_id)
                continue

            try:
                aug = self._augment_task(task_lookup[task_id], generations)
            except Exception as e:
                logger.error("An error occurred when augmenting the test suite for the given task: [%s]", task_id)
                continue

            results.append(aug)

            write_jsonl(self.augmented_output_file, [aug], append=True)

            logger.info(
                "[%s] success=%s  already_detected=%d  undetected=%d  api_calls=%d",
                task_id, aug.success,
                len(aug.already_detected), len(aug.undetected), aug.api_calls,
            )

        logger.info(
            "Done. tasks=%d  succeeded=%d  total_already_detected=%d  total_undetected=%d",
            len(results),
            sum(1 for r in results if r.success),
            sum(len(r.already_detected) for r in results),
            sum(len(r.undetected) for r in results),
        )

        return results
