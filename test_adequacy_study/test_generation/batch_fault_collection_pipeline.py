import ast
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import anthropic

from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.execution_report import Verdict, TestResult
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.file_utils import write_jsonl
from test_adequacy_study.generators.code_generator import CodeGenerator
from test_adequacy_study.helpers.model_operations import convert_system_prompt_to_anthropic
from test_adequacy_study.helpers.parsers import parse_code_from_markdown, is_syntactically_valid, align_code_to_list
from test_adequacy_study.runners.test_runner import TestRunner
from tqdm import tqdm

from test_adequacy_study.types.fault_collection_result import FaultCollectionResult
from test_adequacy_study.types.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class BatchFaultCollectionPipeline:
    """
    Orchestrates the full fault collection loop:

        BenchmarkLoader → CodeGenerator → ProgramBuilder → TestRunner → faults

    The pipeline is fully agnostic of benchmark format, test framework,
    and model backend. All format-specific logic lives in the injected
    components.
    """

    def __init__(
            self,
            loader: BenchmarkLoader,
            runner: TestRunner,
            generator: CodeGenerator,
            builder: ProgramBuilder,
            config: PipelineConfig = None,
            exclude_indexes: list[int] = None
    ):
        self.loader = loader
        self.runner = runner
        self.generator = generator
        self.builder = builder
        self.config = config or PipelineConfig()
        self.exclude_indexes = exclude_indexes

    def _parse_failures(self, stdout: str) -> list[dict]:
        results = []
        for line in stdout.splitlines():
            line = line.strip()
            if line:
                results.append(ast.literal_eval(line))
        return results

    def run_create_and_upload(self):
        """
        Iterates the benchmark, creates a batch file with all the queries that the model shall execute
        Once done, it generates a batch file, uploads it and executes it (via server)

        :return:
        """
        total_number_of_tasks = len(self.loader) - (self.config.skip_n_tasks or 0)
        batch_queries = []

        for idx, task in enumerate(tqdm(self.loader.load(no_tests=True), total=total_number_of_tasks)):

            # Exclude n tasks (if specified under configuration)
            if self.config.skip_n_tasks and idx < self.config.skip_n_tasks:
                continue

            # Exclude tasks outside the ones specified (if any, otherwise iterate all)
            if self.config.tasks and task.task_id not in self.config.tasks:
                continue

            # Step 2: generate n completions
            for i in range(self.config.n_generations):
                history = self.generator.get_history_with_hardcoded_response(prompt_variables={"prompt": task.stub}, response="will be ignored")
                history = history[:-1]
                custom_id: str = self.__create_custom_id(task, i)
                batch_queries.append(self.__create_query(custom_id, history))

        # Step 3: Write queries to a batch file
        logger.info(f"Batch queries: {len(batch_queries)}. Writing into file")
        with open(self.config.output_file, "w") as f:
            for query in batch_queries:
                f.write(json.dumps(query) + "\n")

        logger.info("Batch created. Uploading batch and invoking remote execution")
        batch_id = self.__upload_and_start_batch()
        parts = Path(self.config.output_file).stem.split("_")  # ["batch", "upload", "bcb", "us", "claude"]
        benchmark, variation, model = parts[2], parts[3], parts[4]
        metadata = {
            "batch_id": batch_id,
            "model": model,
            "benchmark": benchmark,
            "variation": variation,
        }
        write_jsonl("batch_id_map_local.jsonl", [metadata], append=True)
        logger.info("Pipeline complete.")

    def run_collect_faults(self, batch_id: str):
        """
        Iterates the results of the batch work. Runs the tests for each case and saves the faulty versions

        :return:
        """

        total_tasks = list(self.loader.load())

        #slicing for HPC parallelism
        if self.config.slice is not None and self.config.slice.strip() != "":
            idx, total = self.config.slice.split("/")
            idx, total = int(idx), int(total)
            total_tasks = [t for i, t in enumerate(total_tasks) if i % total == idx]
        try:
            if self.generator.llm.model.startswith("claude"):
                results_lookup = self.__get_batch_results_lookup_from_anthropic_api(batch_id)
            else:
                results_lookup = self.__get_batch_results_lookup_from_openai_api(batch_id)

            faults_list = FaultCollectionResult()
            logger.info("Iterating each Benchmark record 1 by 1")
            for idx, task in tqdm(enumerate(total_tasks), total=len(total_tasks)):
                faults_list.total_tasks += 1

                if self.exclude_indexes is not None and idx in self.exclude_indexes:
                    continue

                # Step 1: build test suite
                try:
                    test_suite = self.builder.build_tests(task)

                    # Pull the response out of our fresh API download
                    model_responses = results_lookup.get(str(task.task_id), [])
                    faults_list.total_generated += len(model_responses)

                except Exception as e:
                    logger.error("Could not build tests for task {}: {}".format(task, e))
                    continue

                # Step 3: Loop through generated code (above n completions)
                for i, code in tqdm(enumerate(model_responses), total=len(model_responses)):
                    try:
                        if not code.strip():
                            logger.debug("[%s] completion %d is empty, skipping", task.task_id, i)
                            continue

                        # Rename generated code to match the function name of the assertions
                        if self.config.match_function_names is True:
                            code = align_code_to_list(code, test_suite.source['test_inputs'])

                        # Build code under test
                        cut = self.builder.build_program(
                            task=task,
                            code=code)

                        if not is_syntactically_valid(code):
                            logger.debug("[%s] completion %d has syntax error, skipping", task.task_id, i)
                            continue
                    except Exception as e:
                        logger.error("Could not build program for task {}: {}".format(task, e))
                        continue

                    faults_list.total_run += 1

                    # Run tests. In case of failures, save the generated code in the output_file specified in config
                    try:

                        # MBPP compatibility
                        cut.syntactically_valid = cut.syntactically_valid or is_syntactically_valid(cut.implementation)
                        run_result = self.runner.run(
                            cut=cut,
                            suite=test_suite)
                        logger.debug("[%s] completion %d → %s", task.task_id, i, run_result.verdict)

                        # Step 5: collect faults only
                        if run_result.verdict == Verdict.FAILED:

                            # parse the failed assertions
                            if run_result.detailed_test_results and isinstance(run_result.detailed_test_results[0], TestResult) :
                                failures = [asdict(t) for t in run_result.detailed_test_results if
                                            t.outcome == "failed"]
                            else :
                                failures = self._parse_failures(run_result.stdout)

                            fault_dict = {
                                "task_id": task.task_id,
                                "completion": code,
                                "completion_index": i,
                                "failures": failures,
                            }
                            faults_list.faults.append(fault_dict)
                            write_jsonl(self.config.output_file, [fault_dict], append=True)
                    except Exception as e:
                        logging.error(
                            "An exception was thrown when running tests for task {}: {}".format(task.task_id, e))

            logger.info("Pipeline complete. %s", faults_list.summary())
            return faults_list

        except Exception as e:
            print(f"Failed to process batch results: {e}")

    def __upload_and_start_batch(self):

        if self.generator.llm.model.startswith('claude'):
            client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.environ["ANTHROPIC_BASE_URL"]
            )
            logger.debug("Initializing Anthropic Message Batch...")

            requests = []
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        requests.append(json.loads(line))

            # Create the batch job directly
            logger.info(f"Creating an Anthropic Batch Job from: {self.config.output_file}")
            batch_job = client.beta.messages.batches.create(
                requests=requests
            )

            logger.info(f"Anthropic Batch job created! Batch ID: {batch_job.id}")
            return batch_job.id
        else:
            # 1. Upload the file to OpenAI's servers
            logger.info(f"Uploading created batch file: {self.config.output_file}")
            batch_input_file = self.generator.llm.client.files.create(
                file=open(self.config.output_file, "rb"),
                purpose="batch"
            )

            # 2. Create the batch job
            logger.info(f"Creating a Batch Job")
            batch_job = self.generator.llm.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            logger.info(f"Batch job created! Batch ID: {batch_job.id}")
            return batch_job.id

    def __get_batch_results_lookup_from_openai_api(self, batch_job_id: str) -> dict:
        """
        Retrieves a completed batch job from OpenAI, downloads the output file
        directly via the API, and processes it into a results lookup dictionary.
        """
        client = self.generator.llm.client

        # 1. Retrieve the batch job status
        print(f"Retrieving batch job {batch_job_id}...")
        job = client.batches.retrieve(batch_job_id)

        if job.status != "completed":
            raise ValueError(f"Batch job is not ready. Current status: '{job.status}'")

        output_file_id = job.output_file_id
        if not output_file_id:
            raise ValueError("Batch job completed but no output file ID was found.")

        print(f"Downloading results file ({output_file_id})...")

        # 2. Download the file content directly from OpenAI
        # .content() returns the raw text response containing the JSONL lines
        file_response = client.files.content(output_file_id)
        results_text = file_response.text

        # 3. Parse the string data into our lookup dictionary
        results_lookup = {}

        # Split the big text block by newlines to iterate line-by-line
        for line_number, line in enumerate(results_text.strip().split('\n'), 1):
            cleaned_line = line.strip()
            if not cleaned_line:
                continue

            try:
                data = json.loads(cleaned_line)
                custom_id = data.get("custom_id")

                if not custom_id:
                    continue

                task_id = self.__get_task_id_from_custom_id(custom_id)
                response_data = data.get("response", {})
                status_code = response_data.get("status_code")
                if not results_lookup.keys().__contains__(task_id):
                    results_lookup[task_id] = []

                if status_code == 200:
                    content = parse_code_from_markdown(response_data["body"]["choices"][0]["message"]["content"])
                    results_lookup[task_id].append(content)

                # Nebius API typically does not return a status code and has a slightly different output structure
                elif status_code is None:
                    raw_content = response_data["choices"][0]["message"]["content"]

                    if raw_content is not None:
                        content = parse_code_from_markdown(raw_content)
                        results_lookup[task_id].append(content)
                else:
                    error_info = data.get("error") or response_data.get("body", {}).get("error", "Unknown error")
                    print(f"Row error for custom_id '{custom_id}' (Status {status_code}): {error_info}")

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error parsing line {line_number}: {e}")
                continue

        print(f"Successfully loaded {len(results_lookup)} results into memory.")

        return results_lookup

    def __get_batch_results_lookup_from_anthropic_api(self, batch_job_id: str) -> dict:
        """
        Retrieves a completed batch job from Anthropic, streams the output
        directly via the API, and processes it into a results lookup dictionary.
        """
        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.environ["ANTHROPIC_BASE_URL"]
        )

        # 1. Retrieve the batch job status
        print(f"Retrieving Anthropic batch job {batch_job_id}...")
        job = client.beta.messages.batches.retrieve(batch_job_id)

        if job.processing_status != "ended":
            raise ValueError(f"Batch job is not ready. Current status: '{job.processing_status}'")

        print(f"Streaming results for batch {batch_job_id}...")

        # 2. Iterate through results directly
        results_lookup = {}
        line_number = 0

        # The results method returns an iterable object yielding result instances
        response_stream = client.beta.messages.batches.results(batch_job_id)

        for result_row in response_stream:
            line_number += 1
            try:
                # Anthropic SDK might return objects. If it's an object, convert to dict
                # or access properties directly. Safest way is using model_dump() if it's a Pydantic model.
                data = result_row.model_dump() if hasattr(result_row, "model_dump") else dict(result_row)

                custom_id = data.get("custom_id")
                if not custom_id:
                    continue

                task_id = self.__get_task_id_from_custom_id(custom_id)
                if task_id not in results_lookup:
                    results_lookup[task_id] = []

                # 3. Handle Anthropic-specific structure
                result_wrapper = data.get("result", {})
                result_type = result_wrapper.get("type")

                if result_type == "succeeded":
                    # Extract content using Anthropic's message format
                    message_content = result_wrapper["message"]["content"][0]["text"]
                    content = parse_code_from_markdown(message_content)
                    results_lookup[task_id].append(content)

                elif result_type == "errored":
                    error_info = result_wrapper.get("error", "Unknown error")
                    print(f"Row error for custom_id '{custom_id}': {error_info}")

                else:
                    print(f"Row canceled or unknown type for custom_id '{custom_id}'")

            except (KeyError, IndexError, TypeError) as e:
                print(f"Error processing row {line_number}: {e}")
                continue

        print(f"Successfully loaded {len(results_lookup)} results into memory.")

        return results_lookup

    def __get_task_id_from_custom_id(self, custom_id) -> str:
        """
        Returns the task ID from the custom_id field. Typically, your batch should contain a custom_id of
        the following format: TaskId_IterationNumberAsInteger

        :param custom_id:
        :return:
        """
        split_info = custom_id.split("_")

        return str(split_info[0]).replace('-', '/')

    def __create_custom_id(self, task: Task, i: int) -> str:
        return str(task.task_id).replace('/', '-') + "_" + str(i)

    def __create_query(self, custom_id: str, history: list[dict]) -> dict:
        if self.generator.llm.model.startswith("claude"):

            # Anthropic API requires a separate field for system_prompt with a different format
            system_prompt: list[dict] = convert_system_prompt_to_anthropic(history[0])
            history = history[1:]
            return {
                "custom_id": custom_id,
                "params": {
                    "model": self.generator.llm.model,
                    "system": system_prompt,
                    "messages": history,
                    "max_tokens": 32000,
                    **self.generator.llm._temperature_kwargs,
                }
            }
        else:
            return {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.generator.llm.model,
                    "messages": history,
                    **self.generator.llm._temperature_kwargs,
                }
            }