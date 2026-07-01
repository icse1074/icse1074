import ast
import json
import logging
import os

import anthropic

from test_adequacy_study.benchmarks.loader import BenchmarkLoader
from test_adequacy_study.builders.program_builder import ProgramBuilder
from test_adequacy_study.data_models.execution_report import Verdict
from test_adequacy_study.data_models.task import Task
from test_adequacy_study.file_utils import write_jsonl, read_jsonl
from test_adequacy_study.generators.code_generator import CodeGenerator
from test_adequacy_study.generators.prompt_generator import PromptGenerator
from test_adequacy_study.helpers.model_operations import convert_system_prompt_to_anthropic
from test_adequacy_study.helpers.parsers import parse_code_from_markdown, is_syntactically_valid, align_code_to_list
from test_adequacy_study.runners.test_runner import TestRunner
from tqdm import tqdm

from test_adequacy_study.services.batch_processing_service import BatchProcessingService
from test_adequacy_study.types.fault_collection_result import FaultCollectionResult
from test_adequacy_study.types.pipeline_config import PipelineConfig
from test_adequacy_study.types.prompt_input_holder import PromptInputDict

logger = logging.getLogger(__name__)


class BatchPromptGenerationPipeline:

    def __init__(
            self,
            generator: PromptGenerator,
            input_file: str,
            config: PipelineConfig = None,
            exclude_indexes: list[int] = None,
            is_fault_collection_batch: bool = False,
            batch_id: str = None
    ):
        self.config = config or PipelineConfig()
        self.exclude_indexes = exclude_indexes

        # Typically this is the batch file or the prompts jsonl file
        self.input_file = input_file
        self.generator = generator
        self.is_fault_collection_batch = is_fault_collection_batch
        self.batch_id = batch_id

    def run_create_and_upload(self):
        """
        Iterates your local file and creates a new batch file with queries to generate a prompt for the generated code

        :return:
        """
        if self.is_fault_collection_batch:
            records_by_task_id = self.__get_results_from_faults_file()
        else:
            records_by_task_id = BatchProcessingService.get_batch_results_lookup_from_local_file(self.input_file)
        total_number_of_tasks = len(records_by_task_id)
        batch_queries = []

        for task_id, record in tqdm(records_by_task_id.items(), total=total_number_of_tasks):

            # Step 1: Get first code response
            code = record[0]

            # Step 2: generate n completions
            for i in range(self.config.n_generations):
                history = [
                    {"role": "user", "content": self.generator.get_prompt({'code': code})},
                ]

                custom_id: str = task_id
                batch_queries.append(BatchProcessingService.create_query(self.generator.llm, custom_id, history, has_system_prompt=False))

        # Step 3: Write queries to a batch file
        logger.info(f"Batch queries: {len(batch_queries)}. Writing into file")
        with open(self.config.output_file, "w") as f:
            for query in batch_queries:
                f.write(json.dumps(query) + "\n")

        logger.info("Batch created. Uploading batch and invoking remote execution")
        BatchProcessingService.upload_and_start_batch(self.generator.llm, self.config.output_file)
        logger.info("Pipeline complete.")

    def run_create_and_upload_diff_prompt(self):
        """
        Iterates your local file and creates a new batch file with queries to generate a prompt for the difference between the two prompts

        :return:
        """
        input_prompts = PromptInputDict.from_jsonl(self.input_file)
        total_number_of_tasks = len(input_prompts.holders)
        batch_queries = []

        for task_id, record in tqdm(input_prompts.holders.items(), total=total_number_of_tasks):

            # Step 2: generate n completions
            for i in range(self.config.n_generations):
                history = [
                    {"role": "user", "content": self.generator.get_prompt(
                        {
                            'prompt1': record.original_prompt,
                            'prompt2': record.round_trip_prompt
                        })
                     },
                ]

                custom_id: str = task_id
                batch_queries.append(BatchProcessingService.create_query(self.generator.llm, custom_id, history))

        # Step 3: Write queries to a batch file
        logger.info(f"Batch queries: {len(batch_queries)}. Writing into file")
        with open(self.config.output_file, "w") as f:
            for query in batch_queries:
                f.write(json.dumps(query) + "\n")

        logger.info("Batch created. Uploading batch and invoking remote execution")
        BatchProcessingService.upload_and_start_batch(self.generator.llm, self.config.output_file)
        logger.info("Pipeline complete.")


    def __get_results_from_faults_file(self) -> dict[str, list[str]]:
        records = read_jsonl(self.input_file)
        responses_per_task = {}
        for record in records:
            responses_per_task[str(record["task_id"])] = [record["completion"]]

        return responses_per_task


