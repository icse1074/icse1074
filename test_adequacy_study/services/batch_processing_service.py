import json
import logging
import os

import anthropic

from test_adequacy_study.generators.openai_model import OpenAIModel
from test_adequacy_study.helpers.model_operations import convert_system_prompt_to_anthropic
from test_adequacy_study.helpers.parsers import parse_code_from_markdown

logger = logging.getLogger(__name__)


class BatchProcessingService:

    @staticmethod
    def upload_and_start_batch(llm: OpenAIModel, filepath_to_upload: str):
        if llm.model.startswith('claude'):
            client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.environ["ANTHROPIC_BASE_URL"]
            )
            logger.debug("Initializing Anthropic Message Batch...")

            requests = []
            with open(filepath_to_upload, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        requests.append(json.loads(line))

            # Create the batch job directly
            logger.info(f"Creating an Anthropic Batch Job from: {filepath_to_upload}")
            batch_job = client.beta.messages.batches.create(
                requests=requests
            )

            logger.info(f"Anthropic Batch job created! Batch ID: {batch_job.id}")

        else:
            # 1. Upload the file to OpenAI's servers
            logger.info(f"Uploading created batch file: {filepath_to_upload}")
            batch_input_file = llm.client.files.create(
                file=open(filepath_to_upload, "rb"),
                purpose="batch"
            )

            # 2. Create the batch job
            logger.info(f"Creating a Batch Job")
            batch_job = llm.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            logger.info(f"Batch job created! Batch ID: {batch_job.id}")

    @staticmethod
    def get_batch_results_lookup_from_anthropic_api(batch_job_id: str, use_code_parsing: bool = True) -> dict:
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

                task_id = BatchProcessingService.get_task_id_from_custom_id(custom_id)
                if task_id not in results_lookup:
                    results_lookup[task_id] = []

                # 3. Handle Anthropic-specific structure
                result_wrapper = data.get("result", {})
                result_type = result_wrapper.get("type")

                if result_type == "succeeded":
                    # Extract content using Anthropic's message format
                    content = result_wrapper["message"]["content"][0]["text"]

                    if use_code_parsing:
                        content = parse_code_from_markdown(content)
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

    @staticmethod
    def get_batch_results_lookup_from_local_file(file_path: str) -> dict:
        """
        Reads a completed batch job output from a local JSONL file
        and processes it into a results lookup dictionary.
        """
        print(f"Reading local batch results from: {file_path}...")

        results_lookup = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:

                # Enumerate allows us to keep track of line numbers for error logging
                for line_number, line in enumerate(f, 1):
                    cleaned_line = line.strip()
                    if not cleaned_line:
                        continue

                    try:
                        data = json.loads(cleaned_line)
                        custom_id = data.get("custom_id")

                        if not custom_id:
                            continue

                        task_id = BatchProcessingService.get_task_id_from_custom_id(custom_id)
                        response_data = data.get("response", {})
                        status_code = response_data.get("status_code")

                        if task_id not in results_lookup:
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

        except FileNotFoundError:
            raise FileNotFoundError(f"The batch file at {file_path} could not be found.")

        print(f"Successfully loaded {len(results_lookup)} results into memory.")

        return results_lookup

    @staticmethod
    def get_task_id_from_custom_id(custom_id) -> str:
        """
        Returns the task ID from the custom_id field. Typically, your batch should contain a custom_id of
        the following format: TaskId_IterationNumberAsInteger

        :param custom_id:
        :return:
        """
        split_info = custom_id.split("_")

        return str(split_info[0]).replace('-', '/')

    @staticmethod
    def create_query(llm: OpenAIModel, custom_id: str, history: list[dict], has_system_prompt: bool = True) -> dict:
        if llm.model.startswith("claude"):

            # Anthropic API requires a separate field for system_prompt with a different format
            if has_system_prompt:
                system_prompt: list[dict] = convert_system_prompt_to_anthropic(history[0])
                history = history[1:]
            else:
                system_prompt: list[dict] = []

            return {
                "custom_id": custom_id,
                "params": {
                    "model": llm.model,
                    "system": system_prompt,
                    "messages": history,
                    "max_tokens": 32000,
                    "temperature": llm.temperature
                }
            }
        else:
            return {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": llm.model,
                    "messages": history,
                    "temperature": llm.temperature
                }
            }

    @staticmethod
    def get_batch_results_lookup_from_openai_api(llm: OpenAIModel, batch_job_id: str, use_code_parsing: bool = True) -> dict[str, list[str]]:
        """
        Retrieves a completed batch job from OpenAI, downloads the output file
        directly via the API, and processes it into a results lookup dictionary.

        Returns a dictionary with the following structure: {'task_id': ['model response1 as string', 'response2'...]}
        """
        client = llm.client

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

                task_id = BatchProcessingService.get_task_id_from_custom_id(custom_id)
                response_data = data.get("response", {})
                status_code = response_data.get("status_code")
                if not results_lookup.keys().__contains__(task_id):
                    results_lookup[task_id] = []

                if status_code == 200:
                    content: str = response_data["body"]["choices"][0]["message"]["content"]
                    if use_code_parsing:
                        content = parse_code_from_markdown(content)
                    results_lookup[task_id].append(content)

                # Nebius API typically does not return a status code and has a slightly different output structure
                elif status_code is None:
                    content = response_data["choices"][0]["message"]["content"]

                    if content is not None:
                        if use_code_parsing:
                            content = parse_code_from_markdown(content)
                        results_lookup[task_id].append(content)
                else:
                    error_info = data.get("error") or response_data.get("body", {}).get("error", "Unknown error")
                    print(f"Row error for custom_id '{custom_id}' (Status {status_code}): {error_info}")

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error parsing line {line_number}: {e}")
                continue

        print(f"Successfully loaded {len(results_lookup)} results into memory.")

        return results_lookup

    @staticmethod
    def get_batch_results_lookup_from_anthropic_api(llm: OpenAIModel, batch_job_id: str) -> dict:
        """
        Retrieves a completed batch job from Anthropic, streams the output
        directly via the API, and processes it into a results lookup dictionary.
        """
        if llm.model.startswith("claude"):
            client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.environ["ANTHROPIC_BASE_URL"]
            )
        else :
            client = llm.client
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

                task_id = BatchProcessingService.get_task_id_from_custom_id(custom_id)
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
