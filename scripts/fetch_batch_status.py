import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate


def download_batch_results(batch_id, output_filename=None):
    # Initialize the OpenAI client
    try:
        api_key = os.environ["NEBIUS_API_KEY"]
        base_url = os.environ["NEBIUS_BASE_URL"]
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    print(f"Retrieving batch status for: {batch_id}...")

    try:
        # 1. Retrieve the batch details
        batch = client.batches.retrieve(batch_id)

        # Check if the batch is actually ready
        if batch.status.upper() != "COMPLETED":
            print(f"Cannot download. Batch status is currently: {batch.status}")
            if batch.status.upper() == "FAILED" and batch.errors:
                print(f"Batch failed with errors: {batch.errors}")
            return

        # Get the output file ID from the batch object
        output_file_id = batch.output_file_id
        if not output_file_id:
            print("No output file ID found for this batch.")
            return

        print(f"Batch completed successfully! Found output file: {output_file_id}")

        # Set a default filename if one wasn't provided
        if not output_filename:
            output_filename = f"batch_results_{batch_id}.jsonl"

        # 2. Download the file content
        print(f"Downloading results to {output_filename}...")
        file_response = client.files.content(output_file_id)

        # Save the contents to a local file
        with open(output_filename, "wb") as f:
            f.write(file_response.read())

        print("Download complete!")

    except Exception as e:
        print(f"An error occurred: {e}")

def get_openai_batches():
    # Initialize the client. It automatically picks up the OPENAI_API_KEY
    # from your environment variables.
    try:
        api_key = os.environ["NEBIUS_API_KEY"]
        base_url = os.environ["NEBIUS_BASE_URL"]
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as e:
        print(f"Error initializing client: {e}")
        print("Please ensure your environment variables are set.")
        return

    print("Fetching batches from NEBIUS...")

    try:
        # List the batches (defaults to the most recent 20, use 'limit' to change)
        batches_page = client.batches.list(limit=20)

        # Prepare data for the table
        table_data = []

        for batch in batches_page.data:
            # Convert timestamp to human-readable format
            created_at = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M:%S')

            table_data.append([
                batch.id,
                batch.status.upper(),
                created_at,
                batch.input_file_id,
                batch.output_file_id or "N/A"
            ])

        if not table_data:
            print("No batches found.")
            return

        # Define table headers
        headers = ["Batch ID", "Status", "Created At", "Input File ID", "Output File ID"]

        # Print the neatly formatted table
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    except Exception as e:
        print(f"An error occurred while fetching batches: {e}")


if __name__ == "__main__":
    load_dotenv()
    get_openai_batches()
    # download_batch_results("batch_019eb39a-ad70-70c5-ad8b-db7b77d9bfe9")