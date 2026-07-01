import json
from dataclasses import dataclass, field, asdict
from typing_extensions import Self


@dataclass
class PromptInputHolder:
    original_prompt: str
    round_trip_prompt: str
    diff_or_combined_prompt: str  = None

    def is_filled(self):
        """
        Returns whether all prompts have been filled with a non empty value
        :return:
        """
        return (self.original_prompt is not None
                and self.original_prompt is not ""
                and self.round_trip_prompt is not None
                and self.round_trip_prompt is not ""
                and self.diff_or_combined_prompt is not None)

@dataclass
class PromptInputDict:
    # Format: 'task_id': holder
    holders: dict[str, PromptInputHolder] = field(default_factory=dict)

    def set(self, task_id: str, holder: PromptInputHolder):
        self.holders[task_id] = holder

    def get(self, task_id: str) -> PromptInputHolder:
        return self.holders[task_id]

    def to_jsonl(self, file_path: str) -> None:

        """Function 1: Saves the container data into a JSONL file."""
        with open(file_path, "w", encoding="utf-8") as outfile:
            for task_id, holder in self.holders.items():
                # Convert the dataclass instance into a standard dictionary
                record = {"task_id": task_id} | asdict(holder)

                # Write as a single JSON line
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, file_path: str) -> Self:
        """Function 2: Loads data from a JSONL file and returns a new PromptInputDict instance."""
        container = cls()

        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    task_id = str(data.get("task_id"))

                    # Reconstruct the PromptInputHolder
                    holder = PromptInputHolder(
                        original_prompt=data.get("original_prompt", ""),
                        round_trip_prompt=data.get("round_trip_prompt", ""),
                        diff_or_combined_prompt=data.get("diff_or_combined_prompt", None)
                    )

                    container.holders[task_id] = holder

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")

        return container
