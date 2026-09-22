import json
import os

from datasets import load_dataset

from llmperf.ingestion.base import BaseIngestion

class ShareGPTIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["conversations"][0]["from"] == "human" and \
                record["conversations"][1]["from"] == "gpt":
                cleaned_records.append(record)

        return cleaned_records

    def deduplicate(self, records):
        # ! Note: Records may have more than a single round of conversation
        # ! For deduplication, only the first user question is considered
        questions = set()
        
        dedup_records = []
        for record in records:
            q = record["conversations"][0]["value"]
            if q in questions:
                continue
            
            questions.add(q)
            dedup_records.append(record)

        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
