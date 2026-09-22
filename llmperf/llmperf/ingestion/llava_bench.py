import json
import os

from datasets import load_dataset

from llmperf.ingestion.base import BaseIngestion

class LLaVABenchComplexReasoningIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["category"] == "complex":
                cleaned_records.append(record)

        return cleaned_records

    def deduplicate(self, records):
        pairs= set()

        dedup_records = []
        for record in records:
            q = record["question"]
            img = record["image_id"]
            if (q, img) in pairs:
                continue
            
            pairs.add((q, img))
            dedup_records.append(record)

        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )

        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        for record in records:
            file = record["image_id"].replace("jpg", "png")
            path = os.path.join(images_dir, file)
            record["image"].save(path)
            record["image"] = file

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

class LLaVABenchDetailedDescriptionIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["category"] == "detail":
                cleaned_records.append(record)

        return cleaned_records

    def deduplicate(self, records):
        pairs= set()

        dedup_records = []
        for record in records:
            q = record["question"]
            img = record["image_id"]
            if (q, img) in pairs:
                continue
            
            pairs.add((q, img))
            dedup_records.append(record)

        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )

        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        for record in records:
            file = record["image_id"].replace("jpg", "png")
            path = os.path.join(images_dir, file)
            record["image"].save(path)
            record["image"] = file

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

class LLaVABenchConversationsIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["category"] == "conv":
                cleaned_records.append(record)

        return cleaned_records

    def deduplicate(self, records):
        pairs= set()

        dedup_records = []
        for record in records:
            q = record["question"]
            img = record["image_id"]
            if (q, img) in pairs:
                continue
            
            pairs.add((q, img))
            dedup_records.append(record)

        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )

        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        for record in records:
            file = record["image_id"].replace("jpg", "png")
            path = os.path.join(images_dir, file)
            record["image"].save(path)
            record["image"] = file

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

class LLaVABenchQnAIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")

    def deduplicate(self, records):
        pairs= set()

        dedup_records = []
        for record in records:
            q = record["question"]
            img = record["image_id"]
            if (q, img) in pairs:
                continue
            
            pairs.add((q, img))
            dedup_records.append(record)

        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )

        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        for record in records:
            file = record["image_id"].replace("jpg", "png")
            path = os.path.join(images_dir, file)
            record["image"].save(path)
            record["image"] = file

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")