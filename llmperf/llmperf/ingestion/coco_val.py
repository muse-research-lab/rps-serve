import json
import os

from datasets import load_dataset

from llmperf.ingestion.base import BaseIngestion

class COCOValIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("vikhyatk/coco-val", split="validation")

    def deduplicate(self, records):
        captions= set()

        dedup_records = []
        for record in records:
            c = str(record["captions"])
            if c in captions:
                continue
            
            captions.add(c)
            dedup_records.append(record)

        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )

        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        for idx, record in enumerate(records):
            file = f"{idx}.jpg"
            path = os.path.join(images_dir, file)
            record["image"].save(path)
            record["image"] = file

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")