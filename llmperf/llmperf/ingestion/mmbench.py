import base64
import csv
import io
import json
import os
import requests
import sys

from llmperf.ingestion.base import BaseIngestion

csv.field_size_limit(sys.maxsize)

def save_base64_image(b64_string, file, image_dir):
    """Save raw base64 image string as a .jpg file."""
    try:
        image_bytes = base64.b64decode(b64_string)
        image_path = os.path.join(image_dir, file)
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        return image_path
    except Exception as e:
        print(f"⚠️ Could not decode image for {file}: {e}")
        return None

class MMBenchMultipleChoiceIngestion(BaseIngestion):
    def fetch_raw_data(self):
        url = (
            "http://opencompass.openxlab.space/utils/"
            "VLMEval/MMBench_DEV_EN.tsv"
        )

        r = requests.get(url, stream=True)
        r.raise_for_status()

        return csv.DictReader(io.StringIO(r.text), delimiter="\t")
    
    def clean_data(self, records):
        images_dict = {}
        cleaned_records = []
        for record in records:
            img = record.get("image")
            if img and img.strip().startswith("/9j/"):
                cleaned_records.append(record)
                images_dict[record.get("index")] = img

            if not img.strip().startswith("/9j/"):
                index = record["image"]
                record["image"] = images_dict[index]
                cleaned_records.append(record)

        return cleaned_records

    def deduplicate(self, records):
        images = set()
        dedup_records = []
        for record in records:
            q = record["question"]
            a = record["answer"]
            img = record["image"]
            if img in images:
                continue
            
            images.add(img)
            dedup_records.append(record)
        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )
        
        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        for record in records:
            img = record["image"]
            index = int(record["index"])
            if index > 10000:
                # ! Note: Avoid saving the same image
                file = f"{index % 10000}.jpg"
                record["image"] = file
            else:
                file = f"{index}.jpg"
                save_base64_image(img.strip(), file, images_dir)
                record["image"] = file

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")