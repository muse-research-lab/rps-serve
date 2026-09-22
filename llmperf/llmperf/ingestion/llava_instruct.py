import json
import os
import requests
import shutil
import tempfile
import zipfile

from tqdm import tqdm

from llmperf.ingestion.base import BaseIngestion

COCO_ZIP_URL = "http://images.cocodataset.org/zips/train2017.zip"

def download_and_unzip_coco():
    os.makedirs("/srv/muse-lab/datasets/tmp", exist_ok=True)
    tmp_dir = tempfile.TemporaryDirectory(dir="/srv/muse-lab/datasets/tmp")
    zip_path = os.path.join(tmp_dir.name, "train2017.zip")

    # Download ZIP (streaming)
    with requests.get(COCO_ZIP_URL, stream=True) as r:
        r.raise_for_status()

        total_size = int(r.headers.get("Content-Length", 0))
        chunk_size = 8192

        with open(zip_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading COCO"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Unzip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir.name)

    coco_dir = os.path.join(tmp_dir.name, "train2017")
    return tmp_dir, coco_dir

class LLaVAInstructComplexReasoningIngestion(BaseIngestion):
    def fetch_raw_data(self):
        url = (
            "https://huggingface.co/datasets/"
            "liuhaotian/LLaVA-Instruct-150K/"
            "resolve/main/complex_reasoning_77k.json"
            "?download=true"
        )

        r = requests.get(url, stream=True)
        r.raise_for_status()

        return json.loads(r.text)

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["conversations"][0]["from"] == "human" and \
                record["conversations"][1]["from"] == "gpt":
                cleaned_records.append(record)

        return cleaned_records

    def deduplicate(self, records):
        pairs= set()

        dedup_records = []
        for record in records:
            q = record["conversations"][0]["value"]
            img = record["image"]
            if (q, img) in pairs:
                continue
            
            pairs.add((q, img))
            dedup_records.append(record)

        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        images = {record["image"] for record in records}
        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        tmp_dir, coco_dir = download_and_unzip_coco()

        for img in images:
            src_path = os.path.join(coco_dir, img)
            dst_path = os.path.join(images_dir, img)
            shutil.copy2(src_path, dst_path)

        tmp_dir.cleanup()

class LLaVAInstructDetailedDescriptionIngestion(BaseIngestion):
    def fetch_raw_data(self):
        url = (
            "https://huggingface.co/datasets/"
            "liuhaotian/LLaVA-Instruct-150K/"
            "resolve/main/detail_23k.json"
            "?download=true"
        )

        r = requests.get(url, stream=True)
        r.raise_for_status()

        return json.loads(r.text)

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["conversations"][0]["from"] == "human" and \
                record["conversations"][1]["from"] == "gpt":
                cleaned_records.append(record)

        return cleaned_records

    def deduplicate(self, records):
        images= set()

        dedup_records = []
        for record in records:
            q = record["conversations"][0]["value"]
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
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        images = {record["image"] for record in records}
        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        tmp_dir, coco_dir = download_and_unzip_coco()

        for img in images:
            src_path = os.path.join(coco_dir, img)
            dst_path = os.path.join(images_dir, img)
            shutil.copy2(src_path, dst_path)
        
        tmp_dir.cleanup()

class LLaVAInstructConversationsIngestion(BaseIngestion):
    def fetch_raw_data(self):
        url = (
            "https://huggingface.co/datasets/"
            "liuhaotian/LLaVA-Instruct-150K/"
            "resolve/main/conversation_58k.json"
            "?download=true"
        )

        r = requests.get(url, stream=True)
        r.raise_for_status()

        return json.loads(r.text)

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["conversations"][0]["from"] == "human" and \
                record["conversations"][1]["from"] == "gpt":
                cleaned_records.append(record)

        return cleaned_records
    
    def deduplicate(self, records):
        images = set()

        dedup_records = []
        for record in records:
            q = record["conversations"][0]["value"]
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
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        images = {record["image"] for record in records}
        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        tmp_dir, coco_dir = download_and_unzip_coco()

        for img in images:
            src_path = os.path.join(coco_dir, img)
            dst_path = os.path.join(images_dir, img)
            shutil.copy2(src_path, dst_path)

        tmp_dir.cleanup()