import argparse
import json
import logging
import os
import random
import shutil
import tarfile
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, List, LiteralString, Optional, Union

import requests
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

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

def download_extract_and_filter_videos(
    urls: List[str],
    videos_to_keep: set,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for url in urls:
            if not videos_to_keep:
                break

            filename = url.split("/")[-1]
            tar_path = os.path.join(tmp_dir, filename)

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("Content-Length", 0))
                chunk_size = 8192

                with open(tar_path, "wb") as f, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {filename}"
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            extract_root = os.path.join(tmp_dir, filename.replace(".tar.gz", ""))
            os.makedirs(extract_root, exist_ok=True)

            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    
                    if member.name not in videos_to_keep:
                        continue

                    tar.extract(member, path=extract_root)
                    src = os.path.join(extract_root, member.name)
                    dst = os.path.join(output_dir, member.name)

                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    videos_to_keep.remove(member.name)

            if os.path.exists(tar_path):
                os.remove(tar_path)

            if os.path.exists(extract_root):
                shutil.rmtree(extract_root)

@dataclass
class Dataset:
    name: str
    path: Union[str, LiteralString]
    file: str
    alias: str
    data: Optional[List[dict]] = field(default=None, repr=False)

    def __hash__(self):
        return hash((self.name, self.alias))
    
    def __eq__(self, other):
        if isinstance(other, Dataset):
            return self.name == other.name and self.alias == other.alias
        return False

    def load(self) -> List[dict]:
        """
        Loads data from the file if `data` is None, otherwise returns the
        existing data.
        """
        if self.data is not None:
            return self.data
        
        file_path = os.path.join(self.path, self.file)
        loaded_data = []
        with open(file_path, "r") as f:
            for line in f:
                loaded_data.append(json.loads(line))
        self.data = loaded_data
        return self.data

@dataclass(frozen=True, slots=True)
class IngestionConfig:
    dataset_name: str
    storage_path: Union[str, LiteralString]
    records_file: str
    dataset_alias: str
    max_records: int | None = None
    seed: int = 0

    def __post_init__(self):
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive")

@dataclass(slots=True)
class BaseIngestion(ABC):
    """
    Base class for all dataset ingestions.
    Subclasses implement the dataset-specific steps.
    """
    config: "IngestionConfig"

    def __post_init__(self):
        # Use a dataset-scoped logger
        self.logger = logging.getLogger(
            f"ingestion.{self.config.dataset_alias}"
        )

    def run(self) -> "Dataset":
        """
        Executes the ingestion lifecycle.
        """
        self.logger.info("Ingestion started")

        start = time.time()

        raw = self.fetch_raw_data()
        cleaned = self.clean_data(raw)
        deduped = self.deduplicate(cleaned)
        limited = self.limit_records(deduped)
        
        os.makedirs(self.config.storage_path, exist_ok=True)
        self.store(limited)

        duration = round(time.time() - start, 3)

        self.logger.info(
            "Ingestion completed",
            extra={
                "records": len(limited),
                "storage_path": self.config.storage_path,
                "duration_seconds": duration,
            },
        )

        return Dataset(
            name=self.config.dataset_name,
            path=self.config.storage_path,
            file=self.config.records_file,
            alias=self.config.dataset_alias,
            data=limited
        )

    @abstractmethod
    def fetch_raw_data(self) -> Iterable[Any]:
        raise NotImplementedError

    def clean_data(self, records: Iterable[Any]) -> List[Any]:
        return list(records)

    def deduplicate(self, records: List[Any]) -> List[Any]:
        return records

    def limit_records(self, records: List[Any]) -> List[Any]:
        max_records = self.config.max_records
        if max_records is None:
            return records
        return records[:max_records]
    
    @abstractmethod
    def store(self, records: List[Any]):
        """
        Persist records in storage path.
        Images should be stored in images/ directory.
        Videos should be stored in videos/ directory.
        Audios should be stored in audios/ directory.

        Image records should have an "image" field with the image path.
        Video records should have an "video" field with the video path.
        Audio records should have an "audio" field with the audio path.
        
        Image, video, and audio paths should be relative to the modaility dirs:
        - images/
        - videos/
        - audios/
        
        All records should be stored in storage path inside the records file.
        """
        pass

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

class LLaVAVideoCaptionIngestion(BaseIngestion):
    def fetch_raw_data(self):
        # Note: Doesn't include the 15k entries of "llava_hound", since they are
        # not split correctly in HuggingFace.
        configs = [
            "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1",
            "30_60_s_academic_v0_1", "30_60_s_youtube_v0_1",
            "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1",
            "2_3_m_youtube_v0_1", "2_3_m_academic_v0_1"
        ]

        datasets_list = [
            load_dataset("lmms-lab/LLaVA-Video-178K", config, split="caption")
            for config in configs
        ]

        return concatenate_datasets(datasets_list)

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["conversations"][0]["from"] == "human" and \
                record["conversations"][1]["from"] == "gpt":
                cleaned_records.append(record)

        return cleaned_records
    
    def deduplicate(self, records):
        videos = set()
        
        dedup_records = []
        for record in records:
            q = record["conversations"][0]["value"]
            vid = record["video"]
            if vid in videos:
                continue
            
            videos.add(vid)
            dedup_records.append(record)

        return dedup_records

    def limit_records(self, records):
        random.seed(self.config.seed)
        max_records = self.config.max_records
        if max_records is None:
            return records
        
        categories = [
            "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1",
            "30_60_s_academic_v0_1", "30_60_s_youtube_v0_1",
            "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1",
            "2_3_m_youtube_v0_1", "2_3_m_academic_v0_1"
        ]
        buckets = {}
        for cat in categories:
            buckets[cat] = []

        for r in records:
            buckets[r["data_source"]].append(r)

        categories = list(buckets.keys())
        n_cats = len(categories)
        per_category = max_records // n_cats

        selected = []
        leftovers = []

        # First pass: try equal allocation
        for cat in categories:
            pool = buckets[cat]
            random.shuffle(pool)

            take = min(per_category, len(pool))
            selected.extend(pool[:take])

            # Anything not taken goes to leftovers
            leftovers.extend(pool[take:])

        # Second pass: fill deficits from leftovers
        remaining = max_records - len(selected)

        if remaining > 0 and leftovers:
            selected.extend(random.sample(
                leftovers, min(remaining, len(leftovers)))
            )

        return selected

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        videos_to_keep = {record["video"] for record in records}
        videos_dir = os.path.join(self.config.storage_path, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        urls = []

        datasets = [
            ("0_30_s_academic_v0_1", 8),
            ("30_60_s_academic_v0_1", 10),
            ("1_2_m_academic_v0_1", 14),
            ("2_3_m_academic_v0_1", 18),
            ("0_30_s_youtube_v0_1", 19),
            ("30_60_s_youtube_v0_1", 13),
            ("1_2_m_youtube_v0_1", 50),
            ("2_3_m_youtube_v0_1", 98),
        ]

        base_url = (
            "https://huggingface.co/datasets/"
            "lmms-lab/LLaVA-Video-178K/resolve/main"
        )

        for folder, count in datasets:
            urls.extend(
                f"{base_url}/{folder}/{folder}_videos_{i}.tar.gz"
                for i in range(1, count + 1)
            )

        download_extract_and_filter_videos(urls, videos_to_keep, videos_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        action="store_true",
        help="Run ShareGPT ingestion",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Run LLaVA-Instruct-150k ingestion",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Run LLaVA-Video ingestion",
    )
    args = parser.parse_args()

    if not (args.text or args.image or args.video):
        parser.error("Specify at least one of --text, --image, or --video")

    if args.text:
        print("Downloading ShareGPT...")
        start_time = time.perf_counter()

        config = IngestionConfig(
            dataset_name="Text Conversations",
            storage_path="./raw/ShareGPT",
            records_file="sharegpt.jsonl",
            dataset_alias="text-conv",
            max_records=None,
        )

        ShareGPTIngestion(config).run()

        print(
            f"ShareGPT is ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )

    if args.image:
        print("Downloading LLaVA-Instruct-150k...")
        start_time = time.perf_counter()

        config = IngestionConfig(
            dataset_name="Image Reasoning",
            storage_path="./raw/LLaVA-Instruct-150K",
            records_file="complex_reasoning.jsonl",
            dataset_alias="img-reason",
            max_records=1000,
        )

        LLaVAInstructComplexReasoningIngestion(config).run()

        print(
            f"LLaVA-Instruct-150k is ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )

    if args.video:
        print("Downloading LLaVA-Video...")
        start_time = time.perf_counter()

        config = IngestionConfig(
            dataset_name="Video Description",
            storage_path="./raw/LLaVA-Video",
            records_file="description.jsonl",
            dataset_alias="video-desc",
            max_records=1000,
        )

        LLaVAVideoCaptionIngestion(config).run()

        print(
            f"LLaVA-Video is ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )