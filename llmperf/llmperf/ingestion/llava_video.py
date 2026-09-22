import json
import os
import requests
import shutil
import tempfile
import tarfile
import random

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from typing import List

from llmperf.ingestion.base import BaseIngestion

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

class LLaVAVideoOpenEndedIngestion(BaseIngestion):
    def fetch_raw_data(self):
        # Note: Doesn't include the 240k entries of "llava_hound", since they
        # are not split correctly in HuggingFace.
        configs = [
            "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1", "0_30_s_nextqa",
            "30_60_s_academic_v0_1", "30_60_s_youtube_v0_1", "30_60_s_nextqa",
            "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1", "1_2_m_nextqa",
            "2_3_m_youtube_v0_1", "2_3_m_academic_v0_1", "2_3_m_nextqa"
        ]

        datasets_list = [
            load_dataset("lmms-lab/LLaVA-Video-178K", config, split="open_ended")
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
        # ! Note: Some configs may have records with more than a single question
        # ! For deduplication, only the first user question is considered
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
            "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1", "0_30_s_nextqa",
            "30_60_s_academic_v0_1", "30_60_s_youtube_v0_1", "30_60_s_nextqa",
            "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1", "1_2_m_nextqa",
            "2_3_m_youtube_v0_1", "2_3_m_academic_v0_1", "2_3_m_nextqa"
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
            ("0_30_s_nextqa", 1),
            ("30_60_s_nextqa", 2),
            ("1_2_m_nextqa", 2),
            ("2_3_m_nextqa", 1),
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

class LLaVAVideoMultipleChoiceIngestion(BaseIngestion):
    def fetch_raw_data(self):
        configs = [
            "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1", "0_30_s_nextqa",
            "30_60_s_academic_v0_1", "30_60_s_youtube_v0_1", "30_60_s_nextqa",
            "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1", "1_2_m_nextqa",
            "2_3_m_youtube_v0_1", "2_3_m_academic_v0_1", "2_3_m_nextqa",
            "0_30_s_perceptiontest", "30_60_s_perceptiontest",
        ]

        datasets_list = [
            load_dataset(
                "lmms-lab/LLaVA-Video-178K", config, split="multi_choice"
            )
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
        # ! Note: Some configs may have records with more than a single question
        # ! For deduplication, only the first user question is considered
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
            "0_30_s_academic_v0_1", "0_30_s_youtube_v0_1", "0_30_s_nextqa",
            "30_60_s_academic_v0_1", "30_60_s_youtube_v0_1", "30_60_s_nextqa",
            "1_2_m_youtube_v0_1", "1_2_m_academic_v0_1", "1_2_m_nextqa",
            "2_3_m_youtube_v0_1", "2_3_m_academic_v0_1", "2_3_m_nextqa",
            "0_30_s_perceptiontest", "30_60_s_perceptiontest",
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
            ("0_30_s_nextqa", 1),
            ("30_60_s_nextqa", 2),
            ("1_2_m_nextqa", 2),
            ("2_3_m_nextqa", 1),
            ("0_30_s_perceptiontest", 3),
            ("30_60_s_perceptiontest", 2),
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