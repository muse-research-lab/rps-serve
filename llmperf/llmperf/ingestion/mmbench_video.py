import json
import os
import os
import shutil
import tempfile
import requests
import pickle

from datasets import load_dataset
from typing import List
from tqdm import tqdm

from llmperf.ingestion.base import BaseIngestion

def download_extract_and_filter_videos_pkl(
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
            pkl_path = os.path.join(tmp_dir, filename)

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("Content-Length", 0))
                chunk_size = 8192

                with open(pkl_path, "wb") as f, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {filename}",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                raise ValueError(
                    f"Expected dict in {filename}, got {type(data)}"
                )

            for video in list(videos_to_keep):
                if video not in data:
                    continue

                video_bytes = data[video]

                filename = f"{video}.mp4"
                out_path = os.path.join(output_dir, filename)
                with open(out_path, "wb") as f:
                    f.write(video_bytes)

                videos_to_keep.remove(video)

            os.remove(pkl_path)

class MMBenchVideoIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("lscpku/MMBench-Video", split="test")
    
    def deduplicate(self, records):
        pairs = set()

        dedup_records = []
        for record in records:
            q = record["question"]
            vid = record["video"]
            if (q, vid) in pairs:
                continue
            
            pairs.add((q, vid))
            dedup_records.append(record)

        return dedup_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )

        videos_dir = os.path.join(self.config.storage_path, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        videos_to_keep = set()
        for record in records:
            video = record["video"]
            file = f"{video}.mp4"
            record["video"] = file
            videos_to_keep.add(video)

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        base_url = (
            "https://huggingface.co/datasets/"
            "opencompass/MMBench-Video/resolve/main/video_pkl"
        )
        

        urls = [
            f"{base_url}/video_chunk_{i}.pkl"
            for i in range(9)
        ]

        download_extract_and_filter_videos_pkl(urls, videos_to_keep, videos_dir)