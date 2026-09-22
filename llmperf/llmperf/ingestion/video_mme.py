import json
import os
import os
import shutil
import tempfile
import zipfile
import requests

from datasets import load_dataset
from typing import List
from tqdm import tqdm

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
            zip_path = os.path.join(tmp_dir, filename)

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("Content-Length", 0))
                chunk_size = 8192

                with open(zip_path, "wb") as f, tqdm(
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

            extract_root = os.path.join(tmp_dir, filename.replace(".zip", ""))
            os.makedirs(extract_root, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as z:
                for member in z.infolist():
                    if member.is_dir():
                        continue

                    member_name = member.filename
                    base_name = os.path.basename(member_name)

                    if base_name not in videos_to_keep:
                        continue

                    z.extract(member, path=extract_root)

                    src = os.path.join(extract_root, member_name)
                    dst = os.path.join(output_dir, base_name)

                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)

                    videos_to_keep.remove(base_name)

            if os.path.exists(zip_path):
                os.remove(zip_path)

            if os.path.exists(extract_root):
                shutil.rmtree(extract_root)

class VideoMMEIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("lmms-lab/Video-MME", split="test")

    def clean_data(self, records):
        cleaned_records = []
        for record in records:
            if record["duration"] == "short":
                cleaned_records.append(record)
            
        return cleaned_records
    
    def deduplicate(self, records):
        question_ids = set()

        dedup_records = []
        for record in records:
            qid = record["question_id"]
            if qid in question_ids:
                continue
            
            question_ids.add(qid)
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
            video_id = record["videoID"]
            file = f"{video_id}.mp4"
            record["video"] = file
            videos_to_keep.add(file)

        base_url = (
            "https://huggingface.co/datasets/"
            "lmms-lab/Video-MME/resolve/main"
        )

        urls = [
            f"{base_url}/videos_chunked_{i:02d}.zip"
            for i in range(1, 21)
        ]

        download_extract_and_filter_videos(urls, videos_to_keep, videos_dir)

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")