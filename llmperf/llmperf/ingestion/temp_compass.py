import json
import os
import os
import shutil
import tempfile
import requests
import zipfile

from datasets import load_dataset
from tqdm import tqdm

from llmperf.ingestion.base import BaseIngestion

TEMP_COMPASS_ZIP_URL = (
    "https://huggingface.co/datasets/"
    "lmms-lab/TempCompass/resolve/main/tempcompass_videos.zip"
)

def download_and_unzip_tempcompass():
    tmp_dir = tempfile.TemporaryDirectory()
    zip_path = os.path.join(tmp_dir.name, "tempcompass_videos.zip")

    # Download ZIP (streaming)
    with requests.get(TEMP_COMPASS_ZIP_URL, stream=True) as r:
        r.raise_for_status()

        total_size = int(r.headers.get("Content-Length", 0))
        chunk_size = 8192

        with open(zip_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading TempCompass"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Unzip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir.name)

    temp_compass_dir = os.path.join(tmp_dir.name, "videos")

    return tmp_dir, temp_compass_dir

class TempCompassCaptioningIngestion(BaseIngestion):
    def fetch_raw_data(self):
        return load_dataset("lmms-lab/TempCompass", "captioning", split="test")
    
    def deduplicate(self, records):
        pairs = set()

        dedup_records = []
        for record in records:
            q = record["question"]
            vid = record["video_id"]
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

        tmp_dir, temp_compass_dir = download_and_unzip_tempcompass()

        for record in records:
            vid = record["video_id"]
            video = f"{vid}.mp4"
            record["video"] = video
            src_path = os.path.join(temp_compass_dir, video)
            dst_path = os.path.join(videos_dir, video)
            shutil.copy2(src_path, dst_path)

        tmp_dir.cleanup()

        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")