import json
import numpy as np
import os

import imageio.v2 as imageio
from PIL import Image

from llmperf.ingestion.base import BaseIngestion

class DummyImageIngestion(BaseIngestion):
    def fetch_raw_data(self):
        dummy_records = []
        for w in range(0, 2049, 64):
            for h in range(0, 2049, 64):
                if w == 0 or h == 0:
                    continue
                dummy_records.append({
                    "input": "I",
                    "output": "O",
                    "image": f"noise_{w}x{h}.png"
                })

        return dummy_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        for w in range(0, 2049, 64):
            for h in range(0, 2049, 64):
                if h == 0 or w == 0:
                    continue
                noise_data = np.random.randint(0, 256, (w, h, 3), dtype=np.uint8)
                img = Image.fromarray(noise_data, 'RGB')
                # Save the image
                file_path = os.path.join(images_dir, f"noise_{w}x{h}.png")
                img.save(file_path)

class DummyImageSmallIngestion(BaseIngestion):
    def fetch_raw_data(self):
        dummy_records = []
        for w, h in [
            (410, 410), (819, 819), (1229, 1229), (1638, 1638), (2048, 2048)
        ]:
            dummy_records.append({
                "input": "I",
                "output": "O",
                "image": f"noise_{w}x{h}.png"
            })

        return dummy_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        images_dir = os.path.join(self.config.storage_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        for w, h in [
            (410, 410), (819, 819), (1229, 1229), (1638, 1638), (2048, 2048)
        ]:
            noise_data = np.random.randint(0, 256, (w, h, 3), dtype=np.uint8)
            img = Image.fromarray(noise_data, 'RGB')
            # Save the image
            file_path = os.path.join(images_dir, f"noise_{w}x{h}.png")
            img.save(file_path)

class DummyVideoIngestion(BaseIngestion):
    def fetch_raw_data(self):
        dummy_records = []
        for f in [2, 4, 8, 16, 32, 64]:
            for w in range(0, 2049, 128):
                for h in range(0, 2049, 128):
                    if w == 0 or h == 0:
                        continue
                    dummy_records.append({
                        "input": "I",
                        "output": "O",
                        "video": f"noise_{f}_{w}x{h}.mp4"
                    })

        return dummy_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        videos_dir = os.path.join(self.config.storage_path, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        for f in [2, 4, 8, 16, 32, 64]:
            for w in range(0, 2049, 128):
                for h in range(0, 2049, 128):
                    if h == 0 or w == 0:
                        continue
                    video_array = np.random.randint(
                        0, 256,
                        (f, w, h, 3),
                        dtype=np.uint8
                    )
                    # Save the video
                    file_path = os.path.join(videos_dir, f"noise_{f}_{w}x{h}.mp4")
                    imageio.mimsave(
                        file_path,
                        video_array,
                        fps=2,
                        codec="libx264"
                    )

class DummyVideoSmallIngestion(BaseIngestion):
    def fetch_raw_data(self):
        dummy_records = []
        for f in [6, 13, 19, 26, 32]:
            w, h = 512, 512
            dummy_records.append({
                "input": "I",
                "output": "O",
                "video": f"noise_{f}_{w}x{h}.mp4"
            })
        
        for w, h in [
            (205, 205), (410, 410), (614, 614), (819, 819), (1024, 1024)
        ]:
            f = 16
            dummy_records.append({
                "input": "I",
                "output": "O",
                "video": f"noise_{f}_{w}x{h}.mp4"
            })  

        return dummy_records

    def store(self, records):
        output_file = os.path.join(
            self.config.storage_path, self.config.records_file
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        videos_dir = os.path.join(self.config.storage_path, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        for f in [6, 13, 19, 26, 32]:
            w, h = 512, 512
            video_array = np.random.randint(
                0, 256,
                (f, w, h, 3),
                dtype=np.uint8
            )
            # Save the video
            file_path = os.path.join(videos_dir, f"noise_{f}_{w}x{h}.mp4")
            imageio.mimsave(
                file_path,
                video_array,
                fps=2,
                codec="libx264"
            )

        for w, h in [
            (205, 205), (410, 410), (614, 614), (819, 819), (1024, 1024)
        ]:
            f = 16
            video_array = np.random.randint(
                0, 256,
                (f, w, h, 3),
                dtype=np.uint8
            )
            # Save the video
            file_path = os.path.join(videos_dir, f"noise_{f}_{w}x{h}.mp4")
            imageio.mimsave(
                file_path,
                video_array,
                fps=2,
                codec="libx264"
            )