import argparse
import time

from llmperf.ingestion.config import IngestionConfig
from llmperf.ingestion.sharegpt import ShareGPTIngestion
from llmperf.ingestion.llava_instruct import LLaVAInstructComplexReasoningIngestion
from llmperf.ingestion.llava_video import LLaVAVideoCaptionIngestion

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