import argparse
import os
import time

from llmperf.ingestion.dataset import Dataset
from llmperf.preprocessing.config import PreprocessingConfig
from llmperf.preprocessing.sharegpt import ShareGPTPreprocessing, ShareGPTLongPreprocessing
from llmperf.preprocessing.llava_instruct import LLaVAInstructComplexReasoningPreprocessing
from llmperf.preprocessing.llava_video import LLaVAVideoCaptionPreprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        action="store_true",
        help="Run ShareGPT preprocessing",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Run LLaVA-Instruct-150k preprocessing",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Run LLaVA-Video preprocessing",
    )
    args = parser.parse_args()

    if not (args.text or args.image or args.video):
        parser.error("Specify at least one of --text, --image, or --video")

    if args.text:
        print("Preprocessing ShareGPT...")
        start_time = time.perf_counter()

        text_conv_ds = Dataset(
                name="Text Conversations",
                path=os.path.join(os.getcwd(), "raw", "ShareGPT"),
                file="sharegpt.jsonl",
                alias="text-conv",
            )
        
        config = PreprocessingConfig(
            workload_name="Text Conversations",
            storage_path="./static",
            workload_alias="text",
            data_input=text_conv_ds,
            timing_mode="static",
            num_requests=1000
        )

        preprocessing = ShareGPTPreprocessing(config)
        workload = preprocessing.run()

        config = PreprocessingConfig(
            workload_name="Long Text Conversations",
            storage_path="./static",
            workload_alias="text-long",
            data_input=text_conv_ds,
            timing_mode="static",
            num_requests=1000
        )

        preprocessing = ShareGPTLongPreprocessing(config)
        workload = preprocessing.run()

        print(
            f"Text workloads are ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )

    if args.image:
        print("Preprocessing LLaVA-Instruct-150k...")
        start_time = time.perf_counter()

        img_reason_ds = Dataset(
            name="Image Reasoning",
            path=os.path.join(os.getcwd(), "raw", "LLaVA-Instruct-150K"),
            file="complex_reasoning.jsonl",
            alias="img-reason",
        )

        config = PreprocessingConfig(
            workload_name="Image Reasoning",
            storage_path="./static",
            workload_alias="image",
            data_input=img_reason_ds,
            timing_mode="static",
            num_requests=1000
        )

        preprocessing = LLaVAInstructComplexReasoningPreprocessing(config)
        workload = preprocessing.run()

        print(
            f"Image workload is ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )

    if args.video:
        print("Preprocessing LLaVA-Video...")
        start_time = time.perf_counter()

        vid_desc_ds = Dataset(
            name="Video Description",
            path=os.path.join(os.getcwd(), "raw", "LLaVA-Video"),
            file="description.jsonl",
            alias="vid-desc",
        )

        config = PreprocessingConfig(
            workload_name="Video Description",
            storage_path="./static",
            workload_alias="video",
            data_input=vid_desc_ds,
            timing_mode="static",
            num_requests=1000
        )

        preprocessing = LLaVAVideoCaptionPreprocessing(config)
        workload = preprocessing.run()

        print(
            f"Video workload is ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )