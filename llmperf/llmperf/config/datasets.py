import os

from typing import Union

from llmperf.constants import DATASETS_DIR
from llmperf.ingestion.dataset import Dataset

DATASETS = {
    Dataset(
        name="Text Conversations",
        path=os.path.join(DATASETS_DIR, "ShareGPT"),
        file="sharegpt.jsonl",
        alias="text-conv",
    ),
    Dataset(
        name="Image Reasoning",
        path=os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        file="complex_reasoning.jsonl",
        alias="img-reason",
    ),
    Dataset(
        name="Image Description",
        path=os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        file="detail.jsonl",
        alias="img-desc",
    ),
    Dataset(
        name="Image Conversations",
        path=os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        file="conversation.jsonl",
        alias="img-conv",
    ),
    Dataset(
        name="Video Description",
        path=os.path.join(DATASETS_DIR, "LLaVA-Video"),
        file="description.jsonl",
        alias="vid-desc",
    ),
    Dataset(
        name="Multiple Choice (MMBench)",
        path=os.path.join(DATASETS_DIR, "MMBench"),
        file="multiple_choice.jsonl",
        alias="mmbench-mc",
    ),
    Dataset(
        name="Image Reasoning (LLaVABench)",
        path=os.path.join(DATASETS_DIR, "LLaVABench"),
        file="complex_reasoning",
        alias="llavabench-reason",
    ),
    Dataset(
        name="Image Description (LLaVABench)",
        path=os.path.join(DATASETS_DIR, "LLaVABench"),
        file="detail.jsonl",
        alias="llavabench-desc",
    ),
    Dataset(
        name="Image Conversations (LLaVABench)",
        path=os.path.join(DATASETS_DIR, "LLaVABench"),
        file="conversation.jsonl",
        alias="llavabench-conv",
    ),
    Dataset(
        name="Q&A (LLaVABench)",
        path=os.path.join(DATASETS_DIR, "LLaVABench"),
        file="qna.jsonl",
        alias="llavabench-qna",
    ),
    Dataset(
        name="Captioning (COCO-Val)",
        path=os.path.join(DATASETS_DIR, "COCO-Val"),
        file="captioning.jsonl",
        alias="cocoval-captioning",
    ),
    Dataset(
        name="Multiple Choice (Video-MME)",
        path=os.path.join(DATASETS_DIR, "Video-MME"),
        file="multiple_choice.jsonl",
        alias="videomme-mc",
    ),
    Dataset(
        name="Q&A (MMBench-Video)",
        path=os.path.join(DATASETS_DIR, "MMBench-Video"),
        file="qna.jsonl",
        alias="mmbench-video-qna",
    ),
    Dataset(
        name="Captioning (TempCompass)",
        path=os.path.join(DATASETS_DIR, "TempCompass"),
        file="captioning.jsonl",
        alias="tempcompass-captioning",
    )
}

def get_dataset_by_name(name: str) -> Union[None, Dataset]:
    for dataset in DATASETS:
        if getattr(dataset, "name", None) == name:
            return dataset
    return None

def get_dataset_by_alias(alias: str) -> Union[None, Dataset]:
    for dataset in DATASETS:
        if getattr(dataset, "alias", None) == alias:
            return dataset
    return None