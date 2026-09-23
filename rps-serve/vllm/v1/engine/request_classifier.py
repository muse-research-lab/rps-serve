# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from abc import ABC, abstractmethod

import joblib

from vllm.v1.engine import EngineCoreRequest


class BaseRequestClassifier(ABC):
    """Abstract base class for request classifiers."""

    @abstractmethod
    def classify(self, request: EngineCoreRequest) -> str:
        """
        Classify a request and return a queue/category identifier.

        Args:
            request: The engine core request to classify.

        Returns:
            A string identifier representing the request's classification.
        """
        ...


class DummyRequestClassifier(BaseRequestClassifier):
    # Single queue implementation for naive-aging policy
    def classify(self, request: EngineCoreRequest) -> str:
        return "rocks"


class NaiveRequestClassifier(BaseRequestClassifier):
    def classify(self, request: EngineCoreRequest) -> str:
        # Video inputs & multi image inputs
        if (
            request.mm_features
            and request.mm_features[0] is not None
            and request.mm_features[0].modality == "video"
        ) or (
            request.mm_features
            and len(request.mm_features) > 1
            and request.mm_features[0] is not None
            and request.mm_features[0].modality == "image"
        ):
            return "rocks"

        # Single image inputs
        if (
            request.mm_features
            and len(request.mm_features) == 1
            and request.mm_features[0] is not None
            and request.mm_features[0].modality == "image"
        ):
            return "pebbles"

        # Text
        return "sand"


class SmartRequestClassifier(BaseRequestClassifier):
    cache: dict[str, dict[str, int | list[int]]] = {
        "llava-onevision-qwen2-0.5b-ov-hf": {
            "image_token_index": 151646,
            "video_token_index": 151647,
            "classes": [1, 0, 2],  # [sand, pebbles, rocks]
        },
        "Qwen2.5-VL-3B-Instruct": {
            "image_token_index": 151655,
            "video_token_index": 151656,
            "classes": [2, 0, 1],
        },
        "gemma-3-4b-it": {
            "image_token_index": 262144,
            "video_token_index": -1,
            "classes": [2, 0, 1],
        },
        "llava-onevision-qwen2-7b-ov-chat-hf": {
            "image_token_index": 151646,
            "video_token_index": 151647,
            "classes": [1, 0, 2],
        },
        "Qwen2.5-VL-7B-Instruct": {
            "image_token_index": 151655,
            "video_token_index": 151656,
            "classes": [2, 0, 1],
        },
        "gemma-3-12b-it": {
            "image_token_index": 262144,
            "video_token_index": -1,
            "classes": [2, 0, 1],
        },
        "pixtral-12b": {
            "image_token_index": 10,
            "video_token_index": -1,
            "classes": [1, 0, 2],
        },
        "llava-onevision-qwen2-72b-ov-chat-hf": {
            "image_token_index": 151646,
            "video_token_index": 151647,
            "classes": [1, 0, 2],
        },
        "InternVL3_5-38B-HF": {
            "image_token_index": 151671,
            "video_token_index": 151678,
            "classes": [1, 0, 2],
        },
        "gemma-4-31B-it": {
            "image_token_index": 258880,
            "video_token_index": 258884,
            "classes": [2, 0, 1],
        },
        "Qwen3.5-27B": {
            "image_token_index": 248056,
            "video_token_index": 248057,
            "classes": [2, 0, 1],
        },
    }

    def __init__(self, model: str) -> None:
        model_name = model.split("/")[-1]

        if model_name in self.cache:
            entry = self.cache[model_name]
            classifiers_path = os.path.join(os.path.dirname(os.getcwd()), "../artifacts/classifiers")
            (
                "/home/konstantinos.papaioannou/rps-serve/artifacts/classifiers"
            )
            path = os.path.join(classifiers_path, f"{model_name}.pkl")
            self.model = joblib.load(path)
            self.image_token_index: int = int(entry["image_token_index"])  # type: ignore[arg-type]
            self.video_token_index: int = int(entry["video_token_index"])  # type: ignore[arg-type]
            classes = entry["classes"]
            assert isinstance(classes, list)
            self.cluster_to_category: dict[int, str] = {
                classes[0]: "sand",
                classes[1]: "pebbles",
                classes[2]: "rocks",
            }

    def classify(self, request: EngineCoreRequest) -> str:
        assert request.prompt_token_ids is not None, "prompt_token_ids must not be None"
        assert request.request_md is not None, "request_md must not be None"

        modality_tokens_cnt = request.prompt_token_ids.count(
            self.image_token_index
        ) or request.prompt_token_ids.count(self.video_token_index)

        x = [[request.request_md.estimated_time, modality_tokens_cnt]]
        cluster = self.model.predict(x)[0]

        return self.cluster_to_category[int(cluster)]
