import os

from dataclasses import dataclass
from typing import LiteralString, Optional, Union

from llmperf.constants import MODELS_DIR

@dataclass
class Model:
    name: str
    path: Union[str,LiteralString]
    max_model_len: int
    alias: str
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    audio_token_index: Optional[int] = None

    def __hash__(self):
        return hash((self.name, self.alias))

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.name == other.name and self.alias == other.alias
        return False

MODELS = {
    Model(
        name="Mistral-7b",
        path="mistralai/Mistral-7B-Instruct-v0.2",
        max_model_len=32768,
        alias="text-mistral"
    ),
    Model(
        name="LLaVA 1.6 (Mistral-7b)",
        path="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=32768,
        alias="image-mistral",
        image_token_index=32000
    ),
    # Duplicate LLaVA 1.6
    Model(
        name="LLaVA-1.6-7B",
        path="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=32768,
        alias="llava-1.6-small",
        image_token_index=32000
    ),
    Model(
        name="LLaVA-1.6-13B",
        path="llava-hf/llava-v1.6-vicuna-13b-hf",
        max_model_len=4096,
        alias="llava-1.6",
        image_token_index=32000
    ),
    Model(
        name="LLaVA-Next-Video (Mistral-7b)",
        path="llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
        max_model_len=32768,
        alias="video-mistral",
        image_token_index=32001,
        video_token_index=32000
    ),
    Model(
        name="Qwen2-Audio-7b",
        path="Qwen/Qwen2-Audio-7B-Instruct",
        max_model_len=8192,
        alias="audio-qwen",
        audio_token_index=151646
    ),
    Model(
        name="LLaVA-OneVision-7b",
        path=os.path.join(MODELS_DIR, "llava-onevision-qwen2-7b-ov-chat-hf"),
        max_model_len=32768,
        alias="llava-ov",
        image_token_index=151646,
        video_token_index=151647
    ),
    Model(
        name="LLaVA-OneVision-500M",
        path=os.path.join(MODELS_DIR, "llava-onevision-qwen2-0.5b-ov-hf"),
        max_model_len=32768,
        alias="llava-ov-small",
        image_token_index=151646,
        video_token_index=151647
    ),
    Model(
        name="LLaVA-OneVision-72b",
        path=os.path.join(MODELS_DIR, "llava-onevision-qwen2-72b-ov-chat-hf"),
        max_model_len=32768,
        alias="llava-ov-large",
        image_token_index=151646,
        video_token_index=151647
    ),
    Model(
        name="Qwen2.5-7B",
        path=os.path.join(MODELS_DIR, "Qwen2.5-VL-7B-Instruct"),
        max_model_len=128000,
        alias="qwen-2.5",
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        name="Qwen2.5-3B",
        path=os.path.join(MODELS_DIR, "Qwen2.5-VL-3B-Instruct"),
        max_model_len=128000,
        alias="qwen-2.5-small",
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        name="Qwen2-7B",
        path=os.path.join(MODELS_DIR, "Qwen2-VL-7B-Instruct"),
        max_model_len=32768,
        alias="qwen-2",
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        name="Qwen2-2B",
        path=os.path.join(MODELS_DIR, "Qwen2-VL-2B-Instruct"),
        max_model_len=32768,
        alias="qwen-2-small",
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        name="Gemma3-4B",
        path=os.path.join(MODELS_DIR, "gemma-3-4b-it"),
        max_model_len=131072,
        alias="gemma-3-small",
        image_token_index=262144
    ),
    Model(
        name="Gemma3-12B",
        path=os.path.join(MODELS_DIR, "gemma-3-12b-it"),
        max_model_len=131072,
        alias="gemma-3",
        image_token_index=262144
    ),
    Model(
        name="DeepSeek-VL2-1B",
        path="deepseek-ai/deepseek-vl2-tiny",
        max_model_len=4096,
        alias="deepseek-vl2-tiny",
        image_token_index=128815
    ),
    Model(
        name="DeepSeek-VL2-2.8B",
        path="deepseek-ai/deepseek-vl2-small",
        max_model_len=4096,
        alias="deepseek-vl2-small",
        image_token_index=100003
    ),
    Model(
        name="DeepSeek-VL2-4.5B",
        path="deepseek-ai/deepseek-vl2",
        max_model_len=4096,
        alias="deepseek-vl2",
        image_token_index=128815
    ),
    Model(
        name="LLaVA-1.5-7B",
        path="llava-hf/llava-1.5-7b-hf",
        max_model_len=4096,
        alias="llava-1.5-small",
        image_token_index=32000
    ),
    Model(
        name="LLaVA-1.5-13B",
        path="llava-hf/llava-1.5-13b-hf",
        max_model_len=4096,
        alias="llava-1.5",
        image_token_index=32000
    ),
    Model(
        name="Pixtral-12B",
        path=os.path.join(MODELS_DIR, "pixtral-12b"),
        max_model_len=131072,
        alias="pixtral",
        image_token_index=10
    ),
    Model(
        name="Llama-2-13b",
        path=os.path.join(MODELS_DIR, "Llama-2-13b-chat-hf"),
        max_model_len=4096,
        alias="llama-2"
    ),
    Model(
        name="Llama-2-7b",
        path=os.path.join(MODELS_DIR, "Llama-2-7b-chat-hf"),
        max_model_len=4096,
        alias="llama-2-small"
    ),
    Model(
        name="Flow-Judge-v0.1",
        path=os.path.join(MODELS_DIR, "Flow-Judge-v0.1"),
        max_model_len=131072,
        alias="flow-judge"
    ),
    Model(
        name="Gemma4-31B",
        path=os.path.join(MODELS_DIR, "gemma-4-31B-it"),
        max_model_len=262144,
        alias="gemma-4-large",
        image_token_index=258880,
        video_token_index=258884
    ),
    Model(
        name="InternVL3.5-38B",
        path=os.path.join(MODELS_DIR, "InternVL3_5-38B-HF"),
        max_model_len=40960,
        alias="internvl-3.5-large",
        image_token_index=151671,
        video_token_index=151678
    ),
    Model(
        name="Qwen3.5-27B",
        path=os.path.join(MODELS_DIR, "Qwen3.5-27B"),
        max_model_len=262144,
        alias="qwen-3.5-large",
        image_token_index=248056,
        video_token_index=248057
    ),
    
}

def get_model_by_name(name: str) -> Union[None, Model]:
    for model in MODELS:
        if getattr(model, "name", None) == name:
            return model
    return None

def get_model_by_alias(alias: str) -> Union[None, Model]:
    for model in MODELS:
        if getattr(model, "alias", None) == alias:
            return model
    return None