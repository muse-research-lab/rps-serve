from dataclasses import dataclass, field
from typing import Literal

from llmperf.config.models import Model

from transformers import AutoProcessor, AutoTokenizer

SamplingMethod = Literal["uniform", "all", "motion_based", "sharpness_based",
                         "scene_change"]

@dataclass(frozen=True, slots=True)
class PromptPreparationConfig:
    model: Model

    system_prompt: str = None
    multi_image: bool = False

    # ImageAsset configuration
    compression_ratio: float = None

    # Video Asset configuration
    max_sampled_frames: int = -1
    sampling_strategy: SamplingMethod = "uniform"
    strategy_params: dict[str, float] = field(default_factory=dict)
    target_frame_count: int | None = None
    frame_reduction_factor: float | None = None
    
    # For dynamic frame count configuration
    dynamic_frames: float = False
    min_sampled_frames: int = 8
    max_duration: int = 180
    min_duration: int = 1

    # Parameters configuration (vLLM)
    ignore_eos: bool = True
    max_tokens: int = 2048
    temperature: float = 1.0 # Zero means greedy sampling (determinism)
    dynamic_max_tokens: bool = False # Overrides max_tokens if True

    e2e_slo_scale: float = 1.0
    ttft_slo_scale: float = 1.0
    tbt_slo_scale: float = 1.0

    tokenizer: AutoTokenizer = None
    processor: AutoProcessor = None

    def __post_init__(self):
        path = self.model.path
        set_val = object.__setattr__
        
        if not self.tokenizer:
            set_val(self, "tokenizer", AutoTokenizer.from_pretrained(
                path, trust_remote_code=True
            ))
        if not self.processor:
            set_val(self, "processor", AutoProcessor.from_pretrained(
                path, use_fast=True, trust_remote_code=True
            ))

        if self.sampling_strategy == "all":
            set_val(self, "max_sampled_frames", self.total_frames)

        if self.sampling_strategy == "all" and self.dynamic_frames:
            raise ValueError(
                "'all' sampling strategy cannot be combined",
                "with the dynamic_frames option"
            )
        
        if self.dynamic_frames and \
            self.max_sampled_frames <= self.min_sampled_frames:
            raise ValueError(
                "max_sampled_frames must be greater than min_sampled_frames"
            )
        
        if self.dynamic_frames and \
            self.max_duration <= self.min_duration:
            raise ValueError("max_duration must be greater than min_duration")