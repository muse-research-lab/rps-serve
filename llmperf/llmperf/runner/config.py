from dataclasses import dataclass, field
from typing import List, LiteralString, Optional, Union

from llmperf.config.models import Model
from llmperf.preprocessing.workload import Workload
from llmperf.promptpreparation.base import BasePromptPreparation

@dataclass(frozen=True, slots=True)
class RunnerConfig:
    model: Model
    workload: Workload
    prompt_preparation: BasePromptPreparation
    
    save_engine_stats: bool = False
    output_path: Optional[Union[str,LiteralString]] = None
    output_log_path: Optional[Union[str,LiteralString]] = None
    engine_stats_path: Optional[Union[str,LiteralString]] = None
    engine_stats_log_file_path: Union[str,LiteralString] = "engine-stats.log"

    num_requests: int = None
    slo_aware: bool = False
    profiling_data: Optional[List[str]] = field(default_factory=list)
    profiling_data_output_path: Optional[Union[str,LiteralString]] = None

    # vLLM configuration
    gpu_util: float = 0.95
    swap_space: int = 0
    max_model_len: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    num_gpu_blocks_override: Optional[int] = None
    scheduling_policy: Optional[str] = None
    max_num_encoder_input_tokens: Optional[int] = None
    encoder_cache_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = 1
    tensor_parallel_size: Optional[int] = 1

    # OpenAI compatible server configuration
    target_address: str = "http://localhost:8000/"

    # GuideLLM configuration
    profile: str = "synchronous"
    rate: list[float] | None = None
    
    ## Backend configuration
    backend: str = "openai_http"
    max_worker_processes: int = 1
    return_token_ids: bool = True
    
    ## Data configuration
    data_column_mapper: str | dict = "generative_column_mapper"
    data_finalizer: str | dict = "generative"
    data_sampler: str | None = None
    random_seed: int = 42
    
    ## Constraints configuration
    max_seconds: int | float | None = None
    max_requests: int | None = None

    ## Metadata
    engine: str = "vllm"
    approach: str = "default"
