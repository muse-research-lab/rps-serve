import os
from dataclasses import dataclass, field

from vllm import AsyncEngineArgs, LLM, RequestOutput as vllmRequestOutput, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
# from vllm.user_metadata import UserMetadata

import time
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from llmperf.runner.base import AsyncBaseRunner, BaseRunner
from llmperf.postprocessing.output import ExperimentOutput, RequestOutput
from llmperf.preprocessing.workload import Request

class vLLMStaticRunner(BaseRunner):
    def setup_engine(self):
        os.environ["VLLM_USE_V1"] = "1"
        
        limit_mm_per_prompt = None
        multi_image = self.config.prompt_preparation.config.multi_image
        num_frames = self.config.prompt_preparation.config.max_sampled_frames
        if multi_image:
            limit_mm_per_prompt = {"image": num_frames}
        else:
            limit_mm_per_prompt = {"video": 1, "image": 1}

        max_model_len = self.config.max_model_len or \
            self.config.model.max_model_len
        
        max_num_batched_tokens = self.config.max_num_batched_tokens or \
            self.config.model.max_model_len
        
        num_gpu_blocks_override = self.config.num_gpu_blocks_override or \
            (self.config.model.max_model_len // 16)
        
        hf_overrides = {"architectures": ["DeepseekVLV2ForCausalLM"]} \
            if self.config.model.alias.startswith("deepseek-vl2") else None
        
        self.engine: LLM = LLM(
            model=self.config.model.path,
            gpu_memory_utilization=self.config.gpu_util,
            swap_space=self.config.swap_space,
            disable_log_stats=False,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            num_gpu_blocks_override=num_gpu_blocks_override,
            hf_token=True, # requires huggingface-cli login
            hf_overrides=hf_overrides,
            limit_mm_per_prompt=limit_mm_per_prompt,
            trust_remote_code=True,
            max_num_seqs=1,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            tensor_parallel_size=self.config.tensor_parallel_size
        )

    def execute_request(
        self, request: Request, final_prompt: dict, params: dict, user_md: dict
    ) -> RequestOutput:
        """Handle the actual inference call and return a RequestOutput."""
        sampling_params = SamplingParams(
            ignore_eos=params["ignore_eos"],
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
        )
        req_outputs = self.engine.generate(
            prompts=[final_prompt],
            sampling_params=sampling_params,
            use_tqdm=False
        )

        modality_token_index = self.get_modality_token_index(request)
        if req_outputs is None or len(req_outputs) < 1:
            # Aborted request because prompt was longer than max model length
            processed_inputs = self.engine.llm_engine.processor. \
                input_preprocessor.preprocess(
                    final_prompt,
                    lora_request=None,
                    prompt_adapter_request=None,
                    return_mm_hashes=False,
                )
            prompt_token_ids = processed_inputs["prompt_token_ids"]
            return RequestOutput(
                    id=request.id,
                    prompt_tokens_cnt=len(prompt_token_ids),
                    modality_tokens_cnt=prompt_token_ids. \
                        count(modality_token_index),
                    aborted=True
                )
        else:
            return RequestOutput.from_vllm_output(
                req_id=request.id,
                req_output=req_outputs[0],
                modality_token_index=modality_token_index
            )

    def _run(self):
        """Synchronous-style execution, run one-by-one."""
        prompt_prep = self.config.prompt_preparation
        for request in tqdm(self.config.workload.requests):
            params = prompt_prep.get_parameters(request)

            p_start = time.perf_counter()
            final_prompt = prompt_prep.get_final_prompt(request)
            p_time = time.perf_counter() - p_start

            req_output: RequestOutput = self.execute_request(
                request, final_prompt, params, {}
            )
            req_output.prompt_preparation_time = p_time
            self.outputs.append(req_output)
            self.end_time = time.time()

    def create_extra_kwargs(self) -> dict:
        mapping = {
            "pol": self.config.scheduling_policy,
            "maxlen": self.config.max_model_len,
            "batch": self.config.max_num_batched_tokens,
            "blocks": self.config.num_gpu_blocks_override,
            "encbatch": self.config.max_num_encoder_input_tokens,
            "encblocks": self.config.encoder_cache_size,
            "gpu": self.config.gpu_util,
            "swap": self.config.swap_space
        }
        return {k: v if v is not None else "def" for k, v in mapping.items()}

    @classmethod
    def get_backend(cls) -> str:
        return "vllm"
    
    @classmethod
    def get_approach(cls) -> str:
        return "iso"

@dataclass
class vLLMDynamicRunner(AsyncBaseRunner):
    idx_to_req: dict[str, Request] = field(default_factory=dict)
    idx_to_req_id: dict[str, str] = field(default_factory=dict)
    profiling_data: dict[str, RequestOutput] = field(default_factory=dict)

    def setup_engine(self):
        os.environ["VLLM_USE_V1"] = "1"
        
        limit_mm_per_prompt = None
        multi_image = self.config.prompt_preparation.config.multi_image
        num_frames = self.config.prompt_preparation.config.max_sampled_frames
        if multi_image:
            limit_mm_per_prompt = {"image": num_frames}
        else:
            limit_mm_per_prompt = {"video": 1, "image": 1}

        max_model_len = self.config.max_model_len or \
            self.config.model.max_model_len
        
        max_num_batched_tokens = self.config.max_num_batched_tokens or \
            self.config.model.max_model_len
        
        num_gpu_blocks_override = self.config.num_gpu_blocks_override or \
            (self.config.model.max_model_len // 16)
        
        engine_args = AsyncEngineArgs(
            model=self.config.model.path,
            gpu_memory_utilization=self.config.gpu_util,
            swap_space=self.config.swap_space,
            disable_log_requests=True,
            disable_log_stats = False,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            num_gpu_blocks_override=num_gpu_blocks_override,
            max_num_encoder_input_tokens=
                self.config.max_num_encoder_input_tokens,
            encoder_cache_size=self.config.encoder_cache_size,
            limit_mm_per_prompt=limit_mm_per_prompt,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            tensor_parallel_size=self.config.tensor_parallel_size
        )

        self.engine: AsyncLLM = AsyncLLM.from_engine_args(
            engine_args, log_file=self.config.engine_stats_log_file_path
        )

    async def execute_request(
        self, request: Request, final_prompt: dict, params: dict, user_md: dict
    ) -> RequestOutput:
        """Handle the actual inference call and return a RequestOutput."""
        sampling_params = SamplingParams(
            ignore_eos=params["ignore_eos"],
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
        )
        user_metadata = UserMetadata(
            uid=user_md["uid"],
            slo=user_md["slo"],
            ttft_slo=user_md["ttft_slo"],
            tbt_slo=user_md["tbt_slo"]
        )
        req_output = None
        async for output in self.engine.generate(
            final_prompt,
            sampling_params,
            request.id,
            user_md=user_metadata
        ):
            req_output: vllmRequestOutput = output

        assert req_output is not None

        original_request = self.idx_to_req[req_output.request_id]
        original_request.id = self.idx_to_req_id[req_output.request_id]
        modality_token_index = self.get_modality_token_index(original_request)

        if len(req_output.outputs) < 1:
            # Aborted request: prompt was longer than max model length
            return RequestOutput(original_request.id, aborted=True)
        else:
            return RequestOutput.from_vllm_output(
                original_request.id, req_output, modality_token_index
            )

    async def _run(self):
        "Asynchronous execution following workload timestamps."
        prompt_prep = self.config.prompt_preparation
        workload = self.config.workload
        
        async def wrapped_request(request: Request, timestamp: float):
            await asyncio.sleep(timestamp)
            params = prompt_prep.get_parameters(request)
            prof = None
            if self.config.slo_aware:
                prof = self.profiling_data.get(request.id)
            user_md = prompt_prep.get_user_md(
                uid="root",
                slo=prof.e2e if prof else 0.0,
                ttft_slo=prof.ttft if prof else 0.0,
                tbt_slo=prof.tbt if prof else 0.0,
            )
            
            p_start = time.perf_counter()
            final_prompt = prompt_prep.get_final_prompt(request)
            p_time = time.perf_counter() - p_start
            
            req_output: RequestOutput = await self.execute_request(
                request, final_prompt, params, user_md
            )
            req_output.prompt_preparation_time = p_time
            return req_output
        
        for eo_id in self.config.profiling_data:
            eo = ExperimentOutput(
                id=eo_id, output_path=self.config.profiling_data_output_path
            )
            eo.load()
            for o in eo.request_outputs:
                self.profiling_data[o.id] = o

        for idx, request in enumerate(workload.requests):
            self.idx_to_req[str(idx)] = request
            self.idx_to_req_id[str(idx)] = request.id
            request.id = str(idx)

        tasks = [
            asyncio.create_task(wrapped_request(req, ts)) 
            for req, ts in zip(workload.requests, workload.timestamps)
        ]

        try:
            for completed_task in tqdm_asyncio.as_completed(
                tasks, total=len(tasks)
            ):
                res = await completed_task
                self.outputs.append(res)
                self.end_time = time.time()
        except asyncio.CancelledError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        self.end_time = time.time()

    def create_extra_kwargs(self) -> dict:
        mapping = {
            "pol": self.config.scheduling_policy,
            "maxlen": self.config.max_model_len,
            "batch": self.config.max_num_batched_tokens,
            "blocks": self.config.num_gpu_blocks_override,
            "encbatch": self.config.max_num_encoder_input_tokens,
            "encblocks": self.config.encoder_cache_size,
            "gpu": self.config.gpu_util,
            "swap": self.config.swap_space
        }
        return {k: v if v is not None else "def" for k, v in mapping.items()}

    @classmethod
    def get_backend(cls) -> str:
        return "vllm"
    
    @classmethod
    def get_approach(cls) -> str:
        return "online"