
from guidellm.benchmark import benchmark_generative_text
from guidellm.benchmark.schemas.generative.entrypoints import BenchmarkGenerativeTextArgs

import os
import time
import sys

from llmperf.runner.base import AsyncBaseRunner
from llmperf.postprocessing.output import RequestOutput
from llmperf.preprocessing.workload import Request

class GuideLLMRunner(AsyncBaseRunner):
    args: BenchmarkGenerativeTextArgs = None
    
    def setup_engine(self):
        assert self.config.profile != "sweep"
        os.environ["GUIDELLM__MAX_WORKER_PROCESSES"] = str(self.config.max_worker_processes)

        if isinstance(self.config.data_column_mapper, str):
            data_column_mapper = {
                "type": self.config.data_column_mapper
            }
        else:
            data_column_mapper = self.config.data_column_mapper

        self.args = BenchmarkGenerativeTextArgs(
            target=self.config.target_address,
            data=[[]],
            model=self.config.model.path,
            profile=self.config.profile,
            rate=self.config.rate,
            backend=self.config.backend,
            backend_kwargs={
                "extras": {
                    "body": {
                        "return_token_ids": self.config.return_token_ids
                    },
                },
            },
            data_column_mapper=data_column_mapper,
            data_finalizer=self.config.data_finalizer,
            data_sampler=self.config.data_sampler,
            random_seed=self.config.random_seed,
            max_seconds=self.config.max_seconds,
            max_requests=self.config.max_requests,
        )
    
    async def execute_request(
        self, request: Request, final_prompt: dict, params: dict, user_md: dict
    ) -> RequestOutput:
        raise NotImplementedError
    
    async def _run(self):
        """Prepare GuideLLM in-memory dataset and run asynchronously."""
        data = []
        req_to_mod = {}
        prompt_prep = self.config.prompt_preparation
        for request in self.config.workload.requests:
            modality_token_idx = self.get_modality_token_index(request)
            req_to_mod[request.id] = modality_token_idx
            record = {
                "request_id": request.id,
                "input": prompt_prep.process_text(request),
                "system_prompt": prompt_prep.config.system_prompt or "",
                "image": request.modality_path if prompt_prep.is_image(request) else None,
                "video": request.modality_path if prompt_prep.is_video(request) else None,
                "audio": request.modality_path if prompt_prep.is_audio(request) else None,
            }

            params = prompt_prep.get_parameters(request)
            record["output_tokens_count"] = params["max_tokens"]
            data.append(record)

        sys.stdin.flush()
        sys.stdout.flush()
        self.args.data = [data]
        self.start_time = time.time() 
        self.end_time = self.start_time
        report, _ = await benchmark_generative_text(self.args)
        self.end_time = time.time()

        benchmark = report.benchmarks[0]
        for request_status, requests in benchmark.requests:
            for request in (requests or []):
                modality_tokens_cnt = 0
                if request.prompt_token_ids:
                    modality_tokens_cnt = request.prompt_token_ids.count(req_to_mod[request.request_id])
                ro = RequestOutput(
                    id=request.request_id,
                    prompt_tokens_cnt=request.prompt_tokens or 0,
                    modality_tokens_cnt=modality_tokens_cnt,
                    decode_tokens_cnt=request.output_tokens or 0,
                    outputs=[request.output],
                    vllm_id=request.request_id,
                    arrival_time=request.request_start_time,
                    last_token_time=request.info.timings.last_token_iteration,
                    first_token_time=request.info.timings.first_token_iteration,
                    finished_time=request.request_end_time,
                    aborted=(request_status in ["errored", "incomplete"])
                )
                self.outputs.append(ro)

    def create_extra_kwargs(self) -> dict:
        mapping = {
            "pol": self.config.scheduling_policy,
            "maxlen": self.config.max_model_len,
            "batch": self.config.max_num_batched_tokens,
            "blocks": self.config.num_gpu_blocks_override,
            "encbatch": self.config.max_num_encoder_input_tokens,
            "encblocks": self.config.encoder_cache_size,
            "gpu": self.config.gpu_util,
            "swap": self.config.swap_space,
            "engine": self.get_backend(),
            "approach": self.get_approach(),
            "rate": str(self.config.rate or 0),
            "prof": self.config.profile
        }
        return {k: v if v is not None else "def" for k, v in mapping.items()}

    def get_backend(self) -> str:
        return self.config.engine
    
    def get_approach(self) -> str:
        return self.config.approach