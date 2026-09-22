import asyncio
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List
import time
from datetime import datetime

from llmperf.postprocessing.output import ExperimentOutput, RequestOutput
from llmperf.runner.config import RunnerConfig
from llmperf.preprocessing.workload import Request

@dataclass(slots=True)
class BaseRunner(ABC):
    """
    Base class for all workload runners.
    Subclasses implement the runner-specific steps.
    """
    config: "RunnerConfig"

    outputs: List[RequestOutput] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    engine: Any = None

    _id: str = None

    def _term_handler(self, signum, frame):
        """Internal handler to catch signals and raise KeyboardInterrupt."""
        print(f"Received signal {signum}. Gracefully shutting down...")
        if frame is not None and hasattr(frame, "f_code"):
            print(f"Interrupted in {frame.f_code.co_filename}:{frame.f_lineno}")
        
        raise KeyboardInterrupt
    
    @classmethod
    @abstractmethod
    def get_backend(cls) -> str:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def get_approach(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def setup_engine(self):
        """Initialize the underlying inference engine (e.g., vLLM LLM)."""
        pass

    @abstractmethod
    def execute_request(
        self, request: Request, final_prompt: dict, params: dict, user_md: dict
    ) -> RequestOutput:
        """Handle the actual inference call and return a RequestOutput."""
        pass

    @abstractmethod
    def _run(self):
        """Helper for running the workload"""

    def get_modality_token_index(self, request: Request) -> int:
        model = self.config.model
        prompt_prep = self.config.prompt_preparation
        multi_image = prompt_prep.config.multi_image
        
        if request.modality_path is None:
            return -1 

        if request.modality_size["codec"] in prompt_prep.image_codecs:
            return model.image_token_index or -1
        elif request.modality_size["codec"] in prompt_prep.video_codecs:
            if multi_image:
                return model.image_token_index or -1
            return model.video_token_index or -1
        elif request.modality_size["codec"] in prompt_prep.audio_codecs:
            return model.audio_token_index or -1
        else:
            return -1
        
    def create_extra_kwargs(self) -> dict:
        return {}

    def create_experiment_id(self, **kwargs) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        parts = [
            self.config.workload.alias,
            self.config.model.alias,
            self.get_backend(),
            self.get_approach(),
            timestamp,
        ]

        for key, value in kwargs.items():
            parts.append(f"{key}{value}")

        return "__".join(parts)
    
    def save_results(self):
        elapsed = self.end_time - self.start_time
        if elapsed <= 0 or len(self.outputs) == 0:
            print("No outputs collected.")
            return

        extra_kwargs = self.create_extra_kwargs()
        extra_kwargs = {k: v for k, v in extra_kwargs.items() if v is not None}
        
        cfg = self.config.prompt_preparation.config
        mapping = {
            "cr": f"{cfg.compression_ratio * 100:.0f}" 
                if cfg.compression_ratio else None,
            "strat": cfg.sampling_strategy,
            "maxframe": cfg.max_sampled_frames,
            "tarframe": cfg.target_frame_count,
            "redframe": cfg.frame_reduction_factor,
        }
        extra_kwargs.update({k: v for k, v in mapping.items() if v})
        
        exp_id = self.create_experiment_id(**extra_kwargs)
        self._id = exp_id

        experiment_output = ExperimentOutput(
            id=exp_id,
            elapsed_time=elapsed,
            request_outputs=self.outputs,
            output_path=self.config.output_path,
            output_log_path=self.config.output_log_path,
            engine_stats_path=self.config.engine_stats_path
        )
        experiment_output.save()
        print(f"Saved results to {exp_id}")

        if self.config.save_engine_stats:
            log_file = self.config.engine_stats_log_file_path
            experiment_output.save_engine_stats(log_file=log_file)
            print(f"Saved {experiment_output.id} engine stats")

    def run(self):
        """The main experiment loop"""
        original_handler = signal.signal(signal.SIGINT, self._term_handler)

        workload = self.config.workload
        workload.load()
        limit = self.config.num_requests
        workload.requests = workload.requests[:limit]
        workload.timestamps = workload.timestamps[:limit]

        self.setup_engine()
        
        self.start_time = time.time() 
        self.end_time = self.start_time
        try:
            self._run()
        except KeyboardInterrupt:
            print("Experiment interrupted by user.")
        except Exception as e:
            print(f"Experiment failed with error: {e}")
            raise e
        finally:
            self.save_results()
            signal.signal(signal.SIGINT, original_handler)

@dataclass
class AsyncBaseRunner(BaseRunner):
    _main_task = None
    
    def _term_handler(self):
        """Signal handler to trigger graceful shutdown."""
        print("\nTermination signal received. Cancelling pending requests...")
        if self._main_task:
            self._main_task.cancel()

    async def run(self):
        """The main asynchronous experiment loop"""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._term_handler)

        self._main_task = asyncio.current_task()
        
        workload = self.config.workload
        workload.load()
        limit = self.config.num_requests
        workload.requests = workload.requests[:limit]
        workload.timestamps = workload.timestamps[:limit]

        self.setup_engine()
        
        self.start_time = time.time() 
        self.end_time = self.start_time
        try:
            await self._run()
        except asyncio.CancelledError:
            print("Cleanup: Handling cancelled tasks...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.end_time = time.time()
            self.save_results()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)