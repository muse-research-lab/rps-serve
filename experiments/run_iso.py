import os
import importlib

import yaml
import inspect
import asyncio
import argparse
from llmperf.promptpreparation.config import PromptPreparationConfig
from llmperf.runner.config import RunnerConfig
from llmperf.config.models import Model
from llmperf.preprocessing.workload import Workload

models = {
    "llava-ov": Model(
        name="LLaVA-OneVision-7b",
        path="llava-onevision-qwen2-7b-ov-chat-hf",
        max_model_len=32768,
        alias="llava-ov",
        image_token_index=151646,
        video_token_index=151647
    ),
    "llava-ov-large": Model(
        name="LLaVA-OneVision-72b",
        path="llava-onevision-qwen2-72b-ov-chat-hf",
        max_model_len=32768,
        alias="llava-ov-large",
        image_token_index=151646,
        video_token_index=151647
    ),
    "gemma-4-large": Model(
        name="Gemma4-31B",
        path="gemma-4-31B-it",
        max_model_len=262144,
        alias="gemma-4-large",
        image_token_index=258880,
        video_token_index=258884
    ),
    "internvl-3.5-large": Model(
        name="InternVL3.5-38B",
        path="InternVL3_5-38B-HF",
        max_model_len=40960,
        alias="internvl-3.5-large",
        image_token_index=151671,
        video_token_index=151678
    ),
    "qwen-3.5-large": Model(
        name="Qwen3.5-27B",
        path="Qwen3.5-27B",
        max_model_len=262144,
        alias="qwen-3.5-large",
        image_token_index=248056,
        video_token_index=248057
    ),
}

def get_prompt_prep_class(prompt_prep_type: str):
    registry = {
        "default": ("llmperf.promptpreparation.base", "DefaultPromptPreparation"),
        "default-extra": ("llmperf.promptpreparation.base", "DefaultExtraPromptPreparation"),
    }

    if prompt_prep_type not in registry:
        raise TypeError(f"This prompt preparation type is not supported: {prompt_prep_type}")

    module_path, class_name = registry[prompt_prep_type]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_runner_class(runner_type: str):
    registry = {
        "vllm-static": ("llmperf.runner.vllm", "vLLMStaticRunner"),
        "vllm-dynamic": ("llmperf.runner.vllm", "vLLMDynamicRunner"),
        "guidellm": ("llmperf.runner.guidellm", "GuideLLMRunner"),
    }

    if runner_type not in registry:
        raise TypeError(f"This runner type is not supported: {runner_type}")

    module_path, class_name = registry[runner_type]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

async def run_experiments(config_path: str):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    os.environ["GUIDELLM__MAX_CONCURRENCY"] = 1
    
    model = models[config_data['model_alias']]
    model.path = config_data.get("baseline_config", {}).get("model", None)
    workload = Workload(
        name=config_data["workload_alias"],
        path=os.path.join(os.path.dirname(os.getcwd()), "workloads/static"),
        alias=config_data["workload_alias"]
    )

    p_config = PromptPreparationConfig(
        model=model,
        **config_data['prompt_config']
    )
    prompt_prep_class = get_prompt_prep_class(config_data['prompt_prep_type'])
    prompt_prep = prompt_prep_class(p_config)

    r_config = RunnerConfig(
        model=model,
        workload=workload,
        prompt_preparation=prompt_prep,
        **config_data['runner_config']
    )
    runner_class = get_runner_class(config_data['runner_type'])
    runner = runner_class(r_config)

    if inspect.iscoroutinefunction(runner.run):
        await runner.run()
    else:
        runner.run()

def main():
    parser = argparse.ArgumentParser(description="LLM Performance CLI Runner")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_experiments(args.config))

if __name__ == "__main__":
    main()