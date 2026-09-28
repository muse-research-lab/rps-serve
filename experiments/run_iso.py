import importlib

import yaml
import inspect
import asyncio
import argparse
from llmperf.promptpreparation.config import PromptPreparationConfig
from llmperf.runner.config import RunnerConfig
from llmperf.config.models import get_model_by_alias
from llmperf.config.workloads import get_workload_by_alias

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
    
    model = get_model_by_alias(config_data['model_alias'])
    workload = get_workload_by_alias(config_data['workload_alias'])

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