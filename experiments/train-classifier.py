import re
import os
import json
import joblib
import pandas as pd

from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans

from llmperf.postprocessing.output import ExperimentOutput

ISO_OUTPUT_LOG_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/benchmark-log-iso.jsonl")
ISO_OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/outputs-iso")

CLASSIFIER_DIR = os.path.join(os.path.dirname(os.getcwd()), "artifacts/classifiers")

def parse_benchmark_file(filepath: str) -> dict:
    """
    Reads a file of JSON lines, extracts benchmark data, and organizes it
    into a nested dictionary: dict[model][workload]

    ID format example:
    text-static-small__llava-ov-large__vllm__iso__20260526-174501__poldef__
    maxlendef__batchdef__blocksdef__encbatchdef__encblocksdef__gpu0.95__swap0__
    stratuniform__maxframe32
    """
    patterns = {
        "workload": re.compile(r'^([^_]+)__'),
        "model":    re.compile(r'^[^_]+__([^_]+)__'),
    }

    results = defaultdict(dict)

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_id = record.get("id", "")

            extracted = {}
            for key, pattern in patterns.items():
                match = pattern.search(record_id)
                if not match:
                    break
                extracted[key] = match.group(1)
            else:
                results[extracted["model"]][extracted["workload"]] = record

    return results

def generate_classifier(eos, model_name, save = False):
    prefill_times = []
    modality_tokens_cnt = []

    request_outputs = []
    for eo in eos:
        request_outputs.extend(eo.request_outputs)

    t_cnt  = sum(not ro.aborted for ro in eos[0].request_outputs)
    i_cnt  = sum(not ro.aborted for ro in eos[1].request_outputs)
    v_cnt  = sum(not ro.aborted for ro in eos[2].request_outputs)
    lt_cnt = sum(not ro.aborted for ro in eos[3].request_outputs)

    for ro in request_outputs:
        if ro.aborted:
            continue
        prefill_times.append(ro.prefill_time)
        modality_tokens_cnt.append(ro.modality_tokens_cnt)

    data = pd.DataFrame({
        "prefill_time": prefill_times,
        "num_modality_tokens": modality_tokens_cnt,
        "modality_type": t_cnt * ["text"] + i_cnt * ["image"] + v_cnt * ["video"] + lt_cnt * ["long_text"]
    })

    features = ["prefill_time", "num_modality_tokens"]
    
    X = data[features]
    pipeline = Pipeline([
        ("power", PowerTransformer(method="yeo-johnson")),
        ("scale", StandardScaler()),
        ("model", KMeans(n_clusters=3, random_state=42))
    ])

    pipeline.fit(X)
    clusters = pipeline["model"].labels_
    X_proc = pipeline[:-1].transform(X)
    modalities = data["modality_type"].values

    details = {
        "features": features,
        "processed_features": X_proc,
        "clusters": clusters,
        "modalities": modalities,
    }

    if save:
        joblib.dump(pipeline, os.path.join(CLASSIFIER_DIR, f"{model_name}.pkl"))
        joblib.dump(details, os.path.join(CLASSIFIER_DIR, f"{model_name}-details.pkl"))

if __name__ == '__main__':
    res = parse_benchmark_file(ISO_OUTPUT_LOG_PATH)

    model_names = {
        "llava-ov": "llava-onevision-qwen2-7b-ov-chat-hf",
        "llava-ov-large": "llava-onevision-qwen2-72b-ov-chat-hf",
        "internvl-3.5-large": "InternVL3_5-38B-HF",
        "qwen-3.5-large": "Qwen3.5-27B",
        "gemma-4-large": "gemma-4-31B-it"
    }

    save = True
    for model in res.keys():
        if model not in model_names:
            continue
        t_id = res[model]["text"]["id"]
        i_id = res[model]["image"]["id"]
        v_id = res[model]["video"]["id"]
        lt_id = res[model]["text-long"]["id"]

        eo_t = ExperimentOutput(id=t_id, output_path=ISO_OUTPUT_PATH)
        eo_t.load()
        eo_i = ExperimentOutput(id=i_id, output_path=ISO_OUTPUT_PATH)
        eo_i.load()
        eo_v = ExperimentOutput(id=v_id, output_path=ISO_OUTPUT_PATH)
        eo_v.load()
        eo_lt = ExperimentOutput(id=lt_id, output_path=ISO_OUTPUT_PATH)
        eo_lt.load()

        generate_classifier([eo_t, eo_i, eo_v, eo_lt], model_names[model], save)

        print(f"Generated request classifier for {model}")