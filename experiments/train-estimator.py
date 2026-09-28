import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from llmperf.postprocessing.output import ExperimentOutput

ISO_OUTPUT_LOG_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/benchmark-log-iso.jsonl")
ISO_OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/outputs-iso")

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
                if extracted["workload"] != "text-static-long-small":
                    results[extracted["model"]][extracted["workload"]] = record

    return results

def fit_estimator(eo, model_name, plot = False, pct = None, req_type = "Text"):
    prefill_times = []
    prompt_tokens_cnt = []
    for ro in eo.request_outputs:
        if ro.aborted:
            continue
        prefill_times.append(ro.prefill_time)
        prompt_tokens_cnt.append(ro.prompt_tokens_cnt)

    t0 = np.mean(prefill_times)

    data = pd.DataFrame({
        'tokens': prompt_tokens_cnt,
        'latency': prefill_times
    })

    x = data[['tokens']]
    y = data['latency']

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if pct:
        model = QuantileRegressor(quantile=pct, alpha=0)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)

    slope = model.coef_[0]
    intercept = model.intercept_

    y_pred = model.predict(X_test)

    if plot:
        X_vals = X_test.values.flatten()
        plt.scatter(X_vals, y_test, label="Actual")

        idx = np.argsort(X_vals)
        plt.plot(X_vals[idx], y_pred[idx], color="red", label="Linear Fit")

        plt.xlabel("# Prompt Tokens")
        plt.ylabel("Prefill Latency (s)")
        plt.text(0.05, 0.95,
            f"Latency ≈ {intercept:.5f} + {slope:.6f} × tokens",
            transform=plt.gca().transAxes,
            va="top"
        )
        plt.title(f"{model_name} - {req_type} Requests")
        plt.legend()
        plt.show()

    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # print(f"t0 = {t0:.3f} | R² = {r2:.3f} | MAE = {mae:.3f} | MSE = {mse:.3f}")
    # print(f"Latency ≈ {intercept:.5f} + {slope:.6f} × tokens")
    return t0, r2, mae, mse, intercept, slope

if __name__ == '__main__':
    res = parse_benchmark_file(ISO_OUTPUT_LOG_PATH)

    results = {}
    plot = False
    for model in res.keys():
        results[model] = {}
        t_id = res[model]["text"]["id"]
        i_id = res[model]["image"]["id"]
        v_id = res[model]["video"]["id"]

        eo_t = ExperimentOutput(id=t_id, output_path=ISO_OUTPUT_PATH)
        eo_t.load()
        eo_i = ExperimentOutput(id=i_id, output_path=ISO_OUTPUT_PATH)
        eo_i.load()
        eo_v = ExperimentOutput(id=v_id, output_path=ISO_OUTPUT_PATH)
        eo_v.load()

        t0, r2, mae, mse, intercept, slope = fit_estimator(eo_t, model, plot, None, "Text")
        results[model]["text"] = (t0, r2, mae, mse, intercept, slope)

        t0, r2, mae, mse, intercept, slope = fit_estimator(eo_i, model, plot, 0.9, "Image")
        results[model]["image"] = (t0, r2, mae, mse, intercept, slope)

        t0, r2, mae, mse, intercept, slope = fit_estimator(eo_v, model, plot, 0.9, "Video")
        results[model]["video"] = (t0, r2, mae, mse, intercept, slope)

        print(f"Generated impact estimator for {model}")

    # print(json.dumps(results, indent=4))