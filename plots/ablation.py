import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from llmperf.preprocessing.workload import Workload
from llmperf.postprocessing.output import ExperimentOutput
from utils import (
    calculate_metric,
    parse_benchmark_file,
    system_colors,
    system_names
)

OUTPUT_LOG_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/benchmark-log.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/outputs")
MONITOR_STATS_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/monitor-stats")
PAPER_DIR = os.path.join(os.path.dirname(os.getcwd()), "artifacts/figures")

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(os.getcwd()), "artifacts/figures"), exist_ok=True)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    res = parse_benchmark_file(OUTPUT_LOG_PATH)

    title_size       = 24
    axis_label_size  = 24
    ticks_size       = 20
    legend_size      = 20

    # Figure 13a: Ablation - Classifier
    model = "llava-ov"
    rr = "10.0"

    baselines = ["rps-serve-nc", "rps-serve"]
    
    text_ids = set()

    text_workload = Workload(
        name="text",
        path=os.path.join(os.path.dirname(os.getcwd()), "workloads/static"),
        alias="text"
    )
    text_workload.load()

    for r in text_workload.requests:
        text_ids.add(r.id)

    long_text_workload = Workload(
        name="text-long",
        path=os.path.join(os.path.dirname(os.getcwd()), "workloads/static"),
        alias="text-long"
    )
    long_text_workload.load()

    for r in long_text_workload.requests:
        text_ids.add(r.id)
    image_ids = set()

    image_workload = Workload(
        name="image",
        path=os.path.join(os.path.dirname(os.getcwd()), "workloads/static"),
        alias="image"
    )
    image_workload.load()

    for r in image_workload.requests:
        image_ids.add(r.id)

    video_ids = set()

    video_workload = Workload(
        name="video",
        path=os.path.join(os.path.dirname(os.getcwd()), "workloads/static"),
        alias="video"
    )
    video_workload.load()

    for r in video_workload.requests:
        video_ids.add(r.id)

    metrics = {
        # --- normlat (unfiltered) -----------------------------------------
        "normlat":              lambda eo: calculate_metric(eo, "normlat"),
        "normlat_p90":          lambda eo: calculate_metric(eo, "normlat_p90"),
        "normlat_p95":          lambda eo: calculate_metric(eo, "normlat_p95"),
        "normlat_p99":          lambda eo: calculate_metric(eo, "normlat_p99"),
        # --- normlat (filtered) -------------------------------------------
        "normlat_text":         lambda eo: calculate_metric(eo, "normlat",     filter=Filter(category_ids=text_ids)),
        "normlat_p90_text":     lambda eo: calculate_metric(eo, "normlat_p90", filter=Filter(category_ids=text_ids)),
        "normlat_p95_text":     lambda eo: calculate_metric(eo, "normlat_p95", filter=Filter(category_ids=text_ids)),
        "normlat_p99_text":     lambda eo: calculate_metric(eo, "normlat_p99", filter=Filter(category_ids=text_ids)),
        "normlat_image":        lambda eo: calculate_metric(eo, "normlat",     filter=Filter(category_ids=image_ids)),
        "normlat_p90_image":    lambda eo: calculate_metric(eo, "normlat_p90", filter=Filter(category_ids=image_ids)),
        "normlat_p95_image":    lambda eo: calculate_metric(eo, "normlat_p95", filter=Filter(category_ids=image_ids)),
        "normlat_p99_image":    lambda eo: calculate_metric(eo, "normlat_p99", filter=Filter(category_ids=image_ids)),
        "normlat_video":        lambda eo: calculate_metric(eo, "normlat",     filter=Filter(category_ids=video_ids)),
        "normlat_p90_video":    lambda eo: calculate_metric(eo, "normlat_p90", filter=Filter(category_ids=video_ids)),
        "normlat_p95_video":    lambda eo: calculate_metric(eo, "normlat_p95", filter=Filter(category_ids=video_ids)),
        "normlat_p99_video":    lambda eo: calculate_metric(eo, "normlat_p99", filter=Filter(category_ids=video_ids)),
        # --- ttft (unfiltered) --------------------------------------------
        "ttft":                 lambda eo: calculate_metric(eo, "ttft"),
        "ttft_p90":             lambda eo: calculate_metric(eo, "ttft_p90"),
        "ttft_p95":             lambda eo: calculate_metric(eo, "ttft_p95"),
        "ttft_p99":             lambda eo: calculate_metric(eo, "ttft_p99"),
        # --- ttft (filtered) ----------------------------------------------
        "ttft_text":            lambda eo: calculate_metric(eo, "ttft",     filter=Filter(category_ids=text_ids)),
        "ttft_p90_text":        lambda eo: calculate_metric(eo, "ttft_p90", filter=Filter(category_ids=text_ids)),
        "ttft_p95_text":        lambda eo: calculate_metric(eo, "ttft_p95", filter=Filter(category_ids=text_ids)),
        "ttft_p99_text":        lambda eo: calculate_metric(eo, "ttft_p99", filter=Filter(category_ids=text_ids)),
        "ttft_image":           lambda eo: calculate_metric(eo, "ttft",     filter=Filter(category_ids=image_ids)),
        "ttft_p90_image":       lambda eo: calculate_metric(eo, "ttft_p90", filter=Filter(category_ids=image_ids)),
        "ttft_p95_image":       lambda eo: calculate_metric(eo, "ttft_p95", filter=Filter(category_ids=image_ids)),
        "ttft_p99_image":       lambda eo: calculate_metric(eo, "ttft_p99", filter=Filter(category_ids=image_ids)),
        "ttft_video":           lambda eo: calculate_metric(eo, "ttft",     filter=Filter(category_ids=video_ids)),
        "ttft_p90_video":       lambda eo: calculate_metric(eo, "ttft_p90", filter=Filter(category_ids=video_ids)),
        "ttft_p95_video":       lambda eo: calculate_metric(eo, "ttft_p95", filter=Filter(category_ids=video_ids)),
        "ttft_p99_video":       lambda eo: calculate_metric(eo, "ttft_p99", filter=Filter(category_ids=video_ids)),
    }

    series = {}
    for b in baselines:
        series[b] = []
        records = res[model][b]["tp1"][rr]
        accumulated = {m: [] for m in metrics}
        for r in records:
            eo = ExperimentOutput(
                id=r["id"],
                output_path=OUTPUT_PATH,
            )
            eo.load()
            for m, fn in metrics.items():
                accumulated[m].append(fn(eo))

        series[b].append((
            float(rr),
            {m: np.mean(vals) for m, vals in accumulated.items()}
        ))

    systems = list(series.keys())
    n_sys   = len(systems)
            
    percentiles   = ["avg", "p90", "p95", "p99"]
    x_labels      = ["Avg", "P90", "P95", "P99"]

    def get_val(sys, metric_prefix, p, filt):
        suffix = f"_{filt}" if filt else ""
        key = f"{metric_prefix}{suffix}" if p == "avg" else f"{metric_prefix}_{p}{suffix}"
        entry = series[sys][0]
        return float(entry[1][key])

    filters       = ["text",  "image",  "video"]
    filter_titles = ["Text",  "Image",  "Video"]

    x     = np.arange(len(percentiles))
    width = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(6.4 * 3*0.9, 4.8), sharey=False)

    metric_prefix, ylabel = "ttft", "TTFT (s)"

    for col, (filt, filt_title) in enumerate(zip(filters, filter_titles)):
        ax = axes[col]

        for i, sys in enumerate(baselines):
            offset = (i - (n_sys - 1) / 2) * width
            values = [get_val(sys, metric_prefix, p, filt) for p in percentiles]
            ax.bar(x + offset, values, width,
                color=system_colors[sys],
                label=system_names.get(sys, sys))

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=ticks_size)
        ax.tick_params(axis="y", labelsize=ticks_size)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.set_axisbelow(True)
        ax.set_title(filt_title, size=title_size)

        if col == 0:
            ax.set_ylabel(ylabel, fontsize=axis_label_size)
            ax.set_ylim(0, 2.9)
            ax.set_yticks([0.7, 1.4, 2.1, 2.8])
        if col == 1:
            ax.set_yticks([0.5, 2, 3.5, 5])
        if col == 2:
            ax.set_ylim(0, 14.5)
            ax.set_yticks([2, 6, 10, 14])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_sys,
            fontsize=legend_size,
            bbox_to_anchor=(0.5, 1.125))

    fig.tight_layout()
    file_path = os.path.join(PAPER_DIR, "ttft_by_modality_ablation.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)

    # Figure 13b: Ablation - Priority Regulator
    baselines = ["rps-serve-wo", "rps-serve-catfcfs", "edf", "rps-serve"]

    series = {}
    for b in baselines:
        series[b] = []
        records = res[model][b]["tp1"][rr]
        accumulated = {m: [] for m in metrics}
        for r in records:
            eo = ExperimentOutput(
                id=r["id"],
                output_path=OUTPUT_PATH,
            )
            eo.load()
            for m, fn in metrics.items():
                accumulated[m].append(fn(eo))

        series[b].append((
            float(rr),
            {m: np.mean(vals) for m, vals in accumulated.items()}
        ))

    percentiles   = ["avg", "p90", "p95", "p99"]
    x_labels      = ["Avg", "P90", "P95", "P99"]

    def get_val(sys, metric_prefix, p, filt):
        suffix = f"_{filt}" if filt else ""
        key = f"{metric_prefix}{suffix}" if p == "avg" else f"{metric_prefix}_{p}{suffix}"
        entry = series[sys][0]
        return float(entry[1][key])

    filters       = ["text",  "image",  "video"]
    filter_titles = ["Text",  "Image",  "Video"]

    x     = np.arange(len(percentiles))
    width = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(6.4 * 3 * 0.9, 4.8), sharey=False)

    metric_prefix, ylabel = "ttft", "TTFT (s)"

    for col, (filt, filt_title) in enumerate(zip(filters, filter_titles)):
        ax = axes[col]

        for i, sys in enumerate(baselines):
            offset = (i - (n_sys - 1) / 2) * width
            values = [get_val(sys, metric_prefix, p, filt) for p in percentiles]
            ax.bar(x + offset, values, width,
                color=system_colors[sys],
                label=system_names.get(sys, sys))

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=ticks_size)
        ax.tick_params(axis="y", labelsize=ticks_size)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.set_axisbelow(True)
        ax.set_title(filt_title, size=title_size)

        if col == 0:
            ax.set_ylabel(ylabel, fontsize=axis_label_size)
            ax.set_ylim(0, 3.3)
            ax.set_yticks([0.8, 1.6, 2.4, 3.2])
        if col == 1:
            ax.set_yticks([1.5, 3, 4.5, 6])
        if col == 2:
            ax.set_ylim(0, 14.5)
            ax.set_yticks([2, 6, 10, 14])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_sys,
            fontsize=legend_size,
            bbox_to_anchor=(0.5, 1.125))

    fig.tight_layout()
    file_path = os.path.join(PAPER_DIR, "ttft_by_modality_ablation_prio.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)