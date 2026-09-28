import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from llmperf.postprocessing.output import ExperimentOutput
from utils import (
    aggregate_monitor_stats,
    calculate_metric,
    parse_benchmark_file,
    read_monitor_stats, 
    system_colors,
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

    # Figure 10a: Performance at scale - LLaVa-7B
    baseline_to_gpu_cnt = {
        # llava-ov
        ("vllm",         "tp1"):  1,
        ("rps-serve",    "tp1"):  1,
        ("vllm",         "tp2"):  2,
        ("rps-serve",    "tp2"):  2,
        ("mod-serve",    "tp1"):  2,
        ("mod-serve-rps","tp1"):  2,
        ("vllm",         "tp4"):  4,
        ("rps-serve",    "tp4"):  4,
        ("mod-serve",    "tp2"):  4,
        ("mod-serve-rps","tp2"):  4,
    }

    model        = "llava-ov"
    gpu_counts   = [1, 2, 4]
    target_rates = {1: 10.0, 2: 12.0, 4: 15.0}
    titles       = {1: "Single GPU", 2: "2 GPUs", 4: "4 GPUs"}

    title_size       = 24
    axis_label_size  = 24
    ticks_size       = 20
    legend_size      = 20
    all_series = {}   # gpu_cnt -> {sys: [(rate, value), ...]}

    for gpu_cnt in gpu_counts:
        series = {}
        for sys in res[model].keys():
            for tp in res[model][sys].keys():
                if (sys, tp) not in baseline_to_gpu_cnt or baseline_to_gpu_cnt[(sys, tp)] != gpu_cnt:
                    continue

                series[sys] = []
                for rr, records in res[model][sys][tp].items():
                    values = []
                    for r in records:
                        eo = ExperimentOutput(
                            id=r["id"],
                            output_path=OUTPUT_PATH,
                        )
                        eo.load()
                        values.append(calculate_metric(eo, "normlat"))

                        if float(rr) == target_rates[gpu_cnt]:
                            stats = read_monitor_stats(r["id"], MONITOR_STATS_PATH)
                            aggr_stats = aggregate_monitor_stats(stats)

                    series[sys].append((float(rr), np.mean(values)))

        all_series[gpu_cnt] = series

    fig, axes = plt.subplots(1, 3, figsize=(6.4 * 0.8 * 3, 4.8), sharey=False)

    for ax, gpu_cnt in zip(axes, gpu_counts):
        series      = all_series[gpu_cnt]
        target_rate = target_rates[gpu_cnt]

        for system, points in sorted(series.items()):
            rates, values = [], []
            for rate, value in points:
                if rate > target_rate:
                    break
                rates.append(rate)
                values.append(value)

            ax.plot(rates, values, marker="o", linewidth=2,
                    markersize=6, color=system_colors[system], label=system)

        ax.set_title(titles[gpu_cnt], size=title_size)
        ax.set_xlabel("Request Rate (req/s)", fontsize=axis_label_size)
        ax.tick_params(axis="x", labelsize=ticks_size)
        ax.tick_params(axis="y", labelsize=ticks_size)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

        if gpu_cnt == 1:
            ax.set_xticks([0, 2.5, 5, 7.5, 10])
            ax.set_xticklabels(["0","2.5", "5", "7.5", "10"])
            ax.set_yticks([0.02, 0.06, 0.1, 0.14])
        if gpu_cnt == 2:
            ax.set_xticks([0, 3, 6, 9, 12])
            ax.set_yticks([0.05, 0.1, 0.15, 0.20])
        if gpu_cnt == 4:
            ax.set_xticks([0, 5, 10, 15])
            ax.set_yticks([0.05, 0.1, 0.15, 0.20])

        ax.legend()

    # y-label only on the leftmost subplot
    axes[0].set_ylabel("Norm. Lat. (s/token)", fontsize=axis_label_size)
    fig.tight_layout()

    file_path = os.path.join(PAPER_DIR, "norm_lat_baselines.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)

    # Figure 10b: Performance at scale - LLaVa-72B
    baseline_to_gpu_cnt = {
        # llava-ov-large
        ("vllm",         "tp2"): 2,
        ("rps-serve",    "tp2"): 2,
        # ("edf",          "tp2"): 2,
        ("vllm",         "tp4"): 4,
        ("rps-serve",    "tp4"): 4,
        # ("edf",          "tp4"): 4,
        ("mod-serve",    "tp2"): 4,
        ("mod-serve-rps","tp2"): 4,
    }

    model        = "llava-ov-large"
    gpu_counts   = [2, 4]
    target_rates = {2: 2.5, 4: 2.5}
    titles       = {2: "2 GPUs", 4: "4 GPUs"}

    title_size       = 24
    axis_label_size  = 24
    ticks_size       = 20
    legend_size      = 20
    all_series = {}   # gpu_cnt -> {sys: [(rate, value), ...]}

    for gpu_cnt in gpu_counts:
        series = {}
        for sys in res[model].keys():
            for tp in res[model][sys].keys():
                if (sys, tp) not in baseline_to_gpu_cnt or baseline_to_gpu_cnt[(sys, tp)] != gpu_cnt:
                    continue

                series[sys] = []
                for rr, records in res[model][sys][tp].items():
                    values = []
                    for r in records:
                        eo = ExperimentOutput(
                            id=r["id"],
                            output_path=OUTPUT_PATH,
                        )
                        eo.load()
                        values.append(calculate_metric(eo, "normlat"))

                    if float(rr) == target_rates[gpu_cnt]:
                        stats = read_monitor_stats(r["id"], MONITOR_STATS_PATH)
                        aggr_stats = aggregate_monitor_stats(stats)

                    series[sys].append((float(rr), np.mean(values)))

        all_series[gpu_cnt] = series
    
    fig, axes = plt.subplots(1, 2, figsize=(6.4*2*0.9, 4.8), sharey=False)

    for ax, gpu_cnt in zip(axes, gpu_counts):
        series      = all_series[gpu_cnt]
        target_rate = target_rates[gpu_cnt]

        for system, points in sorted(series.items()):
            rates, values = [], []
            for rate, value in points:
                if rate > target_rate:
                    break
                rates.append(rate)
                values.append(value)

            # print(values)
            ax.plot(rates, values, marker="o", linewidth=2,
                    markersize=6, color=system_colors[system], label=system)

        ax.set_title(titles[gpu_cnt], size=title_size)
        ax.set_xlabel("Request Rate (req/s)", fontsize=axis_label_size)
        ax.tick_params(axis="x", labelsize=ticks_size)
        ax.tick_params(axis="y", labelsize=ticks_size)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

        ax.set_yticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
        ax.set_ylim(0, 0.335)

    # y-label only on the leftmost subplot
    axes[0].set_ylabel("Norm. Lat. (s/token)", fontsize=axis_label_size)
    fig.tight_layout()

    file_path = os.path.join(PAPER_DIR, "norm_lat_baselines_large.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)