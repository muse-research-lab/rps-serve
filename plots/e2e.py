import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from llmperf.postprocessing.filter import Filter
from llmperf.postprocessing.output import ExperimentOutput
from llmperf.preprocessing.workload import Workload
from utils import (
    aggregate_monitor_stats,
    calculate_metric,
    model_names,
    parse_benchmark_file,
    read_monitor_stats, 
    system_colors,
    system_markers,
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

    vllm_baselines = [
        ("llava-ov", "vllm", "tp4", 15.0),
        ("llava-ov-large", "vllm", "tp4", 2.5),
        # ("internvl-3.5-large", "vllm", "tp4", 5.0),
        # ("gemma-4-large", "vllm", "tp4", 6.0),
        # ("qwen-3.5-large", "vllm", "tp4", 7.0),
    ]

    mod_serve_baselines = [
        ("llava-ov", "mod-serve", "tp2", 15.0),
        ("llava-ov-large", "mod-serve", "tp2", 2.5),
        # ("internvl-3.5-large", "mod-serve", "tp2", 5.0),
        # ("gemma-4-large", "mod-serve", "tp2", 6.0),
        # ("qwen-3.5-large", "mod-serve", "tp2", 7.0),
    ]

    mod_serve_rps_baselines = [
        ("llava-ov", "mod-serve-rps", "tp2", 15.0),
        ("llava-ov-large", "mod-serve-rps", "tp2", 2.5),
        # ("internvl-3.5-large", "mod-serve-rps", "tp2", 5.0),
        # ("gemma-4-large", "mod-serve-rps", "tp2", 6.0),
        # ("qwen-3.5-large", "mod-serve-rps", "tp2", 7.0),
    ]

    rps_serve_baselines = [
        ("llava-ov", "rps-serve", "tp4", 15.0),
        ("llava-ov-large", "rps-serve", "tp4", 2.5),
        # ("internvl-3.5-large", "rps-serve", "tp4", 5.0),
        # ("gemma-4-large", "rps-serve", "tp4", 6.0),
        # ("qwen-3.5-large", "rps-serve", "tp4", 7.0),
    ]

    vllm_metrics = {}
    for model, sys, appr, rr in vllm_baselines:
        records = res[model][sys][appr][str(rr)]
        values = []
        gpu_util = 0
        for r in records:
            eo = ExperimentOutput(
                id=r["id"],
                output_path=OUTPUT_PATH,
            )
            eo.load()
            values.append(calculate_metric(eo, "normlat"))

            stats = read_monitor_stats(r["id"], MONITOR_STATS_PATH)
            aggr_stats = aggregate_monitor_stats(stats)
            gpu_util = aggr_stats['all']['compute_pct_mean']

        vllm_metrics[model] = (gpu_util, np.mean(values))
    
    rps_metrics = {}
    for model, sys, appr, rr in rps_serve_baselines:
        records = res[model][sys][appr][str(rr)]
        values = []
        gpu_util = 0
        for r in records:
            eo = ExperimentOutput(
                id=r["id"],
                output_path=OUTPUT_PATH,
            )
            eo.load()
            values.append(calculate_metric(eo, "normlat"))

            stats = read_monitor_stats(r["id"], MONITOR_STATS_PATH)
            aggr_stats = aggregate_monitor_stats(stats)
            gpu_util = aggr_stats['all']['compute_pct_mean']

        rps_metrics[model] = (gpu_util, np.mean(values))
    
    mod_serve_metrics = {}
    for model, sys, appr, rr in mod_serve_baselines:
        records = res[model][sys][appr][str(rr)]
        values = []
        gpu_util = 0
        for r in records:
            eo = ExperimentOutput(
                id=r["id"],
                output_path=OUTPUT_PATH,
            )
            eo.load()
            values.append(calculate_metric(eo, "normlat"))

            stats = read_monitor_stats(r["id"], MONITOR_STATS_PATH)
            aggr_stats = aggregate_monitor_stats(stats)
            gpu_util = aggr_stats['all']['compute_pct_mean']

        mod_serve_metrics[model] = (gpu_util, np.mean(values) or 0)
    
    mod_serve_rps_metrics = {}
    for model, sys, appr, rr in mod_serve_rps_baselines:
        records = res[model][sys][appr][str(rr)]
        values = []
        gpu_util = 0
        for r in records:
            eo = ExperimentOutput(
                id=r["id"],
                output_path=OUTPUT_PATH,
            )
            eo.load()
            values.append(calculate_metric(eo, "normlat"))

            stats = read_monitor_stats(r["id"], MONITOR_STATS_PATH)
            aggr_stats = aggregate_monitor_stats(stats)
            gpu_util = aggr_stats['all']['compute_pct_mean']

        mod_serve_rps_metrics[model] = (gpu_util, np.mean(values) or 0)

    title_size = 20
    axis_label_size = 20
    ticks_size = 18
    legend_size = 18
    annot_size = 14

    # Figure 4: Normalize Latency vs GPU Utilization
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8))

    for sys_name, metrics in [
        ("vllm",      vllm_metrics),
        ("mod-serve", mod_serve_metrics),
    ]:
        marker = system_markers[sys_name]
        xs, ys, names = [], [], []

        for model, (gpu_util, lat) in metrics.items():
            xs.append(100 - gpu_util)
            ys.append(float(lat))
            names.append(model_names.get(model, model))

        ax.scatter(xs, ys, marker=marker, color=system_colors[sys_name], s=80,
                label=system_names[sys_name], zorder=3)

        for x, y, name in zip(xs, ys, names):
            if name == "LLaVA-72B" and sys_name == "vllm":
                ax.annotate(name, xy=(x, y),
                        xytext=(10, 5), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            else:
                ax.annotate(name, xy=(x, y),
                        xytext=(10, -5), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])

    ax.set_xlabel("Idle GPU (%)", fontsize=axis_label_size)
    ax.set_ylabel("Norm. Lat. (s/token)", fontsize=axis_label_size)

    ax.tick_params(axis="x", labelsize=ticks_size)
    ax.tick_params(axis="y", labelsize=ticks_size)

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    ax.set_xlim(-2.5, 100)
    ax.set_ylim(0.05, 0.9)

    ax.legend(fontsize=legend_size, ncol=4)

    fig.tight_layout()
    file_path = os.path.join(PAPER_DIR, "gpu_util_vs_lat.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)

    # Figure 9: Normalize Latency (Relative to vLLM) vs GPU Utilization 
    def make_relative(metrics, ref_metrics):
        """Return {model: (idle_gpu, rel_lat)} — idle GPU absolute, lat ratio vs vllm."""
        out = {}
        for model, (gpu_util, lat) in metrics.items():
            if model not in ref_metrics:
                continue
            ref_lat = float(ref_metrics[model][1])
            out[model] = (
                100 - gpu_util,
                float(lat) / ref_lat,
            )
        return out

    rel_mod_serve     = make_relative(mod_serve_metrics,     vllm_metrics)
    rel_rps           = make_relative(rps_metrics,           vllm_metrics)
    rel_mod_serve_rps = make_relative(mod_serve_rps_metrics, vllm_metrics)

    # --- plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4 * 1.5, 4.8))

    for sys_name, rel_metrics in [
        ("rps-serve",     rel_rps),
        ("mod-serve",     rel_mod_serve),
        ("mod-serve-rps", rel_mod_serve_rps),
    ]:
        marker = system_markers[sys_name]
        xs, ys, names = [], [], []
        for model, (idle_gpu, lat_ratio) in rel_metrics.items():
            xs.append(idle_gpu)
            ys.append(lat_ratio)
            names.append(model_names.get(model, model))

        ax.scatter(xs, ys, marker=marker, color=system_colors[sys_name],
                s=80, label=system_names[sys_name], zorder=3)
        for x, y, name in zip(xs, ys, names):
            if name == "InternVL-38B" and sys_name == "mod-serve":
                ax.annotate(name, xy=(x, y),
                        xytext=(-60, 10), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            elif name == "Gemma-31B" and sys_name == "mod-serve":
                ax.annotate(name, xy=(x, y),
                        xytext=(-100, 10), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            elif name == "Qwen-27B" and sys_name == "mod-serve":
                ax.annotate(name, xy=(x, y),
                        xytext=(-100, -15), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            elif name == "LLaVA-72B" and sys_name == "mod-serve-rps":
                ax.annotate(name, xy=(x, y),
                        xytext=(10, 0), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            elif name == "LLaVA-7B" and sys_name == "mod-serve-rps":
                ax.annotate(name, xy=(x, y),
                        xytext=(10, -10), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            elif name == "Gemma-31B" and sys_name == "rps-serve":
                ax.annotate(name, xy=(x, y),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            elif name == "LLaVA-72B" and sys_name == "rps-serve":
                ax.annotate(name, xy=(x, y),
                        xytext=(10, -7), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            elif name == "InternVL-38B" and sys_name == "rps-serve":
                ax.annotate(name, xy=(x, y),
                        xytext=(10, -10), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            elif name == "LLaVA-7B" and sys_name == "rps-serve":
                ax.annotate(name, xy=(x, y),
                        xytext=(10, -10), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])
            else:
                ax.annotate(name, xy=(x, y),
                        xytext=(10, -5), textcoords="offset points",
                        fontsize=annot_size, color=system_colors[sys_name])

    ax.axhline(1.0, color="grey", linewidth=1.0, linestyle="--", alpha=0.5)

    ax.set_xlabel("Idle GPU (%)", fontsize=axis_label_size)
    ax.set_ylabel("Norm. Lat. Ratio vs vLLM (×)", fontsize=axis_label_size)
    ax.tick_params(axis="x", labelsize=ticks_size)
    ax.tick_params(axis="y", labelsize=ticks_size)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_xlim(-2.5, 100)
    ax.set_ylim(0.25, 1.5)
    ax.legend(fontsize=legend_size, ncol=4, loc="upper center",
            bbox_to_anchor=(0.5, 1.2))

    fig.tight_layout()
    file_path = os.path.join(PAPER_DIR, "gpu_util_vs_lat_relative.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)

    # Figure 11: Tail Performance
    baseline_to_gpu_cnt = {
        # llava-ov-large
        ("vllm",         "tp4"): 4,
        ("rps-serve",    "tp4"): 4,
        ("mod-serve",    "tp2"): 4,
        ("mod-serve-rps","tp2"): 4,
    }
    model = "llava-ov-large"
    target_rr = 2.5
    gpu_cnt = 4
    metrics = {
        "normlat_avg":      lambda eo: calculate_metric(eo, "normlat"),
        "normlat_p90":  lambda eo: calculate_metric(eo, "normlat_p90"),
        "normlat_p95":  lambda eo: calculate_metric(eo, "normlat_p95"),
        "normlat_p99":  lambda eo: calculate_metric(eo, "normlat_p99"),
        "ttft_avg":         lambda eo: calculate_metric(eo, "ttft"),
        "ttft_p90":     lambda eo: calculate_metric(eo, "ttft_p90"),
        "ttft_p95":     lambda eo: calculate_metric(eo, "ttft_p95"),
        "ttft_p99":     lambda eo: calculate_metric(eo, "ttft_p99"),
        "tp":           lambda eo: calculate_metric(eo, "tp"),
    }

    series = {}
    for sys in res[model].keys():
        for tp in res[model][sys].keys():
            if (sys, tp) not in baseline_to_gpu_cnt or baseline_to_gpu_cnt[(sys, tp)] != gpu_cnt:
                continue

            series[sys] = []
            for rr, records in res[model][sys][tp].items():
                if float(rr) != target_rr:
                    continue
                # per-metric list of values across records
                accumulated = {m: [] for m in metrics}
                for r in records:
                    eo = ExperimentOutput(
                        id=r["id"],
                        output_path=OUTPUT_PATH,
                    )
                    eo.load()
                    for m, fn in metrics.items():
                        accumulated[m].append(fn(eo))

                series[sys].append((
                    float(rr),
                    {m: np.mean(vals) for m, vals in accumulated.items()}
                ))
    title_size       = 24
    axis_label_size  = 24
    ticks_size       = 20
    legend_size      = 20
    fig, axes = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))

    percentiles  = ["avg", "p90", "p95", "p99"]
    metric_keys  = {"normlat": "normlat_{p}", "ttft": "ttft_{p}"}
    plot_titles  = {"normlat": "Norm. Lat. (s/token)", "ttft": "TTFT (s)"}

    systems = list(series.keys())
    n_sys   = len(systems)
    width   = 0.2
    x       = np.arange(len(percentiles))

    # assume a single target rate — take the first entry per system
    def get_val(sys, metric_prefix, p):
        key = f"{metric_prefix}_{p}"
        return float(series[sys][0][1][key])

    for ax, (metric_prefix, title) in zip(axes, plot_titles.items()):
        for i, sys in enumerate(systems):
            offset = (i - (n_sys - 1) / 2) * width
            values = [get_val(sys, metric_prefix, p) for p in percentiles]
            ax.bar(x + offset, values, width,
                color=system_colors[sys],
                label=system_names.get(sys, sys))

        ax.set_xticks(x)
        ax.set_xticklabels(["Avg", "P90", "P95", "P99"], fontsize=ticks_size)
        ax.set_ylabel(title, fontsize=axis_label_size)
        ax.tick_params(axis="y", labelsize=ticks_size)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.set_axisbelow(True)

    # shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_sys,
            fontsize=legend_size,
            bbox_to_anchor=(0.5, 1.15))

    fig.tight_layout()
    file_path = os.path.join(PAPER_DIR, "tail_normlat_ttft.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)

    # Figure 12
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

    baseline_to_gpu_cnt = {
        # llava-ov-large
        ("vllm",         "tp4"): 4,
        ("rps-serve",    "tp4"): 4,
        ("mod-serve",    "tp2"): 4,
        ("mod-serve-rps","tp2"): 4,
    }
    model = "llava-ov-large"
    target_rr = 2.5
    gpu_cnt = 4
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
    for sys in res[model].keys():
        for tp in res[model][sys].keys():
            if (sys, tp) not in baseline_to_gpu_cnt or baseline_to_gpu_cnt[(sys, tp)] != gpu_cnt:
                continue

            series[sys] = []
            for rr, records in res[model][sys][tp].items():
                if float(rr) != target_rr:
                    continue
                # per-metric list of values across records
                accumulated = {m: [] for m in metrics}
                for r in records:
                    eo = ExperimentOutput(
                        id=r["id"],
                        output_path=OUTPUT_PATH,
                    )
                    eo.load()
                    for m, fn in metrics.items():
                        accumulated[m].append(fn(eo))

                series[sys].append((
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

        for i, sys in enumerate(systems):
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
        
        ax.set_yscale("log", base=10)
        ax.set_yticks([1, 10, 100])
        ax.set_yticklabels([1, 10, 100])
        ax.set_ylim(0.9, 110)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_sys,
            fontsize=legend_size,
            bbox_to_anchor=(0.5, 1.125))

    fig.tight_layout()
    file_path = os.path.join(PAPER_DIR, "ttft_by_modality.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)