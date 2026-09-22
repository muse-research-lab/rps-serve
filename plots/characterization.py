import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from output import ExperimentOutput
from utils import parse_benchmark_iso_file, get_cdf, model_names

ISO_OUTPUT_LOG_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/benchmark-log-iso.jsonl")
ISO_OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts/outputs-iso")
PAPER_DIR = os.path.join(os.path.dirname(os.getcwd()), "artifacts/figures")

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(os.getcwd()), "artifacts/figures"), exist_ok=True)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    title_size = 20
    axis_label_size = 20
    ticks_size = 18
    legend_size = 18

    labels = ["text", "image", "video"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    # Figure 2a: Characterization Memory
    res = parse_benchmark_iso_file(ISO_OUTPUT_LOG_PATH)
    fig, axes = plt.subplots(1, 4, figsize=(6.4*1.5, 2.475*1))
    plt.subplots_adjust(wspace=0.3, top=0.80)

    j = -1
    for model in res.keys():
        if model == "llava-ov":
            continue
        j += 1
        ax = axes[j]
        eos_md = [
            res[model]["text-static-small"],
            res[model]["image-static"],
            res[model]["video-static"],
        ]
        for i, eo_md in enumerate(eos_md):
            eo = ExperimentOutput(id=eo_md["id"], output_path=ISO_OUTPUT_PATH)
            eo.load()
            x, y = get_cdf([r.modality_tokens_cnt or r.prompt_tokens_cnt for r in eo.request_outputs if r.ttft != 0])
            ax.plot(x, y, label=labels[i], color=colors[i], linewidth=3)

        ax.set_title(model_names[model], size=title_size)

        ax.set_xscale("log", base=10)
        ax.set_xticks([10, 100, 1000, 10000, 100000])
        ax.set_xticklabels(["10", "", r"$10^3$", "", r"$10^5$"], fontsize=ticks_size)
        ax.set_xlim(7, 100000 + 30000)

        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0 ", "25", "50", "75", "100"], fontsize=ticks_size)

        if model != "llava-ov-large":
            ax.set_yticklabels(["", "", "", "", ""], fontsize=0)

        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=ticks_size)

    fig.text(0.5, -0.175, "Memory Footprint (#tokens)", ha="center", fontsize=axis_label_size)
    fig.text(0.025, 0.5, "Probability (%)", va="center", rotation="vertical", fontsize=axis_label_size)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=legend_size, handlelength=1.5,
                loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))

    file_path = os.path.join(PAPER_DIR, "mem_cdf.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)

    # Figure 2b: Characterization TTFT
    res = parse_benchmark_iso_file(ISO_OUTPUT_LOG_PATH)
    fig, axes = plt.subplots(1, 4, figsize=(6.4*1.5, 2.475*1))
    plt.subplots_adjust(wspace=0.3, top=0.80)

    j = -1
    for model in res.keys():
        if model == "llava-ov":
            continue
        j += 1
        ax = axes[j]
        eos_md = [
            res[model]["text-static-small"],
            res[model]["image-static"],
            res[model]["video-static"],
        ]
        for i, eo_md in enumerate(eos_md):
            eo = ExperimentOutput(id=eo_md["id"], output_path=ISO_OUTPUT_PATH)
            eo.load()
            x, y = get_cdf([r.ttft for r in eo.request_outputs if r.ttft != 0])
            ax.plot(x, y, label=labels[i], color=colors[i], linewidth=3)

        ax.set_title(model_names[model], size=title_size)

        ax.set_xscale("log", base=10)
        ax.set_xticks([0.1, 1, 10, 50])
        ax.set_xticklabels(["0.1", "1", "10", ""], fontsize=ticks_size)

        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0 ", "25", "50", "75", "100"], fontsize=ticks_size)

        if model != "llava-ov-large":
            ax.set_yticklabels(["", "", "", "", ""], fontsize=0)

        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=ticks_size)

    fig.text(0.5, -0.175, "TTFT Latency (s)", ha="center", fontsize=axis_label_size)
    fig.text(0.025, 0.5, "Probability (%)", va="center", rotation="vertical", fontsize=axis_label_size)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=legend_size, handlelength=1.5,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))

    file_path = os.path.join(PAPER_DIR, "ttft_cdf.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)

    # Figure 3: GPU Prefill Latency Breakdown
    res = parse_benchmark_iso_file(ISO_OUTPUT_LOG_PATH)

    experiment_outputs_t = []
    experiment_outputs_i = []
    experiment_outputs_v = []
    for model in model_names.keys():
        eo = ExperimentOutput(id=res[model]["text-static-small"]["id"], output_path=ISO_OUTPUT_PATH)
        eo.load()
        experiment_outputs_t.append(eo)

        eo = ExperimentOutput(id=res[model]["image-static"]["id"], output_path=ISO_OUTPUT_PATH)
        eo.load()
        experiment_outputs_i.append(eo)

        eo = ExperimentOutput(id=res[model]["video-static"]["id"], output_path=ISO_OUTPUT_PATH)
        eo.load()
        experiment_outputs_v.append(eo)
    title_size = 24
    axis_label_size = 24
    ticks_size = 18
    legend_size = 20

    labels = [v for v in model_names.values()]
    fig, axes = plt.subplots(1, 3, figsize=(6.4*2.5, 2.475))
    ax1, ax2, ax3 = axes
    plt.subplots_adjust(hspace=0.4, left=0.075, bottom=0.1, top=0.92)

    for ax, exp, title, add_xticks, yticks in zip(
        axes,
        [experiment_outputs_t, experiment_outputs_i, experiment_outputs_v],
        ["Text", "Image", "Video"],
        [True, True, True],
        [
            [0.0, 0.05, 0.1, 0.15, 0.2],
            [0, 0.15, 0.3, 0.45, 0.6],
            [0, 0.3, 0.6, 0.9, 1.2]
        ]
    ):
        x = np.arange(len(exp))
        bar_width = 0.8
        ttfts = [np.mean([r.ttft - r.processor_time for r in eo.request_outputs if r.ttft != 0]) for eo in exp]
        encoder_times = [np.mean([r.encoder_time for r in eo.request_outputs if r.ttft != 0]) for eo in exp]

        ax.bar(x, ttfts, bar_width, label="LLM", color="#C9DD84")
        ax.bar(x, encoder_times, bar_width, label="Encoder", color="#859F3D")

        ax.set_title(title, size=title_size)
        if add_xticks:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=ticks_size+6)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([" "] * len(x), fontsize=ticks_size)
        ax.tick_params(axis="y", labelsize=ticks_size)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{i* 1000:.0f}" for i in yticks])
        # ax.set_ylim(0, yticks[-1] + 0.05 * yticks[-1])

    handles, labels_ = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=legend_size, handlelength=2, loc="upper center", bbox_to_anchor=(0.5, 1.375), ncols=2, columnspacing=3.0)

    fig.supylabel("Latency (ms)", fontsize=axis_label_size)

    file_path = os.path.join(PAPER_DIR, "ttft_breakdown.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
