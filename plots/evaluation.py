
import os
import json
from glob import glob

import joblib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

from utils import (
    model_names
)

PAPER_DIR = os.path.join(os.path.dirname(os.getcwd()), "artifacts/figures")
CLASSIFIER_DIR = os.path.join(os.path.dirname(os.getcwd()), "artifacts", "classifiers")
ESTIMATOR_OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()), "artifacts", "impact-estimator-results.json")


if __name__ == '__main__':
    # Figure 6: Estimator Accuracy
    with open(ESTIMATOR_OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = list(data.keys())
    modalities = ["text", "image", "video"]

    mae = {
        modality: [data[m][modality][2] for m in models]  # index 2 is MAE
        for modality in modalities
    }
    title_size = 24
    axis_label_size = 24
    ticks_size = 18
    legend_size = 20

    modalities = ["text", "image", "video"]

    fig, axes = plt.subplots(1, 3, figsize=(6.4*2.5, 2.475))
    ax1, ax2, ax3 = axes
    plt.subplots_adjust(hspace=0.4, left=0.075, bottom=0.1, top=0.92)

    x = np.arange(len(models))
    bar_width = 0.8
    for ax, y, title, color, yticks in zip(
        axes,
        [mae["text"], mae["image"], mae["video"]],
        ["Text", "Image", "Video"],
        ["#4c72b0", "#dd8452", "#55a868", "#000000"],
        [
            [0.0, 0.015, 0.03, 0.045, 0.06],
            [0, 0.002, 0.004, 0.006, 0.008],
            [0, 0.05, 0.1, 0.15, 0.2],
        ]
    ):
        ax.bar(x, y, bar_width, label="Prefill", color=color)

        ax.set_title(title, size=title_size)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names.values(), rotation=90, fontsize=ticks_size+6)
        ax.tick_params(axis="y", labelsize=ticks_size)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{i* 1000:.0f}" for i in yticks])
        ax.set_ylim(0, yticks[-1] + 0.05 * yticks[-1])


    fig.supylabel("MAE (ms)", fontsize=axis_label_size)

    fig.savefig(os.path.join(PAPER_DIR, "estimator_acc.pdf"), dpi=300, bbox_inches="tight", format="pdf", transparent=True)

    # Figure 7: Request Clustering
    model_name = "Qwen3.5-27B"
    classifier_path = os.path.join(CLASSIFIER_DIR, f"{model_name}-details.pkl")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"No classifier details found for {model_name} at {classifier_path}. Run train-classifier.py first.")

    with open(classifier_path, "rb") as f:
        details = joblib.load(f)

    X = np.asarray(details["processed_features"])
    clusters = np.asarray(details["clusters"])
    modality = np.asarray(details["modalities"])
    feature_names = details["features"]

    labels = ["text", "image", "video"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    cmap_clusters = ListedColormap(["#E5992E", "#E16E65", "#56B493"])
    cmap = ListedColormap(["#dd8452","#4c72b0","#4c72b0","#55a868"]) 

    title_size = 20
    axis_label_size = 20
    ticks_size = 18
    legend_size = 18
    d = X.shape[1]

    xlab, ylab = feature_names[0], feature_names[1] if d >= 2 else ""

    fig, ax = plt.subplots(1, 2, figsize=(6.4*2, 4.8*0.8))

    # ---- Plot 1: colored by cluster labels ----
    sc1 = ax[0].scatter(X[:,0], X[:,1], c=clusters, cmap=cmap_clusters)
    ax[0].set_ylabel("Memory", fontsize=axis_label_size)
    ax[0].set_xlabel("Latency", fontsize=axis_label_size)
    ax[0].tick_params(axis="x", labelsize=ticks_size)
    ax[0].tick_params(axis="y", labelsize=ticks_size)

    handles = [
        plt.scatter([], [], color=["#E16E65", "#E5992E", "#56B493"][i], label=["Rocks", "Pebbles", "Sand"][i].capitalize(), s=100)
        for i in range(3)
    ]

    ax[0].legend(handles=handles, loc="upper center", fontsize=legend_size, title_fontsize=legend_size, ncol=3, bbox_to_anchor=(0.5, 1.25), handletextpad=0.2, columnspacing=1)

    # ---- Plot 2: colored by modality ----
    # If modality is categorical strings, convert to numeric codes
    if modality.dtype.kind in {"U", "S", "O"}:
        unique_mods = list(sorted(set(modality.tolist())))
        mod_to_int = {m: i for i, m in enumerate(unique_mods)}
        modality_colors = [mod_to_int[m] for m in modality.tolist()]
        unique_mods = sorted(set(modality.tolist()))
        int_to_mod = {i: m for m, i in mod_to_int.items()}
    else:
        modality_colors = modality

    sc2 = ax[1].scatter(X[:,0], X[:,1], c=modality_colors, cmap=cmap)
    ax[1].set_xlabel("Latency", fontsize=axis_label_size)
    ax[1].tick_params(axis="x", labelsize=ticks_size)
    ax[1].tick_params(axis="y", labelsize=ticks_size)


    handles = [
        plt.scatter([], [], color=colors[i], label=labels[i].capitalize(), s=100)
        for i in range(len(colors))
    ]

    ax[1].legend(handles=handles, loc="upper center", fontsize=legend_size, title_fontsize=legend_size, ncol=3, bbox_to_anchor=(0.5, 1.25), handletextpad=0.2, columnspacing=1)

    file_path = os.path.join(PAPER_DIR, "clustering.pdf")
    fig.savefig(file_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)