# RPS-Serve Artifact

Artifact for **"Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference"**, accepted at ATC '26.

- **Paper DOI:** TODO
- **Artifact DOI (Zenodo):** TODO 
- **License:** Apache License 2.0

---

## 1. Artifact Overview

RPS-Serve is a modality-aware request scheduling framework for multimodal LLM (MLLM) serving, built on top of vLLM (v0.21.1). This artifact contains:

- `artifacts/`: Experiment outputs and figures

- `experiments/`: Scripts for running the paper's experiments and producing summary metrics.

- `guidellm/`: Patched GuideLLM. See changes here: [`guidellm-changes.patch`](guidellm-changes.patch)

- `llmperf/`: Library for generating workloads, running experiments and plotting results.

- `models/`: Models used for evaluation

- `plots/`: Scripts for reproducing the paper's figures from raw experiment outputs

- `rps-serve/`: Modified version of vLLM containing the following components:
    - Impact Estimator: `vllm/v1/engine/prefill_time_estimator.py`
    - Request Classifier: `vllm/v1/engine/request_classifier.py`
    - Queue Manager: `vllm/v1/core/sched/queue_manager.py`
    - Priority Regulator: `vllm/v1/core/sched/slo_expiration_manager.py`

  See changes here [`rps-serve-changes.patch`](rps-serve-changes.patch).

- `vllm/`: Patched vLLM used for distributed serving. See changes here: [`vllm-changes.patch`](vllm-changes.patch)

- `workloads/`: Scripts for generating the workloads

---

## 2. Experimental Environment

### 2.1 Hardware

- GPUs: 4 × NVIDIA H100 (96 GB)
- CPUs: 2 × Intel Xeon Platinum 8592+ (64 cores each)
- Memory: 1 TB DRAM

GPU requirements vary by figure:

- No GPUs: Figures 6 and 7
- 1 GPU: Figures 2, 3, 10, and 13
- 2 GPUs: Figure 10
- 4 GPUs: Figures 4, 9, 10, 11, and 12

### 2.2 Operating System

- Debian 13 (trixie)
- Linux kernel 6.12.107

### 2.3 Software Dependencies

- Python 3.12.8
- CUDA 13.4
- `uv >= 0.10.11`:
  [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
- git-lfs:
  `sudo apt-get install git-lfs`

---

## 3. Installation

To automatically set up the environments and install the dependencies, run:

```sh
bash install.sh
```

**Expected runtime:** ~15 minutes

### 3.1 RPS-Serve

Install RPS-Serve in a dedicated virtual environment. This environment is used
to deploy RPS-Serve instances.

```sh
# From the repository root
cd rps-serve
uv venv --python 3.12.8 --seed --managed-python
source .venv/bin/activate
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=d735968f6d634ec849268f18e3b84ceb494fee79
export SETUPTOOLS_SCM_PRETEND_VERSION=0.21.1rc1.dev120+g70c5a0596
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu130
uv pip install --editable . --torch-backend=cu130
uv pip install joblib
```

### 3.2 Baselines

Install the patched vLLM version in a separate virtual environment. This
environment is used to deploy the baseline systems: vLLM, ModServe, and
ModServe with RPS.

```sh
# From the repository root
cd vllm
uv venv --python 3.12.8 --seed --managed-python
source .venv/bin/activate
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=33ef1941e217a2126d745caec6c6130d6aec3b31
export SETUPTOOLS_SCM_PRETEND_VERSION=0.21.1rc1.dev120+g70c5a0596
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu130
uv pip install --editable . --torch-backend=cu130
```

### 3.3 Additional Dependencies

Create a virtual environment in the repository root for workload generation,
experimentation, and result visualization.

```sh
# From the repository root
uv venv --python 3.12.8 --seed --managed-python
source .venv/bin/activate
uv pip install -r requirements.txt
```

Next, install the patched version of LLMPerf and GuideLLM.

```sh
cd llmperf
uv pip install -e .
cd ..
cd guidellm
uv pip install -e .
```

## 4. Workload Generation

To automatically download raw data and generate the workloads, run:

```sh
bash generate_workloads.sh
```

**Expected runtime:** ~10 minutes

### 4.1 Ingest Datasets

(Optional) To ingest the datasets, run:

```sh
# From the repository root
cd workloads
python3 ingestion.py --text --image --video
```

To avoid the time consuming ingestion of image and video data, we provide the following:

```sh
# From the repository root
source .venv/bin/activate
cd workloads
curl -L "https://cloud.software.imdea.org/index.php/s/fw9DJZ8tkLY9RB6/download" | tar -xf - -C .
```

### 4.2 Generate Modality Workloads

To generate workloads for each modality, run:

```sh
python3 preprocessing.py --text --image --video
```

### 4.3 Generate Mixed Workloads

To generate the mixed workloads used in the paper, run:

```sh
python3 generation.py
```

## 5. Download Models

To download LLaVA-7B (~30GB) and LLaVA-72B (~273GB), run:

```sh
# From the repository root
cd models
git lfs install
git clone https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-chat-hf
git clone https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-chat-hf
```

(Optional) To download the rest of the models:

```
git clone https://huggingface.co/OpenGVLab/InternVL3_5-38B-HF # 144 GB
git clone https://huggingface.co/google/gemma-4-31B-it # 117 GB
git clone https://huggingface.co/Qwen/Qwen3.5-27B # 104 GB
```

The optional step above is only needed if you want to run Experiments 1–4 for
all models reported in the paper. We do not recommend doing this during the
artifact evaluation, because these experiments can take a very long time to complete.

---

## 6. Minimal Working Example

```sh
# From the repository root
source .venv/bin/activate
cd experiments
python3 orchestrator.py --config config-rps-minimal.yaml
```
The orchestrator.py script:
1. Launches RPS-Serve on 1 GPU (LLaVA-7B)
2. Replays a tiny mixed trace (~100 requests)
3. Produces a metrics summary

**Expected output:** Two files are created, one under `artifacts/outputs` and another one under `artifacts/monitor-stats`.
```sh
...
Avg. E2E Latency: 4.484160141944885
Avg. TTFT Latency: 0.4903690218925476
Avg. TBT Latency: 0.019876031017663578
...
```
**Expected runtime:** ~5 minutes on 1×H100

---

## 7. Deployment Configuration

To configure the different baselines, chek the following files:

- RPS-Serve: `experiments/configs/config-rps-tp{1,2,4}.yaml`
- vLLM: `experiments/configs/config-vllm-tp{1,2,4}.yaml`
- Mod-Serve: `experiments/configs/config-mod-serve-tp{1,2}.yaml`
- Mod-Serve & RPS: `experiments/configs/config-mod-serve-rps-tp{1,2}.yaml`

---

## 8. Experimentation

### 8.1 Experiment 1 (End-to-end performance com-parison)

**Claim:**
Evaluated against vLLM, ModServe, and ModServe & RPS across state-of-the-art MLLMs,
RPS-Serve increases GPU utilization by 65% and reduces TTFT by up to 9.3×, on
average across models, compared to disaggregation-based multimodal systems.
Supported by Figures 4, 9, 11, and 12

```sh
# From the repository root
source .venv/bin/activate
cd experiments
bash experiment-1.sh
```

**Expected runtime:** ~1 hour on 4×H100

### 8.2 Experiment 2 (Scale Sensitivity)

**Claim:**
RPS-Serve’s scheduling behavior scales favorably as GPU count and tensor-parallel
configuration (TP2/TP4) increase, in particular relative to disaggregation-based
baselines whose overhead dominates at small scale. Supported by Figure 10.

1 GPU:

```sh
# From the repository root
source .venv/bin/activate
cd experiments
bash experiment-2.sh --gpus 1
```

**Expected runtime:** ~1 hour on 1×H100

2 GPUs:

```sh
# From the repository root
source .venv/bin/activate
cd experiments
bash experiment-2.sh --gpus 2
```

**Expected runtime:** ~2 hours on 2×H100

4 GPUs:

```sh
# From the repository root
source .venv/bin/activate
cd experiments
bash experiment-2.sh --gpus 4
```

**Expected runtime:** ~2 hours on 4×H100

### 8.3 Experiment 3 (Cost-model and classifier characterization)

**Claim:**
The Impact Estimator and Request Classifier produce a meaningful, learnable cost
signal for multimodal requests (per-modality latency characterization and the 𝑘 =3
classification choice). Supported by Figure 13.

To perform the ablation study, run:

```sh
# From the repository root
source .venv/bin/activate
cd experiments
bash experiment-3.sh
```

**Expected runtime:** ~30 minutes on 1xH100


### 8.4 Experiment 4 (Workload characterization)

**Claim:**
The workload mix used in evaluation is representative of realistic multimodal
request distributions. Supported by Figures 2, 3, 6 and 7.

(Optional) To execute charactiraztion of LLaVA-7B, run:

```sh
# From the repository root
source .venv/bin/activate
cd experiments
bash experiment-4-char.sh
```

**Expected runtime:** ~24 hours on 1×H100

To train the impact estimator and the request classifier, run:

```sh
# From the repository root
source .venv/bin/activate
cd experiments
bash experiment-4-train.sh
```

**Expected runtime:** ~5 minutes on CPU

**Important Note:**

Characterization takes very long to execute as we execute requests sequentially
one after another. For this reason we provide our experimental results to use
when generating figures 2 and 3, and when training the classifier and predictor.

You can find all of them in:
- `artifacts/benchmark-log-iso.jsonl`
- `artifacts/outputs-iso`

---

## 9. Visualization

### 9.1 Plotting Environment

The plotting dependencies were already installed in step 3.3.
You can review them in [`requirements.txt`](requirements.txt).

### 9.2 Regenerating Figures from Raw Results

To generate the paper's figures, run:

```sh
cd plots
python3 characterization.py # Experiment 4
python3 evaluation.py # Experiment 4
python3 ablation.py # Experiment 3
python3 scale.py # Experiment 2
python3 e2e.py # Experiment 1
```

The resulting figures can be found under `artifacts/figures`.

| Script | Output | Paper Figure/Table |
|---|---|---|
| `characterization.py` | `mem_cdf.pdf` | Fig. 2a |
| `characterization.py` | `ttft_cdf.pdf` | Fig. 2b |
| `characterization.py` | `ttft_breakdown.pdf` | Fig. 3 |
| `e2e.py` | `gpu_util_vs_lat.pdf` | Fig. 4 |
| `evaluation.py` | `estimator_acc.pdf` | Fig. 6 |
| `evaluation.py` | `clustering.pdf` | Fig. 7 |
| `e2e.py` | `gpu_util_vs_lat_relative.pdf` | Fig. 9 |
| `scale.py` | `norm_lat_baselines.pdf` | Fig. 10a |
| `scale.py` | `norm_lat_baselines_large.pdf` | Fig. 10b |
| `e2e.py` | `tail_normlat_ttft.pdf` | Fig. 11 |
| `e2e.py` | `ttft_by_modality.pdf` | Fig. 12 |
| `ablation.py` | `ttft_by_modality_ablation.pdf` | Fig. 13a |
| `ablation.py` | `ttft_by_modality_ablation_prio.pdf` | Fig. 13b |



## Citation

```bibtex
@inproceedings{rps-serve,
author = {Papaioannou, Konstantinos and Doudali, Thaleia Dimitra},
title = {Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference},
year = {2026},
isbn = {},
publisher = {USENIX Association},
address = {USA},
booktitle = {Proceedings of the 2026 USENIX Conference on Usenix Annual Technical Conference},
location = {Hong Kong},
series = {USENIX ATC '26}
}
```

## Acknowledgements

The work was partially funded by the Madrid Regional Government through the César Nombela grant (2024-T1/COM-31302)
and by the Comunidad de Madrid through the DATIA project, co-funded by the European Union’s FEDER funds.
The work was also supported by grant PID2022-142290OB-I00, funded by MCIN/AEI/10.13039/501100011033 and FEDER, UE,
and by grant CEX2024-001471-M, funded by MICIU/AEI/10.13039/501100011033.

![Acknowledgements](acknowledgements.png)