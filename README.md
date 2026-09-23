# RPS-Serve Artifact

Artifact for **"Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference"**, accepted at ATC '26.

- **Paper DOI:** TODO
- **Artifact DOI (Zenodo):** TODO 
- **License:** Apache License 2.0

---

## 1. Artifact Overview

RPS-Serve is a modality-aware request scheduling framework for multimodal LLM (MLLM) serving, built on top of vLLM (v0.19.2). This artifact contains:

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
- Additional Python packages: [`requirements.txt`](requirements.txt)

---

## 3. Installation

### 3.1 RPS-Serve

Install RPS-Serve in a dedicated virtual environment. This environment is used
to deploy RPS-Serve instances.

```sh
cd rps-serve
uv venv --python 3.12.8 --seed --managed-python
source .venv/bin/activate
export VLLM_USE_PRECOMPILED=1
uv pip install --editable . --torch-backend=cu130
uv pip install joblib
```

### 3.2 Baselines

Install the patched vLLM version in a separate virtual environment. This
environment is used to deploy the baseline systems: vLLM, ModServe, and
ModServe with RPS.

```sh
cd vllm
uv venv --python 3.12.8 --seed --managed-python
source .venv/bin/activate
export VLLM_USE_PRECOMPILED=1
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
cd ../guidellm
uv pip install -e .
```

## 4. Workload Generation

(Optional) To ingest the datasets, run:

```sh
cd workloads
python3 ingestion.py --text --image --video
```

To avoid the time consuming ingestion of image and video data, we provide the following:

```sh
curl -L "https://cloud.software.imdea.org/index.php/s/fw9DJZ8tkLY9RB6/download" | tar -xf - -C .
```

To generate workloads for each modality, run:

```sh
python3 preprocessing.py --text --image --video
```

To generate the mixed workloads used in the paper, run:

```sh
python3 generation.py
```

### 5. Download Models

To download LLaVA-7B (~30GB), run:

```sh
cd models
git lfs install
git clone https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-chat-hf
```

---

## 6. Minimal Working Example

```sh
source .venv/bin/activate
cd experiments
python3 orchestrator.py --config config-rps-minimal.yaml
```
The orchestrator.py script:
1. Launches RPS-Serve on 1 GPU (LLaVA-7B)
2. Replays a tiny mixed trace (~100 requests)
3. Produces a metrics summary

**Expected output:** Two files are created, one under artifacts/outputs and another one under artifacts/monitor-stats.
```sh
...
Avg. E2E Latency: 4.484160141944885
Avg. TTFT Latency: 0.4903690218925476
Avg. TBT Latency: 0.019876031017663578
...
```
**Expected runtime:** ~5 minutes on 1×H100

---


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