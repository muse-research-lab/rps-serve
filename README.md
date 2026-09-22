# RPS-Serve Artifact

Artifact for **"Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference"**, accepted at ATC '26.

- **Paper DOI:** TODO
- **Artifact DOI (Zenodo):** TODO 
- **License:** Apache License 2.0

---

## 1. Artifact Overview

RPS-Serve is a modality-aware request scheduling framework for multimodal LLM (MLLM) serving, built on top of vLLM (v0.19.2). This artifact contains:

- `rps-serve/`: Modified version of vLLM containing the following components:
    - Impact Estimator: `vllm/v1/engine/prefill_time_estimator.py`
    - Request Classifier: `vllm/v1/engine/request_classifier.py`
    - Queue Manager: `vllm/v1/core/sched/queue_manager.py`
    - Priority Regulator: `vllm/v1/core/sched/slo_expiration_manager.py`

  See changes here [`rps-serve-changes.patch`](rps-serve-changes.patch): 

- `vllm/`: Patched vLLM used for distributed serving. See changes here: [`vllm-changes.patch`](vllm-changes.patch)

- `experiments/`: Scripts for running the paper's experiments and producing summary metrics.

- `workloads/`: Scripts for generating the workloads

- `plots/`: Scripts for reproducing the paper's figures from raw experiment outputs

- `artifacts/`: Experiment outputs and figures

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
- `uv >= 0.10.11`
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
export VLLM_USE_PRECOMPILED=0
uv pip install --editable . --torch-backend=cu130
```

### 3.2 Baselines

Install the patched vLLM version in a separate virtual environment. This
environment is used to deploy the baseline systems: vLLM, ModServe, and
ModServe with RPS.

```sh
cd vllm
uv venv --python 3.12.8 --seed --managed-python
source .venv/bin/activate
export VLLM_USE_PRECOMPILED=0
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