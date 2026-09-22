import os

from typing import Union

from llmperf.constants import WORKLOADS_DIR

from llmperf.preprocessing.workload import Workload

_WORKLOADS_STATIC = {
    Workload(
        name="Text Conversations",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="text-static"
    ),
    Workload(
        name="Image Reasoning",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="image-static"
    ),
    Workload(
        name="Video Description",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="video-static"
    ),
    Workload(
        name="Audio Captioning",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="audio-static"
    ),
    Workload(
        name="Long Text Conversations",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="text-static-long"
    ),
}

_WORKLOADS_ACCURACY_BENCHMARK = {
    Workload(
        name="Multiple Choice (MMBench)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="mmbench-mc"
    ),
    Workload(
        name="Multiple Choice (MMBench)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="mmbench-mc-extended"
    ),
    Workload(
        name="Open Ended (LLaVA-Instruct)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="llava-instruct-oe"
    ),
    Workload(
        name="Description (LLaVA-Instruct)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="llava-instruct-desc"
    ),
    Workload(
        name="Q&A (LLaVABench)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="llavabench-qna"
    ),
    Workload(
        name="Captioning (COCO-Val)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="cocoval-captioning"
    ),
    Workload(
        name="Multiple Choice (Video-MME)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="videomme-mc"
    ),
    Workload(
        name="Q&A (MMBench-Video)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="mmbench-video-qna"
    ),
    Workload(
        name="Captioning (TempCompass)",
        path=os.path.join(WORKLOADS_DIR, "static"),
        alias="tempcompass-captioning"
    )
}

################################################################################
rates = [
    0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0
]
# Text Conversations with Poisson | Varying request rate
_WORKLOADS_TEXT_POISSON = {
    Workload(
        name=f"Text Conversations with Poisson {rate}",
        path=os.path.join(WORKLOADS_DIR, "text-poisson"),
        alias=f"text-poisson-{rate}"
    )
    for rate in rates
}

# Mixed Modalities with Poisson | Varying request rate | Top 15% replaced
_WORKLOADS_MIX_POISSON_15 = {
    Workload(
        name=f"Mixed Modalities with Poisson {rate} 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-15"),
        alias=f"mix-poisson-{rate}-15"
    )
    for rate in rates
}

# Mixed Modalities with Poisson | Varying request rate | Top 30% replaced
_WORKLOADS_MIX_POISSON_30 = {
    Workload(
        name=f"Mixed Modalities with Poisson {rate} 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-30"),
        alias=f"mix-poisson-{rate}-30"
    )
    for rate in rates
}

# Mixed Modalities with Poisson | Varying request rate | Top 45% replaced
_WORKLOADS_MIX_POISSON_45 = {
    Workload(
        name=f"Mixed Modalities with Poisson {rate} 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-poisson-45"),
        alias=f"mix-poisson-{rate}-45"
    )
    for rate in rates
}

# Text Conversations with Gamma | Varying request rate
_WORKLOADS_TEXT_GAMMA = {
    Workload(
        name=f"Text Conversations with Gamma {rate}",
        path=os.path.join(WORKLOADS_DIR, "text-gamma"),
        alias=f"text-gamma-{rate}"
    )
    for rate in rates
}

# Mixed Modalities with Gamma | Varying request rate | Top 15% replaced
_WORKLOADS_MIX_GAMMA_15 = {
    Workload(
        name=f"Mixed Modalities with Gamma {rate} 15%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-15"),
        alias=f"mix-gamma-{rate}-15"
    )
    for rate in rates
}

# Mixed Modalities with Gamma | Varying request rate | Top 30% replaced
_WORKLOADS_MIX_GAMMA_30 = {
    Workload(
        name=f"Mixed Modalities with Gamma {rate} 30%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-30"),
        alias=f"mix-gamma-{rate}-30"
    )
    for rate in rates
}

# Mixed Modalities with Gamma | Varying request rate | Top 45% replaced
_WORKLOADS_MIX_GAMMA_45 = {
        Workload(
        name=f"Mixed Modalities with Gamma {rate} 45%",
        path=os.path.join(WORKLOADS_DIR, "mix-gamma-45"),
        alias=f"mix-gamma-{rate}-45"
    )
    for rate in rates
}

################################################################################
rps_rates = [
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0
]
# Rocks Pebbles Sand with Poisson | Varying request rate | 70% - 30% - 0%
_WORKLOADS_RPS_POISSON_70_30_0 = {
    Workload(
        name=f"Rock - Pebbles - Sand with Poisson {rate} 70%-30%-0%",
        path=os.path.join(WORKLOADS_DIR, "rps-poisson-70-30-0"),
        alias=f"rps-poisson-{rate}-70-30-0"
    )
    for rate in rps_rates
}

# Rocks Pebbles Sand with Poisson | Varying request rate | 60% - 30% - 10%
_WORKLOADS_RPS_POISSON_60_30_10 = {
    Workload(
        name=f"Rock - Pebbles - Sand with Poisson {rate} 60%-30%-10%",
        path=os.path.join(WORKLOADS_DIR, "rps-poisson-60-30-10"),
        alias=f"rps-poisson-{rate}-60-30-10"
    )
    for rate in rps_rates
}

# Rocks Pebbles Sand with Poisson | Varying request rate | 45% - 35% - 20%
_WORKLOADS_RPS_POISSON_45_35_20 = {
    Workload(
        name=f"Rock - Pebbles - Sand with Poisson {rate} 45%-35%-20%",
        path=os.path.join(WORKLOADS_DIR, "rps-poisson-45-35-20"),
        alias=f"rps-poisson-{rate}-45-35-20"
    )
    for rate in rps_rates
}

################################################################################
_WORKLOADS_SMALL_BENCHMARK = {
    Workload(
        name="Text Only 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="text-only-1.0"
    ),
    Workload(
        name="Text Only 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="text-only-1.5"
    ),
    Workload(
        name="Text Only 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="text-only-2.0"
    ),
    Workload(
        name="Long Text Light 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="long-text-light-1.0"
    ),
    Workload(
        name="Long Text Light 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="long-text-light-1.5"
    ),
    Workload(
        name="Long Text Light 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="long-text-light-2.0"
    ),
    Workload(
        name="Image Light 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="image-light-1.0"
    ),
    Workload(
        name="Image Light 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="image-light-1.5"
    ),
    Workload(
        name="Image Light 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="image-light-2.0"
    ),
    Workload(
        name="Video Light 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="video-light-1.0"
    ),
    Workload(
        name="Video Light 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="video-light-1.5"
    ),
    Workload(
        name="Video Light 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="video-light-2.0"
    ),
    Workload(
        name="Long Text Heavy 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="long-text-heavy-1.0"
    ),
    Workload(
        name="Long Text Heavy 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="long-text-heavy-1.5"
    ),
    Workload(
        name="Long Text Heavy 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="long-text-heavy-2.0"
    ),
    Workload(
        name="Image Heavy 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="image-heavy-1.0"
    ),
    Workload(
        name="Image Heavy 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="image-heavy-1.5"
    ),
    Workload(
        name="Image Heavy 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="image-heavy-2.0"
    ),
    Workload(
        name="Video Heavy 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="video-heavy-1.0"
    ),
    Workload(
        name="Video Heavy 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="video-heavy-1.5"
    ),
    Workload(
        name="Video Heavy 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="video-heavy-2.0"
    ),
    Workload(
        name="Mixed Light 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-light-1.0"
    ),
    Workload(
        name="Mixed Light 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-light-1.5"
    ),
    Workload(
        name="Mixed Light 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-light-2.0"
    ),
    Workload(
        name="Mixed Heavy 0.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-0.5"
    ),
    Workload(
        name="Mixed Heavy 0.75",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-0.75"
    ),
    Workload(
        name="Mixed Heavy 1.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-1.0"
    ),
    Workload(
        name="Mixed Heavy 1.25",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-1.25"
    ),
    Workload(
        name="Mixed Heavy 1.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-1.5"
    ),
    Workload(
        name="Mixed Heavy 1.75",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-1.75"
    ),
    Workload(
        name="Mixed Heavy 2.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-2.0"
    ),
    Workload(
        name="Mixed Heavy 2.1",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-2.1"
    ),
    Workload(
        name="Mixed Heavy 2.2",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-2.2"
    ),
    Workload(
        name="Mixed Heavy 2.25",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-2.25"
    ),
    Workload(
        name="Mixed Heavy 2.3",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-2.3"
    ),
    Workload(
        name="Mixed Heavy 2.4",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-2.4"
    ),
    Workload(
        name="Mixed Heavy 2.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-2.5"
    ),
    Workload(
        name="Mixed Heavy 3.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-3.0"
    ),
    Workload(
        name="Mixed Heavy 3.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-3.5"
    ),
    Workload(
        name="Mixed Heavy 3.75",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-3.75"
    ),
    Workload(
        name="Mixed Heavy 4.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-4.0"
    ),
    Workload(
        name="Mixed Heavy 4.5",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-4.5"
    ),
    Workload(
        name="Mixed Heavy 5.0",
        path=os.path.join(WORKLOADS_DIR, "small-benchmark"),
        alias="mixed-heavy-5.0"
    ),
}

_WORKLOADS_RPS_SERVE = {
    Workload(
        name="Text Only",
        path=os.path.join(WORKLOADS_DIR, "rps-serve"),
        alias="text-only"
    ),
    Workload(
        name="Mixed Light",
        path=os.path.join(WORKLOADS_DIR, "rps-serve"),
        alias="mixed-light"
    ),
    Workload(
            name="Mixed Light II",
            path=os.path.join(WORKLOADS_DIR, "rps-serve"),
            alias="mixed-light-ii"
        ),
    Workload(
        name="Mixed Heavy",
        path=os.path.join(WORKLOADS_DIR, "rps-serve"),
        alias="mixed-heavy"
    ),
    Workload(
        name="Mixed Heavy II",
        path=os.path.join(WORKLOADS_DIR, "rps-serve"),
        alias="mixed-heavy-ii"
    ),
    Workload(
        name="Mixed Heavy III",
        path=os.path.join(WORKLOADS_DIR, "rps-serve"),
        alias="mixed-heavy-iii"
    ),
    Workload(
        name="Image Only",
        path=os.path.join(WORKLOADS_DIR, "rps-serve"),
        alias="image-only"
    ),
    Workload(
        name="Video Only",
        path=os.path.join(WORKLOADS_DIR, "rps-serve"),
        alias="video-only"
    ),
}

################################################################################
WORKLOADS = _WORKLOADS_STATIC | \
    _WORKLOADS_TEXT_POISSON | \
    _WORKLOADS_MIX_POISSON_15 | \
    _WORKLOADS_MIX_POISSON_30 | \
    _WORKLOADS_MIX_POISSON_45 | \
    _WORKLOADS_TEXT_GAMMA | \
    _WORKLOADS_MIX_GAMMA_15 | \
    _WORKLOADS_MIX_GAMMA_30 | \
    _WORKLOADS_MIX_GAMMA_45 | \
    _WORKLOADS_RPS_POISSON_70_30_0 | \
    _WORKLOADS_RPS_POISSON_60_30_10 | \
    _WORKLOADS_RPS_POISSON_45_35_20 | \
    _WORKLOADS_SMALL_BENCHMARK | \
    _WORKLOADS_ACCURACY_BENCHMARK | \
    _WORKLOADS_RPS_SERVE

def get_workload_by_name(name: str) -> Union[None, Workload]:
    for workload in WORKLOADS:
        if getattr(workload, "name", None) == name:
            return workload
    return None

def get_workload_by_alias(alias: str) -> Union[None, Workload]:
    for workload in WORKLOADS:
        if getattr(workload, "alias", None) == alias:
            return workload
    return None