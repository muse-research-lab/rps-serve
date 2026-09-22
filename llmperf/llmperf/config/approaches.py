from dataclasses import dataclass
from typing import Literal, Union

@dataclass
class Approach:
    name: str
    alias: str
    scheduling_policy: str = "fcfs"
    enable_custom_scheduler: bool = False
    enable_chunked_prefill: bool = False

    def __hash__(self):
        return hash((self.name, self.alias))
    
    def __eq__(self, other):
        if isinstance(other, Approach):
            return self.name == other.name and self.alias == other.alias
        return False

APPROACHES = {
    Approach(
        name="Isolation",
        alias="iso"
    ),
    Approach(
        name="Vanilla vLLM",
        alias="vllm"
    ),
    Approach(
        name="FCFS per Category",
        alias="catfcfs",
        scheduling_policy="catfcfs"
    ),
    Approach(
        name="Earliest Deadline First",
        alias="edf",
        scheduling_policy="edf"
    ),
    Approach(
        name="All Terms",
        alias="full",
        scheduling_policy="full"
    ),
    Approach(
        name="Only Age Term",
        alias="age",
        scheduling_policy="age"
    ),
    Approach(
        name="Only Deadline Term",
        alias="dead",
        scheduling_policy="dead"
    ),
    Approach(
        name="Only Prefill Term",
        alias="pref",
        scheduling_policy="pref"
    ),
    Approach(
        name="Only Memory Term",
        alias="mem",
        scheduling_policy="mem"
    ),
    Approach(
        name="Deadline & Age Term",
        alias="dead-age",
        scheduling_policy="dead-age"
    ),
    Approach(
        name="Deadline & Prefill Term",
        alias="dead-pref",
        scheduling_policy="dead-pref"
    ),
    Approach(
        name="Deadline & Memory Term",
        alias="dead-mem",
        scheduling_policy="dead-mem"
    ),
    Approach(
        name="Deadline & Weight Term",
        alias="dead-w",
        scheduling_policy="dead-w"
    ),
    Approach(
        name="Deadline & Prefill & Memory Term",
        alias="dead-pref-mem",
        scheduling_policy="dead-pref-mem"
    ),
    Approach(
        name="Deadline & Weight & Memory Term",
        alias="dead-w-mem",
        scheduling_policy="dead-w-mem"
    ),
    Approach(
        name="Deadline & Weight & Prefill Term",
        alias="dead-w-pref",
        scheduling_policy="dead-w-pref"
    ),
    Approach(
        name="Deadline & Prefill & Memory & Age Term",
        alias="dead-pref-mem-age",
        scheduling_policy="dead-pref-mem-age"
    ),
    Approach(
        name="Age & Weight Term",
        alias="age-w",
        scheduling_policy="age-w"
    ),
    Approach(
        name="Age & Prefill Term",
        alias="age-pref",
        scheduling_policy="age-pref"
    ),
    Approach(
        name="Age & Memory Term",
        alias="age-mem",
        scheduling_policy="age-mem"
    ),
    Approach(
        name="Age & Prefill & Memory Term",
        alias="age-pref-mem",
        scheduling_policy="age-pref-mem"
    ),
    Approach(
        name="Age & Prefill & Memory & Weight Term",
        alias="age-pref-mem-w",
        scheduling_policy="age-pref-mem-w"
    ),
    Approach(
        name="Naive Aging (only aging & single queue)",
        alias="naive-aging",
        scheduling_policy="naive-aging"
    ),
}

def get_approach_by_name(name: str) -> Union[None, Approach]:
    for approach in APPROACHES:
        if getattr(approach, "name", None) == name:
            return approach
    return None

def get_approach_by_alias(alias: str) -> Union[None, Approach]:
    for approach in APPROACHES:
        if getattr(approach, "alias", None) == alias:
            return approach
    return None