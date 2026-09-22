import ast
import evaluate
import re
from typing import Callable, Dict, List

from llmperf.config.workloads import get_workload_by_alias
from llmperf.postprocessing.output import ExperimentOutput, RequestOutput
from llmperf.config.workloads import Request


def score_acc_mc(requests: List[Request], request_outputs: List[RequestOutput]) -> float:
    correct_answers = 0
    for r, ro in zip(requests, request_outputs):
        gt = r.output # "A"
        answer = ro.outputs[0].extract(r'\b([A-D])\b', flags=re.IGNORECASE)

        if gt == answer:
            correct_answers += 1

    return correct_answers / len(requests)

def score_rougel_qna(requests: List[Request], request_outputs: List[RequestOutput]) -> float:
    rouge = evaluate.load("rouge")
    refs = [r.output for r in requests] # "A creative painting of a dog dressed as the famous Mona Lisa." or # "It's called \"We Are More\"."
    preds = [ro.outputs[0] for ro in request_outputs]
    results = rouge.compute(predictions=preds, references=refs)
    return results["rougeL"]

def score_rougel_cocoval_captioning(requests: List[Request], request_outputs: List[RequestOutput]) -> float:
    rouge = evaluate.load("rouge")
    refs = [ast.literal_eval(r.output) for r in requests] # "['A row of parked cars sitting next to parking meters.', 'A row of cars parked on a street with parking meters.', 'A series of parking meters and cars are located next to each other. ', 'A parking meter on a street by a car with traffic.', 'A parking meter on a street with cars']"
    preds = [ro.outputs[0] for ro in request_outputs]
    results = rouge.compute(predictions=preds, references=refs)
    return results["rougeL"]

def score_rougel_tempcompass_captioning(requests: List[Request], request_outputs: List[RequestOutput]) -> float:
    rouge = evaluate.load("rouge")
    refs = [r.output for r in requests] # "A. solidifying"
    preds = [ro.outputs[0] for ro in request_outputs]
    results = rouge.compute(predictions=preds, references=refs)
    return results["rougeL"]

_METRIC_FN_MAP: Dict[str, Dict[str, Callable[[List[Request], List[RequestOutput]], float]]] = {
    "accuracy": {
        "mmbench-mc": score_acc_mc,
        "videomme-mc": score_acc_mc,
    },
    "rouge-l": {
        "llavabench-qna": score_rougel_qna,
        "mmbench-video-qna": score_rougel_qna,
        "cocoval-captioning": score_rougel_cocoval_captioning,
        "tempcompass-captioning": score_rougel_tempcompass_captioning,
    }
}

def get_metric_fn(metric: str, workload_alias: str) -> Callable[[List[Request], List[RequestOutput]], float]:
    try:
        return _METRIC_FN_MAP[metric][workload_alias]
    except KeyError:
        raise ValueError(f"Metric '{metric}' is not valid for workload '{workload_alias}'")

class Evaluator:
    def __init__(self, workload_alias: str, eo_id: str):
        self.workload = get_workload_by_alias(workload_alias)
        self.workload.load()

        self.eo = ExperimentOutput(id=eo_id)
        self.eo.load()

        assert len(self.workload.requests) == len(self.eo.request_outputs)

    def evaluate(self, metrics: List[str]) -> Dict[str, float]:
        scores = {}
        for metric in metrics:
            metric_fn = get_metric_fn(metric, self.workload.alias)
            score = metric_fn(
                self.workload.requests,
                self.eo.request_outputs
            )
            scores[metric] = score

        return scores