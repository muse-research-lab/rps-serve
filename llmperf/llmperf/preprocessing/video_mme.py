import ast

from llmperf.preprocessing.base import SingleDatasetPreprocessing

class VideoMMEPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        q = record["question"]
        options = ast.literal_eval(record["candidates"])
        return f"Question: {q}\n" + "\n".join(options) + "\nAnswer: "
    
    def get_output(self, record: dict) -> str:
        return record["answer"]
