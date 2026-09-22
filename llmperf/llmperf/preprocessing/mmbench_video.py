from llmperf.preprocessing.base import SingleDatasetPreprocessing

class MMBenchVideoPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        q = record["question"]
        return f"Question: {q}\nAnswer: "
    
    def get_output(self, record: dict) -> str:
        return record["answer"]
