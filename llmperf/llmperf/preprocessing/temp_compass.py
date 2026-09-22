from llmperf.preprocessing.base import SingleDatasetPreprocessing

class TempCompassCaptioningPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        return record["question"]
    
    def get_output(self, record: dict) -> str:
        return record["answer"]
