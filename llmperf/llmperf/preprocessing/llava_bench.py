from llmperf.preprocessing.base import SingleDatasetPreprocessing

class LLaVABenchQnAPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        q = record["question"]
        return q

    def get_output(self, record: dict) -> str:
        return record["caption"]
