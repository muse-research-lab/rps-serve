from llmperf.preprocessing.base import SingleDatasetPreprocessing

class COCOValPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        return ""

    def get_output(self, record: dict) -> str:
        return record["answer"]
