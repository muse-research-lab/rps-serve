from llmperf.preprocessing.base import SingleDatasetPreprocessing

class ShareGPTPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][0]["value"]
    
    def get_output(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][1]["value"]
    
class ShareGPTLongPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][-1]["from"] == "gpt"):
            return None
        if  len(record["conversations"]) < 6:
            return None
        return '\n'.join([c["value"] for c in record["conversations"][:-1]])
    
    def get_output(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][-1]["from"] == "gpt"):
            return None
        if  len(record["conversations"]) < 6:
            return None
        return record["conversations"][-1]["value"]