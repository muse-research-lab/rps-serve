from llmperf.preprocessing.base import SingleDatasetPreprocessing

class LLaVAInstructComplexReasoningPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][0]["value"] \
            .replace("<image>\n", "").replace("\n<image>", "")

    
    def get_output(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][1]["value"]
