from llmperf.preprocessing.base import SingleDatasetPreprocessing

class MMBenchMultipleChoicePreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        q = record.get("question")
        h = record.get("hint")

        options = []
        for label in ["A", "B", "C", "D"]:
            value = record.get(label)
            if value:
                options.append(f"{label}. {value}")

        hint_line = f"Hint: {h}\n" if h else ""

        return f"{hint_line}Question: {q}\nOptions:\n" + "\n".join(options)

    def get_output(self, record: dict) -> str:
        return record["answer"]
