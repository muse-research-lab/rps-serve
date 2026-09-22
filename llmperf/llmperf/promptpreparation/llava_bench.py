from llmperf.promptpreparation.base import DefaultPromptPreparation

class LLaVABenchQnADefaultPromptPreparation(DefaultPromptPreparation):
    def process_text(self, request):
        extra = (
            "Please try to answer the question with short words or phrases "
            "if possible."
        )
        return f"{request.input}\n{extra}"
