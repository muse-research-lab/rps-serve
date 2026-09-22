from llmperf.promptpreparation.base import DefaultPromptPreparation

class MMBenchMultipleChoiceDefaultPromptPreparation(DefaultPromptPreparation):
    def process_text(self, request):
        extra = (
            "Please select the correct answer from the options above. Respond "
            "with only the letter (A, B, C, or D) of the correct option."
        )
        return f"{request.input}\n{extra}"
