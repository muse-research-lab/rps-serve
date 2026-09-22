from llmperf.promptpreparation.base import DefaultPromptPreparation

class COCOValDefaultPromptPreparation(DefaultPromptPreparation):
    def process_text(self, request):
        return request.input + (
            "Please describe this image in general. "
            "Directly provide the description, "
            "do not include prefix like \"This image depicts\"."
        )
