from llmperf.promptpreparation.base import DefaultPromptPreparation

class VideoMMEDefaultPromptPreparation(DefaultPromptPreparation):
    def process_text(self, request):
        extra = (
            "These are the frames of a video. Select the best answer to the "
            "following multiple-choice question based on the video. Respond "
            "with only the letter (A, B, C, or D) of the correct option."
        )
        return f"{extra}\n{request.input}"
