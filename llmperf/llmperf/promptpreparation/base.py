from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from llmperf.promptpreparation.config import PromptPreparationConfig
from llmperf.preprocessing.workload import Request

from llmperf.promptpreparation.audio import AudioAsset
from llmperf.promptpreparation.image import ImageAsset
from llmperf.promptpreparation.video import VideoAsset

Asset = Union[ImageAsset, VideoAsset, AudioAsset]

@dataclass(slots=True)
class BasePromptPreparation(ABC):
    """
    Base class for all prompt preparation.
    Subclasses implement the preparation-specific steps.
    """
    config: "PromptPreparationConfig"

    image_codecs = {"JPEG", "PNG"}
    video_codecs = {"h264", "vp6f", "vp9", "mp4"}
    audio_codecs = {"pcm_s16le"}

    def get_parameters(self, request: Request) -> dict:
        max_tokens = self.config.max_tokens
        if self.config.dynamic_max_tokens:
            max_tokens = len(self.config.tokenizer.encode(request.output))
        return {
            "ignore_eos": self.config.ignore_eos,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature
        }

    def get_user_md(
        self, uid: str, slo: float, ttft_slo: float, tbt_slo: float
    ) -> dict:
        return {
            "uid": uid,
            "slo": slo * self.config.e2e_slo_scale,
            "ttft_slo": ttft_slo * self.config.ttft_slo_scale,
            "tbt_slo": tbt_slo  * self.config.tbt_slo_scale
        }
    
    def get_final_prompt(self, request: Request) -> dict:
        user_prompt = self.process_text(request)

        modality_data = self.process_modality(request)
        if modality_data:
            modality_data = self.compress_modality(modality_data)

        return self.construct_final_prompt(
            user_prompt,
            modality_data
        )
    
    def process_text(self, request: Request) -> str:
        return request.input
    
    @staticmethod
    def duration_to_frames(
            duration: int, min_duration: int, max_duration: int,
            min_frames: int, max_frames: int
        ):
        duration = max(min_duration, min(max_duration, duration))
        scale = (duration - min_duration) / (max_duration - min_duration)
        frames = min_frames + scale * (max_frames - min_frames)
        return int(frames)

    def process_modality(self, request: Request) -> Union[Asset, None]:
        if not (request.modality_path and "codec" in request.modality_size):
            return None
        if self.is_image(request):
            return ImageAsset(request.modality_path)
        
        if self.is_video(request):
            total_frames = request.modality_size.get("frame_count", -1)
            dynamic_frame_count = None
            if self.config.dynamic_frames:
                duration = request.modality_size.get("duration", -1)
                dynamic_frame_count = self.duration_to_frames(
                    duration, self.config.min_duration,
                    self.config.max_duration, self.config.min_sampled_frames,
                    self.config.max_sampled_frames
                )
            max_sampled_frames = dynamic_frame_count or \
                self.config.max_sampled_frames
            return VideoAsset(
                path=request.modality_path,
                max_sampled_frames=max_sampled_frames,
                sampling_strategy=self.config.sampling_strategy,
                strategy_params=self.config.strategy_params,
                total_frames=total_frames
            )
        
        if self.is_audio(request):
            return AudioAsset(request.modality_path)
        
        raise ValueError(
            f"Unsupported modality: {request.modality_path}"
        )
    
    def compress_modality(self, modality_data: Asset) -> Union[Asset, None]:
        if isinstance(modality_data, ImageAsset):
            return self.compress_image(modality_data)
        
        if isinstance(modality_data, VideoAsset):
            return self.compress_video(modality_data)
        
        if isinstance(modality_data, AudioAsset):
            return self.compress_audio(modality_data)
        
        return None
        
    @abstractmethod
    def compress_image(self, image: ImageAsset) -> ImageAsset:
        raise NotImplementedError
    
    @abstractmethod
    def compress_video(self, video: VideoAsset) -> VideoAsset:
        raise NotImplementedError
    
    @abstractmethod
    def compress_audio(self, audio: AudioAsset) -> AudioAsset:
        raise NotImplementedError
    
    @abstractmethod
    def construct_final_prompt(
        self, user_prompt: str, modality_data: Union[Asset, None]
    ) -> dict:
        raise NotImplementedError
    
    @classmethod
    def is_video(cls, request: Request) -> bool:
        return request.modality_size.get("codec", None) in cls.video_codecs
    
    @classmethod
    def is_image(cls, request: Request) -> bool:
        return request.modality_size.get("codec", None) in cls.image_codecs
    
    @classmethod
    def is_audio(cls, request: Request) -> bool:
        return request.modality_size.get("codec", None) in cls.audio_codecs

class DefaultPromptPreparation(BasePromptPreparation):
    def build_messages(self, content):
        system_prompt = self.config.system_prompt
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": [
                {'type': 'text', 'text': system_prompt}
            ]})
        msgs.append({"role": "user", "content": content})
        return msgs

    def image_content(self, user_prompt, modality_data):
        return [{"type": "image"}, {"type": "text", "text": user_prompt}]

    def video_content(self, user_prompt, modality_data: VideoAsset):
        if self.config.multi_image:
            num_frames = modality_data.size[0]
            return [{"type": "image"}] * num_frames + \
                [{"type": "text", "text": user_prompt}]
        return [{"type": "video"}, {"type": "text", "text": user_prompt}]

    def audio_content(self, user_prompt, modality_data):
        prefix = "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
        return [
            {"type": "audio"},
            {"type": "text", "text": prefix + user_prompt},
        ]

    @property
    def content_builders(self):
        return {
            ImageAsset: self.image_content,
            VideoAsset: self.video_content,
            AudioAsset: self.audio_content,
        }

    def fallback_prompt(self, user_prompt, modality_data):
        system_prompt = self.config.system_prompt
        sys = f"<|System|>: {system_prompt}\n\n" if system_prompt else ""

        if isinstance(modality_data, ImageAsset):
            return f"{sys}<|User|>: <image>\n{user_prompt}\n\n<|Assistant|>:"

        if isinstance(modality_data, VideoAsset):
            if self.config.multi_image:
                num_frames = modality_data.size[0]
                frames = "<image>\n" * num_frames
                return f"{sys}<|User|>: {frames}{user_prompt}\n\n<|Assistant|>:"
            return f"{sys}<|User|>: <video>\n{user_prompt}\n\n<|Assistant|>:"

        if isinstance(modality_data, AudioAsset):
            raise NotImplementedError
        
    def extract_mm_data(self, modality_data):
        if isinstance(modality_data, ImageAsset):
            return {"image": modality_data.data}
        
        if isinstance(modality_data, VideoAsset):
            if self.config.multi_image:
                return {"image": modality_data.frame_list}
            return {"video": modality_data.data}
        
        if isinstance(modality_data, AudioAsset):
            return {"audio": modality_data.data}
        
        raise ValueError(
            f"Unsupported modality: {type(modality_data)}"
        )
            
    def compress_image(self, image: ImageAsset) -> ImageAsset:
        compression_ratio = self.config.compression_ratio
        if compression_ratio:
            w, h = image.size
            if self.config.model.alias.startswith("qwen-2"):
                n_w = max(28, int(w - (compression_ratio * w)))
                n_h = max(28, int(h - (compression_ratio * h)))
            else:
                n_w = int(w - (compression_ratio * w))
                n_h = int(h - (compression_ratio * h))
            
            image.compress_image(n_w, n_h)
        
        return image
    
    def compress_video(self, video: VideoAsset) -> VideoAsset:
        target_frame_count = self.config.target_frame_count or \
            self.config.max_sampled_frames
        frame_reduction_factor = self.config.frame_reduction_factor
        n_w = n_h = None
        if frame_reduction_factor:
            w, h = video.size[2], video.size[1]
            if self.config.model.alias.startswith("qwen-2"):
                n_w = max(28, int(w - (frame_reduction_factor * w)))
                n_h = max(28, int(h - (frame_reduction_factor * h)))
            else:
                n_w = int(w - (frame_reduction_factor * w))
                n_h = int(h - (frame_reduction_factor * h))

        if target_frame_count != self.config.max_sampled_frames and \
            frame_reduction_factor:
            video.compress_video(target_frame_count, n_w, n_h)

        return video
    
    def compress_audio(self, audio: AudioAsset) -> AudioAsset:
        return audio
    
    def construct_final_prompt(
        self, user_prompt: str, modality_data: Union[Asset, None]
    ) -> dict:
        system_prompt = self.config.system_prompt
        tokenizer = self.config.tokenizer
        processor = self.config.processor
        
        template_handler = tokenizer if modality_data is None else processor
        chat_template = getattr(template_handler, "chat_template", None)

        if modality_data is None:
            if chat_template:
                prompt = self.build_messages(user_prompt)
                formatted_prompt = template_handler.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False
                )
            else:
                formatted_prompt = (
                    f"<|System|>: {system_prompt}\n\n" if system_prompt else "",
                    + f"<|User|>: {user_prompt}\n\n<|Assistant|>:"
                )
            final_prompt = { "prompt": formatted_prompt }
        else:
            if chat_template:
                builder = self.content_builders.get(type(modality_data))
                if builder is None:
                    raise ValueError(
                        f"Unsupported modality: {type(modality_data)}"
                    )

                prompt = self.build_messages(
                    builder(user_prompt, modality_data)
                )
                formatted_prompt = template_handler.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False
                )
            else:
                formatted_prompt = self.fallback_prompt(
                    user_prompt, modality_data
                )
        
            mm_data = self.extract_mm_data(modality_data)
        
            final_prompt = {
                "prompt": formatted_prompt,
                "multi_modal_data": mm_data
            }
        return final_prompt

class DefaultExtraPromptPreparation(DefaultPromptPreparation):
    def extract_mm_data(self, modality_data):
        if isinstance(modality_data, ImageAsset):
            return {"image": modality_data.data}
        
        if isinstance(modality_data, VideoAsset):
            if self.config.multi_image:
                return {"image": modality_data.frame_list}
            
            if self.config.model.alias in ["gemma-4-large", "qwen-3.5-large"]:
                return {
                    "video": (modality_data.data, modality_data.metadata)
                }
            return {"video": modality_data.data}
        
        if isinstance(modality_data, AudioAsset):
            return {"audio": modality_data.data}
        
        raise ValueError(
            f"Unsupported modality: {type(modality_data)}"
        )