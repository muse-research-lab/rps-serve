# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine import EngineCoreRequest


class PrefillTimeEstimator:
    # intercept, slope
    cache: dict[str, dict[str, list[float]]] = {
        # "llava-onevision-qwen2-7b-ov-chat-hf": { # A100 40GB
        #     "text": [0.00965129201572229, 7.114258735058774e-05],
        #     "image": [0.04782053127560487, 9.877444206212872e-05],
        #     "video": [0.04223119259146503, 0.0001246108308443217],
        # },
        "llava-onevision-qwen2-0.5b-ov-hf": {
            "text": [0.004604158101438752, 5.9372636711886714e-06],
            "image": [0.03001293012286582, 1.9404172214649895e-05],
            "video": [0.0469874803962724, 4.5308649933085995e-05],
        },
        "Qwen2.5-VL-3B-Instruct": {
            "text": [0.007208822979232461, 3.212644982267238e-05],
            "image": [0.050937172928859895, 5.503506783382055e-05],
            "video": [-0.07084554337453441, 0.0001986696026438678],
        },
        "gemma-3-4b-it": {
            "text": [0.008599597649873437, 3.7734497648640356e-05],
            "image": [0.09434673783835049, 6.027112249284893e-05],
            "video": [0.08565421916907176, 0.00022505744889195968],
        },
        "Qwen2.5-VL-7B-Instruct": {
            "text": [0.010600883528447555, 6.869224917719017e-05],
            "image": [0.03958721215825617, 0.0001511681623505576],
            "video": [-0.07786893755228164, 0.0002484057176874437],
        },
        "gemma-3-12b-it": {
            "text": [0.01990582858026893, 0.00011674479041670197],
            "image": [0.1545372152701027, 1.040198840200667e-05],
            "video": [0.11509276966127151, 0.000310104740159892],
        },
        "pixtral-12b": {
            "text": [0.027850550001887853, 0.0001341853661065232],
            "image": [0.06446421307749874, 0.0001438403242202692],
            "video": [-0.6454059303130606, 0.0002304761042188167],
        },
        "llava-onevision-qwen2-7b-ov-chat-hf": {
            "text": [0.010432596427509634, 2.821511489589937e-05],
            "image": [0.01909173512954794, 3.100683525894328e-05],
            "video": [0.004301185911683977, 4.364467756044445e-05],
        },
        "llava-onevision-qwen2-72b-ov-chat-hf": {
            "text": [0.13995626685042, 9.287370662521207e-05],
            "image": [0.0049559710544814315, 0.00018299030347626572],
            "video": [0.036683511410107145, 0.0001910774284488148],
        },
        "InternVL3_5-38B-HF": {
            "text": [0.07819333537157558, 4.5013811089605755e-05],
            "image": [0.03669317742310902, 0.0002018676056106019],
            "video": [0.02770202669398342, 0.00021005774304801722],
        },
        "Qwen3.5-27B": {
            "text": [0.03530299508746881, 6.027831731651396e-05],
            "image": [0.046704020247776305, 5.812866086336343e-05],
            "video": [0.07076368751773693, 0.00010668299209756563],
        },
        "gemma-4-31B-it": {
            "text": [0.0840129659883102, 5.063257790726964e-05],
            "image": [0.06028828101261766, 7.662999996682579e-05],
            "video": [-0.17555549495171396, 0.0006600616225413104],
        },
    }

    def __init__(self, model: str) -> None:
        model_name = model.split("/")[-1]

        self.intercept_text: float = 0.0
        self.slope_text: float = 0.0
        self.intercept_image: float = 0.0
        self.slope_image: float = 0.0
        self.intercept_video: float = 0.0
        self.slope_video: float = 0.0

        if model_name in self.cache:
            self.intercept_text = self.cache[model_name]["text"][0]
            self.slope_text = self.cache[model_name]["text"][1]
            self.intercept_image = self.cache[model_name]["image"][0]
            self.slope_image = self.cache[model_name]["image"][1]
            self.intercept_video = self.cache[model_name]["video"][0]
            self.slope_video = self.cache[model_name]["video"][1]

    def estimate(self, request: EngineCoreRequest) -> float:
        assert request.prompt_token_ids is not None, "prompt_token_ids must not be None"

        # Video inputs
        if (
            request.mm_features
            and request.mm_features[0] is not None
            and request.mm_features[0].modality == "video"
        ) or (
            request.mm_features
            and len(request.mm_features) > 1
            and request.mm_features[0] is not None
            and request.mm_features[0].modality == "image"
        ):
            return self.intercept_video + self.slope_video * len(
                request.prompt_token_ids
            )

        # Image inputs
        if (
            request.mm_features
            and len(request.mm_features) == 1
            and request.mm_features[0] is not None
            and request.mm_features[0].modality == "image"
        ):
            return self.intercept_image + self.slope_image * len(
                request.prompt_token_ids
            )

        # Text inputs
        return self.intercept_text + self.slope_text * len(request.prompt_token_ids)
