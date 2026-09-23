import numpy as np
import random

from llmperf.preprocessing.workload import Workload

if __name__ == '__main__':
    """
    This script generates mixed workloads
    """
    SEED = 0
    NUM_REQUESTS = 3500

    # Modality mix (text, long-text, image, video) for each workload alias.
    WORKLOAD_MIXES = {
        "mixed-i": (0.6, 0.25, 0.1, 0.05),
        "mixed-ii": (0.5, 0.25, 0.15, 0.1),
        "mixed-iii": (0.4, 0.3, 0.2, 0.1),
        "mixed-iv": (0.35, 0.25, 0.25, 0.15),
        "mixed-v": (0.3, 0.2, 0.3, 0.2),
    }

    text_static = Workload(
        name="Text Conversations",
        path="./static",
        alias="text"
    )

    image_static = Workload(
        name="Image Reasoning",
        path="./static",
        alias="image"
    )

    video_static = Workload(
        name="Video Description",
        path="./static",
        alias="video"
    )

    long_text_static = Workload(
        name="Long Text Conversations",
        path="./static",
        alias="text-long"
    )

    text_static.load()
    text_static_requests = text_static.requests

    long_text_static.load()
    long_text_static_requests = long_text_static.requests

    image_static.load()
    image_static_requests = image_static.requests

    video_static.load()
    video_static_requests = video_static.requests

    REQUEST_POOL = {
        "text": text_static_requests,
        "long-text": long_text_static_requests,
        "image": image_static_requests,
        "video": video_static_requests
    }

    random.seed(SEED)
    np.random.seed(SEED)

    final_workloads = {}
    for workload_alias, (text_pct, long_text_pct, image_pct, video_pct) in WORKLOAD_MIXES.items():
        # Generate requests, giving any rounding remainder to video
        n_text = int(text_pct * NUM_REQUESTS)
        n_long_text = int(long_text_pct * NUM_REQUESTS)
        n_image = int(image_pct * NUM_REQUESTS)
        n_video = NUM_REQUESTS - n_text - n_long_text - n_image

        requests = random.choices(REQUEST_POOL["text"], k=n_text) + \
                    random.choices(REQUEST_POOL["long-text"], k=n_long_text) + \
                    random.choices(REQUEST_POOL["image"], k=n_image) + \
                    random.choices(REQUEST_POOL["video"], k=n_video)
        random.shuffle(requests)

        final_workload = Workload(
            name=workload_alias,
            path="./static",
            alias=workload_alias,
            requests=requests
        )
        final_workload.save()