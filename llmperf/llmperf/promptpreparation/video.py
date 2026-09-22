from dataclasses import dataclass, field
from typing import Any, LiteralString, Set, Union

import cv2
import os
import numpy as np
import numpy.typing as npt
from PIL import Image

from scenedetect import SceneManager, VideoCaptureAdapter
from scenedetect.scene_manager import compute_downscale_factor
from scenedetect.detectors import ContentDetector

from llmperf.promptpreparation.config import SamplingMethod

@dataclass
class VideoAsset:
    path: Union[str,LiteralString]
    video: npt.NDArray = field(init=False, repr=False)
    use_cache: bool = False
    cache_path: Union[str,LiteralString] = "/srv/muse-lab/cache/"

    max_sampled_frames: int = -1
    sampling_strategy: SamplingMethod = "uniform"
    strategy_params: dict[str, float] = field(default_factory=dict)
    total_frames: int = -1

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.sampling_strategy in ["uniform", "all"]:
            self.video = self.video_to_ndarrays() 
        else:
            self.video = self.video_to_ndarrays_custom()

    @property
    def size(self) -> tuple[int, int, int, int]:
        return self.video.shape
    
    @property
    def data(self) -> npt.NDArray:
        return self.video
    
    @property
    def frame_list(self) -> list[npt.NDArray]:
        return [frame for frame in self.video]
    
    @property
    def image_list(self) -> list[Image.Image]:
        return [Image.fromarray(frame) for frame in self.video]

    def video_get_metadata(path: str, num_frames: int = -1) -> dict[str, Any]:

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        if num_frames == -1 or num_frames > total_frames:
            num_frames = total_frames

        metadata = {
            "total_num_frames": num_frames,
            "fps": duration / num_frames,
            "duration": duration,
            "video_backend": "opencv",
            "frames_indices": list(range(num_frames)),
            # extra field used to control hf processor's video
            # sampling behavior
            "do_sample_frames": num_frames == total_frames,
        }
        return metadata
    
    def compress_video(
            self,
            target_frame_count: int | None = None,
            width: int | None = None,
            height: int | None = None
        ):
        
        if target_frame_count and target_frame_count < self.size[0]:
            indices = np.linspace(
                0, self.size[0] - 1, target_frame_count
            ).astype(int)
            self.video = self.video[indices]
        
        target_size = (width or self.size[2], height or self.size[1])
        if target_size != (self.size[2], self.size[1]):
            self.video = np.array([
                cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4) 
                for frame in self.video
            ])
    
    def build_cached_path(self) -> str:
        rel_path = self.path.split("videos" + os.sep)[-1]
        rel_dir = os.path.dirname(rel_path)
        filename = os.path.basename(rel_path)
        filename_no_ext = os.path.splitext(filename)[0]

        cache_dir = os.path.join(self.cache_path, rel_dir)

        cached_filename = (
            f"{filename_no_ext}_max_frames_{self.max_sampled_frames}_"
            f"{self.sampling_strategy}"
        )
        for i in self.strategy_params:
            num = str(self.strategy_params[i]).replace(".", "_")
            cached_filename += f"_{i}_{num}"
        cached_filename += ".npy"

        return os.path.join(cache_dir, cached_filename)

    def video_to_ndarrays(self) -> npt.NDArray:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {self.path}")

        if self.total_frames == -1:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = self.total_frames / fps if fps > 0 else 0

        frames = []

        n = self.max_sampled_frames
        num_frames = n if n and n > 0 else self.total_frames
        frame_indices = np.linspace(
            0, self.total_frames - 1, num_frames, dtype=int
        )
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                # OpenCV uses BGR format, we need to convert it to RGB
                # for PIL and transformers compatibility
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # Fallback for corrupted frames or EOF
                continue
        cap.release()

        if len(frames) < num_frames:
            print(
                f"Could not read enough frames from video file {self.path}"
                f" (expected {num_frames} frames, got {len(frames)})"
            )

        self.metadata = {
            "total_num_frames": num_frames,
            "fps": duration / num_frames,
            "duration": duration,
            "video_backend": "opencv",
            "frames_indices": list(range(num_frames)),
            "do_sample_frames": num_frames == self.total_frames,
        }
        
        return np.stack(frames)
    
    def sample_frames_by_motion(self) -> npt.NDArray:
        motion_threshold = self.strategy_params.get("motion_threshold", 1.0)
                
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {self.path}")
        
        if self.total_frames == -1:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        factor = compute_downscale_factor(max(width, height))
        new_w, new_h = int(width / factor), int(height / factor)

        success, first_frame = cap.read()
        if not success:
            cap.release()
            raise ValueError(f"Could not read first frame from {self.path}.")
        
        prev_frame_gray = cv2.resize(
            cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), (new_w, new_h)
        )
        prev_pts = cv2.goodFeaturesToTrack(
            prev_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=10
        )

        selected_idx = [(0, 0)] # Always include the first frame
        current_idx = 1
        while True:
            success, next_frame = cap.read()
            if not success: break
            
            next_frame_gray = cv2.resize(
                cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY), (new_w, new_h)
            )
            
            # Calculate Sparse Optical Flow
            if prev_pts is not None:
                prev_pts = np.array(prev_pts, dtype=np.float32) \
                    .reshape(-1, 1, 2)
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_frame_gray, next_frame_gray, prev_pts, None
                )
            
                # Calculate motion only for successfully tracked points
                if next_pts is not None and status.any():
                    good_new = next_pts[status == 1]
                    good_old = prev_pts[status == 1]
                    # Distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
                    motion = np.sqrt(np.sum((good_new - good_old)**2, axis=1))
                    if np.mean(motion) > motion_threshold:
                        selected_idx.append((current_idx, np.mean(motion)))

            # Refresh points periodically or if tracking fails
            if current_idx % 30 == 0 or len(good_new) < 50:
                prev_pts = cv2.goodFeaturesToTrack(
                    next_frame_gray, maxCorners=200,
                    qualityLevel=0.01, minDistance=10
                )
            else:
                prev_pts = good_new.reshape(-1, 1, 2)

            prev_frame_gray = next_frame_gray
            current_idx += 1

        # Fallback and Thinning
        num_frames = self.max_sampled_frames
        if not selected_idx or len(selected_idx) == 0:
            selected_idx = np.linspace(
                0, self.total_frames - 1, num_frames, dtype=int
            )
        
        if len(selected_idx) > num_frames:
            selected_idx.sort(key=lambda x: x[1], reverse=True)
            selected_idx = [f[0] for f in selected_idx[:num_frames]]

        # Frame Extraction
        frames = []
        current_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        
        target_indices = sorted(list(selected_idx))
        for target in target_indices:
            while current_idx < target:
                cap.grab()
                current_idx += 1
            
            success, frame = cap.read()
            if success:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                current_idx += 1
            else:
                break

        cap.release()

        return np.stack(frames)

    def sample_frames_by_sharpness(self) -> npt.NDArray:
        sharpness_threshold = self.strategy_params \
            .get("sharpness_threshold", 100.0)

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {self.path}")
        
        if self.total_frames == -1:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        factor = compute_downscale_factor(max(width, height))
        new_w, new_h = int(width / factor), int(height / factor)

        selected_idx = []
        current_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Convert to grayscale, as Laplacian works on single-channel images
            gray_frame = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (new_w, new_h)
            )
            
            # Apply Laplacian operator
            # cv2.CV_64F is used as the depth of the out image to avoid overflow
            # when computing variance, as Laplacian can produce negative values.
            # The var of the Laplacian shows the amount of edges (sharpness).
            sharpness_score = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            
            # Select frame if its sharpness score exceeds the threshold
            if sharpness_score > sharpness_threshold:
                selected_idx.append((current_idx, sharpness_score))
            
            current_idx += 1

        # Fallback and Thinning
        num_frames = self.max_sampled_frames
        if not selected_idx or len(selected_idx) == 0:
            selected_idx = np.linspace(
                0, self.total_frames - 1, num_frames, dtype=int
            )
        
        if len(selected_idx) > num_frames:
            selected_idx.sort(key=lambda x: x[1], reverse=True)
            selected_idx = [f[0] for f in selected_idx[:num_frames]]

        # Frame Extraction
        frames = []
        current_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        
        target_indices = sorted(list(selected_idx))
        for target in target_indices:
            while current_idx < target:
                cap.grab()
                current_idx += 1
            
            success, frame = cap.read()
            if success:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                current_idx += 1
            else:
                break

        cap.release()

        return np.stack(frames)

    def sample_frames_by_scene_change(self) -> npt.NDArray:
        content_threshold = self.strategy_params.get("content_threshold", 27.0)

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        
        if self.total_frames == -1:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Scene Detection
        # ContentDetector: detects fast cuts using weighted avg of HSV change.
        # The ContentDetector works by comparing successive frames of a video.
        # If difference <= threshold, the frames are considered part of the same
        # scene. Higher threshold: Fewer scene changes will be detected.
        # The detector will be less sensitive to minor changes and will only
        # mark very distinct cuts
        # TODO: Think about min_scene_len and frame_skip (to reduce latency)
        video_adapter = VideoCaptureAdapter(cap)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=content_threshold, min_scene_len=15)
        )
        scene_manager.detect_scenes(video_adapter, frame_skip=0)
        scene_list = scene_manager.get_scene_list()

        # Pick frames for each scene:
        # Each scene contains a number of frames defined by start and end.
        # We try to keep the start, the middle and the end frame for each scene.
        # If scene smaller than 3 frames, we don't keep the middle frame.
        # If scene has only one frame, we keep that frame only (start).
        selected_idx: Set[int] = set()
        for start_tc, end_tc in scene_list:
            start, end = start_tc.get_frames(), end_tc.get_frames()
            selected_idx.add(start)
            if end - start > 3:
                selected_idx.add((start + end) // 2)
            if end - start >= 1:
                selected_idx.add(end - 1)

        # Fallback and Thinning
        num_frames = self.max_sampled_frames
        if not selected_idx or len(selected_idx) == 0:
            selected_idx = set(
                np.linspace(0, self.total_frames - 1, num_frames, dtype=int)
            )
        
        if len(selected_idx) > num_frames:
            sorted_idx = sorted(selected_idx)
            keep = np.linspace(0, len(sorted_idx) - 1, num_frames, dtype=int)
            selected_idx = {sorted_idx[i] for i in keep}
        
        # Frame Extraction
        frames = []
        current_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        
        target_indices = sorted(list(selected_idx))
        for target in target_indices:
            while current_idx < target:
                cap.grab()
                current_idx += 1
            
            success, frame = cap.read()
            if success:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                current_idx += 1
            else:
                break

        cap.release()

        return np.stack(frames)

    def video_to_ndarrays_custom(self) -> npt.NDArray:
        if self.use_cache:
            cached_path = self.build_cached_path()
        
        if self.use_cache and os.path.exists(cached_path):
            print(f"Found cached frames at {cached_path}!", flush=True)
            return np.load(cached_path)

        try:
            if self.sampling_strategy == "motion_based":
                frames = self.sample_frames_by_motion()
            elif self.sampling_strategy == "sharpness_based":
                frames = self.sample_frames_by_sharpness()
            elif self.sampling_strategy == "scene_change":
                frames = self.sample_frames_by_scene_change()
            else:
                raise ValueError(
                    f"Unknown sampling strategy: {self.sampling_strategy}"
                )
            
            if self.use_cache:
                np.save(cached_path, frames)

        except ValueError as e:
            print(
                f"Error during video processing for strategy",
                f"'{self.sampling_strategy}': {e}", flush=True
            )
            return np.array([])
        except Exception as e:
            print(
                f"An unexpected error occurred during strategy",
                f"'{self.sampling_strategy}': {e}", flush=True
            )
            return np.array([])

        return frames