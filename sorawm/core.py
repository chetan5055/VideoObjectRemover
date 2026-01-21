from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from tqdm import tqdm

import ffmpeg

from sorawm.configs import DEFAULT_DETECT_BATCH_SIZE, ENABLE_E2FGVI_HQ_TORCH_COMPILE
from sorawm.schemas import CleanerType
from sorawm.utils.imputation_utils import (
    find_2d_data_bkps,
    find_idxs_interval,
    get_interval_average_bbox,
    refine_bkps_by_chunk_size,
)
from sorawm.utils.video_utils import VideoLoader, merge_frames_with_overlap
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]

# ✅ Stronger mask expansion to actually remove watermark
MASK_PAD_PX = 18
DILATE_KERNEL = 15  # must be odd
DILATE_ITERS = 1


def _make_mask_from_bbox(height: int, width: int, bbox):
    """
    bbox: (x1,y1,x2,y2) ints
    returns (H,W) uint8 mask 0/255
    """
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    x1 = int(max(0, x1 - MASK_PAD_PX))
    y1 = int(max(0, y1 - MASK_PAD_PX))
    x2 = int(min(width, x2 + MASK_PAD_PX))
    y2 = int(min(height, y2 + MASK_PAD_PX))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    try:
        import cv2
        k = np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8)
        mask = cv2.dilate(mask, k, iterations=DILATE_ITERS)
    except Exception:
        pass

    return mask


def _resize_frame_rgb(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Ensure frame is (H,W,3) in RGB."""
    if frame_rgb.shape[0] == height and frame_rgb.shape[1] == width:
        return frame_rgb

    try:
        import cv2
        resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized
    except Exception:
        # fallback (slower) without cv2
        from PIL import Image
        im = Image.fromarray(frame_rgb.astype(np.uint8))
        im = im.resize((width, height))
        return np.array(im)


class SoraWM:
    def __init__(
        self,
        cleaner_type: CleanerType = CleanerType.LAMA,
        enable_torch_compile=ENABLE_E2FGVI_HQ_TORCH_COMPILE,
        detect_batch_size: int = DEFAULT_DETECT_BATCH_SIZE,
        use_bf16: bool = False,
        propainter_dir: str | None = None,
        propainter_weights_dir: str | None = None,
        propainter_fast_mode: bool = True,
        device: str = "cuda",
    ):
        self.detector = SoraWaterMarkDetector()
        self.cleaner = WaterMarkCleaner(
            cleaner_type=cleaner_type,
            enable_torch_compile=enable_torch_compile,
            use_bf16=use_bf16,
            propainter_dir=propainter_dir,
            propainter_weights_dir=propainter_weights_dir,
            propainter_fast_mode=propainter_fast_mode,
            device=device,
        )
        self.cleaner_type = cleaner_type
        self.detect_batch_size = detect_batch_size

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
        quiet: bool = False,
    ):
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames

        temp_output_path = output_video_path.parent / f"temp_{output_video_path.name}"

        output_options = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "preset": "fast",      # speed
            "crf": "16",           # quality
            "movflags": "+faststart",
           }


        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), **output_options)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True)
        )

        # -------------------------
        # 1) Detect watermarks (batch)
        # -------------------------
        frame_bboxes = {}
        detect_missed = []
        bbox_centers = []
        bboxes = []

        if not quiet:
            logger.debug(f"total frames: {total_frames}, fps: {fps}, width: {width}, height: {height}")

        batch_frames = []
        batch_indices = []

        for idx, frame in enumerate(
            tqdm(input_video_loader, total=total_frames, desc="Detect watermarks", disable=quiet)
        ):
            batch_frames.append(frame)
            batch_indices.append(idx)

            if len(batch_frames) >= self.detect_batch_size:
                batch_results = self.detector.detect_batch(batch_frames, batch_size=self.detect_batch_size)

                for batch_idx, detection_result in zip(batch_indices, batch_results):
                    if detection_result["detected"]:
                        frame_bboxes[batch_idx] = {"bbox": detection_result["bbox"]}
                        x1, y1, x2, y2 = detection_result["bbox"]
                        bbox_centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                        bboxes.append((x1, y1, x2, y2))
                    else:
                        frame_bboxes[batch_idx] = {"bbox": None}
                        detect_missed.append(batch_idx)
                        bbox_centers.append(None)
                        bboxes.append(None)

                    if progress_callback and batch_idx % 10 == 0:
                        progress = 10 + int((batch_idx / total_frames) * 40)
                        progress_callback(progress)

                batch_frames.clear()
                batch_indices.clear()

        if batch_frames:
            batch_results = self.detector.detect_batch(batch_frames, batch_size=self.detect_batch_size)
            for batch_idx, detection_result in zip(batch_indices, batch_results):
                if detection_result["detected"]:
                    frame_bboxes[batch_idx] = {"bbox": detection_result["bbox"]}
                    x1, y1, x2, y2 = detection_result["bbox"]
                    bbox_centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                    bboxes.append((x1, y1, x2, y2))
                else:
                    frame_bboxes[batch_idx] = {"bbox": None}
                    detect_missed.append(batch_idx)
                    bbox_centers.append(None)
                    bboxes.append(None)

                if progress_callback and batch_idx % 10 == 0:
                    progress = 10 + int((batch_idx / total_frames) * 40)
                    progress_callback(progress)

        if not quiet:
            logger.debug(f"detect missed frames: {detect_missed}")

        # -------------------------
        # 2) Fill missed bboxes
        # -------------------------
        bkps_full = [0, total_frames]
        if detect_missed:
            bkps = find_2d_data_bkps(bbox_centers)
            bkps_full = [0] + bkps + [total_frames]
            interval_bboxes = get_interval_average_bbox(bboxes, bkps_full)
            missed_intervals = find_idxs_interval(detect_missed, bkps_full)

            for missed_idx, interval_idx in zip(detect_missed, missed_intervals):
                if interval_idx < len(interval_bboxes) and interval_bboxes[interval_idx] is not None:
                    frame_bboxes[missed_idx]["bbox"] = interval_bboxes[interval_idx]
                else:
                    before = max(missed_idx - 1, 0)
                    after = min(missed_idx + 1, total_frames - 1)
                    before_box = frame_bboxes[before]["bbox"]
                    after_box = frame_bboxes[after]["bbox"]
                    if before_box:
                        frame_bboxes[missed_idx]["bbox"] = before_box
                    elif after_box:
                        frame_bboxes[missed_idx]["bbox"] = after_box

        # -------------------------
        # 3) Clean watermark
        # -------------------------
        if self.cleaner_type == CleanerType.LAMA:
            input_video_loader = VideoLoader(input_video_path)
            for idx, frame in enumerate(
                tqdm(input_video_loader, total=total_frames, desc="Remove watermarks", disable=quiet)
            ):
                bbox = frame_bboxes[idx]["bbox"]
                if bbox is not None:
                    mask = _make_mask_from_bbox(height, width, bbox)
                    cleaned_frame = self.cleaner.clean(frame, mask)
                else:
                    cleaned_frame = frame

                process_out.stdin.write(cleaned_frame.tobytes())

                if progress_callback and idx % 10 == 0:
                    progress = 50 + int((idx / total_frames) * 45)
                    progress_callback(progress)

        elif self.cleaner_type == CleanerType.E2FGVI_HQ:
            input_video_loader = VideoLoader(input_video_path)
            frame_counter = 0
            overlap_ratio = self.cleaner.config.overlap_ratio
            all_cleaned_frames = None

            bkps_full = refine_bkps_by_chunk_size(bkps_full, self.cleaner.chunk_size)
            num_segments = len(bkps_full) - 1

            for segment_idx in tqdm(range(num_segments), desc="Segment", position=0, leave=True, disable=quiet):
                seg_start = bkps_full[segment_idx]
                seg_end = bkps_full[segment_idx + 1]
                seg_length = seg_end - seg_start

                segment_overlap = max(1, int(overlap_ratio * seg_length))
                start = seg_start
                end = seg_end

                if segment_idx > 0:
                    start = max(seg_start - segment_overlap, bkps_full[segment_idx - 1])
                if segment_idx < num_segments - 1:
                    end = min(seg_end + segment_overlap, bkps_full[segment_idx + 2])

                frames = np.array(input_video_loader.get_slice(start, end))
                frames = frames[:, :, :, ::-1].copy()  # BGR -> RGB

                masks = np.zeros((len(frames), height, width), dtype=np.uint8)
                for i in range(start, end):
                    bbox = frame_bboxes[i]["bbox"]
                    if bbox is not None:
                        i_off = i - start
                        masks[i_off] = _make_mask_from_bbox(height, width, bbox)

                cleaned_frames = self.cleaner.clean(frames, masks)

                all_cleaned_frames = merge_frames_with_overlap(
                    result_frames=all_cleaned_frames,
                    chunk_frames=cleaned_frames,
                    start_idx=start,
                    overlap_size=segment_overlap,
                    is_first_chunk=(segment_idx == 0),
                )

                for write_idx in range(seg_start, seg_end):
                    if write_idx < len(all_cleaned_frames) and all_cleaned_frames[write_idx] is not None:
                        cleaned_frame = all_cleaned_frames[write_idx]
                        cleaned_frame_bgr = cleaned_frame[:, :, ::-1]  # RGB -> BGR
                        process_out.stdin.write(cleaned_frame_bgr.astype(np.uint8).tobytes())
                        frame_counter += 1

                        if progress_callback and frame_counter % 10 == 0:
                            progress = 50 + int((frame_counter / total_frames) * 45)
                            progress_callback(progress)

        elif self.cleaner_type == CleanerType.PROPAINTER:
            import tempfile

            input_video_loader = VideoLoader(input_video_path)

            frames_bgr = np.array(list(input_video_loader))
            frames_rgb = frames_bgr[:, :, :, ::-1].copy()

            masks = np.zeros((len(frames_rgb), height, width), dtype=np.uint8)
            for idx in range(len(frames_rgb)):
                bbox = frame_bboxes[idx]["bbox"]
                if bbox is not None:
                    masks[idx] = _make_mask_from_bbox(height, width, bbox)

            with tempfile.TemporaryDirectory() as td:
                tmp_path = Path(td)
                cleaned_frames_rgb = self.cleaner.clean(frames_rgb, masks, tmp_path, fps=fps)

            # ✅ IMPORTANT FIX: ProPainter may output smaller frames (e.g. 240x436),
            # so we resize back to original (width,height) before piping to ffmpeg.
            for idx, fr in enumerate(cleaned_frames_rgb):
                fr = _resize_frame_rgb(fr, width=width, height=height)
                cleaned_bgr = fr[:, :, ::-1].astype(np.uint8)
                process_out.stdin.write(cleaned_bgr.tobytes())

                if progress_callback and idx % 10 == 0:
                    progress = 50 + int((idx / total_frames) * 45)
                    progress_callback(progress)

        else:
            raise ValueError(f"Unsupported cleaner type in run(): {self.cleaner_type}")

        # -------------------------
        # 4) Finalize + merge audio
        # -------------------------
        process_out.stdin.close()
        process_out.wait()

        if progress_callback:
            progress_callback(95)

        self.merge_audio_track(input_video_path, temp_output_path, output_video_path)

        if progress_callback:
            progress_callback(99)

    def merge_audio_track(self, input_video_path: Path, temp_output_path: Path, output_video_path: Path):
        logger.info("Merging audio track...")
        video_stream = ffmpeg.input(str(temp_output_path))
        audio_stream = ffmpeg.input(str(input_video_path)).audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_video_path),
                vcodec="copy",
                acodec="aac",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        temp_output_path.unlink(missing_ok=True)
        logger.info(f"Saved no watermark video with audio at: {output_video_path}")
