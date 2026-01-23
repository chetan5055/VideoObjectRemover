# sorawm/cleaner/propainter_cleaner.py
import subprocess
import sys
from pathlib import Path
import shutil
import cv2
import numpy as np


class ProPainterCleaner:
    """
    Runs ProPainter repo inference as a subprocess.

    Key points:
    - ProPainter expects masks as a DIRECTORY of per-frame PNG files.
    - ProPainter output '-o' is a DIRECTORY, it saves inside: out_dir/<video_name>/...
    - We robustly locate the output video OR output frames and return cleaned frames.
    """

    def __init__(
        self,
        propainter_dir: str,
        weights_dir: str | None = None,
        device: str = "cuda",
        fast_mode: bool = False,
        fast_scale: float = 0.5,  # downscale for 8GB GPUs
    ):
        self.propainter_dir = Path(propainter_dir).expanduser().resolve()
        self.weights_dir = Path(weights_dir).expanduser().resolve() if weights_dir else None
        self.device = device
        self.fast_mode = fast_mode
        self.fast_scale = float(fast_scale)

        script_path = self.propainter_dir / "inference_propainter.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"❌ ProPainter inference script not found: {script_path}\n"
                f"Make sure propainter_dir points to the ProPainter repo root."
            )

    # -------------------------
    # IO helpers
    # -------------------------
    def _write_video_mp4(self, frames_rgb: np.ndarray, out_path: Path, fps: float):
        """
        frames_rgb: (T,H,W,3) uint8 RGB
        """
        h, w = frames_rgb[0].shape[:2]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"❌ Could not open VideoWriter for {out_path}")

        for fr in frames_rgb:
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
        vw.release()

    def _write_masks_png_dir(self, masks: np.ndarray, masks_dir: Path):
        """
        masks: (T,H,W) uint8 0/255
        writes: masks_dir/00000.png ...
        """
        if masks_dir.exists():
            shutil.rmtree(masks_dir, ignore_errors=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for i, m in enumerate(masks):
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            # ensure 0/255
            if m.max() <= 1:
                m = (m * 255).astype(np.uint8)

            fn = masks_dir / f"{i:05d}.png"
            cv2.imwrite(str(fn), m)

    def _read_video_frames_rgb(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"❌ Could not open output video: {video_path}")

        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _read_png_frames_rgb(self, folder: Path):
        pngs = sorted(folder.glob("*.png"))
        if not pngs:
            return []
        frames = []
        for p in pngs:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return frames

    def _resize_even16(self, img: np.ndarray, scale: float):
        """
        resize with scale and make dims divisible by 16 (ProPainter/ffmpeg-friendly)
        """
        h, w = img.shape[:2]
        nh = max(16, int(h * scale))
        nw = max(16, int(w * scale))
        nh = (nh // 16) * 16
        nw = (nw // 16) * 16
        out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return out

    def _resize_masks_even16(self, masks: np.ndarray, scale: float):
        out = []
        for m in masks:
            mm = self._resize_even16(m, scale)
            mm = (mm > 127).astype(np.uint8) * 255
            out.append(mm)
        return np.stack(out, axis=0)

    def _resize_frames_even16(self, frames_rgb: np.ndarray, scale: float):
        out = []
        for fr in frames_rgb:
            out.append(self._resize_even16(fr, scale))
        return np.stack(out, axis=0)

    def _find_best_output(self, out_dir: Path, video_name: str):
        """
        ProPainter typically saves into out_dir/<video_name>/...
        We search for mp4 first; if not found, search for png sequences.
        """
        root = out_dir / video_name
        if not root.exists():
            root = out_dir

        mp4s = list(root.rglob("*.mp4"))
        mp4s = [p for p in mp4s if "mask" not in p.name.lower() and "input" not in p.name.lower()]
        if mp4s:
            mp4s.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return ("mp4", mp4s[0])

        png_folders = {}
        for p in root.rglob("*.png"):
            png_folders[p.parent] = png_folders.get(p.parent, 0) + 1
        if png_folders:
            best_folder = max(png_folders.items(), key=lambda kv: kv[1])[0]
            return ("pngdir", best_folder)

        return (None, None)

    # -------------------------
    # Main
    # -------------------------
    def clean(self, frames_rgb: np.ndarray, masks: np.ndarray, tmp_path: Path, fps: float = 30.0):
        """
        frames_rgb: (T,H,W,3) uint8 RGB
        masks:      (T,H,W)   uint8 0/255
        """
        tmp_path.mkdir(parents=True, exist_ok=True)

        orig_h, orig_w = frames_rgb[0].shape[:2]

        run_frames = frames_rgb
        run_masks = masks
        if self.fast_mode and self.fast_scale < 1.0:
            run_frames = self._resize_frames_even16(run_frames, self.fast_scale)
            run_masks = self._resize_masks_even16(run_masks, self.fast_scale)

        input_mp4 = tmp_path / "input.mp4"
        masks_dir = tmp_path / "masks"
        out_dir = tmp_path / "pp_out"

        # 1) Write input + masks
        self._write_video_mp4(run_frames, input_mp4, fps)
        self._write_masks_png_dir(run_masks, masks_dir)

        # 2) Call ProPainter (SAFE ARGS ADDED HERE)
        python_exe = sys.executable

        cmd = [
            python_exe,
            str(self.propainter_dir / "inference_propainter.py"),
            "-i", str(input_mp4),
            "-m", str(masks_dir),
            "-o", str(out_dir),

            # Safe defaults to reduce VRAM + prevent OOM
            "--fp16",
            "--max_long_edge", "512",
            "--neighbor_length", "4",
            "--ref_stride", "10",
        ]

        # Optional weights path support (only if your ProPainter accepts --ckpt)
        if self.weights_dir:
            ckpt = self.weights_dir / "ProPainter.pth"
            if ckpt.exists():
                cmd += ["--ckpt", str(ckpt)]

        subprocess.check_call(cmd, cwd=str(self.propainter_dir))

        # 3) Locate result
        video_name = input_mp4.stem  # "input"
        kind, path = self._find_best_output(out_dir, video_name)
        if kind is None:
            raise RuntimeError(f"❌ Could not find ProPainter output inside: {out_dir}")

        if kind == "mp4":
            cleaned = self._read_video_frames_rgb(path)
        else:
            cleaned = self._read_png_frames_rgb(path)

        if not cleaned:
            raise RuntimeError("❌ ProPainter produced 0 frames.")

        # 4) If fast_mode, upscale back to original size
        fixed = []
        for fr in cleaned:
            if fr.shape[0] != orig_h or fr.shape[1] != orig_w:
                fr = cv2.resize(fr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            fixed.append(fr)

        return fixed
