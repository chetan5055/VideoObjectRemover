# sorawm/cleaner/propainter_cleaner.py
import subprocess, os, shutil
import numpy as np
import cv2
from pathlib import Path

class ProPainterCleaner:
    def __init__(self, propainter_dir: str, weights_dir: str, device="cuda"):
        self.propainter_dir = Path(propainter_dir)
        self.weights_dir = Path(weights_dir)
        self.device = device

    def _write_frames(self, frames, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    def _write_masks_from_bboxes(self, bboxes, h, w, out_dir: Path, pad=8):
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (x1,y1,x2,y2) in enumerate(bboxes):
            mask = np.zeros((h,w), dtype=np.uint8)
            x1p = max(0, x1-pad); y1p = max(0, y1-pad)
            x2p = min(w-1, x2+pad); y2p = min(h-1, y2+pad)
            mask[y1p:y2p, x1p:x2p] = 255
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), mask)

    def clean(self, frames_rgb, bboxes_xyxy, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        masks_dir  = tmp_path / "masks"
        out_dir    = tmp_path / "out"

        h, w = frames_rgb[0].shape[:2]
        self._write_frames(frames_rgb, frames_dir)
        self._write_masks_from_bboxes(bboxes_xyxy, h, w, masks_dir)

        # Example command – adjust to ProPainter’s actual inference script name/args
        cmd = [
            "python",
            str(self.propainter_dir / "inference_propainter.py"),
            "--img_dir", str(frames_dir),
            "--mask_dir", str(masks_dir),
            "--output_dir", str(out_dir),
            "--weights_dir", str(self.weights_dir),
            "--device", self.device,
        ]
        subprocess.check_call(cmd, cwd=str(self.propainter_dir))

        # read output frames back
        cleaned = []
        for i in range(len(frames_rgb)):
            p = out_dir / f"{i:05d}.png"
            bgr = cv2.imread(str(p))
            cleaned.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return cleaned
