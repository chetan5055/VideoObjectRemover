#!/usr/bin/env python3
"""
End-to-end integration tests for SoraWatermarkCleaner pipeline.

Tests the complete pipeline with all optimizations:
1. Video loading
2. YOLO batch detection
3. E2FGVI_HQ cleaning with AMP and pinned memory
4. Video encoding and audio merging
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# pytest is optional - tests can run standalone
try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

    # Create dummy decorators
    class pytest:
        @staticmethod
        def fixture(func):
            return func

        @staticmethod
        def mark():
            pass

        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(cls_or_func):
                    return cls_or_func

                return decorator

        @staticmethod
        def skip(msg):
            print(f"SKIP: {msg}")
            return None


# Skip if heavy dependencies are not available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sorawm.core import SoraWM
    from sorawm.schemas import CleanerType
    from sorawm.utils.video_utils import VideoLoader
    from sorawm.watermark_detector import SoraWaterMarkDetector

    SORAWM_AVAILABLE = True
except ImportError:
    SORAWM_AVAILABLE = False


# Test video path - adjust this to your test video
TEST_VIDEO_PATH = Path("resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4")


@pytest.fixture
def test_video_path():
    """Get test video path, skip if not available."""
    if TEST_VIDEO_PATH.exists():
        return TEST_VIDEO_PATH

    # Try to find any video in resources
    resources = Path("resources")
    if resources.exists():
        videos = list(resources.glob("*.mp4"))
        if videos:
            return videos[0]

    pytest.skip("No test video available")


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.skipif(not SORAWM_AVAILABLE, reason="SoraWM not available")
class TestVideoLoader:
    """Tests for VideoLoader."""

    def test_video_info(self, test_video_path):
        """Test video info extraction."""
        loader = VideoLoader(test_video_path)

        assert loader.width > 0
        assert loader.height > 0
        assert loader.fps > 0
        assert loader.total_frames > 0

    def test_frame_iteration(self, test_video_path):
        """Test frame iteration."""
        loader = VideoLoader(test_video_path)

        frames = []
        for i, frame in enumerate(loader):
            frames.append(frame)
            if len(frames) >= 10:
                break

        assert len(frames) == 10
        assert frames[0].shape == (loader.height, loader.width, 3)
        assert frames[0].dtype == np.uint8

    def test_get_slice(self, test_video_path):
        """Test frame slicing."""
        loader = VideoLoader(test_video_path)

        frames = loader.get_slice(0, 10)

        assert len(frames) == 10
        assert frames[0].shape == (loader.height, loader.width, 3)


@pytest.mark.skipif(not SORAWM_AVAILABLE, reason="SoraWM not available")
class TestWatermarkDetector:
    """Tests for watermark detector with batch optimization."""

    def test_single_detection(self, test_video_path):
        """Test single frame detection."""
        loader = VideoLoader(test_video_path)
        detector = SoraWaterMarkDetector()

        # Get first frame
        frame = next(iter(loader))

        result = detector.detect(frame)

        assert "detected" in result
        assert "bbox" in result
        assert "confidence" in result
        assert "center" in result

    def test_batch_detection(self, test_video_path):
        """Test batch detection optimization."""
        loader = VideoLoader(test_video_path)
        detector = SoraWaterMarkDetector()

        # Load frames
        frames = []
        for i, frame in enumerate(loader):
            frames.append(frame)
            if len(frames) >= 20:
                break

        # Batch detect
        results = detector.detect_batch(frames, batch_size=4)

        assert len(results) == len(frames)
        for result in results:
            assert "detected" in result
            assert "bbox" in result

    def test_batch_vs_single_consistency(self, test_video_path):
        """Test that batch and single detection give consistent results."""
        loader = VideoLoader(test_video_path)
        detector = SoraWaterMarkDetector()

        # Load frames
        frames = []
        for i, frame in enumerate(loader):
            frames.append(frame)
            if len(frames) >= 10:
                break

        # Single detection
        single_results = [detector.detect(f) for f in frames]

        # Batch detection
        batch_results = detector.detect_batch(frames, batch_size=4)

        # Compare
        for single, batch in zip(single_results, batch_results):
            assert single["detected"] == batch["detected"]
            if single["detected"]:
                # Allow small bbox differences (up to 2 pixels)
                for i in range(4):
                    assert abs(single["bbox"][i] - batch["bbox"][i]) <= 2


@pytest.mark.skipif(not SORAWM_AVAILABLE, reason="SoraWM not available")
@pytest.mark.skipif(
    not TORCH_AVAILABLE or not torch.cuda.is_available(), reason="CUDA not available"
)
class TestE2FGVICleaner:
    """Tests for E2FGVI_HQ cleaner with optimizations."""

    def test_cleaner_with_amp(self, test_video_path):
        """Test cleaner with AMP optimization."""
        from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig

        loader = VideoLoader(test_video_path)

        # Load a few frames
        frames = []
        for i, frame in enumerate(loader):
            frames.append(frame[:, :, ::-1].copy())  # BGR to RGB
            if len(frames) >= 10:
                break

        frames_np = np.array(frames)
        h, w = frames_np.shape[1:3]

        # Create masks
        masks_np = np.zeros((len(frames_np), h, w), dtype=np.uint8)
        masks_np[:, -80:-20, -160:-20] = 255

        # Test with AMP enabled
        config = E2FGVIHDConfig(enable_amp=True, enable_torch_compile=False)
        cleaner = E2FGVIHDCleaner(config=config)

        result = cleaner.clean(frames_np, masks_np)

        assert len(result) == len(frames_np)
        for frame in result:
            assert frame is not None
            assert frame.shape == (h, w, 3)


@pytest.mark.skipif(not SORAWM_AVAILABLE, reason="SoraWM not available")
@pytest.mark.skipif(
    not TORCH_AVAILABLE or not torch.cuda.is_available(), reason="CUDA not available"
)
class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_e2fgvi_hq_pipeline(self, test_video_path, temp_output_dir):
        """Test complete E2FGVI_HQ pipeline."""
        output_path = temp_output_dir / "output_e2fgvi.mp4"

        # Initialize pipeline
        sora_wm = SoraWM(cleaner_type=CleanerType.E2FGVI_HQ, enable_torch_compile=False)

        # Process video
        progress_values = []

        def progress_callback(progress):
            progress_values.append(progress)

        sora_wm.run(
            input_video_path=test_video_path,
            output_video_path=output_path,
            progress_callback=progress_callback,
            quiet=True,
        )

        # Verify output
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify progress was reported
        assert len(progress_values) > 0
        assert max(progress_values) >= 90

        # Verify output video is valid
        output_loader = VideoLoader(output_path)
        assert output_loader.total_frames > 0
        assert output_loader.width > 0
        assert output_loader.height > 0


def run_quick_e2e_test():
    """Run a quick end-to-end test without pytest."""
    print("=" * 60)
    print("Quick End-to-End Test")
    print("=" * 60)

    # Check dependencies
    if not SORAWM_AVAILABLE:
        print("ERROR: SoraWM not available")
        return False

    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    # Find test video
    video_path = TEST_VIDEO_PATH
    if not video_path.exists():
        resources = Path("resources")
        if resources.exists():
            videos = list(resources.glob("*.mp4"))
            if videos:
                video_path = videos[0]
            else:
                print("ERROR: No test video found")
                return False

    print(f"Test video: {video_path}")

    # Create temp output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.mp4"

        print("Initializing SoraWM...")
        sora_wm = SoraWM(cleaner_type=CleanerType.E2FGVI_HQ, enable_torch_compile=False)

        print("Processing video...")
        sora_wm.run(
            input_video_path=video_path,
            output_video_path=output_path,
            quiet=False,
        )

        # Verify
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✓ Output created: {output_path}")
            print(f"✓ Output size: {output_path.stat().st_size / 1024:.2f} KB")

            # Check output video
            loader = VideoLoader(output_path)
            print(f"✓ Output frames: {loader.total_frames}")
            print(f"✓ Output resolution: {loader.width}x{loader.height}")

            print("\n" + "=" * 60)
            print("END-TO-END TEST PASSED ✓")
            print("=" * 60)
            return True
        else:
            print("ERROR: Output not created")
            return False


if __name__ == "__main__":
    import sys

    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    # Re-import after path change
    from sorawm.core import SoraWM
    from sorawm.schemas import CleanerType
    from sorawm.utils.video_utils import VideoLoader

    success = run_quick_e2e_test()
    sys.exit(0 if success else 1)
