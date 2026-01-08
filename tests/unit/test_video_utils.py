from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from sorawm.utils.video_utils import VideoLoader, merge_frames_with_overlap


class TestVideoUtils:
    """Test cases for video utility functions."""

    def test_merge_frames_with_overlap_first_chunk(self, mock_video_frames):
        """Test merging frames when it's the first chunk."""
        chunk_frames = mock_video_frames[:5]
        start_idx = 0
        overlap_size = 2

        result = merge_frames_with_overlap(
            result_frames=None,
            chunk_frames=chunk_frames,
            start_idx=start_idx,
            overlap_size=overlap_size,
            is_first_chunk=True,
        )

        assert len(result) == len(chunk_frames)
        assert result[0] is chunk_frames[0]
        assert result[4] is chunk_frames[4]

    def test_merge_frames_with_overlap_subsequent_chunk_no_overlap(
        self, mock_video_frames
    ):
        """Test merging subsequent chunks with no overlap."""
        existing_frames = mock_video_frames[:3]
        chunk_frames = mock_video_frames[3:6]
        start_idx = 3
        overlap_size = 0

        result = merge_frames_with_overlap(
            result_frames=existing_frames,
            chunk_frames=chunk_frames,
            start_idx=start_idx,
            overlap_size=overlap_size,
            is_first_chunk=False,
        )

        assert len(result) == 6
        assert result[:3] == existing_frames
        assert result[3:] == chunk_frames

    def test_merge_frames_with_overlap_with_blending(self, mock_video_frames):
        """Test merging with overlap blending."""
        # Create simple test frames for predictable blending
        frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 200
        frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 50
        frame4 = np.ones((10, 10, 3), dtype=np.uint8) * 150

        existing_frames = [frame1, frame2]
        chunk_frames = [frame3, frame4]
        start_idx = 1  # Overlap with frame2
        overlap_size = 1

        result = merge_frames_with_overlap(
            result_frames=existing_frames,
            chunk_frames=chunk_frames,
            start_idx=start_idx,
            overlap_size=overlap_size,
            is_first_chunk=False,
        )

        assert len(result) == 3
        assert np.array_equal(result[0], frame1)  # First frame unchanged

        # Second frame should be blended
        # alpha = 0/1 = 0, so result should be all of old frame
        expected_blend = (
            frame2.astype(np.float32) * (1 - 0.0) + frame3.astype(np.float32) * 0.0
        ).astype(np.uint8)
        assert np.array_equal(result[1], expected_blend)

        # Third frame should be the new frame
        assert np.array_equal(result[2], frame4)

    def test_merge_frames_with_overlap_extend_result(self, mock_video_frames):
        """Test that result_frames is extended when needed."""
        existing_frames = mock_video_frames[:2]
        chunk_frames = mock_video_frames[2:5]
        start_idx = 3  # Start beyond current result length
        overlap_size = 1

        result = merge_frames_with_overlap(
            result_frames=existing_frames,
            chunk_frames=chunk_frames,
            start_idx=start_idx,
            overlap_size=overlap_size,
            is_first_chunk=False,
        )

        assert len(result) == 6  # start_idx(3) + chunk_size(3)
        assert result[0] is existing_frames[0]
        assert result[1] is existing_frames[1]
        assert result[2] is None  # Gap filled with None
        assert result[3] is chunk_frames[0]
        assert result[4] is chunk_frames[1]
        assert result[5] is chunk_frames[2]

    def test_merge_frames_with_overlap_none_handling(self):
        """Test handling of None frames in blending."""
        frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        frame2 = None
        frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 200
        frame4 = np.ones((10, 10, 3), dtype=np.uint8) * 50

        existing_frames = [frame1, frame2]
        chunk_frames = [frame3, frame4]
        start_idx = 0
        overlap_size = 2

        result = merge_frames_with_overlap(
            result_frames=existing_frames,
            chunk_frames=chunk_frames,
            start_idx=start_idx,
            overlap_size=overlap_size,
            is_first_chunk=False,
        )

        assert len(result) == 2
        # When existing frame is None, should use chunk frame
        assert np.array_equal(result[0], frame3)
        assert np.array_equal(result[1], frame4)

    @patch("sorawm.utils.video_utils.ffmpeg")
    def test_video_loader_init(self, mock_ffmpeg, temp_dir):
        """Test VideoLoader initialization."""
        video_path = temp_dir / "test.mp4"

        # Mock ffmpeg.probe response
        mock_probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "nb_frames": "900",
                    "bit_rate": "5000000",
                }
            ],
            "format": {"duration": "30.0"},
        }
        mock_ffmpeg.probe.return_value = mock_probe

        loader = VideoLoader(video_path)

        assert loader.video_path == video_path
        assert loader.width == 1920
        assert loader.height == 1080
        assert loader.fps == 30.0
        assert loader.total_frames == 900
        assert loader.original_bitrate == "5000000"

        mock_ffmpeg.probe.assert_called_once_with(video_path)

    @patch("sorawm.utils.video_utils.ffmpeg")
    def test_video_loader_init_no_nb_frames(self, mock_ffmpeg, temp_dir):
        """Test VideoLoader when nb_frames is not available."""
        video_path = temp_dir / "test.mp4"

        # Mock ffmpeg.probe response without nb_frames
        mock_probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "25/1",
                    "bit_rate": "2000000",
                }
            ],
            "format": {"duration": "10.0"},
        }
        mock_ffmpeg.probe.return_value = mock_probe

        loader = VideoLoader(video_path)

        assert loader.fps == 25.0
        assert loader.total_frames == 250  # 10.0 * 25

    def test_video_loader_len(self, temp_dir):
        """Test VideoLoader length."""
        with patch("sorawm.utils.video_utils.ffmpeg"):
            loader = VideoLoader(temp_dir / "test.mp4")
            loader.total_frames = 100

            assert len(loader) == 100

    @patch("sorawm.utils.video_utils.ffmpeg")
    def test_video_loader_get_slice(self, mock_ffmpeg, temp_dir, sample_image):
        """Test VideoLoader get_slice method."""
        video_path = temp_dir / "test.mp4"

        # Mock the loader init
        with patch.object(VideoLoader, "get_video_info"):
            loader = VideoLoader(video_path)
            loader.width = 100
            loader.height = 100
            loader.fps = 30

        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_ffmpeg.input.return_value.output.return_value.global_args.return_value.run_async.return_value = mock_process

        # Mock reading frame data
        frame_data = sample_image.tobytes()
        mock_process.stdout.read.return_value = frame_data
        mock_process.stdout.read.side_effect = [
            frame_data,
            b"",
        ]  # Return data then empty

        frames = loader.get_slice(0, 1)

        assert len(frames) == 1
        assert frames[0].shape == sample_image.shape

        # Verify FFmpeg was called with correct parameters
        mock_ffmpeg.input.assert_called_with(video_path, ss=0.0)

        # Verify cleanup was called
        mock_process.stdout.close.assert_called()
        mock_process.wait.assert_called()

    @patch("sorawm.utils.video_utils.ffmpeg")
    def test_video_loader_iter(self, mock_ffmpeg, temp_dir, sample_image):
        """Test VideoLoader iterator."""
        video_path = temp_dir / "test.mp4"

        # Mock the loader init
        with patch.object(VideoLoader, "get_video_info"):
            loader = VideoLoader(video_path)
            loader.width = 100
            loader.height = 100
            loader.fps = 30

        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_ffmpeg.input.return_value.output.return_value.global_args.return_value.run_async.return_value = mock_process

        # Mock reading frame data
        frame_data = sample_image.tobytes()
        mock_process.stdout.read.side_effect = [
            frame_data,
            b"",
        ]  # Return data then empty

        frames = list(loader)

        assert len(frames) == 1
        assert frames[0].shape == sample_image.shape

        # Verify cleanup was called
        mock_process.stdout.close.assert_called()
        mock_process.wait.assert_called()
