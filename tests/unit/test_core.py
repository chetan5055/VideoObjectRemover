from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from sorawm.core import SoraWM
from sorawm.schemas import CleanerType


class TestSoraWM:
    """Test cases for SoraWM core class."""

    def test_init_default_cleaner(self):
        """Test initialization with default cleaner type."""
        with (
            patch("sorawm.core.SoraWaterMarkDetector") as mock_detector_class,
            patch("sorawm.core.WaterMarkCleaner") as mock_cleaner_factory,
        ):
            mock_detector = MagicMock()
            mock_cleaner = MagicMock()
            mock_detector_class.return_value = mock_detector
            mock_cleaner_factory.return_value = mock_cleaner

            sora_wm = SoraWM()

            assert sora_wm.detector == mock_detector
            assert sora_wm.cleaner == mock_cleaner
            assert sora_wm.cleaner_type == CleanerType.LAMA

            mock_detector_class.assert_called_once()
            mock_cleaner_factory.assert_called_once_with(CleanerType.LAMA)

    @pytest.mark.parametrize("cleaner_type", [CleanerType.LAMA, CleanerType.E2FGVI_HQ])
    def test_init_with_cleaner_type(self, cleaner_type):
        """Test initialization with specific cleaner type."""
        with (
            patch("sorawm.core.SoraWaterMarkDetector") as mock_detector_class,
            patch("sorawm.core.WaterMarkCleaner") as mock_cleaner_factory,
        ):
            mock_detector = MagicMock()
            mock_cleaner = MagicMock()
            mock_detector_class.return_value = mock_detector
            mock_cleaner_factory.return_value = mock_cleaner

            sora_wm = SoraWM(cleaner_type)

            assert sora_wm.detector == mock_detector
            assert sora_wm.cleaner == mock_cleaner
            assert sora_wm.cleaner_type == cleaner_type

            mock_detector_class.assert_called_once()
            mock_cleaner_factory.assert_called_once_with(cleaner_type)

    def test_run_batch_no_output_dir(self, temp_dir):
        """Test run_batch with no output directory specified."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        with (
            patch("sorawm.core.SoraWaterMarkDetector") as mock_detector_class,
            patch("sorawm.core.WaterMarkCleaner") as mock_cleaner_factory,
            patch("sorawm.core.logger") as mock_logger,
        ):
            mock_detector = MagicMock()
            mock_cleaner = MagicMock()
            mock_detector_class.return_value = mock_detector
            mock_cleaner_factory.return_value = mock_cleaner

            sora_wm = SoraWM()
            sora_wm.run = MagicMock()  # Mock the run method

            # Create mock video files
            video_files = ["test1.mp4", "test2.mp4"]
            for video_file in video_files:
                (input_dir / video_file).touch()

            sora_wm.run_batch(input_dir, quiet=True)

            # Verify output directory creation
            expected_output_dir = input_dir.parent / "watermark_removed"
            assert expected_output_dir.exists()

            # Verify run was called for each video
            assert sora_wm.run.call_count == 2
            expected_calls = [
                call(
                    input_dir / "test1.mp4",
                    expected_output_dir / "test1.mp4",
                    progress_callback=None,
                    quiet=True,
                ),
                call(
                    input_dir / "test2.mp4",
                    expected_output_dir / "test2.mp4",
                    progress_callback=None,
                    quiet=True,
                ),
            ]
            sora_wm.run.assert_has_calls(expected_calls)

    def test_run_batch_with_output_dir(self, temp_dir):
        """Test run_batch with specified output directory."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()

        with (
            patch("sorawm.core.SoraWaterMarkDetector") as mock_detector_class,
            patch("sorawm.core.WaterMarkCleaner") as mock_cleaner_factory,
        ):
            mock_detector = MagicMock()
            mock_cleaner = MagicMock()
            mock_detector_class.return_value = mock_detector
            mock_cleaner_factory.return_value = mock_cleaner

            sora_wm = SoraWM()
            sora_wm.run = MagicMock()

            # Create mock video file
            (input_dir / "test.mp4").touch()

            sora_wm.run_batch(input_dir, output_dir, quiet=True)

            # Verify run was called with correct paths
            sora_wm.run.assert_called_once_with(
                input_dir / "test.mp4",
                output_dir / "test.mp4",
                progress_callback=None,
                quiet=True,
            )

    def test_run_batch_with_progress_callback(self, temp_dir):
        """Test run_batch with progress callback."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        progress_callback = MagicMock()

        with (
            patch("sorawm.core.SoraWaterMarkDetector") as mock_detector_class,
            patch("sorawm.core.WaterMarkCleaner") as mock_cleaner_factory,
        ):
            mock_detector = MagicMock()
            mock_cleaner = MagicMock()
            mock_detector_class.return_value = mock_detector
            mock_cleaner_factory.return_value = mock_cleaner

            sora_wm = SoraWM()
            sora_wm.run = MagicMock()

            # Create mock video file
            (input_dir / "test.mp4").touch()

            sora_wm.run_batch(
                input_dir, progress_callback=progress_callback, quiet=True
            )

            # Verify run was called with batch progress callback
            assert sora_wm.run.call_count == 1
            # The progress_callback passed to run should be a wrapper function
            call_args = sora_wm.run.call_args
            assert callable(call_args[1]["progress_callback"])

    def test_run_batch_no_videos_found(self, temp_dir):
        """Test run_batch when no video files are found."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        with (
            patch("sorawm.core.SoraWaterMarkDetector") as mock_detector_class,
            patch("sorawm.core.WaterMarkCleaner") as mock_cleaner_factory,
            patch("sorawm.core.logger") as mock_logger,
        ):
            mock_detector = MagicMock()
            mock_cleaner = MagicMock()
            mock_detector_class.return_value = mock_detector
            mock_cleaner_factory.return_value = mock_cleaner

            sora_wm = SoraWM()

            # Create non-video file
            (input_dir / "test.txt").touch()

            sora_wm.run_batch(input_dir, quiet=True)

            # Verify logger was called
            mock_logger.info.assert_called_with("Found 0 video(s) to process")
            # Verify run was not called
            sora_wm.run.assert_not_called()

    @patch("sorawm.core.VideoLoader")
    @patch("sorawm.core.ffmpeg")
    @patch("sorawm.core.SoraWaterMarkDetector")
    @patch("sorawm.core.WaterMarkCleaner")
    def test_run_lama_cleaner(
        self,
        mock_cleaner_factory,
        mock_detector_class,
        mock_ffmpeg,
        mock_video_loader_class,
        temp_dir,
        sample_image,
    ):
        """Test run method with LAMA cleaner."""
        input_path = temp_dir / "input.mp4"
        output_path = temp_dir / "output.mp4"

        # Setup mocks
        mock_detector = MagicMock()
        mock_cleaner = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_cleaner_factory.return_value = mock_cleaner

        # Mock video loader
        mock_video_loader = MagicMock()
        mock_video_loader.width = 640
        mock_video_loader.height = 480
        mock_video_loader.fps = 30
        mock_video_loader.total_frames = 2
        mock_video_loader.__iter__.return_value = [sample_image, sample_image]
        mock_video_loader_class.return_value = mock_video_loader

        # Mock detection results - first frame has watermark, second doesn't
        mock_detector.detect.side_effect = [
            {"detected": True, "bbox": (10, 10, 50, 50)},
            {"detected": False, "bbox": None},
        ]

        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.global_args.return_value.run_async.return_value = mock_process

        sora_wm = SoraWM(CleanerType.LAMA)

        # Mock the cleaner's clean method
        cleaned_frame = sample_image.copy()
        mock_cleaner.clean.return_value = cleaned_frame

        sora_wm.run(input_path, output_path, quiet=True)

        # Verify video loader was created correctly
        mock_video_loader_class.assert_called_with(input_path)

        # Verify detector was called for each frame
        assert mock_detector.detect.call_count == 2

        # Verify cleaner was called only for frame with watermark
        mock_cleaner.clean.assert_called_once()
        call_args = mock_cleaner.clean.call_args
        assert call_args[0][0] is sample_image  # First argument is the frame

        # Verify FFmpeg was set up correctly
        mock_ffmpeg.input.assert_called()
        mock_process.stdin.write.assert_called()
        mock_process.stdin.close.assert_called()
        mock_process.wait.assert_called()

        # Verify merge_audio_track was called
        # This would require additional mocking of the merge method

    @patch("sorawm.core.VideoLoader")
    @patch("sorawm.core.ffmpeg")
    @patch("sorawm.core.SoraWaterMarkDetector")
    @patch("sorawm.core.WaterMarkCleaner")
    def test_run_e2fgvi_hq_cleaner(
        self,
        mock_cleaner_factory,
        mock_detector_class,
        mock_ffmpeg,
        mock_video_loader_class,
        temp_dir,
        mock_video_frames,
    ):
        """Test run method with E2FGVI-HQ cleaner."""
        input_path = temp_dir / "input.mp4"
        output_path = temp_dir / "output.mp4"

        # Setup mocks
        mock_detector = MagicMock()
        mock_cleaner = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_cleaner_factory.return_value = mock_cleaner

        # Mock video loader
        mock_video_loader = MagicMock()
        mock_video_loader.width = 640
        mock_video_loader.height = 480
        mock_video_loader.fps = 30
        mock_video_loader.total_frames = len(mock_video_frames)
        mock_video_loader_class.return_value = mock_video_loader

        # Mock detection results - all frames have watermarks
        detection_results = [
            {"detected": True, "bbox": (10, 10, 50, 50)} for _ in mock_video_frames
        ]
        mock_detector.detect.side_effect = detection_results

        # Mock FFmpeg process
        mock_process = MagicMock()
        mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.global_args.return_value.run_async.return_value = mock_process

        sora_wm = SoraWM(CleanerType.E2FGVI_HQ)

        # Mock the cleaner's clean method for video
        cleaned_frames = mock_video_frames.copy()
        mock_cleaner.clean.return_value = cleaned_frames

        # Mock the config attribute for E2FGVI cleaner
        mock_cleaner.config.overlap_ratio = 0.1
        mock_cleaner.chunk_size = 100

        sora_wm.run(input_path, output_path, quiet=True)

        # Verify video loader was created
        mock_video_loader_class.assert_called_with(input_path)

        # Verify cleaner was called (E2FGVI processes in chunks)
        mock_cleaner.clean.assert_called()

        # Verify FFmpeg process was used
        mock_process.stdin.write.assert_called()
        mock_process.stdin.close.assert_called()
        mock_process.wait.assert_called()

    def test_merge_audio_track(self, temp_dir):
        """Test audio track merging functionality."""
        input_path = temp_dir / "input.mp4"
        temp_path = temp_dir / "temp.mp4"
        output_path = temp_dir / "output.mp4"

        with (
            patch("sorawm.core.ffmpeg") as mock_ffmpeg,
            patch("sorawm.core.logger") as mock_logger,
        ):
            sora_wm = SoraWM()

            sora_wm.merge_audio_track(input_path, temp_path, output_path)

            # Verify FFmpeg calls for audio merging
            mock_ffmpeg.input.assert_has_calls(
                [call(str(temp_path)), call(str(input_path))]
            )

            # Verify output was configured correctly
            output_call = mock_ffmpeg.output.call_args
            assert output_call[0][0] == mock_ffmpeg.input.return_value  # video stream
            assert (
                output_call[0][1] == mock_ffmpeg.input.return_value.audio
            )  # audio stream
            assert output_call[0][2] == str(output_path)

            # Verify logger calls
            mock_logger.info.assert_has_calls(
                [
                    call("Merging audio track..."),
                    call(f"Saved no watermark video with audio at: {output_path}"),
                ]
            )

            # Verify temp file was unlinked
            temp_path.unlink.assert_called_once()
