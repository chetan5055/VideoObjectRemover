from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sorawm.watermark_detector import SoraWaterMarkDetector


class TestSoraWaterMarkDetector:
    """Test cases for SoraWaterMarkDetector class."""

    @patch("sorawm.watermark_detector.download_detector_weights")
    @patch("sorawm.watermark_detector.YOLO")
    @patch("sorawm.watermark_detector.get_device")
    def test_init(self, mock_get_device, mock_yolo_class, mock_download_weights):
        """Test detector initialization."""
        # Setup mocks
        mock_device = "cuda:0"
        mock_get_device.return_value = mock_device
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Initialize detector
        detector = SoraWaterMarkDetector()

        # Verify calls
        mock_download_weights.assert_called_once()
        mock_yolo_class.assert_called_once()
        mock_get_device.assert_called_once()
        mock_model.to.assert_called_once_with(str(mock_device))
        mock_model.eval.assert_called_once()

        # Verify attributes
        assert detector.model == mock_model

    @patch("sorawm.watermark_detector.download_detector_weights")
    @patch("sorawm.watermark_detector.YOLO")
    @patch("sorawm.watermark_detector.get_device")
    def test_detect_with_watermark(
        self, mock_get_device, mock_yolo_class, mock_download_weights, sample_image
    ):
        """Test detection when watermark is found."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_get_device.return_value = "cpu"

        # Mock YOLO results with detection
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = np.array([[10.0, 20.0, 50.0, 60.0]])  # Mock tensor
        mock_box.conf = np.array([0.85])  # Mock tensor
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        # Initialize detector
        detector = SoraWaterMarkDetector()

        # Run detection
        result = detector.detect(sample_image)

        # Verify result structure
        assert result["detected"] is True
        assert result["bbox"] == (10, 20, 50, 60)
        assert result["confidence"] == 0.85
        assert result["center"] == (30, 40)  # ((10+50)/2, (20+60)/2)

        # Verify YOLO was called
        mock_model.assert_called_once_with(sample_image, verbose=False)

    @patch("sorawm.watermark_detector.download_detector_weights")
    @patch("sorawm.watermark_detector.YOLO")
    @patch("sorawm.watermark_detector.get_device")
    def test_detect_no_watermark(
        self, mock_get_device, mock_yolo_class, mock_download_weights, sample_image
    ):
        """Test detection when no watermark is found."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_get_device.return_value = "cpu"

        # Mock YOLO results with no detections
        mock_result = MagicMock()
        mock_result.boxes = []  # No boxes
        mock_model.return_value = [mock_result]

        # Initialize detector
        detector = SoraWaterMarkDetector()

        # Run detection
        result = detector.detect(sample_image)

        # Verify result structure
        assert result["detected"] is False
        assert result["bbox"] is None
        assert result["confidence"] is None
        assert result["center"] is None

        # Verify YOLO was called
        mock_model.assert_called_once_with(sample_image, verbose=False)

    @patch("sorawm.watermark_detector.download_detector_weights")
    @patch("sorawm.watermark_detector.YOLO")
    @patch("sorawm.watermark_detector.get_device")
    def test_detect_tensor_conversion(
        self, mock_get_device, mock_yolo_class, mock_download_weights, sample_image
    ):
        """Test that tensor results are properly converted to Python types."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_get_device.return_value = "cpu"

        # Mock YOLO results with tensor values
        mock_result = MagicMock()
        mock_box = MagicMock()
        # Simulate tensor values that need conversion
        mock_box.xyxy = MagicMock()
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [15.7, 25.3, 45.9, 55.1]
        mock_box.conf = MagicMock()
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.92
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        # Initialize detector
        detector = SoraWaterMarkDetector()

        # Run detection
        result = detector.detect(sample_image)

        # Verify result structure and types
        assert result["detected"] is True
        assert result["bbox"] == (15, 25, 45, 55)  # Floats converted to ints
        assert result["confidence"] == 0.92
        assert result["center"] == (30, 40)  # ((15+45)/2, (25+55)/2)

    @patch("sorawm.watermark_detector.download_detector_weights")
    @patch("sorawm.watermark_detector.YOLO")
    @patch("sorawm.watermark_detector.get_device")
    def test_detect_multiple_boxes_uses_first(
        self, mock_get_device, mock_yolo_class, mock_download_weights, sample_image
    ):
        """Test that when multiple boxes are detected, only the first one is used."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_get_device.return_value = "cpu"

        # Mock YOLO results with multiple detections
        mock_result = MagicMock()

        # First box (should be used)
        mock_box1 = MagicMock()
        mock_box1.xyxy = np.array([[10.0, 20.0, 50.0, 60.0]])
        mock_box1.conf = np.array([0.85])
        mock_box1.cls = np.array([0])  # class

        # Second box (should be ignored)
        mock_box2 = MagicMock()
        mock_box2.xyxy = np.array([[70.0, 80.0, 100.0, 120.0]])
        mock_box2.conf = np.array([0.75])
        mock_box2.cls = np.array([0])

        mock_result.boxes = [mock_box1, mock_box2]
        mock_model.return_value = [mock_result]

        # Initialize detector
        detector = SoraWaterMarkDetector()

        # Run detection
        result = detector.detect(sample_image)

        # Verify only first detection is used
        assert result["detected"] is True
        assert result["bbox"] == (10, 20, 50, 60)
        assert result["confidence"] == 0.85
        assert result["center"] == (30, 40)

    @patch("sorawm.watermark_detector.download_detector_weights")
    @patch("sorawm.watermark_detector.YOLO")
    @patch("sorawm.watermark_detector.get_device")
    def test_detect_input_validation(
        self, mock_get_device, mock_yolo_class, mock_download_weights
    ):
        """Test detection with various input types."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_get_device.return_value = "cpu"

        mock_result = MagicMock()
        mock_result.boxes = []  # No detections
        mock_model.return_value = [mock_result]

        # Initialize detector
        detector = SoraWaterMarkDetector()

        # Test with different input types (should not raise exceptions)
        # Valid numpy array
        valid_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = detector.detect(valid_image)
        assert isinstance(result, dict)

        # Test with different shapes (grayscale should still work)
        grayscale_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = detector.detect(grayscale_image)
        assert isinstance(result, dict)
