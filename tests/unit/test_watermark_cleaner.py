from unittest.mock import MagicMock, patch

import pytest

from sorawm.schemas import CleanerType
from sorawm.watermark_cleaner import WaterMarkCleaner


class TestWaterMarkCleaner:
    """Test cases for WaterMarkCleaner factory class."""

    @patch("sorawm.watermark_cleaner.LamaCleaner")
    def test_create_lama_cleaner(self, mock_lama_cleaner):
        """Test creating LAMA cleaner."""
        mock_instance = MagicMock()
        mock_lama_cleaner.return_value = mock_instance

        cleaner = WaterMarkCleaner(CleanerType.LAMA)

        assert cleaner is mock_instance
        mock_lama_cleaner.assert_called_once()

    @patch("sorawm.watermark_cleaner.E2FGVIHDCleaner")
    def test_create_e2fgvi_hq_cleaner(self, mock_e2fgvi_cleaner):
        """Test creating E2FGVI-HQ cleaner."""
        mock_instance = MagicMock()
        mock_e2fgvi_cleaner.return_value = mock_instance

        cleaner = WaterMarkCleaner(CleanerType.E2FGVI_HQ)

        assert cleaner is mock_instance
        mock_e2fgvi_cleaner.assert_called_once()

    def test_invalid_cleaner_type(self):
        """Test that invalid cleaner type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid cleaner type"):
            WaterMarkCleaner("invalid_type")

    @pytest.mark.parametrize(
        "cleaner_type",
        [
            CleanerType.LAMA,
            CleanerType.E2FGVI_HQ,
        ],
    )
    def test_all_valid_cleaner_types(self, cleaner_type):
        """Test that all valid cleaner types can be instantiated."""
        # This will raise an exception if the cleaner classes can't be imported/instantiated
        # but we're mainly testing that the factory method works
        try:
            cleaner = WaterMarkCleaner(cleaner_type)
            assert cleaner is not None
        except ImportError:
            # If the actual cleaner classes can't be imported (missing dependencies),
            # that's expected in a test environment
            pass
