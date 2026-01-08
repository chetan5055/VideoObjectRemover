import numpy as np
import pytest

from sorawm.utils.imputation_utils import (
    find_2d_data_bkps,
    find_idxs_interval,
    get_interval_average_bbox,
    refine_bkps_by_chunk_size,
)


class TestImputationUtils:
    """Test cases for imputation utility functions."""

    def test_find_2d_data_bkps_normal_data(self):
        """Test breakpoint detection with normal 2D data."""
        # Create test data with two segments
        segment1 = [(10, 20), (11, 21), (12, 22), (13, 23), (14, 24)]
        segment2 = [(50, 60), (51, 61), (52, 62), (53, 63), (54, 64)]
        data = segment1 + segment2

        bkps = find_2d_data_bkps(data)

        # Should detect the change point around index 5
        assert len(bkps) > 0
        # The exact breakpoint may vary due to the algorithm, but should be around the transition

    def test_find_2d_data_bkps_with_none_values(self):
        """Test breakpoint detection with None/missing values."""
        # Create data with missing values that should be interpolated
        data = [(10, 20), None, (12, 22), None, (50, 60), (51, 61)]

        bkps = find_2d_data_bkps(data)

        # Should handle None values gracefully
        assert isinstance(bkps, list)
        assert all(isinstance(bkp, int) for bkp in bkps)

    def test_find_2d_data_bkps_empty_data(self):
        """Test breakpoint detection with empty data."""
        with pytest.raises(
            Exception
        ):  # ruptures may raise an exception for insufficient data
            find_2d_data_bkps([])

    def test_get_interval_average_bbox_normal_case(self):
        """Test bbox averaging for intervals."""
        bboxes = [
            (10, 20, 30, 40),  # interval 1
            (11, 21, 31, 41),
            None,
            (50, 60, 70, 80),  # interval 2
            (51, 61, 71, 81),
        ]
        bkps = [0, 3, 5]

        averages = get_interval_average_bbox(bboxes, bkps)

        assert len(averages) == 2  # Two intervals

        # First interval average (should average first 3 bboxes, ignoring None)
        avg1 = averages[0]
        assert avg1 is not None
        assert isinstance(avg1, tuple)
        assert len(avg1) == 4

        # Second interval average
        avg2 = averages[1]
        assert avg2 is not None

    def test_get_interval_average_bbox_no_valid_bboxes(self):
        """Test bbox averaging when interval has no valid bboxes."""
        bboxes = [None, None, None, (10, 20, 30, 40), (11, 21, 31, 41)]
        bkps = [0, 3, 5]

        averages = get_interval_average_bbox(bboxes, bkps)

        assert len(averages) == 2
        assert averages[0] is None  # First interval has no valid bboxes
        assert averages[1] is not None  # Second interval has valid bboxes

    def test_find_idxs_interval_normal_case(self):
        """Test finding interval indices for given indices."""
        bkps = [0, 5, 10, 15]
        idxs = [2, 7, 12]

        intervals = find_idxs_interval(idxs, bkps)

        assert len(intervals) == 3
        assert intervals[0] == 0  # idx 2 is in interval [0, 5)
        assert intervals[1] == 1  # idx 7 is in interval [5, 10)
        assert intervals[2] == 2  # idx 12 is in interval [10, 15)

    def test_find_idxs_interval_boundary_cases(self):
        """Test interval finding at boundaries."""
        bkps = [0, 5, 10]
        idxs = [0, 4, 5, 9, 10]

        intervals = find_idxs_interval(idxs, bkps)

        assert intervals[0] == 0  # idx 0 is in interval [0, 5)
        assert intervals[1] == 0  # idx 4 is in interval [0, 5)
        assert intervals[2] == 1  # idx 5 is in interval [5, 10)
        assert intervals[3] == 1  # idx 9 is in interval [5, 10)
        assert intervals[4] == 1  # idx 10 is clamped to last interval

    def test_find_idxs_interval_out_of_bounds(self):
        """Test interval finding for out-of-bounds indices."""
        bkps = [0, 5, 10]
        idxs = [-1, 15]

        intervals = find_idxs_interval(idxs, bkps)

        assert intervals[0] == 0  # Clamped to first interval
        assert intervals[1] == 1  # Clamped to last interval

    def test_refine_bkps_by_chunk_size_normal_case(self):
        """Test refining breakpoints by chunk size."""
        bkps = [0, 10, 20]
        chunk_size = 3

        refined = refine_bkps_by_chunk_size(bkps, chunk_size)

        # Should include original breakpoints and intermediate points
        assert 0 in refined
        assert 10 in refined
        assert 20 in refined

        # Should be sorted
        assert refined == sorted(refined)

        # Should include intermediate points
        assert 3 in refined  # 0 + 3
        assert 6 in refined  # 0 + 6
        assert 9 in refined  # 0 + 9
        assert 13 in refined  # 10 + 3
        assert 16 in refined  # 10 + 6
        assert 19 in refined  # 10 + 9

    def test_refine_bkps_by_chunk_size_small_chunk(self):
        """Test refining with small chunk size."""
        bkps = [0, 5]
        chunk_size = 1

        refined = refine_bkps_by_chunk_size(bkps, chunk_size)

        expected = [0, 1, 2, 3, 4, 5]
        assert refined == expected

    def test_refine_bkps_by_chunk_size_large_chunk(self):
        """Test refining with chunk size larger than intervals."""
        bkps = [0, 3]
        chunk_size = 5

        refined = refine_bkps_by_chunk_size(bkps, chunk_size)

        # Should still include all original breakpoints
        assert refined == [0, 3]
