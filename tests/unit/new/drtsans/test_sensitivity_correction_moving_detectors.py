import numpy as np

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/mono/gpsans/prepare_sensitivity.py
from drtsans.sensitivity_correction_moving_detectors import (
    prepare_sensitivity,
    _mask_zero_count_pixel,
)
import pytest

# This test implements issue #205 to verify
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/205
# DEV - Wenduo Zhou <petersonpf@ornl.gov> and Joe Osborn <osbornjd@ornl.gov>
# SME - William Heller <hellerwt@ornl.gov>, Lisa


# All testing data are from
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/906bfc358e1d6eb12a78439aef615f03/sensitivity_math.xlsx
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/fccd9ac3b9611acda0e4d9efe52ff4f4/
# Test_for_Preparing_Sensitivity_file_for_GP-SANS.pdf


def generate_test_data():
    """Generate test data, 3 flood runs and errors, from sensitivity_math.xlsx

    Returns
    -------
    np.ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
        Matrix A, error of A, Matrix B, error of B, Matrix C, error of C
    """
    # Hard code Lisa's input in the Excel file
    l_matrix_a = np.array([[99, 400, 98], [96, 95, np.nan], [97, 98, 36]])

    l_matrix_b = np.array([[86, 400, 89], [92, np.nan, 91], [95, 97, 36]])

    l_matrix_c = np.array([[99, 400, 93], [np.nan, 105, 94], [95, 99, 36]])

    # Calculate uncertainties
    error_matrix_a = np.sqrt(l_matrix_a)
    error_matrix_b = np.sqrt(l_matrix_b)
    error_matrix_c = np.sqrt(l_matrix_c)

    return (
        l_matrix_a,
        error_matrix_a,
        l_matrix_b,
        error_matrix_b,
        l_matrix_c,
        error_matrix_c,
    )


def get_final_sensitivities():
    """Get the final sensitivities and sigma

    Data is from sensitivity_math.xlsx as gold data to test methods to calculate non-normalized sensitivities
    From PDF: "Apply another weighted average and propate the error"

    Returns
    -------
    ndarray, ndarray

    """
    sen_matrix = np.array(
        [
            [9.91e-01, -np.inf, 9.78e-01],
            [9.93e-01, 1.03e00, 9.77e-01],
            [1.00e00, 1.03e00, -np.inf],
        ]
    )

    sen_sigma_matrix = np.array(
        [
            [6.78e-02, -np.inf, 6.73e-02],
            [8.14e-02, 8.26e-02, 8.06e-02],
            [6.83e-02, 6.94e-02, -np.inf],
        ]
    )

    return sen_matrix, sen_sigma_matrix


def normalize_by_monitor(flood_data, flood_data_error, monitor_counts):
    """Normalize the flood data field data by monitor

    Parameters
    ----------
    flood_data: ndarray
        flood data
    flood_data_error: ndarray
        flood data error
    monitor_counts: int/float
        monitor counts
    Returns
    -------
    ndarray, ndarray
        normalized flood data, normalized flood data error
    """
    return flood_data / monitor_counts, flood_data_error / monitor_counts


def test_prepare_moving_det_sensitivity():
    """Test main algorithm to prepare sensitivity for instrument with moving detector

    Returns
    -------
    None

    """
    # Set up the test data
    test_data_set = generate_test_data()
    monitor_a = 10
    monitor_b = 10
    monitor_c = 10
    threshold_min = 0.5
    threshold_max = 1.5

    # Normalize the flood field data by monitor: A, B and C
    matrix_a, sigma_a = test_data_set[0], test_data_set[1]
    matrix_a, sigma_a = normalize_by_monitor(matrix_a, sigma_a, monitor_a)

    matrix_b, sigma_b = test_data_set[2], test_data_set[3]
    matrix_b, sigma_b = normalize_by_monitor(matrix_b, sigma_b, monitor_b)

    matrix_c, sigma_c = test_data_set[4], test_data_set[5]
    matrix_c, sigma_c = normalize_by_monitor(matrix_c, sigma_c, monitor_c)

    # convert input data to required format
    # Prepare data
    flood_matrix = np.ndarray(shape=(3, matrix_a.size), dtype=float)
    flood_error = np.ndarray(shape=(3, matrix_a.size), dtype=float)

    flood_matrix[0] = matrix_a.flatten()
    flood_matrix[1] = matrix_b.flatten()
    flood_matrix[2] = matrix_c.flatten()

    flood_error[0] = sigma_a.flatten()
    flood_error[1] = sigma_b.flatten()
    flood_error[2] = sigma_c.flatten()

    # Test drtsans.mono.gpsans.prepare_sensitivity()
    test_sens_array, test_sens_sigma_array = prepare_sensitivity(
        flood_matrix, flood_error, threshold_min, threshold_max
    )

    # Get gold data as the sensitivities and error
    gold_final_sen_matrix, gold_final_sigma_matrix = get_final_sensitivities()

    # verify that the refactored high level method renders the same result from prototype
    # compare infinities and convert to NaN
    gold_sens_array = gold_final_sen_matrix.flatten()
    np.testing.assert_allclose(
        np.where(np.isinf(gold_sens_array))[0], np.where(np.isinf(test_sens_array))[0]
    )

    gold_sens_array[np.isinf(gold_sens_array)] = np.nan
    test_sens_array[np.isinf(test_sens_array)] = np.nan

    np.testing.assert_allclose(
        gold_sens_array,
        test_sens_array,
        rtol=1e-3,
        atol=5e-3,
        equal_nan=True,
        verbose=True,
    )

    np.testing.assert_allclose(
        gold_final_sigma_matrix.flatten(),
        test_sens_sigma_array,
        rtol=1e-3,
        atol=1e-3,
        equal_nan=True,
        verbose=True,
    )

    return


def test_mask_zero_pixels():
    """Test the method to mask pixels with zero counts

    Returns
    -------

    """
    # generate a (3 x 8) matrix
    test_data_matrix = np.arange(24).astype("float").reshape((3, 8))
    test_sigma_matrix = np.sqrt(test_data_matrix)

    # set some values to zero
    test_data_matrix[0, 3] = 0.0
    test_data_matrix[1, 5] = 0.0
    test_data_matrix[2, 6] = 0.0

    test_sigma_matrix[0, 3] = 1.0
    test_sigma_matrix[1, 5] = 1.0
    test_sigma_matrix[2, 6] = 1.0

    data_sum = np.sum(test_data_matrix)
    sigma_sum = np.sum(test_sigma_matrix)

    # mask
    _mask_zero_count_pixel(test_data_matrix, test_sigma_matrix)

    # verify
    assert test_data_matrix.shape == (3, 8) and test_sigma_matrix.shape == (
        3,
        8,
    ), "Data shape changed"

    # No zero
    assert (
        test_data_matrix[np.isfinite(test_data_matrix)].min() > 0.5
    ), "Minimum value is not zero"

    # 4 NaN
    assert len(np.where(np.isnan(test_data_matrix))[0]) == 4, "There shall be 4 NaNs"

    # Sum are same
    assert np.nansum(test_data_matrix) == data_sum
    assert np.nansum(test_sigma_matrix) == sigma_sum - 3


if __name__ == "__main__":
    pytest.main()
