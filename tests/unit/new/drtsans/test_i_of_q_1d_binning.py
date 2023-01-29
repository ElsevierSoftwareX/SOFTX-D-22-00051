import numpy as np
from drtsans.dataobjects import IQmod

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/iq.py
from drtsans.determine_bins import Bins
from drtsans.iq import (
    determine_1d_linear_bins,
    determine_1d_log_bins,
    BinningMethod,
    bin_intensity_into_q1d,
)

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/tests/unit/new/drtsans/i_of_q_binning_tests_data.py
from tests.unit.new.drtsans.i_of_q_binning_tests_data import (
    generate_test_data,
    generate_test_data_wavelength,
    get_gold_1d_linear_bins,
    get_gold_1d_log_bins,
)
import pytest

# This module supports testing data for issue #239.
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/239

# DEV - Wenduo Zhou <zhouw@ornl.gov>
# SME - William Heller <hellerwt@ornl.gov>

# All tests data are generated in tests.unit.new.drtsans.i_of_q_binning_tests_data


def test_1d_bin_linear_no_wt():
    """Test case: '1D_bin_linear_no_sub_wt'

    Test methods for 1D linear no-weight binning

    Returns
    -------
    None

    """
    # From Tab '1D_bin_linear_no_sub_wt'
    q_min = 0.000
    q_max = 0.010
    num_bins = 10

    # Verify bin edges and bin center
    linear_bins = determine_1d_linear_bins(q_min, q_max, num_bins)
    gold_edges, gold_centers = get_gold_1d_linear_bins()

    np.testing.assert_allclose(linear_bins.edges, gold_edges, rtol=1.0e-12)
    np.testing.assert_allclose(linear_bins.centers, gold_centers, rtol=1.0e-12)

    # Get Q1D data
    intensities, sigmas, scalar_q_array, scalar_dq_array = generate_test_data(1, True)

    # Binned I(Q) no-weight
    # binned_iq = _do_1d_no_weight_binning(scalar_q_array, scalar_dq_array, intensities, sigmas,
    #                                      linear_bins.centers, linear_bins.edges)

    # Test high level method
    test_iq = IQmod(intensities, sigmas, scalar_q_array, scalar_dq_array, None)
    binned_iq = bin_intensity_into_q1d(test_iq, linear_bins, BinningMethod.NOWEIGHT)

    # Calculate and verify
    # I(0.0035) = 68.92857:    drtsans: 68.92857142857143
    # verify Q[3]
    assert (
        abs(binned_iq.mod_q[3] - 0.0035) < 1e-6
    ), "Q[3] {} shall be {} +/- 1e-6" "".format(binned_iq.delta_mod_q[3], 0.0035)
    # verify I[3]
    assert (
        abs(binned_iq.intensity[3] - 68.92857) < 1e-5
    ), "Intensity[3] shall be 68.92857 but not {}" "".format(binned_iq.intensity[3])
    # verify sigmaI[3] = 2.218889:
    assert abs(binned_iq.error[3] - 2.218889) < 1e-6, "error"
    # verify sigma_Q[3] = 1.154E-02
    assert binned_iq.delta_mod_q[3] == pytest.approx(
        1.154e-02, abs=2.0e-5
    ), "Linear binning: Q resolution {} does not match expected {}".format(
        binned_iq.delta_mod_q[3], 1.135e-02
    )


def test_1d_bin_log_no_wt():
    """Test '1D_bin_log_no_sub_no_wt'

    Test binning methods for 1D no-weight binning with log bins

    Returns
    -------

    """
    # Define Q range from tab '1D_bin_log_no_sub_no_wt' in r4
    q_min = 0.001  # center
    q_max = 0.010  # center
    num_steps_per_10 = 10  # 10 steps per decade

    # Verify bin edges and bin center
    log_bins = determine_1d_log_bins(
        q_min, q_max, decade_on_center=False, n_bins_per_decade=num_steps_per_10
    )
    gold_edges, gold_centers = get_gold_1d_log_bins()

    np.testing.assert_allclose(log_bins.edges, gold_edges, rtol=5.0e-4)
    np.testing.assert_allclose(log_bins.centers, gold_centers, rtol=5.0e-4)

    # Get Q1D data
    intensities, sigmas, scalar_q_array, scalar_dq_array = generate_test_data(1, True)

    # Test the high level method
    test_iq = IQmod(intensities, sigmas, scalar_q_array, scalar_dq_array)
    binned_iq = bin_intensity_into_q1d(test_iq, log_bins, BinningMethod.NOWEIGHT)

    # Verify: 2 I(Q) in bin: Q(3, 2, 3.1), Q(3, 2, 3.2)
    # I(0.0025) between (0.00222397, 0.00279981)
    # (previously) I(0.0022) = 70.00000
    assert binned_iq.intensity[4] == pytest.approx(
        74.333333333333333, abs=1.0e-12
    ), "intensity"
    # dI(0.0022) = 5.9160797831
    assert binned_iq.error[4] == pytest.approx(3.51978534699048, abs=1.0e-12), "error"
    # sigma_Q(0.0022) = 1.135E-02
    assert binned_iq.delta_mod_q[4] == pytest.approx(
        1.154e-2, abs=2.0e-5
    ), "Log binning: Q resolution {} does not match expected {}".format(
        binned_iq.delta_mod_q[3], 1.135e-02
    )


def test_1d_bin_linear_no_wt_no_wl():
    """Test case: linear binning 1D with no-weight binning and keep wavelength"""
    q_min = 0.000
    q_max = 0.010
    num_bins = 10

    # Verify bin edges and bin center
    linear_bins = determine_1d_linear_bins(q_min, q_max, num_bins)
    gold_edges, gold_centers = get_gold_1d_linear_bins()

    np.testing.assert_allclose(linear_bins.edges, gold_edges, rtol=1.0e-12)
    np.testing.assert_allclose(linear_bins.centers, gold_centers, rtol=1.0e-12)

    # Get Q1D data
    (
        intensities,
        sigmas,
        scalar_q_array,
        scalar_dq_array,
        wl_array,
    ) = generate_test_data_wavelength(1, 3)
    test_iq = IQmod(intensities, sigmas, scalar_q_array, scalar_dq_array, wl_array)

    # Binned I(Q) no-weight
    binned_iq_wl = bin_intensity_into_q1d(
        test_iq, linear_bins, BinningMethod.NOWEIGHT, wavelength_bins=None
    )
    binned_iq_no_wl = bin_intensity_into_q1d(
        test_iq, linear_bins, BinningMethod.NOWEIGHT
    )

    # Calculate and verify
    # Check size of output I(Q, wl) and I(Q)
    assert binned_iq_wl.intensity.shape[0] == 3 * binned_iq_no_wl.intensity.shape[0]
    num_base_bins = binned_iq_no_wl.intensity.shape[0]

    # I(0.0035) = 68.92857:    drtsans: 68.92857142857143
    # verify Q[3]
    assert (
        abs(binned_iq_wl.mod_q[3] - 0.0035) < 1e-6
    ), "Q[3] {} shall be {} +/- 1e-6" "".format(binned_iq_wl.delta_mod_q[3], 0.0035)
    assert abs(binned_iq_no_wl.mod_q[3] - 0.0035) < 1e-6, (
        f"Q[3] {binned_iq_wl.delta_mod_q[3]} shall be " f"{0.0035} +/- 1e-6"
    )

    # verify wavelength
    assert binned_iq_no_wl.wavelength is None
    assert binned_iq_wl.wavelength[3] == pytest.approx(1.5, 1e-5)

    # verify I[3]
    assert (
        abs(binned_iq_wl.intensity[3] - 68.92857) < 1e-5
    ), "Intensity[3] shall be 68.92857 but not {}" "".format(binned_iq_wl.intensity[3])
    assert binned_iq_wl.intensity[3 + num_base_bins] == pytest.approx(
        68.92857 * 2, 1e-6
    ), f"diff = {binned_iq_wl.intensity[3 + num_base_bins] - 68.92857 * 2}"
    # 3 wavelengths, 3 times of sample points, 6 times of total intensity (simple sum).
    # thus the binned intensity is increased by 6/3 = 2 times
    assert binned_iq_wl.intensity[3] * 2.0 == pytest.approx(
        binned_iq_no_wl.intensity[3], 1e-6
    ), f"diff = {binned_iq_wl.intensity[3] * 6. - binned_iq_no_wl.intensity[3]}"

    # verify sigmaI[3] = 2.218889:
    assert abs(binned_iq_wl.error[3] - 2.218889) < 1e-6, "error"
    # verify sigma_Q[3] = 1.154E-02
    assert binned_iq_wl.delta_mod_q[3] == pytest.approx(
        1.154e-02, abs=2.0e-5
    ), "Linear binning: Q resolution {} does not match expected {}".format(
        binned_iq_wl.delta_mod_q[3], 1.135e-02
    )


def test_1d_bin_wavelength():
    """Test binning wavelength while the target Q bins are the same as input
    I(Q)

    Returns
    -------

    """
    q_min = 0.000
    q_max = 0.010
    num_bins = 10

    # Verify bin edges and bin center
    linear_bins = determine_1d_linear_bins(q_min, q_max, num_bins)
    gold_edges, gold_centers = get_gold_1d_linear_bins()

    np.testing.assert_allclose(linear_bins.edges, gold_edges, rtol=1.0e-12)
    np.testing.assert_allclose(linear_bins.centers, gold_centers, rtol=1.0e-12)

    # Get Q1D data
    (
        intensities,
        sigmas,
        scalar_q_array,
        scalar_dq_array,
        wl_array,
    ) = generate_test_data_wavelength(1, 3)
    test_iq = IQmod(intensities, sigmas, scalar_q_array, scalar_dq_array, wl_array)

    # Bin I(Q) no-weight with summing wavelength as the correct result
    binned_iq_1step = bin_intensity_into_q1d(
        test_iq, linear_bins, BinningMethod.NOWEIGHT, wavelength_bins=1
    )

    # Bin I(Q) no-weight but not wavelength
    binned_iq_wl = bin_intensity_into_q1d(
        test_iq, linear_bins, BinningMethod.NOWEIGHT, wavelength_bins=None
    )

    # Check NaN
    nan_intensities = np.where(np.isnan(binned_iq_wl.intensity))[0]
    nan_errors = np.where(np.isnan(binned_iq_wl.error))[0]
    assert (
        len(nan_intensities) == 6
    ), f"Expected {6} NaN but got {len(nan_errors)} instead"
    np.testing.assert_allclose(nan_intensities, nan_errors)

    print(f"Number of  intensities = {binned_iq_wl.intensity.shape}")

    #  Remove NaN
    finite_binned_iq_wl = binned_iq_wl.be_finite()

    # Bin I(Q, wavelength) again with same bins
    binned_iq_no_wl = bin_intensity_into_q1d(
        finite_binned_iq_wl, linear_bins, BinningMethod.NOWEIGHT, wavelength_bins=1
    )

    print(f"Number of final binned I(Q): {binned_iq_no_wl.intensity.shape}")

    print("I(Q, wavelength)")
    for i in range(24):
        print(
            f"{i}:  {finite_binned_iq_wl.mod_q[i]:.5f}    {finite_binned_iq_wl.wavelength[i]:.5f}    "
            f"{finite_binned_iq_wl.intensity[i]}"
        )

    # Verify with 1-step binning
    np.testing.assert_allclose(binned_iq_no_wl.mod_q, binned_iq_1step.mod_q)
    np.testing.assert_allclose(binned_iq_no_wl.intensity, binned_iq_1step.intensity)
    np.testing.assert_allclose(binned_iq_no_wl.error, binned_iq_1step.error)
    np.testing.assert_allclose(binned_iq_no_wl.delta_mod_q, binned_iq_1step.delta_mod_q)


def test_1d_weighted_binning():
    """Test 1D weighted binning

    It is expected that weighted binning will make these two approach yield the same results
    - do weighted binning with 1 wavelength bin
    - do weighted binning for each wavelength bin and then do binning on wave length bin to 1 wave length bin

    Returns
    -------

    """
    # Generate test data
    test_qiew_array = np.array(
        [
            [0.013, np.nan, np.nan, 1.0, 0.0005],
            [0.021, np.nan, np.nan, 1.0, 0.0007],
            [0.032, np.nan, np.nan, 1.0, 0.0010],
            [0.041, 10.000, 1.1000, 1.0, 0.0002],
            [0.048, 10.000, 1.5000, 1.0, 0.0001],
            [0.060, 10.000, 1.1300, 1.0, 0.0004],
            [0.070, 10.000, 2.3000, 1.0, 0.0003],
            [0.080, 10.000, 0.7000, 1.0, 0.0006],
            [0.090, 10.000, 0.8000, 1.0, 0.0009],
            [0.099, 10.000, 1.2000, 1.0, 0.00035],
            [0.011, np.nan, np.nan, 2.0, 0.00015],
            [0.021, 12.000, 2.3000, 2.0, 0.0002],
            [0.031, 12.000, 1.5000, 2.0, 0.0006],
            [0.039, 12.000, 1.7000, 2.0, 0.0004],
            [0.049, 12.000, 0.5000, 2.0, 0.0005],
            [0.062, 12.000, 2.4000, 2.0, 0.0006],
            [0.073, 12.000, 0.6000, 2.0, 0.0007],
            [0.079, 12.000, 0.9000, 2.0, 0.0006],
            [0.091, 12.000, 1.3000, 2.0, 0.0009],
            [0.099, 12.000, 1.4000, 2.0, 0.00015],
        ]
    )

    test_iq_1d = IQmod(
        intensity=test_qiew_array[:, 1],
        error=test_qiew_array[:, 2],
        mod_q=test_qiew_array[:, 0],
        delta_mod_q=test_qiew_array[:, 4],
        wavelength=test_qiew_array[:, 3],
    )

    # Generate binning
    test_bins = {
        "edges": np.array([0.0, 0.05, 0.10]),
        "centers": np.array([0.025, 0.075]),
    }
    test_bins = Bins(**test_bins)

    # Expected binned I(Q, wl)
    expected_binned_qiew = np.array(
        [
            [0.025, 10, 0.887045495441276, 1.0, 0.00016502890173410406],
            [0.075, 10, 0.435609182296287, 1.0, 0.0006155217563843883],
            [0.025, 12, 0.448133161649441, 2.0, 0.0004905877304625013],
            [0.075, 12, 0.434869885395938, 2.0, 0.0006426826759447178],
        ]
    )

    # Expected binned I(Q)
    expected_binned_qie = np.array(
        [
            [0.025, 11.5933404636534, 0.399987461455165, 0.00042439192929037884],
            [0.075, 11.0016985965419, 0.307760492837406, 0.0006291252838865729],
        ]
    )

    # Clean NaN
    finite_test_iq_1d = test_iq_1d.be_finite()

    # Do weighted binning on all wavelength
    binned_iq_all_wl = bin_intensity_into_q1d(
        i_of_q=finite_test_iq_1d,
        q_bins=test_bins,
        bin_method=BinningMethod.WEIGHTED,
        wavelength_bins=1,
    )

    np.testing.assert_allclose(binned_iq_all_wl.mod_q, expected_binned_qie[:, 0])
    np.testing.assert_allclose(binned_iq_all_wl.intensity, expected_binned_qie[:, 1])
    np.testing.assert_allclose(binned_iq_all_wl.error, expected_binned_qie[:, 2])
    np.testing.assert_allclose(binned_iq_all_wl.delta_mod_q, expected_binned_qie[:, 3])

    # Do weighted binning on each wavelength
    binned_iq_per_wl = bin_intensity_into_q1d(
        i_of_q=finite_test_iq_1d,
        q_bins=test_bins,
        bin_method=BinningMethod.WEIGHTED,
        wavelength_bins=None,
    )
    assert binned_iq_per_wl.wavelength is not None

    np.testing.assert_allclose(binned_iq_per_wl.mod_q, expected_binned_qiew[:, 0])
    np.testing.assert_allclose(binned_iq_per_wl.intensity, expected_binned_qiew[:, 1])
    np.testing.assert_allclose(binned_iq_per_wl.error, expected_binned_qiew[:, 2])
    np.testing.assert_allclose(binned_iq_per_wl.wavelength, expected_binned_qiew[:, 3])
    np.testing.assert_allclose(binned_iq_per_wl.delta_mod_q, expected_binned_qiew[:, 4])

    # Do weighted binning on Q-binned I(Q, wl)
    binned_iq_all_wl = bin_intensity_into_q1d(
        binned_iq_per_wl, test_bins, BinningMethod.WEIGHTED, 1
    )

    np.testing.assert_allclose(binned_iq_all_wl.mod_q, expected_binned_qie[:, 0])
    np.testing.assert_allclose(binned_iq_all_wl.intensity, expected_binned_qie[:, 1])
    np.testing.assert_allclose(binned_iq_all_wl.error, expected_binned_qie[:, 2])
    np.testing.assert_allclose(binned_iq_all_wl.delta_mod_q, expected_binned_qie[:, 3])


if __name__ == "__main__":
    pytest.main([__file__])
