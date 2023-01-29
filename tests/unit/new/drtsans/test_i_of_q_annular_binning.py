from drtsans.dataobjects import IQazimuthal

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/iq.py
from drtsans.iq import BinningMethod, BinningParams, bin_annular_into_q1d

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/tests/unit/new/drtsans/i_of_q_binning_tests_data.py
from tests.unit.new.drtsans.i_of_q_binning_tests_data import generate_test_data
import numpy as np
import pytest


# This module supports testing data for issue #246.
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/246

# DEV - Wenduo Zhou <zhouw@ornl.gov>
# SME - William Heller <hellerwt@ornl.gov>

# All tests data are generated in tests.unit.new.drtsans.i_of_q_binning_tests_data


def test_1d_annular_no_wt():
    """Test annular binning I(Qx, Qy) with no-weight binning method

    The test data comes from example in '1D_annular_no_sub_no_wt' eqsans_tof_q_binning_tests_R5.xlsx
    File location: https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/5423db9b77dfd4911bf799c247530865/
                   eqsans_tof_q_binning_tests_R5.xlsx
    Returns
    -------
    None

    """
    # Initialize range of theta angle and Q
    theta_min = 0
    theta_max = 360.0
    num_bins = 10

    q_min = 0.003
    q_max = 0.006

    # Generate testing data: Get Q2D data
    intensities, sigmas, qx_array, dqx_array, qy_array, dqy_array = generate_test_data(
        2, True
    )

    # Test the high level method
    # Define input data
    test_i_q = IQazimuthal(
        intensity=intensities,
        error=sigmas,
        qx=qx_array,
        qy=qy_array,
        delta_qx=dqx_array,
        delta_qy=dqy_array,
    )

    # Annular binning
    theta_binning = BinningParams(theta_min, theta_max, num_bins)
    binned_iq = bin_annular_into_q1d(
        test_i_q, theta_binning, q_min, q_max, BinningMethod.NOWEIGHT
    )

    # Verify I(Q), sigma I(Q) and dQ
    assert binned_iq.intensity[1] == pytest.approx(
        63.66666667, abs=1e-8
    ), "Binned intensity is wrong"
    assert binned_iq.error[1] == pytest.approx(
        3.257470048, abs=1e-8
    ), "Binned sigma I is wrong"
    assert binned_iq.delta_mod_q[1] == pytest.approx(1.154e-02, abs=1e-5), (
        "Binned Q resolution {} "
        "is incorrect comparing to {}."
        "".format(binned_iq.delta_mod_q[1], 0.01154)
    )
    # this is actually theta
    np.testing.assert_almost_equal(
        binned_iq.mod_q, np.linspace(start=18, stop=theta_max - 18, num=num_bins)
    )


def test_1d_annular_out_of_range_angles():
    """Test annular binning I(Qx, Qy) supplying the azimuthal angle outside of 0<azimuthal<360deg

    The test data comes from example in '1D_annular_no_sub_no_wt' eqsans_tof_q_binning_tests_R5.xlsx
    File location: https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/5423db9b77dfd4911bf799c247530865/
                   eqsans_tof_q_binning_tests_R5.xlsx
    Returns
    -------
    None

    """
    # Initialize range of theta angle and Q
    theta_min = -90
    theta_max = 270.0
    num_bins = 10

    q_min = 0.003
    q_max = 0.006

    # Generate testing data: Get Q2D data
    intensities, sigmas, qx_array, dqx_array, qy_array, dqy_array = generate_test_data(
        2, True
    )

    # Test the high level method
    # Define input data
    test_i_q = IQazimuthal(
        intensity=intensities,
        error=sigmas,
        qx=qx_array,
        qy=qy_array,
        delta_qx=dqx_array,
        delta_qy=dqy_array,
    )

    # Annular binning
    theta_binning = BinningParams(theta_min, theta_max, num_bins)
    with pytest.raises(ValueError):
        bin_annular_into_q1d(
            test_i_q, theta_binning, q_min, q_max, BinningMethod.NOWEIGHT
        )


def test_1d_flat_data():
    # 2d array of ones with -10 <= Qx/Qy <= 10
    data2d = IQazimuthal(
        intensity=np.full((21, 21), 1.0, dtype=float),
        error=np.full((21, 21), 1.0, dtype=float),
        qx=np.arange(-10, 11, 1, dtype=float),
        qy=np.arange(-10, 11, 1, dtype=float),
    )

    # perform the annular binning
    theta_binning = BinningParams(0, 360, 18)  # 20 deg bins
    binned_iq = bin_annular_into_q1d(
        data2d, theta_binning, q_min=9.0, q_max=10.0, method=BinningMethod.NOWEIGHT
    )

    # verify the results
    assert binned_iq.intensity.size == 18
    np.testing.assert_equal(binned_iq.intensity, 1.0)
    # uncertainties are proportional to the number of input bins contributing to the annullar wedge
    np.testing.assert_allclose(binned_iq.error, 0.6, atol=0.11)
    # actually the azimuthal angle
    np.testing.assert_equal(binned_iq.mod_q, np.arange(10.0, 360.0, 20.0))


if __name__ == "__main__":
    pytest.main([__file__])
