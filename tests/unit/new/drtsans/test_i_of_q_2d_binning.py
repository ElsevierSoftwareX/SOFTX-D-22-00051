from drtsans.dataobjects import IQazimuthal

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/iq.py
from drtsans.iq import determine_1d_linear_bins, BinningMethod, bin_intensity_into_q2d

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/tests/unit/new/drtsans/i_of_q_binning_tests_data.py
from tests.unit.new.drtsans.i_of_q_binning_tests_data import (
    generate_test_data,
    generate_test_data_wavelength,
)
import numpy as np
import pytest

# This module supports testing data for issue #239.
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/245

# DEV - Wenduo Zhou <zhouw@ornl.gov>
# SME - William Heller <hellerwt@ornl.gov>

# All tests data are generated in tests.unit.new.drtsans.i_of_q_binning_tests_data
# Test EXCEL can be found at
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/5423db9b77dfd4911bf799c247530865/
# eqsans_tof_q_binning_tests_R5.xlsx


def test_2d_bin_no_sub_no_wt():
    """Test '2D_bin_no_sub_no_wt'

    2D linear bin no sub pixel with weighted and no-weight summation

    Returns
    -------
    None

    """
    # Calculate and determine the bin edges
    # range of binned (Qx, Qy) is taken from William's Excel
    qx_min = -0.007573828
    qx_max = 0.006825091
    qy_min = -0.005051412
    qy_max = 0.00607504

    qx_bins = determine_1d_linear_bins(qx_min, qx_max, 5)
    qy_bins = determine_1d_linear_bins(qy_min, qy_max, 5)

    # Bin 2D No-weight
    # Get Q1D data
    intensities, sigmas, qx_array, dqx_array, qy_array, dqy_array = generate_test_data(
        2, True
    )

    # Bin I(Qx, Qy) with no-weight binning algorithm
    test_i_q = IQazimuthal(
        intensity=intensities,
        error=sigmas,
        qx=qx_array,
        qy=qy_array,
        delta_qx=dqx_array,
        delta_qy=dqy_array,
    )
    binned_iq_2d = bin_intensity_into_q2d(
        test_i_q, qx_bins, qy_bins, BinningMethod.NOWEIGHT
    )

    # Verify Qx and Qy
    assert qx_bins.centers[1] == pytest.approx(
        -0.003254, abs=1.0e-6
    ), "Qx is not correct"
    assert qy_bins.centers[1] == pytest.approx(
        -0.001713, abs=1.0e-6
    ), "Qy is not correct"

    # verify I(-0.003254,-0.001713) and sigma(-0.003254,-0.001713)
    assert binned_iq_2d.intensity[1][1] == pytest.approx(
        67.0, abs=1e-6
    ), "I(Qx, Qy) is incorrect"
    assert binned_iq_2d.error[1][1] == pytest.approx(
        4.725815626, abs=1e-8
    ), "sigma I(Qx, Qy) is incorrect"

    # verify dQx and dQy
    assert binned_iq_2d.delta_qx[1][1] == pytest.approx(
        0.00816, abs=1e-5
    ), "dQx {} is incorrect comparing to {}." "".format(binned_iq_2d[2][1][1], 0.00816)
    assert binned_iq_2d.delta_qy[1][1] == pytest.approx(
        0.00816, abs=1e-5
    ), "dQy {}is incorrect comparing to {}." "".format(binned_iq_2d[3][1][1], 0.00816)

    # verify Qx and Qy on off diagonal values
    # Qx in row 0 shall be all same as qx bin center [1]
    assert binned_iq_2d.qx[0][1] == pytest.approx(
        qx_bins.centers[0], abs=1e-5
    ), "Qx[0, 1] {} shall be same as Qx bin center [1] {}".format(
        binned_iq_2d.qx[0][1], qx_bins.centers[0]
    )
    # Qx in row 1 shall be all same as qx bin center [0]
    assert binned_iq_2d.qx[1][0] == pytest.approx(
        qx_bins.centers[1], abs=1e-5
    ), "Qx[1, 0] {} shall be same as Qx bin center [1] {}".format(
        binned_iq_2d.qx[1][0], qx_bins.centers[1]
    )

    # Qy in col 0 shall be all same as qy bin center [0]
    assert binned_iq_2d.qy[1][0] == pytest.approx(
        qy_bins.centers[0], abs=1e-5
    ), "Qy[1, 0] {} shall be same as Qy bin center [0] {}".format(
        binned_iq_2d.qy[1][0], qy_bins.centers[0]
    )
    # Qy in col 1 shall be all same as qy bin center [1]

    assert binned_iq_2d.qy[0][1] == pytest.approx(
        qy_bins.centers[1], abs=1e-5
    ), "Qy[0, 1] {} shall be same as Qy bin center [1] {}".format(
        binned_iq_2d.qx[0][1], qy_bins.centers[1]
    )


def test_2d_bin_no_sub_no_wt_wavelength():
    """Test '2D_bin_no_sub_no_wt_wavelength'

    2D linear bin no sub pixel with weighted and no-weight summation
    multiple wavelengths

    Returns
    -------
    None

    """
    # Calculate and determine the bin edges
    # range of binned (Qx, Qy) is taken from William's Excel
    qx_min = -0.007573828
    qx_max = 0.006825091
    qy_min = -0.005051412
    qy_max = 0.00607504

    qx_bins = determine_1d_linear_bins(qx_min, qx_max, 5)
    qy_bins = determine_1d_linear_bins(qy_min, qy_max, 5)

    # Bin 2D No-weight
    # Get Q1D data
    (
        intensities,
        sigmas,
        qx_array,
        dqx_array,
        qy_array,
        dqy_array,
        wl_array,
    ) = generate_test_data_wavelength(2, 3)
    # Bin I(Qx, Qy) with no-weight binning algorithm
    test_i_q = IQazimuthal(
        intensity=intensities,
        error=sigmas,
        qx=qx_array,
        qy=qy_array,
        delta_qx=dqx_array,
        delta_qy=dqy_array,
        wavelength=wl_array,
    )

    # Bin
    binned_iq_2d = bin_intensity_into_q2d(
        test_i_q, qx_bins, qy_bins, BinningMethod.NOWEIGHT, wavelength_bins=None
    )

    # Verify size of output
    num_wl = np.unique(wl_array).size
    assert binned_iq_2d.intensity.size == 5 * 5 * num_wl, (
        f"Expected number of I(Qx, Qy) is "
        f"{5 * 5 * num_wl}; but the binned "
        f"intensities have {binned_iq_2d.intensity.size} "
        f"values"
    )

    # Verify Qx and Qy
    assert qx_bins.centers[1] == pytest.approx(
        -0.003254, abs=1.0e-6
    ), "Qx is not correct"
    assert qy_bins.centers[1] == pytest.approx(
        -0.001713, abs=1.0e-6
    ), "Qy is not correct"

    # verify I(-0.003254,-0.001713) and sigma(-0.003254,-0.001713)
    assert binned_iq_2d.intensity[1][1] == pytest.approx(
        67.0, abs=1e-6
    ), "I(Qx, Qy) is incorrect"
    assert binned_iq_2d.error[1][1] == pytest.approx(
        4.725815626, abs=1e-8
    ), "sigma I(Qx, Qy) is incorrect"

    # verify dQx and dQy
    assert binned_iq_2d.delta_qx[1][1] == pytest.approx(
        0.00816, abs=1e-5
    ), "dQx {} is incorrect comparing to {}." "".format(binned_iq_2d[2][1][1], 0.00816)
    assert binned_iq_2d.delta_qy[1][1] == pytest.approx(
        0.00816, abs=1e-5
    ), "dQy {}is incorrect comparing to {}." "".format(binned_iq_2d[3][1][1], 0.00816)

    # verify Qx and Qy on off diagonal values
    # Qx in row 0 shall be all same as qx bin center [1]
    assert binned_iq_2d.qx[0][1] == pytest.approx(
        qx_bins.centers[0], abs=1e-5
    ), "Qx[0, 1] {} shall be same as Qx bin center [1] {}".format(
        binned_iq_2d.qx[0][1], qx_bins.centers[0]
    )
    # Qx in row 1 shall be all same as qx bin center [0]
    assert binned_iq_2d.qx[1][0] == pytest.approx(
        qx_bins.centers[1], abs=1e-5
    ), "Qx[1, 0] {} shall be same as Qx bin center [1] {}".format(
        binned_iq_2d.qx[1][0], qx_bins.centers[1]
    )

    # Qy in col 0 shall be all same as qy bin center [0]
    assert binned_iq_2d.qy[1][0] == pytest.approx(
        qy_bins.centers[0], abs=1e-5
    ), "Qy[1, 0] {} shall be same as Qy bin center [0] {}".format(
        binned_iq_2d.qy[1][0], qy_bins.centers[0]
    )
    # Qy in col 1 shall be all same as qy bin center [1]
    assert binned_iq_2d.qy[0][1] == pytest.approx(
        qy_bins.centers[1], abs=1e-5
    ), "Qy[0, 1] {} shall be same as Qy bin center [1] {}".format(
        binned_iq_2d.qx[0][1], qy_bins.centers[1]
    )


def test_2d_bin_no_sub_wt():
    """Test '2D_bin_no_sub_wt'

    2D linear bin no sub pixel with weighted and no-weight summation

    Returns
    -------
    None

    """
    # Calculate and determine the bin edges
    # range of binned (Qx, Qy) is taken from William's Excel
    qx_min = -0.007573828
    qx_max = 0.006825091
    qy_min = -0.005051412
    qy_max = 0.00607504

    qx_bins = determine_1d_linear_bins(qx_min, qx_max, 5)
    qy_bins = determine_1d_linear_bins(qy_min, qy_max, 5)

    # Bin 2D No-weight
    # Get Q1D data
    intensities, sigmas, qx_array, dqx_array, qy_array, dqy_array = generate_test_data(
        2, True
    )

    # Bin I(Qx, Qy)
    test_i_q = IQazimuthal(
        intensity=intensities,
        error=sigmas,
        qx=qx_array,
        qy=qy_array,
        delta_qx=dqx_array,
        delta_qy=dqy_array,
    )
    binned_iq_2d = bin_intensity_into_q2d(
        test_i_q, qx_bins, qy_bins, BinningMethod.WEIGHTED
    )

    # verify I(-0.003254,-0.001713) and sigma(-0.003254,-0.001713)
    assert binned_iq_2d.intensity[1][1] == pytest.approx(
        56.8660, abs=1e-4
    ), "Weighted-binned I(Qx, Qy) is incorrect"
    assert binned_iq_2d.error[1][1] == pytest.approx(
        4.353773265, abs=1e-8
    ), "Weighted-binned sigma I(Qx, Qy) is incorrect"

    # verify dQx and dQy
    assert binned_iq_2d.delta_qx[1][1] == pytest.approx(
        0.00815, abs=1e-5
    ), "dQx is incorrect"
    assert binned_iq_2d.delta_qy[1][1] == pytest.approx(
        0.00815, abs=1e-5
    ), "dQy is incorrect"

    return


if __name__ == "__main__":
    pytest.main([__file__])
