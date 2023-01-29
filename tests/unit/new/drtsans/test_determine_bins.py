import numpy as np
import pytest
from drtsans.iq import determine_1d_linear_bins
from drtsans.determine_bins import determine_1d_log_bins
from tests.unit.new.drtsans.i_of_q_binning_tests_data import (
    get_gold_2d_linear_bins,
    get_gold_1d_log_bins,
)


# This module supports testing data for issue #239.
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/263

# DEV - Wenduo Zhou <zhouw@ornl.gov>
# SME - William Heller <hellerwt@ornl.gov>

# Some tests data are generated in tests.unit.new.drtsans.i_of_q_binning_tests_data
def no_more_supported_test_log_bins_backward_compatible():
    """Test log bins determination with 'old' API

    Method determine_1d_log_bins() has been refactored from its previous version by adding more
    method parameters.  While by default value, this method shall be backward compatible such that
    with x-min, x-max and step-per-decade defined, it shall generate a set of bins same as before.

    Here by using data '1D_bin_log_wedget_no_sub_no_wt' (from
    https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/5423db9b77dfd4911bf799c247530865/
    eqsans_tof_q_binning_tests_R5.xlsx), this method is tested

    Returns
    -------

    """
    # Define Q ran
    q_min = 0.001  # Edge
    q_max = 0.010  # Edge
    step_per_decade = 10  # 10 steps per decade

    log_bins = determine_1d_log_bins(q_min, q_max, True, step_per_decade)
    gold_edges, gold_centers = get_gold_1d_log_bins()
    np.testing.assert_allclose(log_bins.edges, gold_edges, rtol=5.0e-4)
    np.testing.assert_allclose(log_bins.centers, gold_centers, rtol=5.0e-4)

    return


# Test EXCEL can be found at
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/5423db9b77dfd4911bf799c247530865/
# eqsans_tof_q_binning_tests_R5.xlsx
def test_linear_bin_determination():
    """Test linear bin determination from '2D_bin_no_sub_no_wt'

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

    # verify
    gold_x_centers, gold_y_centers = get_gold_2d_linear_bins()

    np.testing.assert_allclose(qx_bins.centers, gold_x_centers, atol=5e-6)
    np.testing.assert_allclose(qy_bins.centers, gold_y_centers, atol=5e-6)

    # Check X
    assert qx_bins.edges[1] == pytest.approx(-0.004694044, abs=1e-8)
    assert qx_bins.edges[2] == pytest.approx(-0.001814261, abs=1e-8)
    # Check Y
    assert qy_bins.edges[1] == pytest.approx(-0.002826, abs=1e-6)
    assert qy_bins.edges[2] == pytest.approx(-0.000601, abs=1e-6)


# Tests are from https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/643
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads
# /8d01373d4ae5f582e90d22656ce87d7f/log_bin_definition_testsR2.xlsx
# log_bin_definition_testsR2.xlsx
# All tests' gold data below have 3 columns as bin's left boundary, center and right boundary,
# which are exactly same in the test Excel file
# Example 1:
expected_log_bin_example1 = np.array(
    [
        [0.001766558828858, 0.001995, 0.002224],
        [0.002223965801079, 0.002512, 0.002800],
        [0.00279980706194, 0.003162, 0.003525],
        [0.003524748258397, 0.003981, 0.004437],
        [0.004437395152673, 0.005012, 0.005586],
        [0.005586349519872, 0.006310, 0.007033],
        [0.007032797369731, 0.007943, 0.008854],
        [0.008853767324754, 0.010000, 0.011146],
        [0.011146232675246, 0.012589, 0.014032],
        [0.014032275560638, 0.015849, 0.017666],
        [0.017665588288585, 0.019953, 0.022240],
        [0.022239658010793, 0.025119, 0.027998],
        [0.027998070619399, 0.031623, 0.035247],
        [0.035247482583969, 0.039811, 0.044374],
    ]
)


def test_example1():
    """Example 1: decade on center and bin density are specified

    Returns
    -------

    """
    # user specified Q min
    q_min = 0.002
    # Q max is supposed to be calculated from instrument geometry, but is given here
    q_max = 0.036398139348163
    # number of bins per decade
    n_bins_per_decade = 10

    # Test drtsans.determine_bins.determine_1d_log_bins
    test_bins = determine_1d_log_bins(
        q_min, q_max, decade_on_center=True, n_bins_per_decade=n_bins_per_decade
    )

    # verify bin center
    np.testing.assert_allclose(
        test_bins.centers, expected_log_bin_example1[:, 1], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries min
    np.testing.assert_allclose(
        test_bins.edges[:-1], expected_log_bin_example1[:, 0], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries max
    np.testing.assert_allclose(
        test_bins.edges[1:], expected_log_bin_example1[:, 2], rtol=1e-7, atol=1e-6
    )


# Tests are from https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/643
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads
# /8d01373d4ae5f582e90d22656ce87d7f/log_bin_definition_testsR2.xlsx
# log_bin_definition_testsR2.xlsx
# All tests' gold data below have 3 columns as bin's left boundary, center and right boundary,
# which are exactly same in the test Excel file
# Example 2
expected_log_bin_example2 = np.array(
    [
        [0.001771, 0.002000, 0.002229],
        [0.002229, 0.002518, 0.002806],
        [0.002806, 0.003170, 0.003533],
        [0.003533, 0.003991, 0.004448],
        [0.004448, 0.005024, 0.005600],
        [0.005600, 0.006325, 0.007049],
        [0.007049, 0.007962, 0.008875],
        [0.008875, 0.010024, 0.011173],
        [0.011173, 0.012619, 0.014066],
        [0.014066, 0.015887, 0.017708],
        [0.017708, 0.020000, 0.022292],
        [0.022292, 0.025179, 0.028065],
        [0.028065, 0.031698, 0.035331],
        [0.035331, 0.039905, 0.044479],
    ]
)


def test_example2():
    """Example 2: bin density is specified but decade on center is not required

    Returns
    -------

    """
    # user specified Q min
    q_min = 0.002
    # Q max is supposed to be calculated from instrument geometry, but is given here
    q_max = 0.036398139348163
    # number of bins per decade
    n_bins_per_decade = 10

    # Test drtsans.determine_bins.determine_1d_log_bins
    test_bins = determine_1d_log_bins(
        q_min, q_max, decade_on_center=False, n_bins_per_decade=n_bins_per_decade
    )

    # verify bin center
    np.testing.assert_allclose(
        test_bins.centers, expected_log_bin_example2[:, 1], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries min
    np.testing.assert_allclose(
        test_bins.edges[:-1], expected_log_bin_example2[:, 0], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries max
    np.testing.assert_allclose(
        test_bins.edges[1:], expected_log_bin_example2[:, 2], rtol=1e-7, atol=1e-6
    )


# Tests are from https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/643
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads
# /8d01373d4ae5f582e90d22656ce87d7f/log_bin_definition_testsR2.xlsx
# log_bin_definition_testsR2.xlsx
# All tests' gold data below have 3 columns as bin's left boundary, center and right boundary,
# which are exactly same in the test Excel file
# Example 3
expected_log_bin_example3 = np.array(
    [
        [0.001418, 0.001500, 0.001582],
        [0.001582, 0.001674, 0.001766],
        [0.001766, 0.001869, 0.001972],
        [0.001972, 0.002086, 0.002201],
        [0.002201, 0.002329, 0.002457],
        [0.002457, 0.002599, 0.002742],
        [0.002742, 0.002902, 0.003061],
        [0.003061, 0.003239, 0.003417],
        [0.003417, 0.003615, 0.003814],
        [0.003814, 0.004036, 0.004257],
        [0.004257, 0.004505, 0.004752],
        [0.004752, 0.005028, 0.005305],
        [0.005305, 0.005613, 0.005921],
        [0.005921, 0.006265, 0.006610],
        [0.006610, 0.006994, 0.007378],
        [0.007378, 0.007807, 0.008235],
        [0.008235, 0.008714, 0.009193],
        [0.009193, 0.009727, 0.010261],
        [0.010261, 0.010858, 0.011454],
        [0.011454, 0.012120, 0.012786],
        [0.012786, 0.013529, 0.014272],
        [0.014272, 0.015101, 0.015931],
        [0.015931, 0.016857, 0.017783],
        [0.017783, 0.018816, 0.019850],
        [0.019850, 0.021003, 0.022157],
        [0.022157, 0.023445, 0.024733],
        [0.024733, 0.026170, 0.027608],
        [0.027608, 0.029212, 0.030817],
        [0.030817, 0.032608, 0.034399],
        [0.034399, 0.036398, 0.038397],
    ]
)


def test_example3():
    """Example 3: total number of bins are specified

    Returns
    -------

    """
    # user specified Q min
    q_min = 0.0015
    # Q max is supposed to be calculated from instrument geometry but is given here
    q_max = 0.036398139348163
    # number of bins per decade
    n_bins = 30

    # Test drtsans.determine_bins.determine_1d_log_bins
    test_bins = determine_1d_log_bins(
        q_min, q_max, decade_on_center=False, n_bins_per_decade=None, n_bins=n_bins
    )

    # verify bin center
    np.testing.assert_allclose(
        test_bins.centers, expected_log_bin_example3[:, 1], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries min
    np.testing.assert_allclose(
        test_bins.edges[:-1], expected_log_bin_example3[:, 0], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries max
    np.testing.assert_allclose(
        test_bins.edges[1:], expected_log_bin_example3[:, 2], rtol=1e-7, atol=1e-6
    )


# Tests are from https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/643
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads
# /8d01373d4ae5f582e90d22656ce87d7f/log_bin_definition_testsR2.xlsx
# log_bin_definition_testsR2.xlsx
# All tests' gold data below have 3 columns as bin's left boundary, center and right boundary,
# which are exactly same in the test Excel file
# Example 4
expected_log_bin_example4 = np.array(
    [
        [0.000403, 0.000437, 0.000470],
        [0.000470, 0.000509, 0.000547],
        [0.000547, 0.000592, 0.000638],
        [0.000638, 0.000690, 0.000743],
        [0.000743, 0.000804, 0.000865],
        [0.000865, 0.000936, 0.001008],
        [0.001008, 0.001091, 0.001174],
        [0.001174, 0.001270, 0.001367],
        [0.001367, 0.001479, 0.001592],
        [0.001592, 0.001723, 0.001854],
        [0.001854, 0.002007, 0.002160],
        [0.002160, 0.002338, 0.002516],
        [0.002516, 0.002723, 0.002930],
        [0.002930, 0.003172, 0.003413],
        [0.003413, 0.003694, 0.003975],
        [0.003975, 0.004303, 0.004630],
        [0.004630, 0.005012, 0.005393],
        [0.005393, 0.005838, 0.006282],
        [0.006282, 0.006799, 0.007317],
        [0.007317, 0.007920, 0.008523],
        [0.008523, 0.009225, 0.009927],
        [0.009927, 0.010744, 0.011562],
        [0.011562, 0.012515, 0.013467],
        [0.013467, 0.014577, 0.015686],
        [0.015686, 0.016978, 0.018271],
        [0.018271, 0.019776, 0.021281],
        [0.021281, 0.023034, 0.024787],
        [0.024787, 0.026829, 0.028871],
        [0.028871, 0.031249, 0.033628],
        [0.033628, 0.036398, 0.039168],
    ]
)


def test_example4():
    """Example 4: total number of bins is specified

    Returns
    -------

    """
    # user specified Q min
    q_min = 0.00043672649966
    # Q max is supposed to be calculated from instrument geometry, but is given here
    q_max = 0.036398139348163
    # number of bins per decade
    n_bins = 30

    # Test drtsans.determine_bins.determine_1d_log_bins
    test_bins = determine_1d_log_bins(
        q_min, q_max, decade_on_center=False, n_bins_per_decade=None, n_bins=n_bins
    )

    # verify bin center
    np.testing.assert_allclose(
        test_bins.centers, expected_log_bin_example4[:, 1], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries min
    np.testing.assert_allclose(
        test_bins.edges[:-1], expected_log_bin_example4[:, 0], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries max
    np.testing.assert_allclose(
        test_bins.edges[1:], expected_log_bin_example4[:, 2], rtol=1e-7, atol=1e-6
    )


# Tests are from https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/643
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads
# /8d01373d4ae5f582e90d22656ce87d7f/log_bin_definition_testsR2.xlsx
# log_bin_definition_testsR2.xlsx
# All tests' gold data below have 3 columns as bin's left boundary, center and right boundary,
# which are exactly same in the test Excel file
# Example 5
expected_log_bin_example5 = np.array(
    [
        [0.008854, 0.010000, 0.011146],
        [0.011146, 0.012589, 0.014032],
        [0.014032, 0.015849, 0.017666],
        [0.017666, 0.019953, 0.022240],
        [0.022240, 0.025119, 0.027998],
        [0.027998, 0.031623, 0.035247],
        [0.035247, 0.039811, 0.044374],
        [0.044374, 0.050119, 0.055863],
        [0.055863, 0.063096, 0.070328],
        [0.070328, 0.079433, 0.088538],
        [0.088538, 0.100000, 0.111462],
        [0.111462, 0.125893, 0.140323],
        [0.140323, 0.158489, 0.176656],
        [0.176656, 0.199526, 0.222397],
        [0.222397, 0.251189, 0.279981],
        [0.279981, 0.316228, 0.352475],
        [0.352475, 0.398107, 0.443740],
        [0.443740, 0.501187, 0.558635],
        [0.558635, 0.630957, 0.703280],
        [0.703280, 0.794328, 0.885377],
        [0.885377, 1.000000, 1.114623],
    ]
)


def test_example5():
    """Example 5: total number of bins is specified.  Q min and Q max are specified on decades.
    Thus bin centers have all the decades between Q min and Q max

    Returns
    -------

    """
    # Test data for both Example 1
    # user specified Q min
    q_min = 0.010
    # user specified Q max
    q_max = 1.000
    # number of bins per decade
    n_bins_per_decade = 10

    # Test drtsans.determine_bins.determine_1d_log_bins
    test_bins = determine_1d_log_bins(
        q_min, q_max, decade_on_center=False, n_bins_per_decade=n_bins_per_decade
    )

    # verify bin center
    np.testing.assert_allclose(
        test_bins.centers, expected_log_bin_example5[:, 1], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries min
    np.testing.assert_allclose(
        test_bins.edges[:-1], expected_log_bin_example5[:, 0], rtol=1e-7, atol=1e-6
    )
    # verify bin boundaries max
    np.testing.assert_allclose(
        test_bins.edges[1:], expected_log_bin_example5[:, 2], rtol=1e-7, atol=1e-6
    )


# A unit test from https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/issues/599
def test_issue599():
    """
    Use function in issue 599
    """
    q_min = 0.02
    q_max = 5
    n_bins_per_decade = 5
    delta = 1.0 / n_bins_per_decade
    logqmin = np.log10(q_min)
    logqmax = np.log10(q_max)
    logqmin = delta * np.floor(logqmin / delta)
    expected_values = 10 ** np.arange(logqmin, logqmax + delta * 0.999999, delta)
    test_bin = determine_1d_log_bins(
        q_min, q_max, n_bins_per_decade=n_bins_per_decade, decade_on_center=True
    )

    # verify that the Q min and Q max are in first and last bin
    assert test_bin.edges[0] < q_min < test_bin.edges[1]
    assert test_bin.edges[-1] > q_max > test_bin.edges[-2]
    # very bin centers with expected data
    np.testing.assert_allclose(
        test_bin.centers, expected_values[1:], rtol=1e-7, atol=1e-6
    )


if __name__ == "__main__":
    pytest.main([__file__])
