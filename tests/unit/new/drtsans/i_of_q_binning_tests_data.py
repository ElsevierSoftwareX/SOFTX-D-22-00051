import numpy as np

# This module supports testing data for issue #239, #245, #246 and #247.
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/239
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/245
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/246
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/247
# DEV - Wenduo Zhou <zhouw@ornl.gov>
# SME - William Heller <hellerwt@ornl.gov>

# All tests data is from William's tests in eqsans_tof_q_binning_tests_R5.xlsx
# Intensities for a Generic 2D detector at 3 wavelengths
# Test EXCEL can be found at
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/5423db9b77dfd4911bf799c247530865/
# eqsans_tof_q_binning_tests_R5.xlsx

# The workspace is assumed to have 3 wavelengths
intensities_matrix = np.array(
    [
        [
            [93, 60, 89, 32, 97],
            [43, 61, 82, 97, 55],
            [78, 34, 50, 54, 67],
            [98, 88, 37, 92, 97],
            [72, 97, 100, 71, 39],
        ],  # 3.0 A
        [
            [76, 39, 51, 70, 61],
            [64, 54, 78, 35, 30],
            [67, 98, 100, 56, 79],
            [97, 35, 41, 90, 45],
            [30, 41, 68, 34, 51],
        ],  # 3.1 A
        [
            [78, 36, 46, 75, 91],
            [64, 56, 92, 73, 60],
            [74, 72, 69, 84, 87],
            [36, 78, 40, 68, 72],
            [59, 40, 39, 34, 85],
        ],  # 3.2A
    ]
)

uncertainties_matrix = np.sqrt(intensities_matrix)

# William's input Qx
qx_matrix = np.array(
    [
        [
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
        ],
        [
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
        ],
        [
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
        ],
    ]
)

# Williams's input Qy
qy_matrix = np.array(
    [
        [
            [0.004962, 0.004962, 0.004962, 0.004962, 0.004962],
            [0.002737, 0.002737, 0.002737, 0.002737, 0.002737],
            [0.000512, 0.000512, 0.000512, 0.000512, 0.000512],
            [-0.001713, -0.001713, -0.001713, -0.001713, -0.001713],
            [-0.003939, -0.003939, -0.003939, -0.003939, -0.003939],
        ],
        [
            [0.004802, 0.004802, 0.004802, 0.004802, 0.004802],
            [0.002649, 0.002649, 0.002649, 0.002649, 0.002649],
            [0.000495, 0.000495, 0.000495, 0.000495, 0.000495],
            [-0.001658, -0.001658, -0.001658, -0.001658, -0.001658],
            [-0.003812, -0.003812, -0.003812, -0.003812, -0.003812],
        ],
        [
            [0.004652, 0.004652, 0.004652, 0.004652, 0.004652],
            [0.002566, 0.002566, 0.002566, 0.002566, 0.002566],
            [0.000480, 0.000480, 0.000480, 0.000480, 0.000480],
            [-0.001606, -0.001606, -0.001606, -0.001606, -0.001606],
            [-0.003693, -0.003693, -0.003693, -0.003693, -0.003693],
        ],
    ]
)

# dQx is from rev4/tab 'Setup'/sigma_Q_x
dqx_matrix = np.array(
    [
        # 3.0 A
        [
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
        ],
        # 3.1 A
        [
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
        ],
        # 3.2A
        [
            [0.007896, 0.007896, 0.007896, 0.007896, 0.007896],
            [0.007896, 0.007896, 0.007896, 0.007896, 0.007896],
            [0.007897, 0.007896, 0.007896, 0.007896, 0.007896],
            [0.007897, 0.007896, 0.007896, 0.007896, 0.007896],
            [0.007896, 0.007896, 0.007896, 0.007896, 0.007896],
        ],
    ]
)

# dQy is from rev4/tab 'Setup'/sigma_Q_y
dqy_matrix = np.array(
    [
        # 3.0A
        [
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
            [0.008423, 0.008423, 0.008423, 0.008423, 0.008423],
        ],
        # 3.1 A
        [
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
            [0.008151, 0.008151, 0.008151, 0.008151, 0.008151],
        ],
        # 3.2 A
        [
            [0.007896, 0.007896, 0.007896, 0.007896, 0.007896],
            [0.007896, 0.007896, 0.007896, 0.007896, 0.007896],
            [0.007896, 0.007896, 0.007896, 0.007896, 0.007896],
            [0.007896, 0.007896, 0.007896, 0.007896, 0.007896],
            [0.007896, 0.007896, 0.007896, 0.007896, 0.007896],
        ],
    ]
)

# Scalar dQ matrix copied from revision 4's 1D_bin_linear_no_sub_no_wt and 1D_bin_log_no_sub_no_wt
scalar_dq_matrix = np.array(
    [
        # 3.0 A
        [
            [0.011912, 0.011912, 0.011912, 0.011912, 0.011912],
            [0.011912, 0.011912, 0.011912, 0.011912, 0.011912],
            [0.011912, 0.011912, 0.011912, 0.011912, 0.011912],
            [0.011912, 0.011912, 0.011912, 0.011912, 0.011912],
            [0.011912, 0.011912, 0.011912, 0.011912, 0.011912],
        ],
        # 3.1 A
        [
            [0.011527, 0.011527, 0.011527, 0.011527, 0.011527],
            [0.011527, 0.011527, 0.011527, 0.011527, 0.011527],
            [0.011527, 0.011527, 0.011527, 0.011527, 0.011527],
            [0.011527, 0.011527, 0.011527, 0.011527, 0.011527],
            [0.011527, 0.011527, 0.011527, 0.011527, 0.011527],
        ],
        # 3.2 A
        [
            [0.011167, 0.011167, 0.011167, 0.011167, 0.011167],
            [0.011167, 0.011167, 0.011167, 0.011167, 0.011167],
            [0.011167, 0.011167, 0.011167, 0.011167, 0.011167],
            [0.011167, 0.011167, 0.011167, 0.011167, 0.011167],
            [0.011167, 0.011167, 0.011167, 0.011167, 0.011167],
        ],
    ]
)


def generate_test_data(q_dimension, drt_standard):
    """Generate test data

    Test data including I(Q), Q and dQ depending on 1D or 2D

    Parameters
    ----------
    q_dimension : int
        Scalar Q or Qx, Qy
    drt_standard: bool
        flag to convert test 3D data (detector view + wave length) to drt standard 1D

    Returns
    -------
    ~tuple
        (intensity, sigma, q, dq) or (intensity, sigma, qx, dqx, qy, dqy)

    """
    # Check input: dimension must be either 1 or 2
    if q_dimension not in [1, 2]:
        raise RuntimeError("Q-dimension must be 1 or 2")

    if q_dimension == 1:
        # Calculate scalar Q
        scalar_q_matrix = np.sqrt(qx_matrix**2 + qy_matrix**2)
        # Scalar dQ is defined
    else:
        # No-op
        scalar_q_matrix = None

    # Define a None returning object
    returns = None
    if drt_standard:
        # Convert the Q, dQ, I, sigma I in 2D matrix to 1D arrays to match drtsans binning methods' requirements
        intensity_array = intensities_matrix.flatten()
        sigma_array = uncertainties_matrix.flatten()
        if q_dimension == 1:
            # Q1D: scalar Q and scalar dQ from matrix to 1D array
            scalar_q_array = scalar_q_matrix.flatten()
            scalar_dq_array = scalar_dq_matrix.flatten()
            returns = intensity_array, sigma_array, scalar_q_array, scalar_dq_array
        else:
            # Q2D: Qx, Qy, dQx and dQy from matrix to 1D array
            qx_array = qx_matrix.flatten()
            qy_array = qy_matrix.flatten()
            dqx_array = dqx_matrix.flatten()
            dqy_array = dqy_matrix.flatten()
            returns = (
                intensity_array,
                sigma_array,
                qx_array,
                dqx_array,
                qy_array,
                dqy_array,
            )
    else:
        # Raw matrix format
        if q_dimension == 1:
            # 1D: scalar Q
            returns = (
                intensities_matrix,
                uncertainties_matrix,
                scalar_q_matrix,
                scalar_dq_matrix,
            )
        elif q_dimension == 2:
            # 2D: Qx, Qy
            returns = (
                intensities_matrix,
                uncertainties_matrix,
                qx_matrix,
                dqx_matrix,
                qy_matrix,
                dqy_matrix,
            )

    return returns


def generate_test_data_wavelength(q_dimension, num_wavelengths):
    """

    Parameters
    ----------
    q_dimension
    num_wavelengths: int
        number of wavelength

    Returns
    -------
    ~tuple
        (intensity, sigma, q, dq, wavelength) or (intensity, sigma, qx, dqx, qy, dqy, wavelength)

    """
    assert isinstance(num_wavelengths, int) and num_wavelengths > 0, (
        "Number of wavelength must be greater" "than 0."
    )

    if q_dimension == 1:
        # get initial 1D arrays
        i_array, sigma_array, q_array, dq_array = generate_test_data(1, True)

        # size
        num_pts = len(i_array)

        # tile to number of wavelengths
        i_array = np.tile(i_array, num_wavelengths)
        sigma_array = np.tile(sigma_array, num_wavelengths)
        q_array = np.tile(q_array, num_wavelengths)
        dq_array = np.tile(dq_array, num_wavelengths)

        for i_wl in range(1, num_wavelengths):
            i_array[i_wl * num_pts : (1 + i_wl) * num_pts] *= i_wl + 1

        # repeat for wavelength: 1.5, 2.5, 3.5, ...
        wl_array = np.arange(num_wavelengths) * 1.0 + 1.5
        wl_array = np.repeat(wl_array, num_pts)

        returns = i_array, sigma_array, q_array, dq_array, wl_array

    elif q_dimension == 2:
        # get initial 2D arrays
        (
            i_array,
            sigma_array,
            qx_array,
            dqx_array,
            qy_array,
            dqy_array,
        ) = generate_test_data(2, True)

        # size
        num_pts = len(i_array)
        # tile to number of wavelengths
        i_array = np.tile(i_array, num_wavelengths)
        sigma_array = np.tile(sigma_array, num_wavelengths)
        qx_array = np.tile(qx_array, num_wavelengths)
        dqx_array = np.tile(dqx_array, num_wavelengths)
        qy_array = np.tile(qy_array, num_wavelengths)
        dqy_array = np.tile(dqy_array, num_wavelengths)

        # repeat for wavelength: 1.5, 2.5, 3.5, ...
        wl_array = np.arange(num_wavelengths) * 1.0 + 1.5
        wl_array = np.repeat(wl_array, num_pts)
        returns = (
            i_array,
            sigma_array,
            qx_array,
            dqx_array,
            qy_array,
            dqy_array,
            wl_array,
        )

    else:
        raise RuntimeError(f"Q dimension equal to {q_dimension} is not supported")

    return returns


def get_gold_1d_linear_bins():
    """Get the gold array for 1D linear bins

    This is to test the method to create linear bins

    Returns
    -------
    ndarray, ndarray
        bin edges, bin centers
    """
    # Create bin edges (N + 1)
    edge_array = np.array(
        [
            0.0000,
            0.0010,
            0.0020,
            0.0030,
            0.0040,
            0.0050,
            0.0060,
            0.0070,
            0.0080,
            0.0090,
            0.0100,
        ]
    )

    center_array = np.array(
        [0.0005, 0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0095]
    )

    return edge_array, center_array


def get_gold_1d_log_bins():
    """Get the gold array for 1D logarithm bins

    The gold data is re-generated due to the change in log bins calculation equation required by
    issue https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/issues/643

    This is to test the method to create logarithm bins with the setup as
    for bin centers: q_min = 0.001, q_max = 0.010, points density = 9
    match the previous one best (with 10 bins in total)

    Returns
    -------
    ndarray, ndarray
        bin edges, bin centers
    """
    edge_array = np.array(
        [
            0.000885,
            0.001115,
            0.001403,
            0.001767,
            0.002224,
            0.0028,
            0.003525,
            0.004437,
            0.005586,
            0.007033,
            0.008854,
            0.011146,
        ]
    )

    center_array = np.array(
        [
            0.001000,
            0.001259,
            0.001585,
            0.001995,
            0.002512,
            0.003162,
            0.003981,
            0.005012,
            0.006310,
            0.007943,
            0.0100,
        ]
    )

    return edge_array, center_array


def get_gold_2d_linear_bins():
    """Get the gold (Qx, Qy) bins

    Data are from William's Excel tests

    Returns
    -------
    ndafray, ndarray
        Qx centers, Qy centers
    """
    qx_center = np.array([-0.006134, -0.003254, -0.000374, 0.002505, 0.005385])

    # Qy shall increase monotonically
    qy_center = np.array([-0.003939, -0.001713, 0.000512, 0.002737, 0.004962])

    return qx_center, qy_center
