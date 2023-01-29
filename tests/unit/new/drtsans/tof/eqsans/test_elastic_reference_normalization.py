import pytest
from drtsans.dataobjects import IQmod

from drtsans.tof.eqsans.elastic_reference_normalization import (
    determine_common_mod_q_range_mesh,
)
from drtsans.tof.eqsans.elastic_reference_normalization import (
    calculate_scale_factor_mesh_grid,
)
from drtsans.tof.eqsans.elastic_reference_normalization import (
    reshape_q_wavelength_matrix,
)
from drtsans.tof.eqsans.elastic_reference_normalization import (
    determine_reference_wavelength_q1d_mesh,
)
from drtsans.tof.eqsans.elastic_reference_normalization import normalize_intensity_q1d
from drtsans.tof.eqsans.elastic_reference_normalization import (
    normalize_by_elastic_reference,
)
import numpy as np


def test_reshaped_imodq_2d():
    """Test the method to reshape I(Q) to 2D  mesh grid:"""
    # Get test data
    test_data = create_testing_iq1d()
    i_of_q = test_data[0]

    # Reshape
    wl_vec, q_vec, i_array, error_array, dq_array = reshape_q_wavelength_matrix(i_of_q)

    # Verify shape
    assert wl_vec.shape == (4,)
    assert q_vec.shape == (20,)
    assert i_array.shape == (20, 4)
    assert error_array.shape == (20, 4)

    # Verify value
    # intensity from q = 0.07 to 0.12
    gold_intensity_array = np.array(
        [
            [28.727, 31.600, 34.473, 25.855],
            [18.262, 20.088, 21.914, 16.435],
            [12.076, 13.283, 14.491, 10.868],
            [8.264, 9.091, 9.917, 7.438],
            [5.827, 6.410, 6.993, 5.244],
        ]
    )
    for i_q in range(6, 11):
        np.testing.assert_allclose(i_array[i_q], gold_intensity_array[i_q - 6])
    assert i_array[19, 0] == 0.595

    # error for wl = 3
    wl3_errors = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            5.360,
            4.273,
            3.475,
            2.875,
            2.414,
            2.053,
            1.767,
            1.535,
            1.346,
            1.189,
            1.058,
            0.947,
            0.852,
            0.771,
        ]
    )
    np.testing.assert_allclose(error_array[:, 0], wl3_errors, equal_nan=True)
    assert error_array[0, 3] == 27.273
    assert error_array[16, 1] == 1.028


def test_determine_common_q_range():
    """Test method to determine q range common to all wavelength"""
    # Get test data
    test_data = create_testing_iq1d()
    i_of_q = test_data[0]

    # Reshape
    wl_vec, q_vec, i_array, error_array, dq_array = reshape_q_wavelength_matrix(i_of_q)

    # Call the general algorithm
    q_min, q_max = determine_common_mod_q_range(i_of_q)

    # Call the mesh-grid algorithm
    qmin_index, qmax_index = determine_common_mod_q_range_mesh(q_vec, i_array)

    # Verify by comparing results from 2 algorithms
    assert q_min == q_vec[qmin_index]
    assert q_max == q_vec[qmax_index]


def test_determine_reference_wavelength():
    """Test method to determine the reference wavelength's intensity and error
    for each momentum transfer Q1D
    """
    # Get test data
    test_data = create_testing_iq1d()
    i_of_q = test_data[0]

    # Reshape
    wl_vec, q_vec, i_array, error_array, dq_array = reshape_q_wavelength_matrix(i_of_q)

    # Calculate reference wavelength and the related intensities and errors
    ref_wl_intensities = determine_reference_wavelength_q1d_mesh(
        wl_vec, q_vec, i_array, error_array, qmin_index=6, qmax_index=10
    )

    # Verify
    # Must be all at wl = 3.0
    np.testing.assert_allclose(
        ref_wl_intensities.ref_wl_vec, np.zeros_like(q_vec) + 3.0
    )
    np.testing.assert_allclose(
        ref_wl_intensities.intensity_vec[6:11],
        np.array([28.727, 18.262, 12.076, 8.264, 5.827]),
    )


def test_determine_q_range():
    """Test the method to determine the common q-range among all the wavelength vectors

    Test data
        - L wavelength, 1., 2., 3.,
        - N Q on regular grids for each wavelength
        - For each wavelength, there are P (P <= N) consecutive arbitrary finite values
        WL/Q   q1  q2  q3  ... ... q_N
        wl1    nan v12 v13 ... ... nan
        wl2    nan nan v23 ... ....
        ...
        wl_L
    """
    # Create test data set and manually set the range of valid wavelength for each Q
    num_wl = 10
    num_q = 100
    vec_intensity, vec_error, vec_q, vec_wl = create_configurable_testing_iq1d(
        num_q, num_wl
    )
    # set wl index = 1 with min q = 300, max q = 70000
    vec_intensity[num_q : num_q + 2] = np.nan
    vec_intensity[num_q * 2 - 3 : num_q * 2] = np.nan
    # create the IQmod
    test_i_q1d = IQmod(
        intensity=vec_intensity, error=vec_error, mod_q=vec_q, wavelength=vec_wl
    )

    min_q, max_q = determine_common_mod_q_range(test_i_q1d)

    assert min_q == 300.0
    assert max_q == 9700.0


def test_determine_reference_wavelength_q1d():
    """Test method to determine the reference wavelength for each Q in I(Q, wavelength)

    Test data
        - L wavelength, 1., 2., 3.,
        - N Q on regular grids for each wavelength
        - For each wavelength, there are P (P <= N) consecutive arbitrary finite values
        WL/Q   q1  q2  q3  ... ... q_N
        wl1    nan nan nan ... ... nan
        wl2    nan nan v13 ... ....
        wl3    nan v32 v33
        wl4    v41 v42 v43 ... ..
        wl_L
    """
    # Create test data set and manually set the range of valid wavelength for each Q
    num_wl = 10
    num_q = 100
    vec_intensity, vec_error, vec_q, vec_wl = create_configurable_testing_iq1d(
        num_q, num_wl
    )
    # set minimum wavelength range: first 3 from wl_i, such as (wl_1, wl_2, wl_3)
    # q1, q2, q3
    vec_intensity[0:3] = np.nan
    vec_intensity[num_q : num_q + 2] = np.nan
    vec_intensity[num_q * 2 : num_q * 2 + 1] = np.nan

    # create the IQmod
    test_i_q1d = IQmod(
        intensity=vec_intensity, error=vec_error, mod_q=vec_q, wavelength=vec_wl
    )

    # Call method to calculate reference
    test_ref_lambda_vec = prototype_determine_reference_wavelength_q1d(test_i_q1d)

    # Test: shape... (num Q, 2) as Q and wavelength
    assert test_ref_lambda_vec.shape == (num_q, 4)

    # Verify the values
    np.testing.assert_allclose(test_ref_lambda_vec[0][0], 100.0, rtol=1e-4)
    np.testing.assert_allclose(test_ref_lambda_vec[0][1], 0.4, rtol=1e-4)
    np.testing.assert_allclose(test_ref_lambda_vec[1][1], 0.3, rtol=1e-4)
    np.testing.assert_allclose(test_ref_lambda_vec[2][1], 0.2, rtol=1e-4)
    for i in range(3, num_q):
        assert test_ref_lambda_vec[i][1] == 0.1, (
            f"{i}-th Q has a wrong reference wavelength " f"{test_ref_lambda_vec[i]}"
        )


def test_calculate_scale_factor():
    """Test the method to calculate scale factor K(wavelength) and delta K(wavelength)"""
    # Get testing data and gold data
    test_i_of_q, gold_k_vec, gold_intensity_vec, gold_error_vec = create_testing_iq1d()
    # Reshape
    wl_vec, q_vec, i_array, error_array, dq_array = reshape_q_wavelength_matrix(
        test_i_of_q
    )

    # Calculate Qmin and Qmax
    q_min, q_max = determine_common_mod_q_range(test_i_of_q)
    qmin_index, qmax_index = determine_common_mod_q_range_mesh(q_vec, i_array)

    # Calculate reference
    ref_wl_ie = determine_reference_wavelength_q1d_mesh(
        wl_vec, q_vec, i_array, error_array, qmin_index, qmax_index
    )

    # Calculate scale factor
    (
        k_vec,
        delta_k_vec,
        p_vec,
        s_vec,
        unique_wl_vec,
        ref_q_wl_vec,
    ) = calculate_scale_factor_prototype(test_i_of_q, q_min, q_max)
    mk_vec, mk_error_vec, mp_vec, ms_vec = calculate_scale_factor_mesh_grid(
        wl_vec, i_array, error_array, ref_wl_ie, qmin_index, qmax_index
    )

    np.testing.assert_allclose(k_vec, mk_vec)
    np.testing.assert_allclose(p_vec, mp_vec)
    np.testing.assert_allclose(s_vec, ms_vec)
    np.testing.assert_allclose(delta_k_vec, mk_error_vec)


def test_workflow_q1d():
    """Test method normalize_by_elastic_reference"""
    # Get testing data and gold data
    test_i_of_q, gold_k_vec, gold_intensity_vec, gold_error_vec = create_testing_iq1d()

    # Normalize
    normalized_iq1d, k_vec, delta_k_vec = normalize_by_elastic_reference(
        test_i_of_q, ref_i_of_q=test_i_of_q
    )

    # Verify
    np.testing.assert_allclose(normalized_iq1d.mod_q, test_i_of_q.mod_q, rtol=1e-6)
    np.testing.assert_allclose(
        normalized_iq1d.wavelength, test_i_of_q.wavelength, rtol=1e-6
    )
    np.testing.assert_allclose(normalized_iq1d.intensity, gold_intensity_vec, 1e-3)
    np.testing.assert_allclose(normalized_iq1d.error, gold_error_vec, 1e-3)


def test_normalize_i_of_q1d():
    """Test the refined method to normalize I(Q1D)"""
    # Get testing data and gold data
    test_i_of_q, gold_k_vec, gold_intensity_vec, gold_error_vec = create_testing_iq1d()
    # Reshape
    wl_vec, q_vec, i_array, error_array, dq_array = reshape_q_wavelength_matrix(
        test_i_of_q
    )

    # Calculate Qmin and Qmax
    qmin_index, qmax_index = determine_common_mod_q_range_mesh(q_vec, i_array)

    # Calculate reference
    ref_wl_ie = determine_reference_wavelength_q1d_mesh(
        wl_vec, q_vec, i_array, error_array, qmin_index, qmax_index
    )

    # Calculate scale factor
    mk_vec, mk_error_vec, mp_vec, ms_vec = calculate_scale_factor_mesh_grid(
        wl_vec, i_array, error_array, ref_wl_ie, qmin_index, qmax_index
    )

    # Normalize
    normalized = normalize_intensity_q1d(
        wl_vec,
        q_vec,
        i_array,
        error_array,
        ref_wl_ie,
        gold_k_vec,
        mp_vec,
        ms_vec,
        qmin_index,
        qmax_index,
    )
    normalized_intensity = normalized[0].flatten()
    normalized_error = normalized[1].flatten()

    np.testing.assert_allclose(
        normalized_intensity, gold_intensity_vec, rtol=8e-4, equal_nan=True
    )
    np.testing.assert_allclose(
        normalized_error, gold_error_vec, rtol=1e-3, equal_nan=True
    )


def test_normalize_i_of_q1d_prototype():
    """Test method to calculate scale factor K(lambda) and error delta K(lambda)

    Test data: using data from attached Excel
    """
    # Get testing data and gold data
    test_i_of_q, gold_k_vec, gold_intensity_vec, gold_error_vec = create_testing_iq1d()

    # Calculate Qmin and Qmax
    q_min, q_max = determine_common_mod_q_range(test_i_of_q)
    assert q_min == 0.07
    assert q_max == 0.11

    # Calculate reference wavelength
    ref_q_wl_vec = prototype_determine_reference_wavelength_q1d(test_i_of_q)
    # vector Q
    vec_q = ref_q_wl_vec[:, 0]
    np.testing.assert_allclose(vec_q, np.arange(1, 21) * 0.01)
    # wavelength
    vec_wl = ref_q_wl_vec[:, 1]
    np.testing.assert_allclose(vec_wl[6:], 3.0)
    np.testing.assert_allclose(vec_wl[0:6], np.array([6.0, 6.0, 5.0, 5.0, 4.0, 4.0]))

    # Calculate scale factor
    (
        k_vec,
        delta_k_vec,
        p_vec,
        s_vec,
        unique_wl_vec,
        ref_q_wl_vec,
    ) = calculate_scale_factor_prototype(test_i_of_q, q_min, q_max)

    # Verify scale factor, P[3] and S[3]
    # considering numerical precision between original Excel with more effective decimals than the copied version
    # to python text
    assert k_vec.shape == (4,)
    assert delta_k_vec.shape == (4,)
    np.testing.assert_allclose(k_vec, gold_k_vec, atol=1.0e-3)
    assert p_vec[3] == pytest.approx(1266.145, 0.005)
    assert s_vec[3] == pytest.approx(1139.531, 0.005)
    assert delta_k_vec[3] == pytest.approx(
        0.2063066, abs=0.00002
    )  # numerical precision difference

    # Normalize
    # Due to the numerical precision issue when import data from Excel
    # it is more precise to use expected k vector
    corrected_iqmod = prototype_normalize_q1d(
        test_i_of_q, gold_k_vec, ref_q_wl_vec, p_vec, s_vec, unique_wl_vec, q_min, q_max
    )

    # verify: shape
    assert corrected_iqmod
    assert corrected_iqmod.mod_q.shape == test_i_of_q.mod_q.shape

    # verify: mod_q and wavelength
    # original: each row has same wavelength; output: each column has same wavelength
    # solution: transpose original
    np.testing.assert_allclose(
        corrected_iqmod.wavelength,
        test_i_of_q.wavelength.reshape((20, 4)).transpose().flatten(),
    )
    # transpose as above
    np.testing.assert_allclose(
        corrected_iqmod.mod_q, test_i_of_q.mod_q.reshape((20, 4)).transpose().flatten()
    )

    np.testing.assert_allclose(
        corrected_iqmod.intensity,
        gold_intensity_vec.reshape((20, 4)).transpose().flatten(),
        equal_nan=True,
        atol=1e-3,
    )

    # verify: corrected intensity error
    gold_errors = gold_error_vec.reshape((20, 4)).transpose().flatten()
    for i in range(80):
        buf = f"{corrected_iqmod.error[i]} -  {gold_errors[i]}"
        if corrected_iqmod.intensity[i] is not None and gold_errors[i] is not None:
            buf += f"  = {corrected_iqmod.error[i] - gold_errors[i]}"
        print(buf)

    np.testing.assert_allclose(
        corrected_iqmod.error, gold_errors, equal_nan=True, rtol=8e-4
    )


def create_configurable_testing_iq1d(num_q, num_wl):
    # create a testing I(Q)
    vec_q = (np.arange(num_q).astype("float") + 1.0) * 100.0
    vec_wl = (np.arange(num_wl).astype("float") + 1.0) * 0.1

    # make a full list of Q and lambda such in order as
    # q1, wl1
    # q2, wl1
    # ...
    # q1, wl2
    vec_q = np.tile(vec_q, num_wl)
    vec_wl = np.repeat(vec_wl, num_q)

    # intensity
    vec_intensity = np.zeros(shape=(num_q * num_wl,), dtype="float") + 123.0
    vec_error = np.sqrt(vec_intensity)

    return vec_intensity, vec_error, vec_q, vec_wl


def create_testing_iq1d():
    """Create a test data I(Q, wavelength) as the attached EXCEL spreadsheet attached in gitlab story
    Returns
    -------

    """
    # Intensity vector
    intensity_vec = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            743.802,
            np.nan,
            np.nan,
            np.nan,
            459.184,
            np.nan,
            np.nan,
            332.410,
            249.307,
            np.nan,
            np.nan,
            177.515,
            133.136,
            np.nan,
            89.796,
            97.959,
            73.469,
            np.nan,
            51.985,
            56.711,
            42.533,
            28.727,
            31.600,
            34.473,
            25.855,
            18.262,
            20.088,
            21.914,
            16.435,
            12.076,
            13.283,
            14.491,
            10.868,
            8.264,
            9.091,
            9.917,
            7.438,
            5.827,
            6.410,
            6.993,
            5.244,
            4.217,
            4.638,
            5.060,
            np.nan,
            3.121,
            3.433,
            3.745,
            np.nan,
            2.356,
            2.592,
            2.828,
            np.nan,
            1.811,
            1.992,
            2.173,
            np.nan,
            1.413,
            1.555,
            np.nan,
            np.nan,
            1.119,
            1.230,
            np.nan,
            np.nan,
            0.896,
            np.nan,
            np.nan,
            np.nan,
            0.727,
            np.nan,
            np.nan,
            np.nan,
            0.595,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    print(f"intensity: {intensity_vec.shape}")

    # Error
    error_vec = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            27.273,
            np.nan,
            np.nan,
            np.nan,
            21.429,
            np.nan,
            np.nan,
            18.232,
            15.789,
            np.nan,
            np.nan,
            13.323,
            11.538,
            np.nan,
            16.792,
            9.897,
            8.571,
            np.nan,
            10.611,
            7.531,
            6.522,
            5.360,
            4.605,
            5.871,
            5.085,
            4.273,
            4.374,
            4.681,
            4.054,
            3.475,
            3.640,
            3.807,
            3.297,
            2.875,
            2.985,
            3.149,
            2.727,
            2.414,
            2.470,
            2.644,
            2.290,
            2.053,
            2.095,
            2.249,
            np.nan,
            1.767,
            1.772,
            1.935,
            np.nan,
            1.535,
            1.522,
            1.682,
            np.nan,
            1.346,
            1.322,
            1.474,
            np.nan,
            1.189,
            1.161,
            np.nan,
            np.nan,
            1.058,
            1.028,
            np.nan,
            np.nan,
            0.947,
            np.nan,
            np.nan,
            np.nan,
            0.852,
            np.nan,
            np.nan,
            np.nan,
            0.771,
            np.nan,
            np.nan,
            np.nan,
        ]
    )

    # Q vector
    vec_q = np.arange(1, 21) * 0.01
    vec_q = np.repeat(vec_q, 4)
    print(f"q: {vec_q.shape}")

    # Wavelength vector
    wavelength_vec = np.arange(3, 7) * 1.0
    wavelength_vec = np.tile(wavelength_vec, 20)
    print(f"lambda: {wavelength_vec.shape}")

    # Construct IQmod
    i_of_q = IQmod(
        intensity=intensity_vec, error=error_vec, mod_q=vec_q, wavelength=wavelength_vec
    )

    # Expected K vector
    expected_k_vec = np.array(
        [1.0000, 0.9090909090909, 0.833333333333333, 1.11111111111111]
    )

    expected_corrected_intensities = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            826.446,
            np.nan,
            np.nan,
            np.nan,
            510.204,
            np.nan,
            np.nan,
            277.008,
            277.008,
            np.nan,
            np.nan,
            147.929,
            147.929,
            np.nan,
            81.633,
            81.633,
            81.633,
            np.nan,
            47.259,
            47.259,
            47.259,
            28.727,
            28.727,
            28.727,
            28.727,
            18.262,
            18.262,
            18.262,
            18.262,
            12.076,
            12.076,
            12.076,
            12.076,
            8.264,
            8.264,
            8.264,
            8.264,
            5.827,
            5.827,
            5.827,
            5.827,
            4.217,
            4.217,
            4.217,
            np.nan,
            3.121,
            3.121,
            3.121,
            np.nan,
            2.356,
            2.356,
            2.356,
            np.nan,
            1.811,
            1.811,
            1.811,
            np.nan,
            1.413,
            1.413,
            np.nan,
            np.nan,
            1.119,
            1.119,
            np.nan,
            np.nan,
            0.896,
            np.nan,
            np.nan,
            np.nan,
            0.727,
            np.nan,
            np.nan,
            np.nan,
            0.595,
            np.nan,
            np.nan,
            np.nan,
        ]
    )

    expected_corrected_errors = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            156.415,
            np.nan,
            np.nan,
            np.nan,
            97.679,
            np.nan,
            np.nan,
            50.281,
            54.344,
            np.nan,
            np.nan,
            27.900,
            30.312,
            np.nan,
            20.395,
            16.357,
            17.901,
            np.nan,
            12.424,
            10.308,
            11.380,
            5.360,
            4.429,
            4.534,
            4.788,
            4.273,
            4.179,
            4.241,
            4.708,
            3.475,
            3.561,
            3.513,
            3.958,
            2.875,
            2.920,
            2.875,
            3.263,
            2.414,
            2.394,
            2.374,
            2.708,
            2.053,
            2.028,
            2.011,
            np.nan,
            1.767,
            1.692,
            1.701,
            np.nan,
            1.535,
            1.437,
            1.459,
            np.nan,
            1.346,
            1.239,
            1.268,
            np.nan,
            1.189,
            1.081,
            np.nan,
            np.nan,
            1.058,
            0.952,
            np.nan,
            np.nan,
            0.947,
            np.nan,
            np.nan,
            np.nan,
            0.852,
            np.nan,
            np.nan,
            np.nan,
            0.771,
            np.nan,
            np.nan,
            np.nan,
        ]
    )

    return (
        i_of_q,
        expected_k_vec,
        expected_corrected_intensities,
        expected_corrected_errors,
    )


def determine_common_mod_q_range(iqmod):
    """Determine the common Q1D range among all the wavelengths such that I(q, lambda) does exist

    Detailed requirement:
        Determine q_min and q_max  that exist in all I(q, lambda) for the fitting (minimization) process

    Parameters
    ----------
    iqmod :  ~drtsans.dataobjects.IQmod
        Input I(Q, wavelength) to find common Q range from

    Returns
    -------
    tuple
        minimum Q, maximum Q
    """
    # verify
    if iqmod.wavelength is None:
        raise RuntimeError("Input IQmod is required to have wavelength vector")

    # description of the algorithm for Q1D
    # get unique wave length
    wl_vec = np.unique(iqmod.wavelength)
    wl_vec.sort()

    # create a 2D array: wavelength, Q1D, intensity
    combo_vec = np.array([iqmod.wavelength, iqmod.mod_q, iqmod.intensity])
    # transpose for numpy selection and slicing
    combo_vec = combo_vec.transpose()

    # then for each unique wave length, determine the min and max
    min_q_vec = np.zeros_like(wl_vec, dtype=float)
    max_q_vec = np.zeros_like(wl_vec, dtype=float)
    for iw, wave_length in enumerate(wl_vec):
        # filter by wave length: column 0
        single_wl_q_vec = combo_vec[combo_vec[:, 0] == wave_length]
        # filter by intensity: column 2
        single_wl_q_vec = single_wl_q_vec[np.isfinite(single_wl_q_vec[:, 2])]
        # get minimum and max Q
        min_q_wl = np.min(single_wl_q_vec[:, 1])
        max_q_wl = np.max(single_wl_q_vec[:, 1])
        # set value
        min_q_vec[iw] = min_q_wl
        max_q_vec[iw] = max_q_wl

    # get the common qmin and qmax
    return np.max(min_q_vec), np.min(max_q_vec)


def calculate_scale_factor_prototype(i_of_q, q_min, q_max):
    """Prototype to calculate scale factor

    Parameters
    ----------
    i_of_q: ~drtsans.dataobjects.IQmod
        Input I(Q, wavelength) to find common Q range from
    q_min
    q_max

    Returns
    -------
    tuple
        K, delta K, P, S, unique Q, reference wavelength I(q) and dI(q)

    """
    # Retrieve unique wave length in ascending order
    unique_wavelength_vec = np.unique(i_of_q.wavelength)
    assert len(unique_wavelength_vec.shape) == 1
    unique_wavelength_vec.sort()

    # Retrieve reference wavelength for each q value within q_min and q_max
    ref_q_wl_vec = prototype_determine_reference_wavelength_q1d(i_of_q)

    # Create a matrix for q, wavelength, intensity and error
    # output I'(Q, lambda) will be a complete new i_of_q instance
    i_q_wl_matrix = np.array(
        [i_of_q.mod_q, i_of_q.wavelength, i_of_q.intensity, i_of_q.error]
    )
    i_q_wl_matrix = i_q_wl_matrix.transpose()
    # print(f'Q-WL-I-Sigma matrix shape = {i_q_wl_matrix.shape}')

    # Calculate P(wl), S(wl)
    p_vec = np.zeros_like(unique_wavelength_vec)
    s_vec = np.zeros_like(unique_wavelength_vec)
    k_vec = np.zeros_like(unique_wavelength_vec)
    k_error2_vec = np.zeros_like(unique_wavelength_vec)

    for i_wl, lambda_i in enumerate(unique_wavelength_vec):
        # filter i_q_wl_matrix for certain wavelength
        i_q_matrix = i_q_wl_matrix[i_q_wl_matrix[:, 1] == lambda_i]
        # print(f'[DEBUG-INFO] wavelength = {lambda_i}, values size = {i_q_matrix.shape}')

        # get the index of q_min and q_max
        i_q_min = np.argmin(np.abs(i_q_matrix[:, 0] - q_min))  # numpy int64
        assert q_min == i_q_matrix[:, 0][i_q_min]
        i_q_max = np.argmin(np.abs(i_q_matrix[:, 0] - q_max))
        assert q_max == i_q_matrix[:, 0][i_q_max]

        # loop over each Q for S(wl) and P(wl)
        for i_q in range(i_q_min, i_q_max + 1):
            q_i = i_q_matrix[:, 0][i_q]
            # print(f'Over {i_q}-th Q {q_i}')
            # acquire index to reference lambda
            ref_index = np.argmin(np.abs(ref_q_wl_vec[:, 0] - q_i))
            # ref_wl = ref_q_wl_vec[ref_index, 1]
            i_q_ref_wl = ref_q_wl_vec[ref_index, 2]
            # print(f'  ref index = {ref_index}  '
            #       f'reference wl = {ref_wl}  I(q, ref) = {i_q_ref_wl}  I(q, wl) = {i_q_matrix[i_q][2]}')
            p_vec[i_wl] += i_q_ref_wl * i_q_matrix[i_q][2]
            # print(f'q-index {i_q}  increment P = {i_q_ref_wl * i_q_matrix[i_q][2]}')
            s_vec[i_wl] += i_q_matrix[i_q][2] * i_q_matrix[i_q][2]

        # calculate K(wl)
        k_vec[i_wl] = p_vec[i_wl] / s_vec[i_wl]

        # calculate delta K(wl)
        for i_q in range(i_q_min, i_q_max + 1):
            q_i = i_q_matrix[:, 0][i_q]
            # acquire index to reference lambda
            ref_index = np.argmin(np.abs(ref_q_wl_vec[:, 0] - q_i))
            i_q_ref_wl = ref_q_wl_vec[ref_index, 2]
            err_q_ref_wl = ref_q_wl_vec[ref_index, 3]
            # delta I(q, lambda) = delta I^{lambda}(q)
            term0 = i_q_matrix[i_q, 3]
            # I(q, ref_wl(q)) * S(wl) - 2 * I(q, wl) * P(wl) / S(wl)**2
            term1 = (
                i_q_ref_wl * s_vec[i_wl] - 2 * i_q_matrix[i_q][2] * p_vec[i_wl]
            ) / s_vec[i_wl] ** 2
            # delta I(q, lambda^ref)
            term2 = err_q_ref_wl
            # I(q, lambda)/S(lambda) = I^{lambda}(q) / S(lambda)
            term3 = i_q_matrix[i_q, 2] / s_vec[i_wl]
            # increment = (t0 * t1)**2 + (t2 * t3)**2
            k_error2_vec[i_wl] += (term0 * term1) ** 2 + (term2 * term3) ** 2
    # END-FOR

    # Get K error vector
    k_error_vec = np.sqrt(k_error2_vec)

    return k_vec, k_error_vec, p_vec, s_vec, unique_wavelength_vec, ref_q_wl_vec


def prototype_determine_reference_wavelength_q1d(i_of_q):
    """Determine the reference wavelength for each Q.

    The reference wavelength of a specific Q or (qx, qy)
    is defined as the shortest wavelength for all the finite I(Q, wavelength) or
    I(qx, qy, wavelength)

    They shall be the same between qmin and qmax

    Parameters
    ----------
    i_of_q: ~drtsans.dataobjects.IQmod
        I(Q, wavelength)

    Returns
    -------
    numpy.ndarray
        2D array of (Q, wavelength, intensity, error)

    """
    # Construct nd array for Q, I and wavelength
    combo_matrix = np.array(
        [i_of_q.mod_q, i_of_q.intensity, i_of_q.wavelength, i_of_q.error]
    )
    combo_matrix = combo_matrix.transpose()  # good for filter and slicing
    # Create a list of unique Q
    unique_q_vec = np.unique(i_of_q.mod_q)

    # Init return vector
    ref_wavelength_vec = np.ndarray(shape=(unique_q_vec.shape[0], 4), dtype="float")

    # Remove all the I(q, wl) with intensity as nan or infinity:

    # For each Q, search wavelength min
    for index, q_value in enumerate(unique_q_vec):
        # filter out the items with desired Q value
        filtered_combo_matrix = combo_matrix[combo_matrix[:, 0] == q_value]
        filtered_combo_matrix = filtered_combo_matrix[
            np.isfinite(filtered_combo_matrix[:, 1])
        ]
        # find minimum wavelength and add to output array
        min_index = np.argmin(filtered_combo_matrix[:, 2])
        # set values
        ref_wavelength_vec[index][0] = q_value
        ref_wavelength_vec[index][1] = filtered_combo_matrix[:, 2][min_index]
        ref_wavelength_vec[index][2] = filtered_combo_matrix[:, 1][min_index]
        ref_wavelength_vec[index][3] = filtered_combo_matrix[:, 3][min_index]

    return ref_wavelength_vec


def prototype_normalize_q1d(
    i_of_q, k_vec, ref_wl_vec, p_vec, s_vec, unique_wavelength_vec, q_min, q_max
):
    """Prototype of normalizing I(Q)

    Parameters
    ----------
    i_of_q
    k_vec
    ref_wl_vec
    p_vec
    s_vec
    unique_wavelength_vec

    Returns
    -------

    """
    # Create a matrix for q, wavelength, intensity and error
    # output I'(Q, lambda) will be a complete new i_of_q instance
    if i_of_q.delta_mod_q is None:
        # create a fake deltaQ for slicing
        delta_q_vec = i_of_q.mod_q
        delta_q_none = True
    else:
        delta_q_vec = i_of_q.delta_mod_q
        delta_q_none = False

    # TODO FIXME - this can be improved/refactor under the assumption that (Q, wavelength) are on mesh grid
    # unique Q vector
    unique_q_vec = np.unique(i_of_q.mod_q)
    unique_q_vec.sort()
    qmin_index = np.argmin(np.abs(unique_q_vec - q_min))
    qmax_index = np.argmin(np.abs(unique_q_vec - q_max))
    print(f"qmin = {q_min} @ {qmin_index}; qmax = {q_max} @ {qmax_index}")

    i_q_wl_matrix = np.array(
        [i_of_q.mod_q, i_of_q.wavelength, i_of_q.intensity, i_of_q.error, delta_q_vec]
    )
    i_q_wl_matrix = i_q_wl_matrix.transpose()

    # Init arrays to start
    new_mod_q = np.ndarray(shape=(0,), dtype=i_q_wl_matrix.dtype)
    new_q_error = np.ndarray(shape=(0,), dtype=i_q_wl_matrix.dtype)
    new_wavelength = np.ndarray(shape=(0,), dtype=i_q_wl_matrix.dtype)
    new_intensity = np.ndarray(shape=(0,), dtype=i_q_wl_matrix.dtype)
    new_error_sq = np.ndarray(shape=(0,), dtype=i_q_wl_matrix.dtype)

    # Calculate the normalized intensity error
    for i_wl, lambda_i in enumerate(unique_wavelength_vec):
        # filter i_q_wl_matrix for certain wavelength
        i_q_matrix = i_q_wl_matrix[i_q_wl_matrix[:, 1] == lambda_i]
        print(
            f"[DEBUG...INFO] number of Q = {i_q_matrix.shape[0]}, P({lambda_i} = {p_vec[i_wl]}, "
            f"S({lambda_i}) = {s_vec[i_wl]}"
        )
        print(f"[Q]: {i_q_matrix[:, 0]}")

        # normalize intensity by K^{lambda} * I^{lambda}(q)
        corrected_intensity_vec = i_q_matrix[:, 2] * k_vec[i_wl]

        # normalize error
        corrected_error_vec = np.zeros_like(corrected_intensity_vec)
        for i_q in range(i_q_matrix.shape[0]):
            #
            print(
                f"delta I({i_q_matrix[i_q, 0]}, {lambda_i}), I = {i_q_matrix[i_q, 2]}"
            )

            # directly pass the nan values
            if np.isnan(i_q_matrix[i_q, 2]):
                corrected_error_vec[i_q] = np.nan
                print("\tNaN")
                continue

            # Using original equations to match with Changwoo's result

            # On reference wavelength (inside qmin and qmax) and thus no correction
            if i_wl == 0:
                # FIXME - need an elegant method to determine minimum wavelength (index)
                # replace qmin_index <= i_q <= qmax_index and lambda_i == ref_wl_vec[i_q, 1]:
                corrected_error_vec[i_q] = i_q_matrix[i_q, 3] ** 2
                print("\tNo correction")
                continue

            # Term 1
            if i_q < qmin_index or i_q > qmax_index:
                # t1 = [delta I(q, wl)]**2 * [P(wl) / S(wl)]**2
                t1_sum = (i_q_matrix[i_q, 3] * p_vec[i_wl] / s_vec[i_wl]) ** 2
                inside = False
            else:
                t1_sum = 0.0
                inside = True

            t2_sum = t3_sum = 0.0
            print("Q     Y     T2     T3")
            for j_q in range(qmin_index, qmax_index + 1):

                # calculate Y
                # Y(q, q', wl) = I(q, wl) * I (q', ref_wl) * S(wl) - I(q, wl) * 2 * I(q', wl) * P(wl)
                y_value = (
                    i_q_matrix[i_q, 2] * ref_wl_vec[j_q, 2] * s_vec[i_wl]
                    - i_q_matrix[i_q, 2] * 2.0 * i_q_matrix[j_q, 2] * p_vec[i_wl]
                )

                if inside and j_q == i_q:
                    # inc = delta I_j(wl_i) * (P^2 * S^2 + 2 * P * S * Y_{q, q}) / S4
                    t1_inc = (
                        i_q_matrix[j_q, 3] ** 2
                        * (
                            p_vec[i_wl] ** 2 * s_vec[i_wl] ** 2
                            + 2 * p_vec[i_wl] * s_vec[i_wl] * y_value
                        )
                        / s_vec[i_wl] ** 4
                    )
                    t1_sum += t1_inc
                    print(f"[{j_q}]  t1 inc = {t1_inc}")

                # calculate t2_i
                # t2 += [delta I(q', wl)]**2 * Y(q, q'', wl)**2 / S(lw)**4
                t2_inc = i_q_matrix[j_q, 3] ** 2 * y_value**2 / s_vec[i_wl] ** 4
                t2_sum += t2_inc

                # calculate t3_i
                # t3: increment = [delta I(q_j, ref_wl[q_j]]^2 * [I(q_j, wl) * I(q, wl)]^2 / S(wl)^2
                # reference: i_q_matrix[i_q, 2] ** 2 / s_vec[i_wl] ** 2
                t3_inc = (
                    ref_wl_vec[j_q, 3] ** 2
                    * i_q_matrix[j_q, 2] ** 2
                    * i_q_matrix[i_q, 2] ** 2
                    / s_vec[i_wl] ** 2
                )
                t3_sum += t3_inc

                # DEBUG OUTPUT 1: print(f'{i_q_matrix[i_q, 0]}    {y_value}    {t2_inc}    {t3_inc}')

            print(
                f"WL = {lambda_i}  Q = {i_q_matrix[i_q, 0]}. e^2 = {t1_sum + t2_sum + t3_sum}: "
                f"t1 = {t1_sum}, t2 = {t2_sum}, t3 = {t3_sum}"
            )
            corrected_error_vec[i_q] = t1_sum + t2_sum + t3_sum

        # update
        new_mod_q = np.concatenate((new_mod_q, i_q_matrix[:, 0]))
        new_q_error = np.concatenate((new_q_error, i_q_matrix[:, 4]))
        new_wavelength = np.concatenate((new_wavelength, i_q_matrix[:, 1]))
        new_intensity = np.concatenate((new_intensity, corrected_intensity_vec))
        new_error_sq = np.concatenate((new_error_sq, corrected_error_vec))

    # construct a new instance of I(Q)
    if delta_q_none:
        new_q_error = None
    corrected_i_of_q = IQmod(
        new_intensity, np.sqrt(new_error_sq), new_mod_q, new_q_error, new_wavelength
    )

    return corrected_i_of_q


if __name__ == "__main__":
    pytest.main([__file__])
