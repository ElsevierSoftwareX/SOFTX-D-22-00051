import pytest
from drtsans.dataobjects import IQazimuthal
from drtsans.tof.eqsans.elastic_reference_normalization import (
    calculate_K_2d,
    determine_common_mod_q2d_range_mesh,
    determine_reference_wavelength_q2d,
    normalize_intensity_q2d
)
import numpy as np


def create_testing_iq2d():
    """Create a test data I(Q, wavelength) as the attached EXCEL spreadsheet attached in gitlab story SANS 834
    https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/uploads/d268b5ddc440becf9c677e5e0e69e9b8/test_elastic_1_.xlsx
    Returns
    -------

    """
    # Intensity vector
    intensity_vec_3 = np.ones((11, 11), dtype=np.float64)
    intensity_vec_3[3:8, 3:8] = np.nan
    intensity_vec_4 = np.full((11, 11), np.nan, dtype=np.float64)
    intensity_vec_4[1:10, 1:10] = 1.2
    intensity_vec_4[4:7, 4:7] = np.nan
    intensity_vec_5 = np.full((11, 11), np.nan, dtype=np.float64)
    intensity_vec_5[2:9, 2:9] = 0.9
    intensity_vec_5[5, 5] = np.nan
    intensity_vec = intensity_vec_3
    intensity_vec = np.concatenate((intensity_vec, intensity_vec_4), axis=1)
    intensity_vec = np.concatenate((intensity_vec, intensity_vec_5), axis=1)

    # Error
    error_vec_3 = np.full((11, 11), 0.02, dtype=np.float64)
    error_vec_3[3:8, 3:8] = np.nan
    error_vec_4 = np.full((11, 11), np.nan, dtype=np.float64)
    error_vec_4[1:10, 1:10] = 0.03
    error_vec_4[4:7, 4:7] = np.nan
    error_vec_5 = np.full((11, 11), np.nan, dtype=np.float64)
    error_vec_5[2:9, 2:9] = 0.02
    error_vec_5[5, 5] = np.nan

    error_vec = error_vec_3
    error_vec = np.concatenate((error_vec, error_vec_4), axis=1)
    error_vec = np.concatenate((error_vec, error_vec_5), axis=1)

    # Q vector
    vec_q = np.linspace(-0.1, 0.1, num=11, endpoint=True, dtype=np.float64)
    qx_matrix, qy_matrix = np.meshgrid(vec_q, vec_q, indexing='ij')
    qx_vec = qx_matrix
    qx_vec = np.concatenate((qx_vec, qx_matrix), axis=1)
    qx_vec = np.concatenate((qx_vec, qx_matrix), axis=1)
    qy_vec = qx_matrix
    qy_vec = np.concatenate((qy_vec, qy_matrix), axis=1)
    qy_vec = np.concatenate((qy_vec, qy_matrix), axis=1)

    # Wavelength vector
    wavelength_vec = np.full((11, 11), 3., dtype=np.float64)
    wavelength_vec = np.concatenate((wavelength_vec, np.full((11, 11), 4., dtype=np.float64)), axis=1)
    wavelength_vec = np.concatenate((wavelength_vec, np.full((11, 11), 5., dtype=np.float64)), axis=1)

    # Construct IQmod
    i_of_q = IQazimuthal(intensity=intensity_vec,
                         error=error_vec,
                         qx=qx_vec,
                         qy=qy_vec,
                         wavelength=wavelength_vec)

    expected_output = np.ones((11, 11), dtype=np.float64)
    expected_output[5, 5] = np.nan

    expected_output_before_renormalization = np.ones((11, 11), dtype=np.float64)
    expected_output_before_renormalization[1:10, 1:10] = 1.1
    expected_output_before_renormalization[2:9, 2:9] = 31. / 30.
    expected_output_before_renormalization[3:8, 3:8] = 1.05
    expected_output_before_renormalization[4:7, 4:7] = 0.9
    expected_output_before_renormalization[5, 5] = np.nan

    expected_normalized_error_vec_3 = np.full((11, 11), 0.022360679774998, dtype=np.float64)
    expected_normalized_error_vec_3[3:8, 3:8] = np.nan
    expected_normalized_error_vec_4 = np.full((11, 11), np.nan, dtype=np.float64)
    expected_normalized_error_vec_4[1:10, 1:10] = 0.027239931881135
    expected_normalized_error_vec_4[3:8, 3:8] = 0.026266529187458
    expected_normalized_error_vec_4[4:7, 4:7] = np.nan
    expected_normalized_error_vec_5 = np.full((11, 11), np.nan, dtype=np.float64)
    expected_normalized_error_vec_5[2:9, 2:9] = 0.023921166824012
    expected_normalized_error_vec_5[3:8, 3:8] = 0.023044955171311
    expected_normalized_error_vec_5[5, 5] = np.nan

    expected_normalized_error_vec = expected_normalized_error_vec_3
    expected_normalized_error_vec = np.concatenate(
        (expected_normalized_error_vec, expected_normalized_error_vec_4),
        axis=1
    )
    expected_normalized_error_vec = np.concatenate(
        (expected_normalized_error_vec, expected_normalized_error_vec_5),
        axis=1
    )

    return i_of_q, expected_output, expected_output_before_renormalization, expected_normalized_error_vec


def test_verify_q2d_functions():
    np.set_printoptions(edgeitems=30, linewidth=100000)
    i_of_q, expected_output, expected_output_before_renormalization, expected_normalized_error_vec = create_testing_iq2d() # noqa E501

    # print(i_of_q)
    k_vec, k_error2_vec, p_vec, s_vec = calculate_K_2d(i_of_q)

    np.testing.assert_allclose(k_vec, [1, 0.833333333333333, 1.11111111111111], rtol=1e-5, atol=0)
    np.testing.assert_allclose(p_vec, [24,28.8,21.6], rtol=1e-5, atol=0)
    np.testing.assert_allclose(s_vec, [24,34.56,19.44], rtol=1e-5, atol=0)
    np.testing.assert_allclose(k_error2_vec, [3.33333E-05, 2.96586E-05, 4.59788E-05], rtol=1e-5, atol=0)

    mask = determine_common_mod_q2d_range_mesh(i_of_q.qx, i_of_q.qy, i_of_q.wavelength, i_of_q.intensity)

    ref_wavelength_vec = determine_reference_wavelength_q2d(i_of_q, mask)

    normalized_intensity_array, normalized_error2_array = normalize_intensity_q2d(
        i_of_q.wavelength,
        i_of_q.qx,
        i_of_q.qy,
        i_of_q.intensity,
        i_of_q.error,
        ref_wavelength_vec,
        k_vec,
        p_vec,
        s_vec,
        mask
    )
    ma = np.ma.MaskedArray(normalized_intensity_array.transpose(), mask=np.isnan(normalized_intensity_array))
    averaged_normalized_intensity_array = np.ma.average(ma,axis=2)
    np.testing.assert_allclose(averaged_normalized_intensity_array, expected_output, rtol=1e-5, atol=0)

    np.testing.assert_allclose(normalized_error2_array, expected_normalized_error_vec.transpose().reshape((3, 11, 11)), rtol=1e-5, atol=0) # noqa E501


if __name__ == '__main__':
    pytest.main([__file__])
