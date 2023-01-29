from drtsans.dataobjects import IQazimuthal, q_azimuthal_to_q_modulo

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/iq.py
from drtsans.iq import select_i_of_q_by_wedge

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/tests/unit/new/drtsans/test_q_azimuthal_to_q_modulo.py
from tests.unit.new.drtsans.i_of_q_binning_tests_data import generate_test_data
import pytest

# This module supports testing data for issue #373
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/373

# DEV - Jean Bilheux <bilheuxjm@ornl.gov>
# SME - ??

# All tests data are generated in tests.unit.new.drtsans.i_of_q_binning_tests_data


def test_i_q_azimuthal_to_i_q_modulo():
    """
    Returns
    -------

    """
    """Test 'i_q_azimuthal_to_i_q_modulo
    """
    min_wedge_angle = -45.0
    max_wedge_angle = 45

    # Get data
    intensities, sigmas, qx_array, dqx_array, qy_array, dqy_array = generate_test_data(
        2, True
    )

    # Test high level method
    # Define input data
    test_i_q = IQazimuthal(
        intensity=intensities,
        error=sigmas,
        qx=qx_array,
        qy=qy_array,
        delta_qx=dqx_array,
        delta_qy=dqy_array,
    )

    # Select I(Q) inside wedge
    wedge_i_of_q = select_i_of_q_by_wedge(test_i_q, min_wedge_angle, max_wedge_angle)

    q_modulo = q_azimuthal_to_q_modulo(wedge_i_of_q)

    mod_q = q_modulo.mod_q
    delta_mod_q = q_modulo.delta_mod_q
    assert mod_q[0] == pytest.approx(0.007322, abs=1e-5)
    assert mod_q[-1] == pytest.approx(0.006255, abs=1e-5)
    assert delta_mod_q[0] == pytest.approx(0.0119119, abs=1e-7)
    assert delta_mod_q[-1] == pytest.approx(0.0111666, abs=1e-7)


if __name__ == "__main__":
    pytest.main([__file__])
