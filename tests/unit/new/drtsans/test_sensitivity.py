import pytest
import numpy as np

# testing https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/sensitivity.py
from drtsans.sensitivity import apply_sensitivity_correction
import os
from tests.conftest import data_dir

WKSPNAME_IN = "EQSANS_87680"
FILENAME_IN = os.path.join(data_dir, "EQSANS_87680_integrated.nxs")
MIN, MAX = 0.5, 2.0


@pytest.mark.parametrize(
    "workspace_with_instrument", [{"Nx": 3, "Ny": 3}], indirect=True
)
def test_apply_simple_sensitivity(workspace_with_instrument):
    r"""
    Testing section 5 in the master document
    Apply sensitivity to a 3x3 workspace. Check if the output is masked where sensitivity is masked
    Functions to test: drtsans.sensitivity.apply_sensitivity_correction
    Underlying Mantid algorithms:
        Divide https://docs.mantidproject.org/nightly/algorithms/Divide-v1.html
        MaskDetectorsIf https://docs.mantidproject.org/nightly/algorithms/MaskDetectorsIf-v1.html
    See also https://docs.mantidproject.org/nightly/concepts/ErrorPropagation.html
    dev - Andrei Savici <saviciat@ornl.gov>
    SME - Venky Pingali <pingalis@ornl.gov>
    """
    data = np.array([[7.0, 8.0, 12.0], [10.0, 17.0, 13.0], [10.0, 10.0, 9.0]])
    sensitivity = np.array([[1.03, 0.99, 1.01], [0.96, 1.02, 0.98], [0.90, 1.00, 0.31]])
    data_error = np.sqrt(data)
    sensitivity_error = np.array(
        [[0.02, 0.02, 0.01], [0.03, 0.01, 0.01], [0.03, 0.01, 0.04]]
    )
    # create workspaces
    data_ws = workspace_with_instrument(
        axis_values=[6.0],  # fake wavelength
        intensities=data,
        uncertainties=data_error,
        view="pixel",
    )
    sensitivity_ws = workspace_with_instrument(
        axis_values=[6.0],  # fake wavelength
        intensities=sensitivity,
        uncertainties=sensitivity_error,
        view="pixel",
    )
    # run the function
    data_ws = apply_sensitivity_correction(
        data_ws, sensitivity_workspace=sensitivity_ws, min_threshold=0.5
    )
    # check the results
    # masked pixels will show up as 0 (and they have a mask flag)
    assert data_ws.extractY() == pytest.approx(
        [
            6.796116,
            8.080808,
            11.881188,
            10.41666,
            16.666666,
            13.265306,
            11.111111,
            10.000000,
            0.000000,
        ],
        abs=1e-5,
    )
    assert data_ws.extractE() == pytest.approx(
        [
            2.572078,
            2.861657,
            3.431820,
            3.310084,
            4.045561,
            3.681623,
            3.533108,
            3.163858,
            0.000000,
        ],
        abs=1e-5,
    )
    # check that the pixel with low sensitivity is masked (instead of NaN)
    assert data_ws.spectrumInfo().isMasked(8)
    for sp in range(8):
        assert not data_ws.spectrumInfo().isMasked(sp)


if __name__ == "__main__":
    pytest.main([__file__])
