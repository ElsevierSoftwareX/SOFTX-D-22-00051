import pytest
from pytest import approx
import numpy as np
from mantid.simpleapi import MoveInstrumentComponent
from drtsans.transmission import apply_transmission_correction


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [dict(name="MYSANS", Nx=5, Ny=1, dx=0.1, dy=0.01, zc=1.3)],
    indirect=True,
)
def test_calculate_theta_dependent_transmission_single_value(workspace_with_instrument):
    r"""
    This implements Issue #176, addressing master document section 7 and test 17a.

    dev - Jose Borreguero <borreguerojm@ornl.gov>
    SME - Lilin He <hel3@ornl.gov>

    This tests uses a simplified detector panel with the following properties:
    - made up of 5 tubes, each tube containing only one square pixel of side 10cm.
    - detector positioned 130cm away from the sample.
    - one edge of the detector intersects the beam, the other edge is 50cm away from the beam along the X-axis
    These properties result in pixel detectors with two_theta angles matching those of test 17a

    Mantid algorithms employed:
    - MoveInstrumentComponent <https://docs.mantidproject.org/nightly/algorithms/MoveInstrumentComponent-v1.html>

    drtsans functions employed:
    - apply_transmission_correction:
          <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/transmission.py#L172>
    """

    # Generate a detector with five pixels, and embed the detector in a Mantid workspace
    n_pixels = 5
    wavelength_bin = [3.0, 3.2]  # some arbitrary wavelength bin
    sample_counts = np.ones(n_pixels).reshape(
        (n_pixels, 1)
    )  # each pixel has an intensity of one
    sample_errors = np.zeros(n_pixels).reshape(
        (n_pixels, 1)
    )  # only interested in errors from transmission
    sample_workspace = workspace_with_instrument(
        axis_values=wavelength_bin,
        intensities=sample_counts,
        uncertainties=sample_errors,
        view="array",
    )

    # Displace the detector 20 cm along the X-axis from its current location so that the pixel located on the
    # right edge (when viewed from the sample) intersects the beam
    MoveInstrumentComponent(
        sample_workspace, ComponentName="detector1", X=0.20, RelativePosition=True
    )

    # Verify the two-theta angles subtended by the pixels do match those of test 17a
    pixel_ids = sample_workspace.getNumberHistograms()
    spectrum_info = sample_workspace.spectrumInfo()
    two_thetas = np.array(
        [spectrum_info.twoTheta(pixel_id) for pixel_id in range(pixel_ids)]
    )
    # start comparison from the pixel closest to the beam, thus we reverse order of two_thetas with two_thetas[::-1]
    assert two_thetas[::-1] == approx(
        [np.arctan(x * 10 / 130) for x in range(5)], abs=0.001
    )

    # Apply the transmission correction to our sample intensities
    sample_workspace = apply_transmission_correction(
        sample_workspace, trans_value=0.9, trans_error=0.1, theta_dependent=True
    )

    # Compare transmissions from the sample workspace with those of test 17a
    transmissions = (
        1.0 / sample_workspace
    )  # recall sample_workspace contains corrected intensities
    test17a_transmissions = [0.9, 0.8999, 0.8994, 0.8988, 0.8978]
    assert transmissions.extractY().flatten()[::-1] == approx(
        test17a_transmissions, abs=0.0001
    )

    # Compare transmission errors from the sample workspace with those of test 17a.
    # Our precision is restricted to four decimal points instead of five. The reason is that test17a included
    # errors in two_theta while function apply_transmission_correction ignores these errors.
    test17a_transmissions_sigma_square = [0.01, 0.01003, 0.01011, 0.01024, 0.01042]
    transmissions_sigma_square = np.square(transmissions.extractE().flatten()[::-1])
    assert transmissions_sigma_square == pytest.approx(
        test17a_transmissions_sigma_square, abs=0.0001
    )
