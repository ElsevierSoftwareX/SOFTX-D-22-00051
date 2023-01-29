import pytest
from pytest import approx
import numpy as np

"""
https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html
https://docs.mantidproject.org/nightly/algorithms/LoadInstrument-v1.html
https://docs.mantidproject.org/nightly/algorithms/SumSpectra-v1.html
"""
from mantid.simpleapi import CreateWorkspace, LoadInstrument, SumSpectra
from drtsans.settings import unique_workspace_dundername as uwd
from drtsans.mask_utils import apply_mask


@pytest.mark.parametrize(
    "generic_IDF",
    [{"Nx": 3, "Ny": 3, "dx": 0.00425, "dy": 0.0055, "xc": 0.32, "yc": -0.16}],
    indirect=True,
)
def test_apply_mask_single_bin(generic_IDF):
    r"""
    Apply a mask to a simple 3 x 3 detector
    dev - Jose Borreguero <borreguerojm@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>
    """
    # Nine histograms, each containing only one bin
    wavelength = np.array([1, 2] * 9)
    intensities = np.array([8, 15, 30, 9, 17, 25, 6, 10, 31])
    ws = CreateWorkspace(
        DataX=wavelength,
        DataY=intensities,
        DataE=np.sqrt(intensities),
        Nspec=9,
        OutputWorkspace=uwd(),
    )
    LoadInstrument(
        Workspace=ws,
        InstrumentXML=generic_IDF,
        RewriteSpectraMap=True,
        InstrumentName="GenericSANS",
    )

    # detectors to be masked with detector ID's 0 and 3
    masked_detectors = [3, 0]
    apply_mask(ws, mask=masked_detectors)

    # check mask-flags
    si = ws.spectrumInfo()
    mask_flags = [False] * 9
    for i in masked_detectors:
        mask_flags[i] = True
    assert [si.isMasked(i) for i in range(9)] == mask_flags

    # check masked data
    intensities_after_mask = np.copy(intensities)
    intensities_after_mask[masked_detectors] = 0
    assert ws.extractY().flatten() == approx(intensities_after_mask, abs=0.1)

    # errors also set to zero for the masked spectra
    errors_after_mask = np.sqrt(intensities_after_mask)
    assert ws.extractE().flatten() == approx(errors_after_mask, abs=1e-9)

    # check the value of a masked detector is irrelevant when doing
    # operations on the whole workspace
    big_value = 1.0e6
    ws.setY(3, [big_value])
    ws_sum = SumSpectra(ws, OutputWorkspace=uwd())
    assert ws_sum.dataY(0)[0] == sum(intensities_after_mask)
    intensities_after_mask[3] = big_value

    # Now apply an additional mask mimicking the trap beam
    x_c, y_c = (1, 1)  # location of the trap beam, in pixel coordinates
    pixels_per_tube = 3
    trap_detector_id = x_c * pixels_per_tube + y_c
    apply_mask(ws, mask=[trap_detector_id])
    intensities_after_mask[trap_detector_id] = 0
    assert ws.extractY().flatten() == approx(intensities_after_mask, abs=0.1)

    # Clean up
    ws.delete()
    ws_sum.delete()


@pytest.mark.parametrize(
    "generic_IDF",
    [{"Nx": 2, "Ny": 2, "dx": 0.00425, "dy": 0.0055, "xc": 0.32, "yc": -0.16}],
    indirect=True,
)
def test_apply_mask_simple_histogram(generic_IDF):
    r"""
    Apply a mask to a  2 x 2 detector with histograms
    dev - Andrei Savici <asviciat@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>
    """
    # Four histograms, each containing three bins
    wavelength = np.array([1.0, 2.0, 3.0, 4.0] * 4)
    intensities = np.array(
        [[9, 10, 11, 3], [8, 12, 4, 14], [11, 15, 3, 16]]
    ).transpose()
    ws = CreateWorkspace(
        DataX=wavelength,
        DataY=intensities,
        DataE=np.sqrt(intensities),
        Nspec=4,
        OutputWorkspace=uwd(),
    )
    LoadInstrument(
        Workspace=ws,
        InstrumentXML=generic_IDF,
        RewriteSpectraMap=True,
        InstrumentName="GenericSANS",
    )

    # detector to be masked with detector ID 2
    masked_detectors = [2]
    apply_mask(ws, mask=masked_detectors)

    # check mask-flags
    si = ws.spectrumInfo()
    mask_flags = [False, False, True, False]
    assert [si.isMasked(i) for i in range(4)] == mask_flags

    # check masked data
    intensities_after_mask = np.copy(intensities)
    intensities_after_mask[masked_detectors] = 0
    assert ws.extractY() == approx(intensities_after_mask, abs=0.1)

    # errors also set to zero for the masked spectra
    errors_after_mask = np.sqrt(intensities_after_mask)
    assert ws.extractE() == approx(errors_after_mask, abs=1e-6)

    # check the value of a masked detector is irrelevant when doing
    # operations on the whole workspace
    big_value = 1.0e6
    ws.setY(2, [big_value] * 3)
    ws_sum = SumSpectra(ws, OutputWorkspace=uwd())
    assert ws_sum.dataY(0) == approx(np.sum(intensities_after_mask, axis=0))
    intensities_after_mask[2] = big_value

    # Now apply an additional mask mimicking the trap beam
    x_c, y_c = (1, 1)  # location of the trap beam, in pixel coordinates
    pixels_per_tube = 2
    trap_detector_id = x_c * pixels_per_tube + y_c
    apply_mask(ws, mask=[trap_detector_id])
    intensities_after_mask[trap_detector_id] = 0
    assert ws.extractY() == approx(intensities_after_mask, abs=0.1)

    # Clean up
    ws.delete()


@pytest.mark.parametrize(
    "generic_workspace",
    [{"name": "EQ-SANS", "dx": 0.01, "dy": 0.01, "Nx": 192, "Ny": 256}],
    indirect=True,
)
def test_apply_mask_btp_and_angle(generic_workspace):
    r"""
    Apply a circular mask
    """
    w = generic_workspace
    apply_mask(w, Pixel="1-8", MaxAngle=10)
    spectrum_info = w.spectrumInfo()
    for i in range(256):  # central tube
        if i < 8:
            # bottom 8 pixels should be masked
            spectrum_info.isMasked(i + 96 * 256)
        else:
            # only detectors with two theta<10 degrees should be masked
            if spectrum_info.twoTheta(i + 96 * 256) > np.radians(10.0):
                assert not spectrum_info.isMasked(i + 96 * 256)
            else:
                assert spectrum_info.isMasked(i + 96 * 256)


if __name__ == "__main__":
    pytest.main([__file__])
