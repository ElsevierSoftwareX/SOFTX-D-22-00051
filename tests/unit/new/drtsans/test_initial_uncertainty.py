import numpy as np
import pytest

# https://docs.mantidproject.org/nightly/algorithms/LoadEmptyInstrument-v1.html
from mantid.simpleapi import LoadEmptyInstrument
from drtsans.process_uncertainties import set_init_uncertainties


# This implements Issue #163
# in master document section 3.3
# dev - Pete Peterson <petersonpf@ornl.gov>
# SME - William Heller <hellerwt@ornl.gov>
@pytest.mark.parametrize(
    "generic_IDF",
    [{"Nx": 2, "Ny": 2, "dx": 0.005, "dy": 0.004, "zc": 2.5}],
    indirect=True,
)
def test_initial_uncertainty(generic_IDF, cleanfile):
    """
    Test initial uncertainty after histogram data is converted to unit
    wavelength for a TOF instrument
    :param generic_IDF,: IDF to generate
    :return:i
    """
    # Range of TOF
    wave_length_range = np.array([2.5, 6.5])  # A
    intensity = np.array([0.0, 187.0, 1.0, np.nan])
    init_delta_intensity = np.random.randn(
        intensity.shape[0],
    )
    gold_delta_intensity = np.array([1.0, np.sqrt(187.0), 1.0, np.nan])

    # Generate a generic SANS instrument with a pixel of
    # the size and position specified in
    # sans-backend/documents/Master_document_022219.pdf
    with open(r"/tmp/GenericSANS2_Definition.xml", "w") as tmp:
        tmp.write(generic_IDF)
        tmp.close()
    cleanfile(tmp.name)
    ws = LoadEmptyInstrument(
        Filename=tmp.name,
        InstrumentName="GenericSANS2",
        OutputWorkspace="test_uncertainty",
    )
    ws.getAxis(0).setUnit("Wavelength")
    # assume that the TOF is already frame corrected
    for i in range(4):
        ws.dataX(i)[:] = wave_length_range  # A
        ws.dataY(i)[0] = intensity[i]
        ws.dataE(i)[0] = init_delta_intensity[i]
    # #### ABOVE THIS POINT WILL BE A TEST FIXTURE

    # Set uncertainties
    ws = set_init_uncertainties(ws)

    print(
        "[TEST INFO] Workspace {} has {} spectra".format(ws, ws.getNumberHistograms())
    )
    for ws_index in range(4):
        if np.isnan(gold_delta_intensity[ws_index]):
            assert np.isnan(ws.dataE(ws_index)[0])
        else:
            assert abs(ws.readE(ws_index)[0] - gold_delta_intensity[ws_index]) < 1.0e-10
        # END-IF
    # END-FOR


if __name__ == "__main__":
    pytest.main([__file__])
