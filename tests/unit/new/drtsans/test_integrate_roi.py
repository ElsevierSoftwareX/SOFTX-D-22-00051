import pytest
import numpy as np
from drtsans.integrate_roi import integrate_detector_roi
from mantid.simpleapi import LoadEmptyInstrument


counts_array = np.array(
    [
        [1, 1, 0, 1, 1, 1, 0, 1, 1],
        [25, 25, 25, 25, 7, 5, 1, 5, 5],
        [50, 100, 100, 50, 8, 1, 1, 1, 4],
        [25, 50, 50, 25, 8, 1, 5, 1, 1],
        [5, 5, 5, 5, 0, 5, 5, 5, 5],
        [5, 1, 1, 1, 10, 25, 50, 50, 25],
        [1, 1, 5, 1, 8, 50, 100, 100, 50],
        [1, 3, 2, 1, 1, 25, 50, 50, 25],
        [1, 1, 0, 1, 0, 1, 1, 1, 0],
    ]
)


@pytest.mark.parametrize(
    "generic_IDF",
    [
        {
            "Nx": counts_array.shape[0],
            "Ny": counts_array.shape[1],
            "dx": 0.006,
            "dy": 0.004,
            "zc": 1.25,
            "l1": -5.0,
        }
    ],
    indirect=True,
)
def test_integrate_detector_roi(generic_IDF, cleanfile):
    """
    Create a 2D generic SANS instrument, set detector counts value stated in
    https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/178
    Parameters
    ----------
    generic_IDF

    Returns
    -------

    """
    # Generate a generic SANS instrument with detector dimension stated in
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/178
    with open(r"/tmp/GenericSANS_Definition.xml", "w") as tmp:
        tmp.write(generic_IDF)
        tmp.close()
    cleanfile(tmp.name)
    ws = LoadEmptyInstrument(
        Filename=tmp.name,
        InstrumentName="GenericSANS",
        OutputWorkspace="test_integration_roi",
    )
    ws.getAxis(0).setUnit("Wavelength")

    # Set the detector counts: columns will be stacked to 1D array
    data_y = counts_array.transpose().flatten()
    assert counts_array.shape == (9, 9)
    for iws in range(data_y.shape[0]):
        ws.dataY(iws)[0] = data_y[iws]
        # Set uncertainty
        if data_y[iws] == 0:
            error_i = 1.0
        elif np.isnan(data_y[iws]):
            error_i = np.nan
        else:
            error_i = np.sqrt(data_y[iws])
        ws.dataE(iws)[0] = error_i
    # END-FOR

    # Set mask
    pid_list = list(range(1, 5))
    pid_list.extend(list(range(9, 14)))
    pid_list.extend(list(range(18, 23)))
    pid_list.extend(list(range(28, 31)))

    roi_counts, roi_error = integrate_detector_roi(ws, pid_list)

    assert abs(roi_counts - 566) < 1e-10
    # sqrt(567) = 23.817
    assert abs(roi_error - np.sqrt(567.0)) < 1e-12

    return
