import pytest
import numpy as np

# CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
# DeleteWorkspaces <https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspaces-v1.html>
from mantid.simpleapi import mtd, CreateWorkspace, DeleteWorkspaces

# unique_workspace_dundername within <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py> # noqa: 501
# SampleLogs within <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
# subtract_dark_current <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/mono/dark_current
from drtsans.settings import unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.mono.dark_current import subtract_dark_current


@pytest.fixture(scope="module")
def data_test_16a():
    r"""
    Input and expected output taken from the intro to issue #174
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/174#dark-current-normalized-at-eq-sans>
    """
    return dict(
        dark_current_duration=3600.0,  # dark current collected for 1 hr.
        sample_run_duration=5.0,  # sample run collected for 5s
        number_of_pixels=25,
        wavelength_bin_boundaries=[
            2.5,
            3.5,
        ],  # the actual numbers are irrelevant for the test
        # dark current intensities and errors
        I_dc=[
            [1.0, 2.0, 3.0, 4.0, 5.0],  # dark current intensities per pixels
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ],
        I_dc_err=[
            [1.0, 1.4142, 1.7321, 2.0, 2.2361],  # associated intensity errors
            [1.4142, 1.7321, 2.0, 2.2361, 2.4495],
            [1.7321, 2.0, 2.2361, 2.4495, 2.6458],
            [2.0, 2.2361, 2.4495, 2.6458, 2.8284],
            [2.2361, 2.4495, 2.6458, 2.8284, 3.0],
        ],
        # sample data intensities and errors. Numers are selected in such a way that subtracting the dark
        # current will yield no intensities.
        I_data=[
            [0.0014, 0.0028, 0.0042, 0.0056, 0.0069],  # data intensities
            [0.0028, 0.0042, 0.0056, 0.0069, 0.0083],
            [0.0042, 0.0056, 0.0069, 0.0083, 0.0097],
            [0.0056, 0.0069, 0.0083, 0.0097, 0.0111],
            [0.0069, 0.0083, 0.0097, 0.0111, 0.0125],
        ],
        I_data_err=[
            [
                0.0014,
                0.002,
                0.0024,
                0.0028,
                0.0031,
            ],  # associated errors in data intensities
            [0.002, 0.0024, 0.0028, 0.0031, 0.0034],
            [0.0024, 0.0028, 0.0031, 0.0034, 0.0037],
            [0.0028, 0.0031, 0.0034, 0.0037, 0.0039],
            [0.0031, 0.0034, 0.0037, 0.0039, 0.0042],
        ],
        # Sample data normalized by the dark current. As stated before, the sample data was selected such
        # that the normalization will yield no intensities.
        I_data_norm=[
            [0.0, 0.0, 0.0, 0.0, 0.0],  # data minus normalized dark current
            [0.0, 0.0, 0.0, 0.0, 0.0],  # is all zeroes by construction
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        I_data_norm_err=[
            [0.002, 0.0028, 0.0034, 0.004, 0.0044],  # error in the normalized data is
            [0.0028, 0.0034, 0.004, 0.0044, 0.0048],  # sqrt(2) * I_data_err
            [0.0034, 0.004, 0.0044, 0.0048, 0.0052],
            [0.004, 0.0044, 0.0048, 0.0052, 0.0055],
            [0.0044, 0.0048, 0.0052, 0.0055, 0.0059],
        ],
        precision=1.0e-4,  # precision to compare reduction framework to test results
    )


def test_subtract_dark_current(data_test_16a):
    r"""Test of dark current subtraction from data. Dark current must be normalized

    For details see https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/156
    and also https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/174

    dev - Jose Borreguero <borreguerojm@ornl.gov>, Steven Hahn <hahnse@ornl.gov>, Jiao Lin <linjiao@ornl.gov>
    SME - Changwoo Do <doc1@ornl.gov>

    **Mantid algorithms used:**
    :ref:`CreateWorkspace <algm-CreateWorkspace-v1>`,
        <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>

    **drtsans functions used:**
    ~drtsans.samplelogs.SampleLogs
        <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
    ~drtsans.mono.dark_current.subtract_dark_current
        <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/mono/dark_current.py>
    """
    # Create dark current workspace, insert the duration of the dark current run as one of the log entries in the
    # dark current workspace.
    dark_current_workspace = (
        unique_workspace_dundername()
    )  # arbitrary name for the dark current workspace
    CreateWorkspace(
        DataX=data_test_16a["wavelength_bin_boundaries"],
        DataY=np.array(data_test_16a["I_dc"]).ravel(),
        DataE=np.array(data_test_16a["I_dc_err"]).ravel(),
        NSpec=data_test_16a["number_of_pixels"],
        OutputWorkspace=dark_current_workspace,
    )
    SampleLogs(dark_current_workspace).insert(
        "duration", data_test_16a["dark_current_duration"], "second"
    )

    # Create a sample run workspace.
    data_workspace = (
        unique_workspace_dundername()
    )  # arbitrary name for the sample workspace
    CreateWorkspace(
        DataX=data_test_16a["wavelength_bin_boundaries"],
        DataY=np.array(data_test_16a["I_data"]).ravel(),
        DataE=np.array(data_test_16a["I_data_err"]).ravel(),
        NSpec=data_test_16a["number_of_pixels"],
        OutputWorkspace=data_workspace,
    )
    # Insert the duration of the sample run. The log key must be the same as that used for the dark current,
    # which turns out to be 'duration'
    SampleLogs(data_workspace).insert(
        "duration", data_test_16a["sample_run_duration"], "second"
    )

    # Call the reduction workflow function
    subtract_dark_current(data_workspace, dark_current_workspace)

    # Compare the normalized intensities.
    computed_intensities = mtd[data_workspace].extractY().ravel()
    test_intensities = np.array(data_test_16a["I_data_norm"]).ravel()
    assert computed_intensities == pytest.approx(
        test_intensities, abs=data_test_16a["precision"]
    )

    # Compare the errors of the normalized intensities
    computed_errors = mtd[data_workspace].extractE().ravel()
    test_errors = np.array(data_test_16a["I_data_norm_err"]).ravel()
    assert computed_errors == pytest.approx(test_errors, abs=data_test_16a["precision"])

    # some cleanup
    DeleteWorkspaces([dark_current_workspace, data_workspace])


if __name__ == "__main__":
    pytest.main([__file__])
