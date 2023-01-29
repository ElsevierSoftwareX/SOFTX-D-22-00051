from os.path import join as pjn
import pytest
import numpy as np

r""" Links to mantid algorithms
CompareWorkspaces <https://docs.mantidproject.org/nightly/algorithms/CompareWorkspaces-v1.html>
CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
Load <https://docs.mantidproject.org/nightly/algorithms/Load-v1.html>
LoadNexus <https://docs.mantidproject.org/nightly/algorithms/LoadNexus-v1.html>
SumSpectra <https://docs.mantidproject.org/nightly/algorithms/SumSpectra-v1.html>
"""
from mantid.simpleapi import (
    CompareWorkspaces,
    CreateWorkspace,
    Load,
    LoadNexus,
    SumSpectra,
)

r"""
Hyperlinks to drtsans functions
amend_config, unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
dark_current <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof.eqsans/dark_current.py>
"""  # noqa: E501
from drtsans.settings import amend_config, unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.tof.eqsans import dark_current


def test_flatten_TOF():
    r"""
    Check that the counts are added together in each spectra

    Function tested: drtsans.tof.eqsans.dark_current.counts_in_detector
    Undelying Mantid algorithms:
        Integration https://docs.mantidproject.org/nightly/algorithms/Integration-v1.html
        Transpose   https://docs.mantidproject.org/nightly/algorithms/Transpose-v1.html

    dev - Andrei Savici <saviciat@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>
    """
    # create the workspace
    tof = [1.0, 2.0, 3.0, 4.0] * 9  # wavelength boundaries
    cts = [
        23,
        5,
        15,
        18,
        50,
        13,
        9,
        7,
        15,
        48,
        41,
        34,
        79,
        45,
        33,
        85,
        78,
        1,
        50,
        20,
        105,
        53,
        23,
        45,
        47,
        30,
        45,
    ]
    err = np.sqrt(cts)
    ws = CreateWorkspace(DataX=tof, DataY=cts, DataE=err, NSpec=9)
    # run the function
    y, e = dark_current.counts_in_detector(ws)
    # check the results
    expected_counts = [43, 81, 31, 123, 157, 164, 175, 121, 122]
    expected_errors = np.sqrt(expected_counts)
    assert np.allclose(y, expected_counts)
    assert np.allclose(e, expected_errors)


@pytest.fixture(scope="module")
def wss(reference_dir):
    with amend_config(data_dir=reference_dir.new.eqsans):
        name = pjn(reference_dir.new.eqsans, "test_dark_current", "data.nxs")
        # data is a Workspace2D in wavelength
        data = Load(name, OutputWorkspace=unique_workspace_dundername())
        # dark is an EventsWorkspace in time-of-flight
        dark = Load("EQSANS_89157", OutputWorkspace=unique_workspace_dundername())
        return dict(data=data, dark=dark)


def test_normalize_to_workspace(wss, reference_dir):
    r"""
    (This test was introduced prior to the testset with the instrument team)
    """
    _w0 = dark_current.normalize_dark_current(
        wss["dark"], wss["data"], output_workspace=unique_workspace_dundername()
    )
    _w1 = SumSpectra(_w0, OutputWorkspace=unique_workspace_dundername())
    name = pjn(reference_dir.new.eqsans, "test_dark_current", "dark_norm_sum.nxs")
    _w2 = LoadNexus(name, OutputWorkspace=unique_workspace_dundername())
    assert CompareWorkspaces(_w1, _w2)
    [_w.delete() for _w in (_w0, _w1, _w2)]


def test_subtract_normalized_dark(wss, reference_dir):
    r"""
    (This test was introduced prior to the testset with the instrument team)
    """
    file_path = pjn(reference_dir.new.eqsans, "test_dark_current", "dark_norm_sum.nxs")
    dark_normalized = LoadNexus(
        file_path, OutputWorkspace=unique_workspace_dundername()
    )
    data_normalized = dark_current.subtract_normalized_dark_current(
        wss["data"], dark_normalized, output_workspace=unique_workspace_dundername()
    )
    assert SampleLogs(data_normalized).normalizing_duration.value == "duration"
    summed_normalized = SumSpectra(
        data_normalized, OutputWorkspace=unique_workspace_dundername()
    )

    # Compare to stored data
    file_path = pjn(
        reference_dir.new.eqsans, "test_dark_current", "data_minus_dark.nxs"
    )
    stored_summed_normalized = LoadNexus(
        file_path, OutputWorkspace=unique_workspace_dundername()
    )
    assert CompareWorkspaces(summed_normalized, stored_summed_normalized).Result

    # Some cleanup
    [
        ws.delete()
        for ws in (
            dark_normalized,
            data_normalized,
            summed_normalized,
            stored_summed_normalized,
        )
    ]


if __name__ == "__main__":
    pytest.main([__file__])
