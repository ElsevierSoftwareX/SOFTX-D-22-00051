import pytest

r"""
Hyperlinks to Mantid algorithms
LoadHFIRSANS <https://docs.mantidproject.org/nightly/algorithms/LoadHFIRSANS-v1.html>
"""
from mantid.simpleapi import LoadHFIRSANS
from mantid import mtd

r"""
Hyperlinks to drtsans functions
unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
normalize_by_monitor, normalize_by_time <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/mono/normalization.py>
"""  # noqa: E501
from drtsans.settings import unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.mono.gpsans import normalize_by_monitor, normalize_by_time


def test_normalize_by_monitor(gpsans_f):
    r"""
    Load GPSANS file CG2_exp245_scan0010_0001.xml and normalize by monitor count.
    (This test was introduced prior to the testset with the instrument team)
    """
    input_sample_workspace_mame = unique_workspace_dundername()
    LoadHFIRSANS(
        Filename=gpsans_f["sample_scattering"],
        OutputWorkspace=input_sample_workspace_mame,
    )
    input_sample_workspace = mtd[input_sample_workspace_mame]

    sample_logs = SampleLogs(input_sample_workspace)
    monitor_counts = sample_logs.monitor.value
    assert monitor_counts == 1284652

    unnormalized_values = input_sample_workspace.extractY().flatten()
    normalized_workspace = normalize_by_monitor(input_sample_workspace)
    normalized_values = normalized_workspace.extractY().flatten()

    assert normalized_values == pytest.approx(
        1.0e08 * unnormalized_values / monitor_counts, abs=0.1
    )


def test_normalize_by_time(gpsans_f):
    r"""
    Load GPSANS file CG2_exp245_scan0010_0001.xml and normalize by run duration.
    (This test was introduced prior to the testset with the instrument team)
    """
    input_sample_workspace_mame = unique_workspace_dundername()
    LoadHFIRSANS(
        Filename=gpsans_f["sample_scattering"],
        OutputWorkspace=input_sample_workspace_mame,
    )
    input_sample_workspace = mtd[input_sample_workspace_mame]

    sample_logs = SampleLogs(input_sample_workspace)
    run_duration = sample_logs.single_value("timer")
    assert run_duration == 60.0  # in seconds

    unnormalized_values = input_sample_workspace.extractY().flatten()
    normalized_workspace = normalize_by_time(input_sample_workspace)
    normalized_values = normalized_workspace.extractY().flatten()

    assert normalized_values == pytest.approx(
        unnormalized_values / run_duration, abs=1.0 - 3
    )
