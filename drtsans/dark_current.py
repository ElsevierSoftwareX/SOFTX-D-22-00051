from dateutil.parser import parse as parse_date
import numpy as np

r"""
Links to mantid algorithms
Integration <https://docs.mantidproject.org/nightly/algorithms/Integration-v1.html>
DeleteWorkspace <https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html>
"""
from mantid.simpleapi import Integration, DeleteWorkspace
from mantid.api import mtd

r"""
Hyperlinks to drtsans functions
namedtuplefy, unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
"""  # noqa: E501
from drtsans.settings import namedtuplefy, unique_workspace_dundername
from drtsans.samplelogs import SampleLogs


@namedtuplefy
def duration(input_workspace, log_key=None):
    """
    Compute the duration of the workspace by iteratively searching the logs for
    the following keys: 'duration', 'start_time/end_time', and 'timer'.

    Parameters
    ----------
    input_workspace: str, MatrixWorkspace
        Usually the dark current workspace
    log_key: str
        If a log entry is passed, only the contents under this log entry are
        searched. No iterative search over the default values is performed.

    Returns
    -------
    namedtuple
        Fields of the namedtuple:
        - value: float, contents under the log
        - log_key: str, log used to return the duration

    """
    # Determine which log keys to use when finding out the duration of the run
    log_keys = (
        ("duration", "start_time", "proton_charge", "timer")
        if log_key is None
        else (log_key,)
    )

    sample_logs = SampleLogs(input_workspace)

    def from_start_time(log_entry):
        r"""Utility function to find the duration using the start_time and end_time log entries"""
        st = parse_date(sample_logs[log_entry].value)
        et = parse_date(sample_logs["end_time"].value)
        return (et - st).total_seconds()

    def from_proton_charge(log_entry):
        r"""Utility function to find the duration using the start_time and end_time log entries"""
        return sample_logs[log_entry].getStatistics().duration

    # Dictionary storing the utility functions that find the duration of the run, based on different log
    # keys
    calc = dict(start_time=from_start_time, proton_charge=from_proton_charge)

    # Iterate over all possible log entries until we are able to find the duration of the run
    for key in log_keys:
        try:
            return dict(value=calc.get(key, sample_logs.single_value)(key), log_key=key)
        except RuntimeError:
            continue  # check next log entry
    raise AttributeError("Could not determine the duration of the run")


def counts_in_detector(input_workspace):
    r"""
    Find the total number of neutron counts in each detector pixel.

    In a detector pixel has no counts, then the error of the zero counts is set to one.

    Parameters
    ----------
    input_workspace: str, EventsWorkspace
        Usually a dark current workspace for which we need to know the total number of counts per pixel-detector

    Returns
    -------
    tuple
        counts, error in the counts
    """
    # Create a workspace containing the total counts per pixel, and starting errors
    counts_workspace = unique_workspace_dundername()
    Integration(input_workspace, OutputWorkspace=counts_workspace)

    counts = mtd[counts_workspace].extractY().flatten()
    errors = mtd[counts_workspace].extractE().flatten()
    errors[np.where(counts == 0)[0]] = 1

    DeleteWorkspace(counts_workspace)  # some clean-up
    return counts, errors
