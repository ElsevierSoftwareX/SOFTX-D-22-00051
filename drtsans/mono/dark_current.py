r"""Links to mantid algorithms
Minus <https://docs.mantidproject.org/nightly/algorithms/Minus-v1.html>
DeleteWorkspace <https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html>
Integration <https://docs.mantidproject.org/nightly/algorithms/Integration-v1.html>
Scale <https://docs.mantidproject.org/nightly/algorithms/Scale-v1.html>
"""
from mantid.simpleapi import Minus, mtd, DeleteWorkspace, Scale, Integration

r""" links to drtsans imports
unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
duration <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/dark_current.py>
set_init_uncertainties <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/process_uncertainties.py>
"""  # noqa: E501
from drtsans.settings import unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.dark_current import duration
from drtsans.mono.load import load_mono
from drtsans.path import exists, registered_workspace
from drtsans.process_uncertainties import set_init_uncertainties

__all__ = [
    "subtract_dark_current",
    "load_dark_current_workspace",
    "normalize_dark_current",
]


def normalize_dark_current(dark_workspace, output_workspace=None):
    r"""
    Divide a dark current workspace by its duration.

    Entry 'normalizing_duration' is added to the logs of the normalized
    dark current to annotate what log entry was used to find the duration

    **Mantid algorithms used:**
    :ref:`Scale <algm-Scale-v1>`,
    :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`,

    Parameters
    ----------
    dark_workspace: str, ~mantid.api.MatrixWorkspace
        Dark current workspace
    output_workspace : str
        Name of the normalized dark workspace. If None, the name of the input
        workspace `dark_workspace` is chosen (and the input workspace is overwritten).

    Returns
    -------
    MatrixWorkspace
        Output workspace
    """
    if output_workspace is None:
        output_workspace = str(dark_workspace)

    # Find out the duration of the dark current from the logs, and divide
    dark_duration = duration(dark_workspace)
    Scale(
        InputWorkspace=dark_workspace,
        Factor=1.0 / dark_duration.value,
        Operation="Multiply",
        OutputWorkspace=output_workspace,
    )
    # Save the name of the log used to calculate the duration
    SampleLogs(output_workspace).insert("normalizing_duration", dark_duration.log_key)
    return mtd[output_workspace]


def load_dark_current_workspace(dark_current_filename, output_workspace):
    """Loads dark current workspace. Useful to avoid multiple loads from disk.

    **Mantid algorithms used:**
    :ref:`LoadEventNexus <algm-LoadEventNexus-v1>`,

    Parameters
    ----------
    dark_current_filename: str
        file containing previously calculated sensitivity correction
    output_workspace: int, str
        run number or file path for dark current
    """
    if (
        isinstance(dark_current_filename, str) and exists(dark_current_filename)
    ) or isinstance(dark_current_filename, int):
        load_mono(dark_current_filename, output_workspace=output_workspace)
    else:
        message = "Unable to find or load the dark current {}".format(
            dark_current_filename
        )
        raise RuntimeError(message)
    return mtd[output_workspace]


def subtract_dark_current(data_workspace, dark, output_workspace=None):
    r"""
    Subtract normalized dark from data, taking into account the duration of both the data and dark runs.

    ``normalized_data = data - (data_duration / dark_duration) * dark``

    **Mantid algorithms used:**
    :ref:`Scale <algm-Scale-v1>`,
    :ref:`Minus <algm-Minus-v1>`
    :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`

    Parameters
    ----------
    data_workspace: MatrixWorkspace
        Sample scattering with intensities versus wavelength.
    dark: int, str, ~mantid.api.MatrixWorkspace
        run number, file path, workspace name, or :py:obj:`~mantid.api.MatrixWorkspace`
        for dark current.
    output_workspace : str
        Name of the output workspace. If None, the name of the input
        workspace `data_workspace` is chosen (and the input workspace is overwritten).

    Returns
    -------
    MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = str(data_workspace)

    if registered_workspace(dark):
        dark_workspace = dark
    else:
        dark_workspace = load_dark_current_workspace(
            dark, output_workspace=unique_workspace_dundername()
        )

    # Integrate and set uncertainties
    dark_integrated = Integration(
        dark_workspace, OutputWorkspace=unique_workspace_dundername()
    )
    dark_integrated = set_init_uncertainties(dark_integrated)
    # Normalize the dark current
    normalized_dark_current = unique_workspace_dundername()  # temporary workspace
    normalize_dark_current(dark_integrated, output_workspace=normalized_dark_current)

    # Find the duration of the data run using the same log key than that of the dark current
    duration_log_key = SampleLogs(normalized_dark_current).normalizing_duration.value
    data_duration = duration(data_workspace, log_key=duration_log_key).value
    Scale(
        InputWorkspace=normalized_dark_current,
        Factor=data_duration,
        Operation="Multiply",
        OutputWorkspace=normalized_dark_current,
    )
    Minus(
        LHSWorkspace=data_workspace,
        RHSWorkspace=normalized_dark_current,
        OutputWorkspace=output_workspace,
    )

    DeleteWorkspace(normalized_dark_current)  # some clean-up
    return mtd[output_workspace]
