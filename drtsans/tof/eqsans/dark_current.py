import numpy as np

r""" Links to mantid algorithms
CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
Minus <https://docs.mantidproject.org/nightly/algorithms/Minus-v1.html>
Scale <https://docs.mantidproject.org/nightly/algorithms/Scale-v1.html>
"""
from mantid.simpleapi import mtd, CreateWorkspace, Minus, Scale

r"""
Hyperlinks to drtsans functions
amend_config, unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
exists, registered_workspace <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/path.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
clipped_bands_from_logs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/correct_frame.py>
duration, counts_in_detector <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/dark_current.py>
"""  # noqa: E501
from drtsans.settings import amend_config, unique_workspace_dundername
from drtsans.path import exists, registered_workspace
from drtsans.samplelogs import SampleLogs
from drtsans.tof.eqsans.correct_frame import clipped_bands_from_logs
from drtsans.dark_current import duration, counts_in_detector
from drtsans.tof.eqsans.load import load_events

__all__ = [
    "subtract_dark_current",
    "load_dark_current_workspace",
    "normalize_dark_current",
]


def normalize_dark_current(dark_workspace, data_workspace, output_workspace=None):
    r"""
    Scale and Rebin in wavelength a ``dark`` current workspace with information
    from a ``data`` workspace.

    Rescale and rebin to the ``data`` workspace according to:

    .. math:: frame\_width\_clipped / (frame\_width * n\_bins * duration) * I\_dc(x, y)

    Entry 'normalizing_duration' is added to the logs of the normalized
    dark current to annotate what log entry was used to find the duration

    **Mantid algorithms used:**
    :ref:`Integration <algm-Integration-v1>`,
    :ref:`CreateWorkspace <algm-CreateWorkspace-v1>`
    :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`

    Parameters
    ----------
    dark_workspace: str, EventsWorkspace
        Dark current workspace with units in time-of-flight
    data_workspace: str, MatrixWorkspace
        Sample scattering with intensities versus wavelength
    output_workspace : str
        Name of the normalized dark workspace. If None, the name of the input
        workspace `dark_workspace` is chosen (and the input workspace is overwritten).

    Returns
    -------
    MatrixWorkspace
        Output workspace, dark current rebinned to wavelength and rescaled
    """
    if output_workspace is None:
        output_workspace = str(dark_workspace)

    # work with the names of the workspaces
    dark_workspace_name = str(dark_workspace)
    data_workspace_name = str(data_workspace)

    # rescale counts by the duration of the dark current run
    dark_duration = duration(dark_workspace_name)

    # rescale counts by the ratio of trusted TOF frame and available frame
    sample_logs = SampleLogs(data_workspace_name)
    tof_clipping_factor = (
        sample_logs.tof_frame_width_clipped.value / sample_logs.tof_frame_width.value
    )

    # rescale counts by the range of wavelengths over which there should be measurable intensities
    bands = clipped_bands_from_logs(data_workspace_name)  # lead and pulse bands
    wavelength_range = (
        bands.lead.max - bands.lead.min
    )  # wavelength range from lead skipped pulse
    if bands.skip is not None:
        wavelength_range += (
            bands.skip.max - bands.skip.min
        )  # add the wavelength range from the skipped pulse

    # Find out the binning of the sample run
    bin_boundaries = mtd[data_workspace_name].readX(0)
    bin_widths = bin_boundaries[1:] - bin_boundaries[0:-1]

    # Gather all factors into a "rescaling" array, of size len(bin_widths)
    rescalings = (
        tof_clipping_factor * bin_widths / (dark_duration.value * wavelength_range)
    )

    # If running in skip-mode, find the range of wavelengths between the lead and skip pulses.
    # Also find the indexes of the bins that fall in this wavelength gap.
    # Set the rescalings to zero for the bins falling in the wavelength gap.
    gap_bin_indexes = None
    if bands.skip is not None:
        bin_centers = 0.5 * (bin_boundaries[0:-1] + bin_boundaries[1:])
        gap_bin_indexes = np.where(
            (bin_centers > bands.lead.max) & (bin_centers < bands.skip.min)
        )[0]
        rescalings[gap_bin_indexes] = 0.0

    counts, errors = counts_in_detector(dark_workspace_name)
    pixel_indexes_with_no_counts = np.where(counts == 0)[0]

    # Multiply the rescalings array by the counts-per-pixel array
    normalized_counts = (
        counts[:, np.newaxis] * rescalings
    )  # array.shape = (#pixels, #bins)
    normalized_errors = errors[:, np.newaxis] * rescalings

    # Recall that if a pixel had no counts, then we insert a special error values: error is one for all
    # wavelength bins, and zero for the bins falling in the wavelength gap.
    special_errors = np.ones(len(bin_widths)) * rescalings
    if gap_bin_indexes is not None:
        special_errors[gap_bin_indexes] = 0.0
    normalized_errors[pixel_indexes_with_no_counts] = special_errors

    # Create the normalized dark current workspace
    CreateWorkspace(
        DataX=bin_boundaries,
        UnitX="Wavelength",
        DataY=normalized_counts,
        DataE=normalized_errors,
        Nspec=len(counts),  # number of detector pixels
        ParentWorkspace=data_workspace,
        OutputWorkspace=output_workspace,
    )

    SampleLogs(output_workspace).insert("normalizing_duration", dark_duration.log_key)
    return mtd[output_workspace]


def subtract_normalized_dark_current(input_workspace, dark_ws, output_workspace=None):
    r"""
    Subtract normalized dark current from data, taking into account
    the duration of both 'data' and 'dark' runs.

    Entry 'normalizing_duration' is added to the logs of the output workspace
    to annotate what log entry was used to find the duration of both
    'data' and 'dark' runs. Log entry 'normalizing_duration' must be
    present in the logs of workspace 'dark'.

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
        Sample scattering with intensities versus wavelength
    dark_ws: str, ~mantid.api.MatrixWorkspace
        Normalized dark current after being normalized with
        `normalize_dark_current`
    output_workspace : str
        Name of the workspace after dark current subtraction. If :py:obj:`None`,
        the name of the input workspace is chosen (and the input workspace
        is overwritten).

    **Mantid algorithms used:**
    :ref:`Scale <algm-Scale-v1>`,
    :ref:`Minus <algm-Minus-v1>`,

    Returns
    -------
    ~mantid.api.MatrixWorkspace
        'data' minus 'dark' current
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)

    duration_log_key = SampleLogs(dark_ws).normalizing_duration.value
    d = duration(input_workspace, log_key=duration_log_key).value
    scaled = Scale(
        InputWorkspace=dark_ws, Factor=d, OutputWorkspace=unique_workspace_dundername()
    )
    Minus(
        LHSWorkspace=input_workspace,
        RHSWorkspace=scaled,
        OutputWorkspace=output_workspace,
    )
    scaled.delete()
    SampleLogs(output_workspace).insert("normalizing_duration", duration_log_key)
    return mtd[output_workspace]


def load_dark_current_workspace(dark_current_filename, output_workspace):
    """Loads dark current workspace. Useful to avoid multiple loads from disk.

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
        with amend_config({"default.instrument": "EQSANS"}):
            load_events(run=dark_current_filename, output_workspace=output_workspace)
    else:
        message = "Unable to find or load the dark current {}".format(
            dark_current_filename
        )
        raise RuntimeError(message)
    return mtd[output_workspace]


def subtract_dark_current(input_workspace, dark, output_workspace=None):
    r"""


    Parameters
    ----------
    input_workspace : int, str, ~mantid.api.IEventWorkspace
        The workspace to be normalized
    dark: int, str, ~mantid.api.IEventWorkspace
        run number, file path, workspace name, or :py:obj:`~mantid.api.IEventWorkspace`
        for dark current.
    output_workspace : str
        Name of the workspace after dark current subtraction. If None,
        the name of the input workspace is chosen (and the input workspace
        is overwritten).

    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)

    if registered_workspace(dark):
        _dark = dark
    else:
        _dark = load_dark_current_workspace(dark, unique_workspace_dundername())

    _dark_normal = normalize_dark_current(
        _dark, input_workspace, output_workspace=unique_workspace_dundername()
    )
    subtract_normalized_dark_current(
        input_workspace, _dark_normal, output_workspace=output_workspace
    )
    _dark_normal.delete()
    if _dark is not dark:
        _dark.delete()

    return mtd[output_workspace]
