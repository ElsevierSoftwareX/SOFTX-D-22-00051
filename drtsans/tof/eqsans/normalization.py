r"""
Links to mantid algorithms
CloneWorkspace <https://docs.mantidproject.org/nightly/algorithms/CloneWorkspace-v1.html>
ConvertToHistogram <https://docs.mantidproject.org/nightly/algorithms/ConvertToHistogram-v1.html>
DeleteWorkspace <https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html>
Divide <https://docs.mantidproject.org/nightly/algorithms/Divide-v1.html>
Load <https://docs.mantidproject.org/nightly/algorithms/Load-v1.html>
LoadAscii <https://docs.mantidproject.org/nightly/algorithms/LoadAscii-v1.html>
Multiply <https://docs.mantidproject.org/nightly/algorithms/Multiply-v1.html>
NormaliseByCurrent <https://docs.mantidproject.org/nightly/algorithms/NormaliseByCurrent-v1.html>
RebinToWorkspace <https://docs.mantidproject.org/nightly/algorithms/RebinToWorkspace-v1.html>
RemoveSpectra <https://docs.mantidproject.org/nightly/algorithms/RemoveSpectra-v1.html>
Scale <https://docs.mantidproject.org/nightly/algorithms/Scale-v1.html>
SplineInterpolation <https://docs.mantidproject.org/nightly/algorithms/SplineInterpolation-v1.html>
"""
from mantid.simpleapi import (
    CloneWorkspace,
    ConvertToHistogram,
    DeleteWorkspace,
    Divide,
    Load,
    LoadAscii,
    Multiply,
    NormaliseByCurrent,
    RebinToWorkspace,
    RemoveSpectra,
    Scale,
    SplineInterpolation,
)
from mantid.api import mtd

r"""
Hyperlinks to drtsans functions
unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
path <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/path.py>
duration <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/dark_current.py>
"""  # noqa: E501
from drtsans.settings import unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans import path
from drtsans.dark_current import duration as run_duration

__all__ = [
    "normalize_by_flux",
    "normalize_by_time",
    "normalize_by_monitor",
    "normalize_by_proton_charge_and_flux",
]


def load_beam_flux_file(flux, data_workspace=None, output_workspace=None):
    r"""
    Loads the beam flux spectrum file.

    **Mantid algorithms used:**
    :ref:`LoadAscii <algm-LoadAscii-v1>`,
    :ref:`ConvertToHistogram <algm-ConvertToHistogram-v1>`,
    :ref:`ConvertToDistribution <algm-ConvertToDistribution-v1>`,
    :ref:`NormaliseToUnity <algm-NormaliseToUnity-v1>`,
    :ref:`RebinToWorkspace <algm-RebinToWorkspace-v1>`,

    Parameters
    ----------
    flux: str
        Path to file with the wavelength distribution of the neutron
        flux. Loader is Mantid `LoadAscii` algorithm.
    data_workspace : str, MatrixWorkspace
        Workspace to rebin the flux to. If None, then no rebin is performed
    output_workspace: str
        Name of the output workspace. If None, a hidden random name
        will be assigned.

    Returns
    -------
    MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = unique_workspace_dundername()  # make a hidden workspace

    # Load flux filename to a point-data workspace (we have as many intensities as wavelength values)
    LoadAscii(
        Filename=flux,
        Separator="Tab",
        Unit="Wavelength",
        OutputWorkspace=output_workspace,
    )
    # In histogram data we have as many intensities as wavelength bins
    ConvertToHistogram(
        InputWorkspace=output_workspace, OutputWorkspace=output_workspace
    )
    if data_workspace is not None:
        RebinToWorkspace(
            WorkspaceToRebin=output_workspace,
            WorkspaceToMatch=data_workspace,
            OutputWorkspace=output_workspace,
        )
    return mtd[output_workspace]


def normalize_by_proton_charge_and_flux(input_workspace, flux, output_workspace=None):
    r"""
    Normalizes the input workspace by proton charge and measured flux

    **Mantid algorithms used:**
    :ref:`RebinToWorkspace <algm-Divide-v1>`,
    :ref:`Divide <algm-Divide-v1>`,
    :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`,
    :ref:`NormaliseByCurrent <algm-NormaliseByCurrent-v1>`,

    Parameters
    ----------
    input_workspace : str, MatrixWorkspace
        Workspace to be normalized, rebinned in wavelength.
    flux : Workspace
        Measured beam flux file ws, usually the output of `load_beam_flux_file`
    output_workspace : str
        Name of the normalized workspace. If None, the name of the input
        workspace is chosen (the input workspace is overwritten).

    Returns
    -------
    MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)
    # Match the binning of the input workspace prior to carry out the division
    rebinned_flux = unique_workspace_dundername()
    RebinToWorkspace(
        WorkspaceToRebin=flux,
        WorkspaceToMatch=input_workspace,
        OutputWorkspace=rebinned_flux,
    )
    # Normalize by the flux
    Divide(
        LHSWorkspace=input_workspace,
        RHSWorkspace=rebinned_flux,
        OutputWorkspace=output_workspace,
    )
    DeleteWorkspace(rebinned_flux)  # remove the temporary rebinned flux workspace
    # Normalize by the proton charge
    NormaliseByCurrent(
        InputWorkspace=output_workspace, OutputWorkspace=output_workspace
    )
    return mtd[output_workspace]


def load_flux_to_monitor_ratio_file(
    flux_to_monitor_ratio_file,
    data_workspace=None,
    loader_kwargs=dict(),
    output_workspace=None,
):
    r"""
    Loads the flux-to-monitor ratio

    **Mantid algorithms used:**
    :ref:`Load <algm-Load-v1>`,
    :ref:`ConvertToHistogram <algm-Divide-v1>`,
    :ref:`SplineInterpolation <algm-Divide-v1>`,

    Parameters
    ----------
    flux_to_monitor_ratio_file: str
        Path to file with the flux-to-monitor ratio data. Loader is
        Mantid `LoadAscii` algorithm.
    data_workspace: str, MatrixWorkspace
        Match the binning of the flux-to-monitor workspace to that of the data workspace.
    loader_kwargs: dict
        optional keyword arguments to Mantid's Load algorithm.
    output_workspace: str
        Name of the output workspace. If None, a hidden random name
        will be assigned.

    Returns
    -------
    MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = unique_workspace_dundername()  # make a hidden workspace

    # Let Mantid figure out what kind file format is the flux file
    Load(
        Filename=flux_to_monitor_ratio_file,
        OutputWorkspace=output_workspace,
        **loader_kwargs
    )
    ConvertToHistogram(
        InputWorkspace=output_workspace, OutputWorkspace=output_workspace
    )
    if data_workspace is not None:
        SplineInterpolation(
            WorkspaceToMatch=data_workspace,
            WorkspaceToInterpolate=output_workspace,
            OutputWorkspace=output_workspace,
        )
    return mtd[output_workspace]


def normalize_by_monitor(
    input_workspace, flux_to_monitor, monitor_workspace, output_workspace=None
):
    r"""
    Normalizes the input workspace by monitor count and flux-to-monitor
    ratio.

    **Mantid algorithms used:**
    :ref:`RebinToWorkspace <algm-Divide-v1>`,
    :ref:`RemoveSpectra <algm-RemoveSpectra-v1>`,
    :ref:`CloneWorkspace <algm-CloneWorkspace-v1>`,
    :ref:`SplineInterpolation <algm-SplineInterpolation-v1>`,
    :ref:`Multiply <algm-Multiply-v1>`,
    :ref:`Divide <algm-Divide-v1>`,
    :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`

    Parameters
    ----------
    input_workspace : str, MatrixWorkspace
        Workspace to be normalized, rebinned in wavelength.
    flux_to_monitor : str, MatrixWorkspace
        Flux to monitor ratio. A file path or a workspace resulting from
        calling `load_flux_to_monitor_ratio_file`.
    monitor_workspace : str, MatrixWorkspace
        Counts from the monitor.
    output_workspace : str
        Name of the normalized workspace. If None, the name of the input
        workspace is chosen (the input workspace is overwritten).

    Returns
    -------
    MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)

    # Check non-skip mode
    if bool(SampleLogs(input_workspace).is_frame_skipping.value) is True:
        msg = "Normalization by monitor not possible in frame-skipping mode"
        raise ValueError(msg)

    # Only the first spectrum of the monitor is required
    monitor_workspace_rebinned = unique_workspace_dundername()
    RebinToWorkspace(
        monitor_workspace, input_workspace, OutputWorkspace=monitor_workspace_rebinned
    )
    excess_idx = range(
        1, mtd[monitor_workspace_rebinned].getNumberHistograms()
    )  # only one spectrum is needed
    RemoveSpectra(
        monitor_workspace_rebinned,
        WorkspaceIndices=excess_idx,
        OutputWorkspace=monitor_workspace_rebinned,
    )

    # Elucidate the nature of the flux to monitor input
    flux_to_monitor_workspace = unique_workspace_dundername()
    if isinstance(flux_to_monitor, str) and path.exists(flux_to_monitor):
        load_flux_to_monitor_ratio_file(
            flux_to_monitor,
            data_workspace=input_workspace,
            output_workspace=flux_to_monitor_workspace,
        )
    else:
        CloneWorkspace(flux_to_monitor, OutputWorkspace=flux_to_monitor_workspace)
        # Match the binning to that of the input workspace. Necessary prior to division
        SplineInterpolation(
            WorkspaceToMatch=input_workspace,
            WorkspaceToInterpolate=flux_to_monitor_workspace,
            OutputWorkspace=flux_to_monitor_workspace,
        )

    # the neutron flux integrated over the duration of the run is the product of the monitor counts and the
    # flux-to-monitor ratios
    flux_workspace = unique_workspace_dundername()
    Multiply(
        monitor_workspace_rebinned,
        flux_to_monitor_workspace,
        OutputWorkspace=flux_workspace,
    )

    # Normalize our input workspace
    Divide(
        LHSWorkspace=input_workspace,
        RHSWorkspace=flux_workspace,
        OutputWorkspace=output_workspace,
    )

    # Clean the dust balls
    [
        DeleteWorkspace(name)
        for name in (
            flux_to_monitor_workspace,
            flux_workspace,
            monitor_workspace_rebinned,
        )
    ]
    return mtd[output_workspace]


def normalize_by_time(input_workspace, log_key=None, output_workspace=None):
    r"""
    Divide the counts by the duration of the run

    **Mantid algorithms used:**
    :ref:`Scale <algm-Scale-v1>`

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
    log_key: str
        Use this log entry to figure out the run duration. If :py:obj:`None`,
        logs are sequentially searched for keys ``duration``, ``start_time``,
        ``proton_charge``, and ``timer``, in order to find out the duration.
    output_workspace : str
        Name of the normalized workspace. If :py:obj:`None`, the name of the input
        workspace is chosen (and the input workspace is overwritten).

    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)
    duration = run_duration(input_workspace, log_key=log_key)
    Scale(
        input_workspace,
        Factor=1.0 / duration.value,
        Operation="Multiply",
        OutputWorkspace=output_workspace,
    )
    SampleLogs(output_workspace).insert("normalizing_duration", duration.log_key)
    return mtd[output_workspace]


def normalize_by_flux(
    input_workspace,
    flux,
    method="proton charge",
    monitor_workspace=None,
    output_workspace=None,
):
    r"""
    Normalize counts by several methods to estimate the neutron flux.

    This function calls specialized normalizing functions based on ``method`` argument.
    Those functions are:
    - normalize_by_time
    - normalize_by_monitor
    - normalize_by_proton_charge_and_flux


    Parameters
    ----------
    input_workspace: ~mantid.api.MatrixWorkspace
        Input workspace, binned in wavelength
    flux: str
        If ``method`` is 'proton charge', flux is the path to the file
        containing the wavelength distribution of the neutron flux. If
        ``method`` is 'monitor', then flux is the path to the file containing
        a pre-measured flux-to-monitor ratio spectrum. If ``flux_method``
        is 'time', then pass one log entry name such as 'duration' or pass
        :py:obj:`None` for automatic log search.
    method: str
        Either 'proton charge', 'monitor', or 'time'
    monitor_workspace: str, ~mantid.api.MatrixWorkspace
        Prepared monitor workspace
    output_workspace : str
        Name of the normalized workspace. If :py:obj:`None`, the name of the input
        workspace is chosen (the input workspace is overwritten).

    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)

    # Use the appropriate flux file loader
    if method == "proton charge":
        w_flux = load_beam_flux_file(flux, data_workspace=input_workspace)
    elif method == "monitor":
        w_flux = load_flux_to_monitor_ratio_file(flux, data_workspace=input_workspace)
    else:
        w_flux = None

    # Select the normalization function
    normalizer = {
        "proton charge": normalize_by_proton_charge_and_flux,
        "monitor": normalize_by_monitor,
        "time": normalize_by_time,
    }

    # Arguments specific to the normalizer
    args = {"proton charge": [w_flux], "monitor": [w_flux, monitor_workspace]}
    args = args.get(method, list())
    kwargs = {"time": dict(log_key=flux)}
    kwargs = kwargs.get(method, dict())

    normalizer[method](
        input_workspace, *args, output_workspace=output_workspace, **kwargs
    )

    # A bit of cleanup
    if w_flux:
        w_flux.delete()

    return mtd[output_workspace]
