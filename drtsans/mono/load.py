import os

# https://docs.mantidproject.org/nightly/algorithms/LoadHFIRSANS-v1.html
from mantid.simpleapi import LoadHFIRSANS, HFIRSANS2Wavelength, mtd, SaveNexusProcessed
from mantid.kernel import logger

# the generic version is feature complete for monochromatic data
from drtsans.load import load_events, sum_data
from drtsans.load import load_and_split as drt_load_and_split
from drtsans.process_uncertainties import set_init_uncertainties
from drtsans.instruments import extract_run_number, instrument_enum_name
from drtsans.mono.meta_data import get_sample_detector_offset
from drtsans.load import move_instrument
from drtsans.geometry import sample_detector_distance
from drtsans.samplelogs import SampleLogs


__all__ = [
    "load_events",
    "sum_data",
    "load_histogram",
    "transform_to_wavelength",
    "load_mono",
    "load_events_and_histogram",
    "load_and_split",
    "set_init_uncertainties",
]


def load_histogram(
    filename,
    output_workspace=None,
    wavelength=None,
    wavelength_spread=None,
    sample_det_cent=None,
):
    """Loads a SANS data file produce by the HFIR instruments at ORNL.
    The instrument geometry is also loaded. The center of the detector is
    placed at (0, 0, :ref:`sample_det_cent <devdocs-standardnames>` )

    Parameters
    ----------
    filename : str
        The name of the input xml file to load
    output_workspace : str, optional
        The optional name of the output workspace. If :py:obj:`None` is the filename stripped of the extension.
    wavelength : float
        The wavelength value to use when loading the data file (Angstrom).
        This value will be used instead of the value found in the data file.
    wavelength_spread : float
        wavelength spread value to use when loading the data file (Angstrom).
        This value will be used instead of the value found in the data file.
    sample_det_cent : float
        Sample to detector distance to use (overrides meta data) in mm

    Returns
    -------
    ~mantid.api.MatrixWorkspace
        A reference for the workspace created.
    """

    if output_workspace is None:
        output_workspace = os.path.basename(filename).split(".")[0]

    ws = LoadHFIRSANS(
        Filename=filename,
        Wavelength=wavelength,
        WavelengthSpread=wavelength_spread,
        SampleDetectorDistance=sample_det_cent,
        OutputWorkspace=output_workspace,
    )
    return ws


def transform_to_wavelength(input_workspace, output_workspace=None):
    r"""
    Transforms the event files with fake time of flight from the SANS instruments at HFIR into histograms
    in wavelength.

    **Mantid Algorithms used:**
    :ref:`HFIRSANS2Wavelength <algm-HFIRSANS2Wavelength-v1>`,

    Parameters
    ----------
    input_workspace: str, ~mantid.api.IEventWorkspace
        Events workspace in time-of-flight.
    output_workspace: str
        Name of the output workspace. If :py:obj:`None`, the name of the input_workspace will be
        used, thus overwriting the input workspace.
    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)

    HFIRSANS2Wavelength(
        InputWorkspace=input_workspace, OutputWorkspace=output_workspace
    )

    return mtd[output_workspace]


def load_mono(filename, **kwargs):
    r"""
    Loads a SANS data file produce by the HFIR instruments at ORNL.

    Parameters
    ----------

    filename: int, str
        Examples: ``55555`` or ``CG3_55555`` or file path.
    kwargs:
        keyword arguments for load_events or load_histogram.
    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """
    try:
        return load_events(filename, **kwargs)
    except Exception:
        return load_histogram(filename, **kwargs)


def load_events_and_histogram(
    run,
    data_dir=None,
    output_workspace=None,
    output_suffix="",
    overwrite_instrument=True,
    pixel_calibration=False,
    reuse_workspace=False,
    sample_to_si_name=None,
    si_nominal_distance=None,
    sample_to_si_value=None,
    sample_detector_distance_value=None,
    **kwargs,
):
    r"""Load one or more event Nexus file produced by the instruments at
    HFIR. Convert to wavelength and sums the data.

    Parameters
    ----------
    run: str, list of runs to load
        Examples: ``CG3_55555``, ``CG355555``, file path, ``CG3_55555,CG3_55556``
    data_dir: str, list
        Additional data search directories
    output_workspace: str
        If not specified it will be ``BIOSANS_55555`` determined from the supplied value of ``run``.
    output_suffix: str
        If the ``output_workspace`` is not specified, this is appended to the automatically generated
        output workspace name.
    overwrite_instrument: bool, str
        If not :py:obj:`False`, ignore the instrument embedeed in the Nexus file. If :py:obj:`True`, use the
        latest instrument definition file (IDF) available in Mantid. If ``str``, then it should be the filepath to the
        desired IDF.
    pixel_calibration: bool
        Adjust pixel heights and widths according to bar-scan and tube-width calibrations.
    sample_to_si_name: str
        Meta data name for sample to Silicon window distance
    si_nominal_distance: float
        distance between nominal position to silicon window.  unit = meter
    sample_to_si_value: float or None
        Sample to silicon window distance to overwrite the EPICS value.  None for no operation.  unit = meter
    sample_detector_distance_value: float or None
        Sample to detector distance to overwrite the EPICS value.  None for no operation. unit = meter
    reuse_workspace: bool
        When true, return the ``output_workspace`` if it already exists
    kwargs: dict
        Additional positional arguments for :ref:`LoadEventNexus <algm-LoadEventNexus-v1>`.

    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """
    # Check inputs
    if sample_to_si_name is None:
        raise NotImplementedError(
            f"For {run} Sample to Si window name must be specified thus cannot be None"
        )

    # If needed convert comma separated string list of workspaces in list of strings
    if isinstance(run, str):
        runs = [r.strip() for r in run.split(",")]
    else:
        runs = run
    # sanity check
    if not isinstance(runs, list):
        raise RuntimeError(
            f"runs {runs} of type {type(runs)} must be a list at this stage"
        )
    single_run = len(runs) == 1

    # Specify a default name for non-single run output workspace
    if not single_run and (
        (output_workspace is None)
        or (not output_workspace)
        or (output_workspace == "None")
    ):
        # create default name for output workspace, uses all input
        instrument_unique_name = instrument_enum_name(
            runs[0]
        )  # determine which SANS instrument
        output_workspace = "{}_{}{}".format(
            instrument_unique_name,
            "_".join(str(extract_run_number(r)) for r in runs),
            output_suffix,
        )

    # Load NeXus file(s)
    # define list of workspace to sum
    temp_workspaces = list()
    for index, run in enumerate(runs):
        # load and transform to wavelength and return workspace
        if single_run:
            output_ws_name = output_workspace
        else:
            output_ws_name = "__tmp_ws_{}".format(index)
        # Load event but not move sample or detector position by meta data
        ws = load_events(
            run=run,
            data_dir=data_dir,
            output_workspace=output_ws_name,
            overwrite_instrument=overwrite_instrument,
            output_suffix=output_suffix,
            pixel_calibration=pixel_calibration,
            detector_offset=0.0,
            sample_offset=0.0,
            reuse_workspace=reuse_workspace,
            **kwargs,
        )

        # Calculate offset with overwriting to sample-detector-distance
        set_sample_detector_position(
            ws,
            sample_to_si_window_name=sample_to_si_name,
            si_window_to_nominal_distance=si_nominal_distance,
            sample_si_window_overwrite_value=sample_to_si_value,
            sample_detector_distance_overwrite_value=sample_detector_distance_value,
        )
        # Transform to wavelength
        ws = transform_to_wavelength(ws)

        # Append
        temp_workspaces.append(ws)

    # Sum over all the workspaces if needed
    if single_run:
        out_ws = temp_workspaces[0]
    else:
        # Sum temporary loaded workspaces
        out_ws = sum_data(temp_workspaces, output_workspace=output_workspace)
        # Remove temporary workspaces
        for ws_i in temp_workspaces:
            ws_name = str(ws_i)
            if mtd.doesExist(ws_name):
                mtd.remove(ws_name)

    # Set uncertainty: After summing data re-calculate initial uncertainties
    out_ws = set_init_uncertainties(out_ws)

    return out_ws


def set_sample_detector_position(
    ws,
    sample_to_si_window_name,
    si_window_to_nominal_distance,
    sample_si_window_overwrite_value,
    sample_detector_distance_overwrite_value,
):
    """Calculate sample and detector offset from default position from geometry-related meta data
    and move the main detector and/or sample to correct position

    Parameters
    ----------
    ws: ~mantid.api.MatrixWorkspace
        Workspace where the instrument is for sample detector position to set correctly
    sample_to_si_window_name: str
        meta data name for Sample to Silicon window distance
    si_window_to_nominal_distance: float
        Silicon window to nominal position distance in unit of meter
    sample_si_window_overwrite_value: float or None
        value to overwrite sample to silicon window distance in unit of meter
        None for not overwriting
    sample_detector_distance_overwrite_value: float or None
        value to overwrite sample to detector distance in unit of meter
        None for not overwriting

    Returns
    -------

    """
    # Information output before
    logs = SampleLogs(ws)

    # Input verification: DAS record SDD must be same as calculated SDD
    das_sdd = sample_detector_distance(
        ws, search_logs=True, unit="mm", forbid_calculation=True
    )
    real_sdd = sample_detector_distance(ws, search_logs=False, unit="mm")
    if abs(das_sdd - real_sdd) > 1.0:
        raise RuntimeError(
            f"Workspace {str(ws)}: after loading and initial setup, DAS SDD ({das_sdd})"
            f"is not equal to calculated/real SDD ({real_sdd}) by proportion as "
            f"{abs(das_sdd - real_sdd)/das_sdd}"
        )

    # Get original sample detector distance: find expected SDD for further verification
    if sample_detector_distance_overwrite_value is None:
        # respect the das-recorded SDD
        expected_sdd = sample_detector_distance(ws, search_logs=True, unit="mm")
        if sample_si_window_overwrite_value is not None:
            das_sample_si_distance = (
                ws.getRun().getProperty(sample_to_si_window_name).value.mean() * 1e-3
            )  # meter
            shift = sample_si_window_overwrite_value - das_sample_si_distance  # meter
            expected_sdd += shift * 1e3

    else:
        # sample overwrite value: input is meter
        expected_sdd = sample_detector_distance_overwrite_value * 1000

    # record some raw (prior to any processing) geometry information
    prior_geom_info = (
        f"{ws}: \n"
        f"Prior to any geometry correction:\n"
        f"Sample to detector distance = {sample_detector_distance(ws, search_logs=False)}"
        f"(calculated)  vs {sample_detector_distance(ws, search_logs=True)} (meta) mm.\n"
        f' SampleToSi = {logs.find_log_with_units(sample_to_si_window_name, unit="mm")} mm\n'
        f"Overwrite Values = {sample_si_window_overwrite_value}, "
        f"{sample_detector_distance_overwrite_value}\n"
    )

    # Calculate sample and detector offsets for moving
    sample_offset, detector_offset = get_sample_detector_offset(
        ws,
        sample_si_meta_name=sample_to_si_window_name,
        zero_sample_offset_sample_si_distance=si_window_to_nominal_distance,
        overwrite_sample_si_distance=sample_si_window_overwrite_value,
        overwrite_sample_detector_distance=sample_detector_distance_overwrite_value,
    )
    # log
    prior_geom_info += "Sample offset = {}, Detector offset = {}\n".format(
        sample_offset, detector_offset
    )

    # Move sample and detector
    ws = move_instrument(
        ws,
        sample_offset,
        detector_offset,
        is_mono=True,
        sample_si_name=sample_to_si_window_name,
        si_window_to_nominal_distance=si_window_to_nominal_distance,
    )

    # Check current instrument setup and meta data (sample logs)
    logger.notice(
        "{} Sample to detector distance = {} (calculated) vs {} (meta) mm"
        "".format(
            str(ws),
            sample_detector_distance(ws, search_logs=False),
            sample_detector_distance(ws, search_logs=True),
        )
    )

    # Verification
    calculated_sdd = sample_detector_distance(ws, search_logs=False, unit="mm")

    # FIXME - absolute 0.01 mm is not a criteria restrict enough: 10E-2 mm will fail the test
    criteria_mm = 1e-3
    if (
        abs(expected_sdd - calculated_sdd) > criteria_mm
    ):  # absolute difference: 0.02 mm.  not good!
        logs = SampleLogs(ws)
        prior_geom_info += (
            f"Result from geometry operation:\n"
            f"Sample position = {ws.getInstrument().getSample().getPos()}\n"
            f'SampleToSi = {logs.find_log_with_units(sample_to_si_window_name, unit="mm")}'
            f"mm (From Log)\n"
        )
        # add detector information
        prior_geom_info += f"Detector[0] pos = {ws.getDetector(0).getPos()}\n"
        prior_geom_info += (
            f"Detector[{192 * 256 - 1}] = {ws.getDetector(192 * 256 - 1).getPos()}"
        )

        shift_det_x = ws.getRun().getProperty("detector_trans_Readback").value
        shift_det_x_unit = ws.getRun().getProperty("detector_trans_Readback").units
        prior_geom_info += (
            f"Detector translation X-axis = {shift_det_x} ({shift_det_x_unit})\n"
        )

        # form error message
        error_msg = (
            f"Error: ws = {str(ws)}:\n"
            f"Expected SDD = {expected_sdd} (mm), "
            f"Overwrite SDD = {sample_detector_distance_overwrite_value}, "
            f"Calculated SDD = {calculated_sdd} (mm)."
            f"Error = {abs(expected_sdd - calculated_sdd)} > {criteria_mm}.\n"
            f"FYI:\n {prior_geom_info}\n"
            f"Failed workspace is saved to mono_sans_run_geometry_error.nxs"
        )

        logger.error(error_msg)

        # Save workspace for further investigation
        SaveNexusProcessed(
            InputWorkspace=ws,
            Filename="mono_sans_run_geometry_error.nxs",
            Title=f"from workspace {str(ws)}",
        )

        raise RuntimeError(error_msg)

    return ws


def load_and_split(
    run,
    sample_to_si_name,
    si_nominal_distance,
    data_dir=None,
    output_workspace=None,
    output_suffix="",
    overwrite_instrument=True,
    pixel_calibration=False,
    time_interval=None,
    log_name=None,
    log_value_interval=None,
    sample_to_si_value=None,
    sample_detector_distance_value=None,
    reuse_workspace=False,
    monitors=False,
    **kwargs,
):
    r"""Load an event NeXus file and filter into a WorkspaceGroup depending
    on the provided filter options. Either a time_interval must be
    provided or a log_name and log_value_interval.

    Metadata added to output workspace includes the ``slice`` number,
    ``number_of_slices``, ``slice_parameter``, ``slice_interval``,
    ``slice_start`` and ``slice_end``.

    For EQSANS two WorkspaceGroup's are return, one for the filtered data and one for filtered monitors

    Parameters
    ----------
    run: str, ~mantid.api.IEventWorkspace
        Examples: ``CG3_55555``, ``CG355555`` or file path.
    sample_to_si_name: str
        Meta data name for sample to Silicon window distance
    si_nominal_distance: float
        distance between nominal position to silicon window.  unit = meter
    data_dir: str, list
        Additional data search directories
    output_workspace: str
        If not specified it will be ``BIOSANS_55555`` determined from the supplied value of ``run``.
    output_suffix: str
        If the ``output_workspace`` is not specified, this is appended to the automatically generated
        output workspace name.
    overwrite_instrument: bool, str
        If not :py:obj:`False`, ignore the instrument embedeed in the Nexus file. If :py:obj:`True`, use the
        latest instrument definition file (IDF) available in Mantid. If ``str``, then it should be the filepath to the
        desired IDF.
    pixel_calibration: bool
        Adjust pixel heights and widths according to bar-scan and tube-width calibrations.
    time_interval: float or list of floats
        Array for lengths of time intervals for splitters.  If the array has one value,
        then all splitters will have same time intervals. If the size of the array is larger
        than one, then the splitters can have various time interval values.
    sample_to_si_value: float or None
        Sample to silicon window distance to overwrite the EPICS value.  None for no operation.  unit = meter
    sample_detector_distance_value: float or None
        Sample to detector distance to overwrite the EPICS value.  None for no operation. unit = meter
    log_name: string
        Name of the sample log to use to filter. For example, the pulse charge is recorded in 'ProtonCharge'.
    log_value_interval: float
        Delta of log value to be sliced into from min log value and max log value.
    reuse_workspace: bool
        When true, return the ``output_workspace`` if it already exists
    monitors: bool
        flag to load monitors
    kwargs: dict
        Additional positional arguments for :ref:`LoadEventNexus <algm-LoadEventNexus-v1>`.

    Returns
    -------
    WorkspaceGroup
        Reference to the workspace groups containing all the split workspaces

    """
    # Load workspace
    ws = load_events(
        run=run,
        data_dir=data_dir,
        output_workspace="_load_tmp",
        overwrite_instrument=overwrite_instrument,
        pixel_calibration=pixel_calibration,
        output_suffix=output_suffix,
        detector_offset=0.0,
        sample_offset=0.0,
        reuse_workspace=reuse_workspace,
        **dict(kwargs, LoadMonitors=True),
    )

    # Calculate offset with overwriting to sample-detector-distance
    ws = set_sample_detector_position(
        ws,
        sample_to_si_name,
        si_nominal_distance,
        sample_to_si_value,
        sample_detector_distance_value,
    )

    # determine which SANS instrument from the data file
    instrument_name = instrument_enum_name(run)

    # create default name for output workspace
    if (
        (output_workspace is None)
        or (not output_workspace)
        or (output_workspace == "None")
    ):
        run_number = extract_run_number(run) if isinstance(run, str) else ""
        output_workspace = "{}_{}{}".format(instrument_name, run_number, output_suffix)

    # Split the workspace
    split_ws_group = drt_load_and_split(
        run=ws,
        data_dir=data_dir,
        output_workspace=output_workspace,
        overwrite_instrument=overwrite_instrument,
        output_suffix=output_suffix,
        detector_offset=0.0,
        sample_offset=0.0,
        time_interval=time_interval,
        log_name=log_name,
        log_value_interval=log_value_interval,
        reuse_workspace=reuse_workspace,
        monitors=monitors,
        instrument_unique_name=instrument_name,
        is_mono=True,
        **kwargs,
    )

    return split_ws_group
