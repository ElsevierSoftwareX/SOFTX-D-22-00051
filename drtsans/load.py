import h5py
import re

import mantid
from mantid.simpleapi import mtd
from mantid.simpleapi import (
    LoadEventNexus,
    MergeRuns,
    GenerateEventsFilter,
    FilterEvents,
)
from mantid.simpleapi import AddSampleLogMultiple
from mantid.kernel import logger
from drtsans.geometry import (
    translate_detector_by_z,
    translate_sample_by_z,
    translate_source_by_z,
)
from drtsans.instruments import (
    extract_run_number,
    instrument_enum_name,
    InstrumentEnumName,
    is_time_of_flight,
)
from drtsans.path import abspath, registered_workspace, exists as path_exists
from drtsans.pixel_calibration import apply_calibrations
from drtsans.samplelogs import SampleLogs
from drtsans.settings import amend_config


__all__ = ["load_events", "sum_data", "load_and_split", "move_instrument"]


def __monitor_counts(filename, monitor_name="monitor1"):
    r"""Get the total number of counts in a single monitor

    Parameters
    ----------
    filename: str
        Absolute path to the HDF5 file to be read
    monitor_name: str
        Name of the monitor to determine the total counts of

    Raises
    ------
    RuntimeError
        The HDF5 file does not contain a monitor entry
    """
    counts = 0  # default value is zero
    with h5py.File(filename, "r") as handle:
        if monitor_name not in handle["entry"]:
            raise RuntimeError(
                'File "{}" does not contain /entry/{}'.format(filename, monitor_name)
            )
        # open the monitor group
        nxmonitor = handle["entry"][monitor_name]

        # get the number of counts from the total counts array or the monitor array
        if "total_counts" in nxmonitor:
            counts = nxmonitor["total_counts"][0]
        else:
            counts = nxmonitor["event_time_offset"].shape[0]
    return int(counts)


def load_events(
    run,
    data_dir=None,
    output_workspace=None,
    overwrite_instrument=True,
    output_suffix="",
    pixel_calibration=False,
    detector_offset=0.0,
    sample_offset=0.0,
    reuse_workspace=False,
    **kwargs,
):
    r"""
    Load an event Nexus file produced by the instruments at ORNL.

    Parameters
    ----------
    run: str, ~mantid.api.IEventWorkspace
        Examples: ``CG3_55555``, ``CG355555`` or file path.
    output_workspace: str
        If not specified it will be ``BIOSANS_55555`` determined from the supplied value of ``run``.
    data_dir: str, list
        Additional data search directories
    overwrite_instrument: bool, str
        If not :py:obj:`False`, ignore the instrument embedeed in the Nexus file. If :py:obj:`True`, use the
        latest instrument definition file (IDF) available in Mantid. If ``str``, then it should be the filepath to the
        desired IDF.
    output_suffix: str
        If the ``output_workspace`` is not specified, this is appended to the automatically generated
        output workspace name.
    pixel_calibration: bool, str
        Adjust pixel heights and widths according to bar-scan and tube-width calibrations. Options are
        (1) No calibration (2) Using default calibration file (True) and
        (3) User specified calibration file (str)
    detector_offset: float
        Additional translation of the detector along the Z-axis, in mm. Positive
        moves the detector downstream.
    sample_offset: float
        Additional translation of the sample, in mm. The sample flange remains
        at the origin of coordinates. Positive moves the sample downstream.
    reuse_workspace: bool
        When true, return the ``output_workspace`` if it already exists
    kwargs: dict
        Additional positional arguments for :ref:`LoadEventNexus <algm-LoadEventNexus-v1>`.

    Returns
    -------
    ~mantid.api.IEventWorkspace
        Reference to the events workspace
    """
    instrument_unique_name = instrument_enum_name(
        run
    )  # determine which SANS instrument
    run_number = extract_run_number(run) if isinstance(run, str) else ""
    filename = (
        run if path_exists(run) else "{}{}".format(instrument_unique_name, run_number)
    )

    # create default name for output workspace
    if (
        (output_workspace is None)
        or (not output_workspace)
        or (output_workspace == "None")
    ):
        output_workspace = "{}_{}{}".format(
            instrument_unique_name, run_number, output_suffix
        )

    # determine if this is a monochromatic measurement
    is_mono = not is_time_of_flight(instrument_unique_name)

    # retrieve or load workspace
    if reuse_workspace and mtd.doesExist(output_workspace):
        # if it exists skip loading
        return mtd[output_workspace]
    else:
        # load the data into the appropriate workspace
        with amend_config(
            {"default.instrument": str(instrument_unique_name)}, data_dir=data_dir
        ):
            # not loading the instrument xml from the nexus file will use the correct one that is inside mantid
            # decide the value of LoadNexusInstrumentXML
            # if not specified, determine by overwrite_instrument
            if "LoadNexusInstrumentXML" not in kwargs:
                kwargs["LoadNexusInstrumentXML"] = not overwrite_instrument

            logger.notice(f"Loading {filename} to {output_workspace}")
            LoadEventNexus(
                Filename=filename, OutputWorkspace=output_workspace, **kwargs
            )

            # FIXME - what is the difference from: pixel_calibration is True?
            if pixel_calibration is not False:
                # pixel calibration is specified as not False
                if isinstance(pixel_calibration, str):
                    calib_file = pixel_calibration
                else:
                    calib_file = None
                apply_calibrations(output_workspace, database=calib_file)

    # insert monitor counts for monochromatic instruments
    if is_mono:
        # determine the fully qualified file path
        if "Filename" in mtd[output_workspace].run():
            # from the existing workspace
            filename = str(mtd[output_workspace].run()["Filename"].value)
        else:
            # use archive search
            filename = str(abspath(filename))

        # create new log with the monitor counts if monitor counts exists
        try:
            SampleLogs(output_workspace).insert("monitor", __monitor_counts(filename))
        except RuntimeError as e:
            logger.warning(
                str(e)
            )  # log a warning that monitor info not found in filename

    # move instrument components - sample position must happen first

    # Translate source along Z-axis
    translate_source_by_z(output_workspace, z=None, relative=False)

    # FIXME (485) - This shall be modified accordingly
    from drtsans.geometry import sample_detector_distance

    if is_mono:
        # HFIR-SANS: use new method
        out_ws = mtd[str(output_workspace)]
        logger.notice(
            f"Before translate source and sample:\n"
            f"Sample position = {out_ws.getInstrument().getSample().getPos()}\n"
            f"DAS SDD = "
            f"{sample_detector_distance(output_workspace, search_logs=True, forbid_calculation=True)}\n"
            f"Calculated SDD =  {sample_detector_distance(output_workspace, search_logs=False)}"
        )
        # Determine detector and sample offset from meta data afterwards

    else:
        # For TOF (i.e., EQSANS), still translate sample and detector as usual
        try:
            # get DAS recorded SDD
            das_sdd = sample_detector_distance(
                output_workspace, search_logs=True, forbid_calculation=True
            )
        except RuntimeError as run_err:
            # it may  not exist
            if "Unable to find any meta data related to SDD" in str(run_err):
                # it is fine if das recorded SDD does not exist
                das_sdd = None
            else:
                raise run_err

        translate_sample_by_z(
            output_workspace, 1e-3 * float(sample_offset)
        )  # convert sample offset from mm to meter
        translate_detector_by_z(
            output_workspace, None
        )  # search logs and translate if necessary
        translate_detector_by_z(output_workspace, 1e-3 * float(detector_offset))

        real_sdd = sample_detector_distance(output_workspace, search_logs=False)
        logger.notice(
            f"EQSANS workspace {str(output_workspace)} SDD is equal to {real_sdd}"
        )

        if das_sdd is not None:
            assert (
                real_sdd == das_sdd
            ), f"EQSANS DAS SDD = {das_sdd}, Calculated SDD = {real_sdd}"

    return mtd[output_workspace]


def move_instrument(
    workspace,
    sample_offset,
    detector_offset,
    is_mono=False,
    sample_si_name=None,
    si_window_to_nominal_distance=None,
):
    """Move instrument sample and detector

    Parameters
    ----------
    workspace:
        Workspace with instrument's sample or detector translated along z-axis
    sample_offset: float
        sample offset in unit meter
    detector_offset: float
        detector offset in unit meter
    is_mono: bool
        Flag that it belongs to a mono-SANS
    sample_si_name: str
        Name of Sample to silicon window name
    si_window_to_nominal_distance : float or None
        distance between Silicon window and sample in unit of meter

    Returns
    -------
    ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Workspace with instrument moved

    """
    # Get workspace name
    workspace_name = str(workspace)

    # Move sample and detector
    translate_sample_by_z(workspace, sample_offset)
    translate_detector_by_z(workspace, detector_offset)

    # Reset the SampleToSi log and sample_detector_distance log for mono-SANS
    if is_mono:
        logs = SampleLogs(workspace)

        # Update sample-silicon-window distance
        # curr_value = logs.find_log_with_units(sample_si_name, unit='mm')
        # sample offset is at same direction to +Y, while 'SampleToSi' is toward -Y
        # convert from meter to mm
        new_value = (si_window_to_nominal_distance - sample_offset) * 1e3
        AddSampleLogMultiple(
            Workspace=workspace,
            LogNames="{}".format(sample_si_name),
            LogValues="{}".format(new_value),
            LogUnits="mm",
        )

        # Adjust sample_to_detector_distance
        curr_sdd = logs.find_log_with_units("sample_detector_distance", unit="m")
        # shift shall be (-sample_offset + detector_offset)
        curr_sdd += -sample_offset + detector_offset
        # Set
        AddSampleLogMultiple(
            Workspace=workspace,
            LogNames="{}".format("sample_detector_distance"),
            LogValues="{}".format(curr_sdd),
            LogUnits="m",
        )
    # END-IF

    return mtd[workspace_name]


def load_and_split(
    run,
    data_dir=None,
    output_workspace=None,
    output_suffix="",
    overwrite_instrument=True,
    pixel_calibration=False,
    detector_offset=0.0,
    sample_offset=0.0,
    time_interval=None,
    log_name=None,
    log_value_interval=None,
    reuse_workspace=False,
    monitors=False,
    instrument_unique_name=None,
    is_mono=None,
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
    detector_offset: float
        Additional translation of the detector along the Z-axis, in mm. Positive
        moves the detector downstream.
    sample_offset: float
        Additional translation of the sample, in mm. The sample flange remains
        at the origin of coordinates. Positive moves the sample downstream.
    time_interval: float or list of floats
        Array for lengths of time intervals for splitters.  If the array has one value,
        then all splitters will have same time intervals. If the size of the array is larger
        than one, then the splitters can have various time interval values.
    log_name: string
        Name of the sample log to use to filter. For example, the pulse charge is recorded in 'ProtonCharge'.
    log_value_interval: float
        Delta of log value to be sliced into from min log value and max log value.
    reuse_workspace: bool
        When true, return the ``output_workspace`` if it already exists
    kwargs: dict
        Additional positional arguments for :ref:`LoadEventNexus <algm-LoadEventNexus-v1>`.

    Returns
    -------
    WorkspaceGroup
        Reference to the workspace groups containing all the split workspaces

    """
    if not (time_interval or (log_name and log_value_interval)):
        raise ValueError(
            "Must provide with time_interval or log_name and log_value_interval"
        )

    # Check whether need to load or not
    run = str(run)
    if registered_workspace(run):
        # Input is a workspace or name of a workspace in ADS
        ws = mtd[run]
        monitors = monitors or is_mono
        assert instrument_unique_name is not None, "Instrument name must be given!"
    else:
        # determine if this is a monochromatic measurement
        instrument_unique_name = instrument_enum_name(
            run
        )  # determine which SANS instrument
        is_mono = (instrument_unique_name == InstrumentEnumName.BIOSANS) or (
            instrument_unique_name == InstrumentEnumName.GPSANS
        )

        # monitors are required for gpsans and biosans
        monitors = monitors or is_mono

        ws = load_events(
            run=run,
            data_dir=data_dir,
            output_workspace="_load_tmp",
            overwrite_instrument=overwrite_instrument,
            pixel_calibration=pixel_calibration,
            output_suffix=output_suffix,
            detector_offset=detector_offset,
            sample_offset=sample_offset,
            reuse_workspace=reuse_workspace,
            **dict(kwargs, LoadMonitors=monitors or is_mono),
        )

    # create default name for output workspace
    if (
        (output_workspace is None)
        or (not output_workspace)
        or (output_workspace == "None")
    ):
        run_number = extract_run_number(run) if isinstance(run, str) else ""
        output_workspace = "{}_{}{}".format(
            instrument_unique_name, run_number, output_suffix
        )

    # Create event filter workspace
    GenerateEventsFilter(
        InputWorkspace=str(ws),
        OutputWorkspace="_filter",
        InformationWorkspace="_info",
        TimeInterval=time_interval,
        LogName=log_name,
        LogValueInterval=log_value_interval,
    )

    # Filter data
    FilterEvents(
        InputWorkspace=str(ws),
        SplitterWorkspace="_filter",
        OutputWorkspaceBaseName=output_workspace,
        InformationWorkspace="_info",
        FilterByPulseTime=True,
        GroupWorkspaces=True,
        OutputWorkspaceIndexedFrom1=True,
    )

    # Remove empty workspaces from event filtering
    split_ws_list = [
        mtd[output_workspace].getItem(n)
        for n in range(mtd[output_workspace].getNumberOfEntries())
    ]
    for split_ws in split_ws_list:
        num_events = split_ws.getNumberEvents()
        if num_events == 0:
            logger.notice(f"Remove empty sliced workspace {str(split_ws)}")
            mtd[output_workspace].remove(str(split_ws))

    if monitors:
        # Filter monitors
        FilterEvents(
            InputWorkspace=str(ws) + "_monitors",
            SplitterWorkspace="_filter",
            OutputWorkspaceBaseName=output_workspace + "_monitors",
            InformationWorkspace="_info",
            FilterByPulseTime=True,
            GroupWorkspaces=True,
            SpectrumWithoutDetector="Skip only if TOF correction",
            OutputWorkspaceIndexedFrom1=True,
        )

    # Check is_mono
    assert is_mono is not None, "is_mono shall be either set or specified"
    if is_mono:
        # Set monitor log to be correct for each workspace
        for n in range(mtd[output_workspace].getNumberOfEntries()):
            SampleLogs(mtd[output_workspace].getItem(n)).insert(
                "monitor",
                mtd[output_workspace + "_monitors"].getItem(n).getNumberEvents(),
            )

    # Add metadata for each slice with details
    for n in range(mtd[output_workspace].getNumberOfEntries()):
        samplelogs = SampleLogs(mtd[output_workspace].getItem(n))
        samplelogs.insert("slice", n + 1)
        samplelogs.insert(
            "number_of_slices", mtd[output_workspace].getNumberOfEntries()
        )
        slice_info = mtd[output_workspace].getItem(n).getComment()
        samplelogs.insert("slice_info", slice_info)
        if time_interval:
            samplelogs.insert("slice_parameter", "relative time from start")
            samplelogs.insert("slice_interval", time_interval)
            # Calculate relative start and end time
            samplelogs.insert(
                "slice_start",
                (mtd["_filter"].cell(n, 0) - samplelogs.startTime().totalNanoseconds())
                / 1e9,
                "seconds",
            )
            samplelogs.insert(
                "slice_end",
                (mtd["_filter"].cell(n, 1) - samplelogs.startTime().totalNanoseconds())
                / 1e9,
                "seconds",
            )
        else:
            samplelogs.insert("slice_parameter", log_name)
            samplelogs.insert("slice_interval", log_value_interval)
            slice_start, slice_end = re.sub(
                r".*\.From\.|\.Value.*", "", slice_info
            ).split(".To.")
            samplelogs.insert(
                "slice_start", float(slice_start), samplelogs[log_name].units
            )
            samplelogs.insert("slice_end", float(slice_end), samplelogs[log_name].units)

    if is_mono or not monitors:
        return mtd[output_workspace]
    else:  # If EQSANS and the filtered monitors are also being returned
        return mtd[output_workspace], mtd[output_workspace + "_monitors"]


def sum_data(
    data_list, output_workspace, sum_logs=("duration", "timer", "monitor", "monitor1")
):
    r"""
    Merge data set together, summing the listed logs

    Parameters
    ----------
    data_list: list of Workspace2D, list of workspace names, comma separated list of workspace names, WorkspaceGroup
        Examples: [ws1, ws1], ['ws1', 'ws2'] or 'ws1, ws2'
    output_workspace: str
        Name of output workspace, required
    sum_logs: list of str
        numeric logs that will be summed during merging

    Returns
    -------
    Workspace2D
    """
    # If needed convert comma separated string list of workspaces in list of strings
    if isinstance(data_list, str):
        data_list = [data.strip() for data in data_list.split(",")]

    # If only one input workpsace then just return that workspace
    if len(data_list) == 0:
        return mtd[data_list[0]]

    # Check workspaces are of correct type
    for data in data_list:
        if not mtd.doesExist(str(data)):
            raise ValueError("Workspace " + data + " does not exist")
        if not isinstance(mtd[str(data)], mantid.dataobjects.Workspace2D):
            raise ValueError(
                data
                + " is not a Workspace2D, this currently only works correctly for Workspace2D"
            )

    # Filter sum_logs list to only include logs that exist in data
    sum_logs = [
        log for log in sum_logs if mtd[str(data_list[0])].getRun().hasProperty(log)
    ]

    # Merge workspaces together
    MergeRuns(
        InputWorkspaces=",".join(str(data) for data in data_list),
        OutputWorkspace=output_workspace,
        SampleLogsSum=",".join(sum_logs),
    )

    return mtd[output_workspace]
