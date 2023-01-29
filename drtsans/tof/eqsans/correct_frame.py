import numpy as np

# https://docs.mantidproject.org/nightly/algorithms/ConvertUnits-v1.html
# https://docs.mantidproject.org/nightly/algorithms/CropWorkspace-v1.html
# https://docs.mantidproject.org/nightly/algorithms/EQSANSCorrectFrame-v1.html
# https://docs.mantidproject.org/nightly/algorithms/MaskBins-v1.html
# https://docs.mantidproject.org/nightly/algorithms/Rebin-v1.html
# https://docs.mantidproject.org/nightly/algorithms/RebinToWorkspace-v1.html
from mantid.simpleapi import (
    mtd,
    ConvertUnits,
    CropWorkspace,
    EQSANSCorrectFrame,
    MaskBins,
    Rebin,
    RebinToWorkspace,
    logger,
    SetInstrumentParameter,
    ModeratorTzero,
    ChangeBinOffset,
)
from drtsans.samplelogs import SampleLogs
from drtsans.tof.eqsans.chopper import EQSANSDiskChopperSet
from drtsans.frame_mode import FrameMode
from drtsans import wavelength as wlg
from drtsans.settings import namedtuplefy
from drtsans.geometry import source_detector_distance
from drtsans.tof.eqsans.geometry import source_monitor_distance
from drtsans.process_uncertainties import set_init_uncertainties


__all__ = ["transform_to_wavelength"]


def _is_frame_skipping(input_workspace):
    r"""
    Find whether the run was created in frame-skip mode

    Parameters
    ----------
    input_workspace: str, MatrixWorkspace

    Returns
    -------
    bool
    """
    sample_logs = SampleLogs(input_workspace)
    if "is_frame_skipping" in sample_logs.keys():
        return bool(sample_logs.is_frame_skipping.value)
    else:
        return EQSANSDiskChopperSet(input_workspace).frame_mode == FrameMode.skip


@namedtuplefy
def transmitted_bands(input_workspace):
    r"""
    Wavelength bands of the lead and skipped pulses transmitted by the choppers of the workspace.

    Parameters
    ----------
    input_workspace: str, MatrixWorkspace
        Input Workspace containing all necessary info

    Returns
    -------
    namedtuple
        Fields of the namedtuple:
        - lead, WBand object for the wavelength band of the lead pulse
        - skipped, Wband for the skipped pulse. None if not operating in the skipped frame mode
    """
    sample_logs = SampleLogs(input_workspace)
    try:
        # 10^6/60 micro-seconds
        pulse_period = 1.0e6 / sample_logs.single_value("frequency")
    except RuntimeError:
        pulse_period = 1.0e6 / 60.0  # reasonable default
    ch = EQSANSDiskChopperSet(input_workspace)  # object representing the four choppers
    # Wavelength band of neutrons from the leading pulse transmitted
    # by the chopper system
    lead_band = ch.transmission_bands(pulsed=True)[0]
    # Wavelength from the previous, skipped pulse.
    skip_band = (
        ch.transmission_bands(delay=pulse_period, pulsed=True)[0]
        if ch.frame_mode == FrameMode.skip
        else None
    )
    return dict(lead=lead_band, skip=skip_band)


@namedtuplefy
def limiting_tofs(input_workspace, sdd):
    r"""
    Minimimum and maximum TOF's for neutrons coming from the lead and skip
    pulses, assuming a detector were placed at `distance`.

    Parameters
    ----------
    input_workspace: str, MatrixWorkspace
            Input Workspace containing all necessary info
    sdd: float
        Distance from source to detector, in meters

    Returns
    -------
    namedtuple
        Fields of the namedtuple
        lead: tuple, tof_min and tof_max for the neutrons of the lead pulse
        skip: tuple, tof_min and tof_max for the neutrons of the skip pulse. `None` if the frame mode is 'non-skipping'
    """
    ws = mtd[str(input_workspace)]
    ch = EQSANSDiskChopperSet(ws)  # object representing the four choppers
    bands = transmitted_bands(ws)
    lead = (wlg.tof(bands.lead.min, sdd, ch.pulse_width), wlg.tof(bands.lead.max, sdd))
    skip = (
        None
        if ch.frame_mode == FrameMode.not_skip
        else (
            wlg.tof(bands.skip.min, sdd, ch.pulse_width),
            wlg.tof(bands.skip.max, sdd),
        )
    )
    return dict(lead=lead, skip=skip)


@namedtuplefy
def clipped_bands_from_logs(input_workspace):
    r"""
    Retrieve the wavelength bands over which we expect non-zero intensity. We
    inspect the log.

    Parameters
    ----------
    input_workspace: str, MatrixWorkspace
        Input Workspace containing all necessary info

    Returns
    -------
    namedtuple
        Fields of the namedtuple:
        - lead, WBand object for the wavelength band of the lead pulse
        - skipped, Wband for the skipped pulse. None if not operating in the skipped frame mode
    """
    sl = SampleLogs(input_workspace)
    lead = wlg.Wband(sl.wavelength_lead_min.value, sl.wavelength_lead_max.value)
    if _is_frame_skipping(input_workspace):
        skip = wlg.Wband(sl.wavelength_skip_min.value, sl.wavelength_skip_max.value)
    else:
        skip = None
    return dict(lead=lead, skip=skip)


@namedtuplefy
def transmitted_bands_clipped(
    input_workspace,
    source_detector_dist=None,
    low_tof_clip=None,
    high_tof_clip=None,
    interior_clip=True,
    search_in_logs=True,
):
    r"""
    Wavelength bands of the lead and skipped pulses transmitted by the choppers taking into account
    the TOF clippings for neutrons arriving at the detector, assuming a detector were placed at `distance`.

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventWorkspace
        Data workspace
    source_detector_dist: float
        Distance from source to detector, in meters. if :py:obj:`None`, the distance will be obtained from the
        information stored in the input workspace.
    low_tof_clip: float
        trim neutrons of the leading pulse with a TOF smaller than the minimal TOF plus this value. Units in
        micro-seconds. If py:obj:`None`, the value is retrieved from log entry `'low_tof_clip``.
    high_tof_clip: float
        trim neutrons of the leading pulse with TOF bigger than the maximal TOF minus this value. Units in
        micro-seconds. If py:obj:`None`, the value is retrieved from log entry `'low_tof_clip``.
    interior_clip: bool
        If True, trim slow neutrons from the lead pulse (using `high_tof_clip`) and fast neutrons from the skip
        pulse (using `low_tof_clip`)
    search_in_logs: True
        If :py:obj:`True`, function clipped_bands_from_logs is tried first in order to retrieve the clipped bands
        from the logs.

    Returns
    -------
    namedtuple
        Fields of the namedtuple:
        - lead, WBand object for the wavelength band of the lead pulse
        - skipped, Wband for the skipped pulse. None if not operating in the skipped frame mode
    """
    # First see if the clipped bands are already present in the logs
    if search_in_logs is True:
        try:
            bands = clipped_bands_from_logs(input_workspace)
            return dict(lead=bands.lead, skip=bands.skip)
        except AttributeError:
            logger.warning(
                "Clipped bands not present in the logs. Calculating using the chopper settings."
            )

    # If necessary, retrieve the clips from the logs
    sample_logs = SampleLogs(input_workspace)
    if low_tof_clip is None:
        low_tof_clip = sample_logs.low_tof_clip.value
    if high_tof_clip is None:
        high_tof_clip = sample_logs.low_tof_clip.value

    # If necessary, retrieve the source_detector_distance from the input workspace
    if source_detector_dist is None:
        source_detector_dist = source_detector_distance(input_workspace, unit="m")

    ch = EQSANSDiskChopperSet(input_workspace)  # object representing the four choppers
    lwc = wlg.from_tof(
        low_tof_clip, source_detector_dist, ch.pulse_width
    )  # low wavel. clip
    hwc = wlg.from_tof(high_tof_clip, source_detector_dist)  # high wavelength clip
    bands = transmitted_bands(input_workspace)
    if ch.frame_mode == FrameMode.not_skip:
        lead = wlg.Wband(bands.lead.min + lwc, bands.lead.max - hwc)
        skip = None
    else:
        l_i = bands.lead.max - hwc if interior_clip is True else bands.lead.max
        l_a = bands.skip.min + lwc if interior_clip is True else bands.skip.min
        lead = wlg.Wband(bands.lead.min + lwc, l_i)
        skip = wlg.Wband(l_a, bands.skip.max - hwc)
    return dict(lead=lead, skip=skip)


def log_tof_structure(input_workspace, low_tof_clip, high_tof_clip, interior_clip=True):
    r"""
    Append to the logs relevant information about the time of flight
    frame and structure

    Insert clipping times

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
        Input workspace to amend its logs
    sdd: float
        Distance from source to detector, in meters
    low_tof_clip: float
        trim neutrons of the leading pulse with a TOF smaller than the
        minimal TOF plus this value. Units in micro-seconds
    high_tof_clip: float
        trim neutrons of the leading pulse with TOF bigger than the maximal
        TOF minus this value.  Units in micro-seconds
    interior_clip: bool
        If True, also trim slow neutrons from the lead pulse (using `htc`) and
        fast neutrons from the skip pulse (using `ltc`)

    Returns
    -------
    ~mantid.api.MatrixWorkspace
        The input workspace, with the logs ammended
    """
    ws = mtd[str(input_workspace)]
    ch = EQSANSDiskChopperSet(ws)
    sample_logs = SampleLogs(ws)
    sample_logs.insert("low_tof_clip", low_tof_clip, unit="ms")
    sample_logs.insert("high_tof_clip", high_tof_clip, unit="ms")
    sample_logs.insert("tof_frame_width", ch.period, unit="ms")
    clip_times = 1 if interior_clip is False else 2
    tof_width_clipped = ch.period - clip_times * (low_tof_clip + high_tof_clip)
    sample_logs.insert("tof_frame_width_clipped", tof_width_clipped, unit="ms")
    return ws


def log_band_structure(input_workspace, bands):
    r"""
    Insert bands information in the logs

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
    bands: namedtuple
        Output of running `transmitted_bands_clipped` on the workspace
    """
    is_frame_skipping = _is_frame_skipping(input_workspace)
    sample_logs = SampleLogs(input_workspace)
    w_min = bands.lead.min
    w_max = bands.skip.max if is_frame_skipping else bands.lead.max
    sample_logs.insert("wavelength_min", w_min, unit="Angstrom")
    sample_logs.insert("wavelength_max", w_max, unit="Angstrom")
    sample_logs.insert("wavelength_lead_min", bands.lead.min, unit="Angstrom")
    sample_logs.insert("wavelength_lead_max", bands.lead.max, unit="Angstrom")
    if is_frame_skipping:
        sample_logs.insert("wavelength_skip_min", bands.skip.min, unit="Angstrom")
        sample_logs.insert("wavelength_skip_max", bands.skip.max, unit="Angstrom")


@namedtuplefy
def metadata_bands(input_workspace):
    r"""
    Scan the logs for the wavelength bands of the lead and skipped pulses transmitted by the choppers

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
    Returns
    -------
    namedtuple
        Fields of the namedtuple:
        - lead, WBand object for the wavelength band of the lead pulse
        - skipped, Wband for the skipped pulse. None if not operating in the skipped frame mode
    """
    sample_logs = SampleLogs(input_workspace)
    try:
        lead = wlg.Wband(
            sample_logs.wavelength_lead_min.value, sample_logs.wavelength_lead_max.value
        )
    except AttributeError as e:
        raise RuntimeError("Band structure not found in the logs") from e
    skip = None
    if _is_frame_skipping(input_workspace):
        try:
            skip = wlg.Wband(
                sample_logs.wavelength_skip_min.value,
                sample_logs.wavelength_skip_max.value,
            )
        except AttributeError as e:
            raise RuntimeError(
                "Bands from the skipped pulse missing in the logs"
            ) from e

    return dict(lead=lead, skip=skip)


def correct_tof_frame(
    input_workspace, source_to_component_distance, path_to_pixel=True
):
    r"""
    Assign the correct TOF to each event.

    TOFS may need to be shifted one (or more) frame widths and if working
    in frame skipped mode, one pulse period.

    Parameters
    ----------
    input_workspace: str, ~mantid.api.IEventsWorkspace
        Data workspace
    source_to_component_distance: float
        Distance from source to detecting component (detector or monitor), in meters
    path_to_pixel: bool
        When correcting the recorded time of flight of each neutron, use the path from the moderator
        to the detector pixel (`True`) or to the center of the detector panel (`False`). The latter
        for comparison to the old data reduction.
    Returns
    -------
    namedtuple
        Fields of the namedtuple:
        - ws: ~mantid.api.IEventsWorkspace, the input workspace
        - lead_band: WBand wavelength band for the lead pulse
        - skip_band: WBand wavelength band for the skip pulse. `None` if not working in frame-skipping mode
    """
    ws = mtd[str(input_workspace)]
    sl = SampleLogs(ws)
    pulse_period = 1.0e6 / sl.frequency.value.mean()  # 10^6/60 micro-seconds
    ch = EQSANSDiskChopperSet(ws)  # object representing the four choppers
    # The TOF values recorded are never bigger than the frame width,
    # which is also the choppers' rotational period.
    frame_width = ch.period  # either 10^6/60 or 10^6/30 micro-seconds

    tof_min, tof_max = limiting_tofs(ws, source_to_component_distance).lead
    EQSANSCorrectFrame(
        ws,
        PulsePeriod=pulse_period,
        MinTOF=tof_min,
        FrameWidth=frame_width,
        FrameSkipping=(ch.frame_mode is FrameMode.skip),
        PathToPixel=path_to_pixel,
    )
    # Amend the logs
    fr_skip = 1 if ch.frame_mode == FrameMode.skip else 0
    sl.insert("is_frame_skipping", fr_skip)


def correct_detector_frame(ws, path_to_pixel=True):
    correct_tof_frame(
        ws, source_detector_distance(ws, unit="m"), path_to_pixel=path_to_pixel
    )


def correct_monitor_frame(input_workspace):
    r"""
    Assign the correct TOF to each event.

    Parameters
    ----------
    input_workspace: ~mantid.api.IEventsWorkspace
        Monitor events workspace

    Returns
    -------
    ~mantid.api.IEventsWorkspace
    """
    # check we are not running in skip-frame mode
    ws = mtd[str(input_workspace)]
    if EQSANSDiskChopperSet(ws).frame_mode == FrameMode.skip:
        raise RuntimeError('cannot correct monitor frame in "skip" mode')
    # correct TOF's
    correct_tof_frame(ws, source_monitor_distance(ws, unit="m"), path_to_pixel=False)


def correct_emission_time(input_workspace):
    r"""
    This correct the TOF values on a workspace for the moderator
    emission time as a function of wavelength.

    Parameters
    ----------
    input_workspace: ~mantid.api.IEventsWorkspace
        Data workspace
    """

    # set the t0 formula, this equation first converts the
    # incidentEnergy value in wavelength the calculates one of two
    # equations depending if the wavelength is greater than or less
    # than 2.
    #
    # if λ>=2 then 0.5*(231.99+6.4787λ-0.5233λ^2+0.0148λ^3)
    # else 0.5*(1280.5-7448.4*λ+16509*λ^2-17872*λ^3+10445*λ^4-3169.3*λ^5+392.31*λ^6)
    SetInstrumentParameter(
        Workspace=input_workspace,
        ParameterName="t0_formula",
        Value="incidentEnergy=sqrt(81.80420249996277/incidentEnergy), (incidentEnergy < 2.0) ? 0.5*(1280.5-7448.4*incidentEnergy+16509*incidentEnergy^2-17872*incidentEnergy^3+10445*incidentEnergy^4-3169.3*incidentEnergy^5+392.31*incidentEnergy^6) : 0.5*(231.99+6.4797*incidentEnergy-0.5233*incidentEnergy^2+0.0148*incidentEnergy^3)",
    )
    ModeratorTzero(
        InputWorkspace=input_workspace,
        OutputWorkspace=input_workspace,
        EMode="Elastic",
        Niter=10,
    )


def correct_tof_offset(input_workspace):
    r"""
    This corrects the TOF offset when running in 60Hz, non-frameskip. A
    constant 664.7us needs to be removed, otherwise nothing is changed.

    Parameters
    ----------
    input_workspace: ~mantid.api.IEventsWorkspace
        Data workspace
    """
    # If 60Hz (non-frameskip) then subtract 664.7us from all tofs
    if not _is_frame_skipping(input_workspace):
        ChangeBinOffset(
            InputWorkspace=input_workspace,
            OutputWorkspace=input_workspace,
            Offset=-664.7,
        )


def smash_monitor_spikes(input_workspace, output_workspace=None):
    r"""
    Detect and remove spikes in monitor data TOF between min and max TOF's.
    These spikes will spoil rebinning to wavelength

    This function will transform to histogram data.

    Parameters
    ----------
    input_workspace: ~mantid.api.IEventsWorkspace
        Monitor events workspace
    output_workspace : str
        Name of the normalized workspace. If None, the name of the input
        workspace is chosen (the input workspace is overwritten).

    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """

    def remove_spikes(y, to_median=20):
        median = np.median(np.abs(y[1:] - y[:-1]))
        spikes_idx = np.ones(1)
        while np.any(spikes_idx):
            diff = y[1:] - y[:-1]
            spikes_idx = 1 + np.where(diff > median * to_median)[0]
            y[spikes_idx] = y[spikes_idx - 1]

    w = mtd[str(input_workspace)]
    if output_workspace is None:
        output_workspace = str(input_workspace)

    # We detect spikes in histogram mode
    bin_width = 1  # 1 micro second
    w = Rebin(
        input_workspace,
        Params=bin_width,
        PreserveEvents=False,
        OutputWorkspace=output_workspace,
    )

    # Find intensities in the range of valid time-of-flight
    intensity = w.dataY(0)
    tofs = w.dataX(0)
    smd = source_monitor_distance(w, unit="m")
    tof_min, tof_max = limiting_tofs(w, smd).lead
    valid_idx = np.where(np.logical_and(tofs >= tof_min, tofs <= tof_max))[0]
    if valid_idx[-1] == len(intensity):
        valid_idx = valid_idx[:-1]  # dataX is one element longer than dataY
    intensity = intensity[valid_idx]

    # Is this a flat monitor?
    if len(np.where(intensity < 1)[0]) > 0.1 * len(intensity):
        raise RuntimeError("Monitor spectrum is flat")

    remove_spikes(intensity)
    w.dataY(0)[valid_idx] = intensity

    # reset the uncertainties now that the data has been modified
    w = set_init_uncertainties(w)

    return w


def band_gap_indexes(input_workspace, bands):
    r"""
    Convention to define the indexes of the gap band.

    For runs in skipped frame mode, there is a wavelength band in between
    the bands of the lead and skipped pulse. This range has zero neutron
    counts. This function returns the indexes of `ws.dataY(0)` array
    corresponding to the band gap.

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
        Input workspace with units of Wavelength on the X-axis
    bands: namedtuple
        Output of running `transmitted_bands_clipped` on the workspace
    Returns
    -------
    list
        Indexes of array `ws.dataY(i)` where intensity is zero. Empty list if
        working frame mode is not skipped-mode
    """
    ws = mtd[str(input_workspace)]
    if bands.skip is None:
        return list()
    else:
        return (
            np.where((ws.dataX(0) > bands.lead.max) & (ws.dataX(0) < bands.skip.min))[0]
        ).tolist()


# flake8: noqa: C901
def convert_to_wavelength(
    input_workspace, bands=None, bin_width=0.1, events=True, output_workspace=None
):
    r"""
    Convert a time-of-flight events workspace to a wavelength workspace

    Parameters
    ----------
    input_workspace: ~mantid.api.IEventsWorkspace
        Input workspace, after TOF frame correction
    bands: namedtuple
        Output of running `transmitted_bands_clipped` on the workspace. If None, the
        band structure will be read from the logs.
    bin_width: float
        Bin width in Angstroms
    events: bool
        Do we preserve events?
    Returns
    -------
    ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
    """
    input_workspace = str(input_workspace)
    if output_workspace is None:
        output_workspace = str(input_workspace)

    ConvertUnits(
        InputWorkspace=input_workspace,
        Target="Wavelength",
        Emode="Elastic",
        OutputWorkspace=output_workspace,
    )

    # Rebin to the clipped bands
    is_frame_skipping = _is_frame_skipping(input_workspace)
    w_min, w_max = None, None
    if bands is None:
        try:
            bands = metadata_bands(input_workspace)
        except RuntimeError:
            # metadata not set get from the workspace
            pass
    else:
        w_min = bands.lead.min
        w_max = bands.skip.max if is_frame_skipping else bands.lead.max
    if bin_width:
        if w_min is not None and w_max is not None:
            params = (w_min, bin_width, w_max)
        else:
            params = bin_width

        Rebin(
            InputWorkspace=output_workspace,
            Params=params,
            PreserveEvents=events,
            OutputWorkspace=output_workspace,
        )
    else:
        # crop the workspace if wavelength range is found
        kwargs = dict()
        if w_min is not None:
            kwargs["XMin"] = w_min
        if w_max is not None:
            kwargs["XMax"] = w_max
        if kwargs:
            CropWorkspace(
                InputWorkspace=output_workspace,
                OutputWorkspace=output_workspace,
                **kwargs
            )

    # Discard neutrons in between bands.lead.max and bands.skip.min
    if is_frame_skipping:
        # for some reason event and histogram mode of MaskBins do not give the same result
        # they differ by 2-3 events in the tests
        if mtd[output_workspace].id() == "EventWorkspace":
            MaskBins(
                InputWorkspace=output_workspace,
                OutputWorkspace=output_workspace,
                XMax=bands.skip.min,
                XMin=bands.lead.max,
            )
        else:  # histogram
            _ws = mtd[output_workspace]
            to_zero = band_gap_indexes(_ws, bands)
            if to_zero:
                for i in range(_ws.getNumberHistograms()):
                    _ws.dataY(i)[to_zero] = 0.0
                    _ws.dataE(i)[to_zero] = 1.0

    # convert to a histogram if requested
    if (mtd[output_workspace].id() == "EventWorkspace") and (not events):
        RebinToWorkspace(
            WorkspaceToRebin=output_workspace,
            WorkspaceToMatch=output_workspace,
            OutputWorkspace=output_workspace,
            PreserveEvents=False,
        )

    return mtd[output_workspace]


def transform_to_wavelength(
    input_workspace,
    bin_width=0.1,
    low_tof_clip=0.0,
    high_tof_clip=0.0,
    keep_events=True,
    interior_clip=True,
    output_workspace=None,
    bands=None,
):
    r"""
    API function that converts corrected TOF's to Wavelength.

    Parameters
    ----------
    input_workspace: str, ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Events workspace in time-of-flight
    bin_width: float
        Bin width for the output workspace, in Angstroms.
    low_tof_clip: float
        Ignore events with a time-of-flight (TOF) smaller than the minimal
        TOF plus this quantity.
    high_tof_clip: float
        Ignore events with a time-of-flight (TOF) bigger than the maximal
        TOF minus this quantity.
    keep_events: bool
        The final histogram will be an EventsWorkspace if True.
    interior_clip: bool
        If True, trim slow neutrons from the lead pulse (using
        ``high_tof_clip``) and fast neutrons from the skip pulse (using
        ``low_tof_clip``)
    output_workspace: str
        Name of the output workspace. If None, the input_workspace will be
        overwritten.

    Returns
    -------
    tuple
        ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
    """
    input_workspace = mtd[str(input_workspace)]
    if output_workspace is None:
        output_workspace = str(input_workspace)

    # generate bands if not given
    if bands is None:
        if low_tof_clip > 0.0 or high_tof_clip > 0.0:
            sdd = source_detector_distance(input_workspace, unit="m")
            bands = transmitted_bands_clipped(
                input_workspace, sdd, low_tof_clip, high_tof_clip
            )
        else:
            bands = transmitted_bands(input_workspace)

    # convert input workspaces to wavelength
    convert_to_wavelength(
        input_workspace,
        bands=bands,
        bin_width=bin_width,
        events=keep_events,
        output_workspace=output_workspace,
    )
    log_band_structure(output_workspace, bands)
    w = log_tof_structure(
        output_workspace, low_tof_clip, high_tof_clip, interior_clip=interior_clip
    )

    return w, bands
