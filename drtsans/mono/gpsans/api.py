""" GPSANS API """
import copy
from datetime import datetime
import os
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mantid.simpleapi import mtd, MaskDetectors, logger, SaveNexusProcessed, MoveInstrumentComponent

from drtsans import getWedgeSelection
from drtsans.path import abspath, abspaths, registered_workspace
from drtsans.instruments import extract_run_number, instrument_filesystem_name
from drtsans.settings import namedtuplefy
from drtsans.samplelogs import SampleLogs
from drtsans.plots import plot_IQmod, plot_IQazimuthal, plot_detector
from drtsans.reductionlog import savereductionlog
from drtsans.solid_angle import solid_angle_correction
from drtsans.beam_finder import center_detector, find_beam_center, fbc_options_json
from drtsans.mask_utils import apply_mask, load_mask
from drtsans.mono.load import (
    load_events,
    transform_to_wavelength,
    load_events_and_histogram,
    load_and_split,
    set_init_uncertainties,
)
from drtsans.mono.normalization import normalize_by_monitor, normalize_by_time
from drtsans.mono.dark_current import subtract_dark_current
from drtsans.sensitivity import apply_sensitivity_correction, load_sensitivity_workspace
from drtsans.mono.transmission import (
    apply_transmission_correction,
    calculate_transmission,
)
from drtsans.thickness_normalization import normalize_by_thickness
from drtsans.mono.absolute_units import empty_beam_scaling
from drtsans.mono.gpsans.attenuation import attenuation_factor
from drtsans.mono.gpsans import convert_to_q
from drtsans import subtract_background
from drtsans.iq import bin_all
from drtsans.save_ascii import save_ascii_binned_2D
from drtsans.dataobjects import save_iqmod
from drtsans.mono.meta_data import set_meta_data, get_sample_detector_offset
from drtsans.load import move_instrument
from drtsans.path import allow_overwrite
from drtsans.mono.meta_data import parse_json_meta_data
import drtsans.mono.meta_data as meta_data

# Functions exposed to the general user (public) API
__all__ = [
    "prepare_data",
    "prepare_data_workspaces",
    "process_single_configuration",
    "load_all_files",
    "plot_reduction_output",
    "reduce_single_configuration",
    'adjust_back_panels_to_effective_position',
]


SI_WINDOW_NOMINAL_DISTANCE_METER = 0.0  # meter, (i.e., 0. mm)
SAMPLE_SI_META_NAME = "CG2:CS:SampleToSi"


@namedtuplefy
def load_all_files(
    reduction_input,
    prefix="",
    load_params=None,
    path=None,
    use_nexus_idf: bool = False,
    debug_output: bool = False,
    back_panel_correction: bool = False
):
    """load all required files at the beginning, and transform them to histograms

    Parameters
    ----------
    reduction_input
    prefix
    load_params
    path : str or None
        Path to search the NeXus file
    use_nexus_idf: bool
        Flag to use IDF inside NeXus file.  True for SPICE data
    debug_output: bool
        Flag to save out internal result
    back_panel_correction: bool
        Move the z direction of back panel to remove artifacts in azimuthal plots

    Returns
    -------

    """
    reduction_config = reduction_input["configuration"]

    instrument_name = reduction_input["instrumentName"]
    ipts = reduction_input["iptsNumber"]
    sample = reduction_input["sample"]["runNumber"]
    sample_trans = reduction_input["sample"]["transmission"]["runNumber"]
    bkgd = reduction_input["background"]["runNumber"]
    bkgd_trans = reduction_input["background"]["transmission"]["runNumber"]
    empty = reduction_input["emptyTransmission"]["runNumber"]
    center = reduction_input["beamCenter"]["runNumber"]
    blocked_beam = reduction_config["blockedBeamRunNumber"]

    # Remove existing workspaces, this is to guarantee that all the data is loaded correctly
    # In the future this should be made optional
    ws_to_remove = [
        f"{prefix}_{instrument_name}_{run_number}_raw_histo"
        for run_number in (
            sample,
            center,
            bkgd,
            empty,
            sample_trans,
            bkgd_trans,
            blocked_beam,
        )
    ]
    ws_to_remove.append(f"{prefix}_{instrument_name}_{sample}_raw_histo_slice_group")
    ws_to_remove.append(f"{prefix}_sensitivity")
    ws_to_remove.append(f"{prefix}_mask")
    if reduction_config["darkFileName"]:
        run_number = extract_run_number(reduction_config["darkFileName"])
        ws_to_remove.append(f"{prefix}_{instrument_name}_{run_number}_raw_histo")
    for ws_name in ws_to_remove:
        if registered_workspace(ws_name):
            mtd.remove(ws_name)

    # sample offsets, etc
    if load_params is None:
        load_params = {}
    # load nexus idf
    if use_nexus_idf:
        load_params["LoadNexusInstrumentXML"] = use_nexus_idf

    # Adjust pixel heights and widths
    load_params["pixel_calibration"] = reduction_config.get(
        "usePixelCalibration", False
    )

    # wave length and wave length spread
    (
        wave_length_dict,
        wave_length_spread_dict,
    ) = meta_data.parse_json_wave_length_and_spread(reduction_input)

    if reduction_config["useDefaultMask"]:
        # reduction_config["defaultMask"] is a list of python dictionaries
        default_mask = (
            reduction_config["defaultMask"]
            if reduction_config["defaultMask"] is not None
            else []
        )
    else:
        default_mask = []

    if path is None:
        path = f"/HFIR/{instrument_filesystem_name(instrument_name)}/IPTS-{ipts}/nexus"
    assert os.path.exists(path), "NeXus file path {} does not exist".format(path)

    # check for time/log slicing
    timeslice = reduction_config["useTimeSlice"]
    logslice = reduction_config["useLogSlice"]
    if timeslice or logslice:
        if len(sample.split(",")) > 1:
            raise ValueError("Can't do slicing on summed data sets")

    # sample thickness
    # thickness is written to sample log if it is defined...
    thickness = reduction_input["sample"]["thickness"]

    # sample aperture diameter in mm
    sample_aperture_diameter = reduction_config["sampleApertureSize"]
    # source aperture diameter in mm
    source_aperture_diameter = reduction_config["sourceApertureDiameter"]

    # Overwriting pixel size X and pixel size Y
    smearing_pixel_size_x_dict = parse_json_meta_data(
        reduction_input,
        "smearingPixelSizeX",
        1e-3,
        beam_center_run=True,
        background_run=True,
        empty_transmission_run=True,
        transmission_run=True,
        background_transmission=True,
        block_beam_run=True,
        dark_current_run=True,
    )

    smearing_pixel_size_y_dict = parse_json_meta_data(
        reduction_input,
        "smearingPixelSizeY",
        1e-3,
        beam_center_run=True,
        background_run=True,
        empty_transmission_run=True,
        transmission_run=True,
        background_transmission=True,
        block_beam_run=True,
        dark_current_run=True,
    )

    # Retrieve parameters for overwriting geometry related meta data
    # Sample to Si-window distance
    swd_value_dict = parse_json_meta_data(
        reduction_input,
        "sampleToSi",
        1e-3,
        beam_center_run=True,
        background_run=True,
        empty_transmission_run=True,
        transmission_run=True,
        background_transmission=True,
        block_beam_run=True,
        dark_current_run=False,
    )
    # Sample to detector distance
    sdd_value_dict = parse_json_meta_data(
        reduction_input,
        "sampleDetectorDistance",
        1.0,
        beam_center_run=True,
        background_run=True,
        empty_transmission_run=True,
        transmission_run=True,
        background_transmission=True,
        block_beam_run=True,
        dark_current_run=False,
    )

    # special loading case for sample to allow the slicing options
    logslice_data_dict = {}
    if timeslice or logslice:
        # Load data and split
        ws_name = f"{prefix}_{instrument_name}_{sample}_raw_histo_slice_group"
        if not registered_workspace(ws_name):
            filename = abspath(
                sample.strip(), instrument=instrument_name, ipts=ipts, directory=path
            )
            logger.notice(f"Loading filename {filename} to slice")
            if timeslice:
                timesliceinterval = reduction_config["timeSliceInterval"]
                logslicename = logsliceinterval = None
            elif logslice:
                timesliceinterval = None
                logslicename = reduction_config["logslicename"]
                logsliceinterval = reduction_config["logsliceinterval"]
            load_and_split(
                filename,
                output_workspace=ws_name,
                time_interval=timesliceinterval,
                log_name=logslicename,
                log_value_interval=logsliceinterval,
                sample_to_si_name=SAMPLE_SI_META_NAME,
                si_nominal_distance=SI_WINDOW_NOMINAL_DISTANCE_METER,
                sample_to_si_value=swd_value_dict[meta_data.SAMPLE],
                sample_detector_distance_value=sdd_value_dict[meta_data.SAMPLE],
                **load_params,
            )
            for _w in mtd[ws_name]:
                # Overwrite meta data
                set_meta_data(
                    str(_w),
                    wave_length=wave_length_dict[meta_data.SAMPLE],
                    wavelength_spread=wave_length_spread_dict[meta_data.SAMPLE],
                    sample_thickness=thickness,
                    sample_aperture_diameter=sample_aperture_diameter,
                    source_aperture_diameter=source_aperture_diameter,
                    smearing_pixel_size_x=smearing_pixel_size_x_dict[meta_data.SAMPLE],
                    smearing_pixel_size_y=smearing_pixel_size_y_dict[meta_data.SAMPLE],
                )
                # Transform X-axis to wave length with spread
                _w = transform_to_wavelength(_w)
                _w = set_init_uncertainties(_w)
                for btp_params in default_mask:
                    apply_mask(_w, **btp_params)

                if not (logslicename is None):
                    for n in range(mtd[ws_name].getNumberOfEntries()):
                        samplelogs = SampleLogs(mtd[ws_name].getItem(n))
                        logslice_data_dict[str(n)] = {
                            "data": list(samplelogs[logslicename].value),
                            "units": samplelogs[logslicename].units,
                            "name": logslicename,
                        }
    else:
        # Load single data
        ws_name = f"{prefix}_{instrument_name}_{sample}_raw_histo"
        if not registered_workspace(ws_name):
            filename = abspaths(
                sample, instrument=instrument_name, ipts=ipts, directory=path
            )
            logger.notice(f"Loading filename {filename} to {ws_name}")
            load_events_and_histogram(
                filename,
                output_workspace=ws_name,
                sample_to_si_name=SAMPLE_SI_META_NAME,
                si_nominal_distance=SI_WINDOW_NOMINAL_DISTANCE_METER,
                sample_to_si_value=swd_value_dict[meta_data.SAMPLE],
                sample_detector_distance_value=sdd_value_dict[meta_data.SAMPLE],
                **load_params,
            )
            # Overwrite meta data
            set_meta_data(
                ws_name,
                wave_length=wave_length_dict[meta_data.SAMPLE],
                wavelength_spread=wave_length_spread_dict[meta_data.SAMPLE],
                sample_thickness=thickness,
                sample_aperture_diameter=sample_aperture_diameter,
                source_aperture_diameter=source_aperture_diameter,
                smearing_pixel_size_x=smearing_pixel_size_x_dict[meta_data.SAMPLE],
                smearing_pixel_size_y=smearing_pixel_size_y_dict[meta_data.SAMPLE],
            )
            # Re-transform to wave length if overwriting values are specified
            if wave_length_dict[meta_data.SAMPLE]:
                transform_to_wavelength(ws_name)
            logger.information(
                "[META] Wavelength range is from {} to {}"
                "".format(mtd[ws_name].readX(0)[0], mtd[ws_name].readX(0)[1])
            )
            # Apply mask
            for btp_params in default_mask:
                apply_mask(ws_name, **btp_params)

    reduction_input["logslice_data"] = logslice_data_dict

    # load all other files
    for run_number, run_type in [
        (center, meta_data.BEAM_CENTER),
        (bkgd, meta_data.BACKGROUND),
        (empty, meta_data.EMPTY_TRANSMISSION),
        (sample_trans, meta_data.TRANSMISSION),
        (bkgd_trans, meta_data.TRANSMISSION_BACKGROUND),
        (blocked_beam, meta_data.BLOCK_BEAM),
    ]:
        if run_number:
            ws_name = f"{prefix}_{instrument_name}_{run_number}_raw_histo"
            if not registered_workspace(ws_name):
                filename = abspaths(
                    run_number, instrument=instrument_name, ipts=ipts, directory=path
                )
                logger.notice(f"Loading {run_type} filename {filename} to {ws_name}")
                load_events_and_histogram(
                    filename,
                    output_workspace=ws_name,
                    sample_to_si_name=SAMPLE_SI_META_NAME,
                    si_nominal_distance=SI_WINDOW_NOMINAL_DISTANCE_METER,
                    sample_to_si_value=swd_value_dict[run_type],
                    sample_detector_distance_value=sdd_value_dict[run_type],
                    **load_params,
                )
                # Set the wave length and wave length spread
                set_meta_data(
                    ws_name,
                    wave_length=wave_length_dict[run_type],
                    wavelength_spread=wave_length_spread_dict[run_type],
                    sample_thickness=None,
                    sample_aperture_diameter=None,
                    source_aperture_diameter=None,
                    smearing_pixel_size_x=smearing_pixel_size_x_dict[run_type],
                    smearing_pixel_size_y=smearing_pixel_size_y_dict[run_type],
                )
                if wave_length_dict[run_type]:
                    # Transform X-axis to wave length with spread
                    transform_to_wavelength(ws_name)
                for btp_params in default_mask:
                    apply_mask(ws_name, **btp_params)

    # do the same for dark current if exists
    dark_current_file = reduction_config["darkFileName"]
    if dark_current_file:
        # dark current file is specified and thus loaded
        run_number = extract_run_number(dark_current_file)
        ws_name = f"{prefix}_{instrument_name}_{run_number}_raw_histo"
        if not registered_workspace(ws_name):
            # load dark current file
            logger.notice(f"Loading dark current file {dark_current_file} to {ws_name}")
            # identify to use exact given path to NeXus or use OnCat instead
            temp_name = os.path.join(
                path, "{}_{}.nxs.h5".format(instrument_name, run_number)
            )
            if os.path.exists(temp_name):
                dark_current_file = temp_name
            load_events_and_histogram(
                dark_current_file,
                output_workspace=ws_name,
                sample_to_si_name=SAMPLE_SI_META_NAME,
                si_nominal_distance=SI_WINDOW_NOMINAL_DISTANCE_METER,
                sample_to_si_value=swd_value_dict[meta_data.DARK_CURRENT],
                sample_detector_distance_value=sdd_value_dict[meta_data.DARK_CURRENT],
                **load_params,
            )
            # Set the wave length and wave length spread
            set_meta_data(
                ws_name,
                wave_length=wave_length_dict[meta_data.DARK_CURRENT],
                wavelength_spread=wave_length_spread_dict[meta_data.DARK_CURRENT],
                sample_thickness=None,
                sample_aperture_diameter=None,
                source_aperture_diameter=None,
                smearing_pixel_size_x=smearing_pixel_size_x_dict[
                    meta_data.DARK_CURRENT
                ],
                smearing_pixel_size_y=smearing_pixel_size_y_dict[
                    meta_data.DARK_CURRENT
                ],
            )
            # Re-transform X-axis to wave length with spread
            if wave_length_dict[meta_data.DARK_CURRENT]:
                transform_to_wavelength(ws_name)
            for btp_params in default_mask:
                apply_mask(ws_name, **btp_params)
            dark_current = mtd[ws_name]
        else:
            # being loaded previously
            dark_current = mtd[ws_name]
    else:
        # No dark current (correction) is specified
        dark_current = None

    # load required processed_files
    sensitivity_ws_name = None
    flood_file = reduction_config["sensitivityFileName"]
    if flood_file:
        sensitivity_ws_name = f"{prefix}_sensitivity"
        if not registered_workspace(sensitivity_ws_name):
            logger.notice(f"Loading flood file {flood_file} to {sensitivity_ws_name}")
            load_sensitivity_workspace(flood_file, output_workspace=sensitivity_ws_name)

    mask_ws = None
    custom_mask_file = reduction_config["maskFileName"]
    if custom_mask_file is not None:
        mask_ws_name = f"{prefix}_mask"
        if not registered_workspace(mask_ws_name):
            logger.notice(
                f"Loading user mask file {custom_mask_file} to {mask_ws_name}"
            )
            mask_ws = load_mask(custom_mask_file, output_workspace=mask_ws_name)
        else:
            mask_ws = mtd[mask_ws_name]

    if registered_workspace(
        f"{prefix}_{instrument_name}_{sample}_raw_histo_slice_group"
    ):
        raw_sample_ws = mtd[
            f"{prefix}_{instrument_name}_{sample}_raw_histo_slice_group"
        ]
        raw_sample_ws_list = [w for w in raw_sample_ws]
    else:
        raw_sample_ws_list = [mtd[f"{prefix}_{instrument_name}_{sample}_raw_histo"]]
    raw_bkgd_ws = mtd[f"{prefix}_{instrument_name}_{bkgd}_raw_histo"] if bkgd else None
    raw_blocked_ws = (
        mtd[f"{prefix}_{instrument_name}_{blocked_beam}_raw_histo"]
        if blocked_beam
        else None
    )
    raw_center_ws = mtd[f"{prefix}_{instrument_name}_{center}_raw_histo"]
    raw_empty_ws = (
        mtd[f"{prefix}_{instrument_name}_{empty}_raw_histo"] if empty else None
    )
    raw_sample_trans_ws = (
        mtd[f"{prefix}_{instrument_name}_{sample_trans}_raw_histo"]
        if sample_trans
        else None
    )
    raw_bkg_trans_ws = (
        mtd[f"{prefix}_{instrument_name}_{bkgd_trans}_raw_histo"]
        if bkgd_trans
        else None
    )
    sensitivity_ws = mtd[sensitivity_ws_name] if sensitivity_ws_name else None

    # Plot
    if debug_output:
        for raw_sample in raw_sample_ws_list:
            plot_detector(
                input_workspace=str(raw_sample),
                filename=form_output_name(raw_sample),
                backend="mpl",
            )
        for ws in [
            raw_bkgd_ws,
            raw_center_ws,
            raw_empty_ws,
            raw_sample_trans_ws,
            raw_bkg_trans_ws,
            raw_blocked_ws,
            dark_current,
        ]:
            if ws is not None:
                plot_detector(
                    input_workspace=str(ws),
                    filename=form_output_name(ws),
                    backend="mpl",
                )

    # Apply the backpanel correction on z direction to remove the
    # artifacts in 2D azimuthal patterns
    if back_panel_correction == True:
        for ws in raw_sample_ws_list:
            if ws is not None:
                adjust_back_panels_to_effective_position(ws)
        if raw_bkgd_ws is not None:
            adjust_back_panels_to_effective_position(raw_bkgd_ws)
        if raw_center_ws is not None:
            adjust_back_panels_to_effective_position(raw_center_ws)
        if raw_empty_ws is not None:
            adjust_back_panels_to_effective_position(raw_empty_ws)
        if raw_sample_trans_ws is not None:
            adjust_back_panels_to_effective_position(raw_sample_trans_ws)
        if raw_bkg_trans_ws is not None:
            adjust_back_panels_to_effective_position(raw_bkg_trans_ws)
        if raw_blocked_ws is not None:
            adjust_back_panels_to_effective_position(raw_blocked_ws)
        if dark_current is not None:
            adjust_back_panels_to_effective_position(dark_current)
        if sensitivity_ws is not None:
            adjust_back_panels_to_effective_position(sensitivity_ws)
        if mask_ws is not None:
            adjust_back_panels_to_effective_position(mask_ws)

    return dict(
        sample=raw_sample_ws_list,
        background=raw_bkgd_ws,
        center=raw_center_ws,
        empty=raw_empty_ws,
        sample_transmission=raw_sample_trans_ws,
        background_transmission=raw_bkg_trans_ws,
        blocked_beam=raw_blocked_ws,
        dark_current=dark_current,
        sensitivity=sensitivity_ws,
        mask=mask_ws,
    )


def prepare_data(
    data,
    pixel_calibration=False,
    mask_detector=None,
    detector_offset=0,
    sample_offset=0,
    center_x=None,
    center_y=None,
    dark_current=None,
    flux_method=None,
    monitor_fail_switch=False,
    mask=None,
    mask_panel=None,
    btp=dict(),
    solid_angle=True,
    sensitivity_file_path=None,
    sensitivity_workspace=None,
    wave_length=None,
    wavelength_spread=None,
    sample_aperture_diameter=None,
    sample_thickness=None,
    source_aperture_diameter=None,
    smearing_pixel_size_x=None,
    smearing_pixel_size_y=None,
    output_workspace=None,
    output_suffix="",
    **kwargs,
):
    r"""
    Load a GPSANS data file and bring the data to a point where it can be used. This includes applying basic
    corrections that are always applied regardless of whether the data is background or scattering data.

    Parameters
    ----------
    data: int, str, ~mantid.api.IEventWorkspace
        Run number as int or str, file path, :py:obj:`~mantid.api.IEventWorkspace`
    pixel_calibration: bool, str
        Adjust pixel heights and widths according to barscan and tube-width calibrations.
        Options are
        (1) No calibration (False), (2) default calibration file (True)
        (3) user specified calibration file (str)
    mask_detector: str
        Name of an instrument component to mask
    detector_offset: float
        Additional translation of the detector along Z-axis, in millimeters.
    sample_offset: float
        Additional translation of the sample along the Z-axis, in millimeters.
    center_x: float
        Move the center of the detector to this X-coordinate. If :py:obj:`None`, the
        detector will be moved such that the X-coordinate of the intersection
        point between the neutron beam and the detector array will have ``x=0``.
    center_y: float
        Move the center of the detector to this Y-coordinate. If :py:obj:`None`, the
        detector will be moved such that the Y-coordinate of the intersection
        point between the neutron beam and the detector array will have ``y=0``.
    dark_current: int, str, ~mantid.api.IEventWorkspace
        Run number as int or str, file path, :py:obj:`~mantid.api.IEventWorkspace`
    flux_method: str
        Method for flux normalization. Either 'monitor', or 'time'.
    monitor_fail_switch: bool
        Resort to 'time' normalization if 'monitor' was selected but no monitor counts are available
    mask_panel: str
        Either 'front' or 'back' to mask whole front or back panel.
    mask: mask file path, MaskWorkspace, list
        Additional mask to be applied. If `list`, it is a list of
        detector ID's.
    btp: dict
        Additional properties to Mantid's MaskBTP algorithm
    solid_angle: bool
        Apply the solid angle correction
    sensitivity_file_path: str
        file containing previously calculated sensitivity correction
    sensitivity_workspace: str, ~mantid.api.MatrixWorkspace
        workspace containing previously calculated sensitivity correction. This
        overrides the sensitivity_filename if both are provided.
    wave_length: float, None
        wave length in Angstrom
    wavelength_spread: float, None
        wave length spread in Angstrom
    sample_aperture_diameter: float, None
        sample aperture diameter in mm
    sample_thickness: None, float
        sample thickness in unit cm
    source_aperture_diameter: float, None
        source aperture size radius in unit mm
    smearing_pixel_size_x: float, None
        pixel size in x direction in unit as meter, only for Q-resolution calculation
    smearing_pixel_size_y: float, None
        pixel size in Y direction in unit as meter, only for Q-resolution calculation
    output_workspace: str
        Name of the output workspace. If not supplied, will be determined from the supplied value of ``data``.
    output_suffix: str
        If the ``output_workspace`` is not specified, this is appended to the automatically generated
        output workspace name.

    Returns
    -------
    ~mantid.api.IEventWorkspace
        Reference to the events workspace
    """
    # Detector offset and sample offset are disabled
    if abs(detector_offset) > 1e-8 or abs(sample_offset) > 1e-8:
        raise RuntimeError(
            "gpsans.api.prepare_data does not work with detector_offset or sample_offset"
        )

    # Load data and enforce to use nexus IDF
    if "enforce_use_nexus_idf" in kwargs:
        enforce_use_nexus_idf = kwargs["enforce_use_nexus_idf"]
    else:
        enforce_use_nexus_idf = False

    # GPSANS: detector offset is fixed to 0. Only detector sample distance is essential.
    #         So one offset is sufficient
    ws = load_events(
        data,
        overwrite_instrument=True,
        output_workspace=output_workspace,
        output_suffix=output_suffix,
        pixel_calibration=pixel_calibration,
        detector_offset=0,
        sample_offset=sample_offset,
        LoadNexusInstrumentXML=enforce_use_nexus_idf,
    )

    # Reset the offset
    sample_offset, detector_offset = get_sample_detector_offset(
        ws, SAMPLE_SI_META_NAME, SI_WINDOW_NOMINAL_DISTANCE_METER
    )
    # Translate instrument with offsets
    move_instrument(ws, sample_offset, detector_offset)

    ws_name = str(ws)
    transform_to_wavelength(ws_name)
    set_init_uncertainties(ws_name)

    if center_x is not None and center_y is not None:
        center_detector(ws_name, center_x=center_x, center_y=center_y)

    # Dark current
    if dark_current is not None:
        if registered_workspace(str(dark_current)):
            dark_ws = mtd[str(dark_current)]
        else:
            dark_ws = load_events(dark_current, overwrite_instrument=True)
            dark_ws = transform_to_wavelength(dark_ws)
            dark_ws = set_init_uncertainties(dark_ws)
    else:
        dark_ws = None

    # load sensitivity
    if sensitivity_workspace is None and sensitivity_file_path:
        sensitivity_workspace = os.path.split(sensitivity_file_path)[-1]
        sensitivity_workspace = sensitivity_workspace.split(".")[0]
        sensitivity_workspace = load_sensitivity_workspace(
            sensitivity_file_path, sensitivity_workspace
        )

    # Mask either detector
    if mask_detector is not None:
        MaskDetectors(ws_name, ComponentList=mask_detector)

    # Overwrite meta data
    set_meta_data(
        ws_name,
        wave_length,
        wavelength_spread,
        sample_offset,
        sample_aperture_diameter,
        sample_thickness,
        source_aperture_diameter,
        smearing_pixel_size_x,
        smearing_pixel_size_y,
    )

    return prepare_data_workspaces(
        ws_name,
        center_x=center_x,
        center_y=center_y,
        dark_current=dark_current,
        flux_method=flux_method,
        monitor_fail_switch=monitor_fail_switch,
        mask_ws=mask,
        mask_panel=mask_panel,
        mask_btp=btp,
        solid_angle=solid_angle,
        sensitivity_workspace=sensitivity_workspace,
        output_workspace_name=output_workspace,
        debug=False,
    )


def prepare_data_workspaces(
    data,
    center_x=None,
    center_y=None,
    dark_current=None,
    flux_method=None,  # normalization (time/monitor)
    monitor_fail_switch=False,
    mask_ws=None,  # apply a custom mask from workspace
    mask_panel=None,  # mask back or front panel
    mask_btp=None,  # mask bank/tube/pixel
    solid_angle=True,
    sensitivity_workspace=None,
    output_workspace_name=None,
    debug=False,
):
    r"""
    Given a " raw"data workspace, this function provides the following:

        - centers the detector
        - subtracts dark current
        - normalize by time or monitor
        - applies masks
        - corrects for solid angle
        - corrects for sensitivity

    All steps are optional. data, mask_ws, dark_current are either None
    or histogram workspaces. This function does not load any file.

    Parameters
    ----------
    data: ~mantid.dataobjects.Workspace2D
        raw workspace (histogram)
    center_x: float
        Move the center of the detector to this X-coordinate. If :py:obj:`None`, the
        detector will be moved such that the X-coordinate of the intersection
        point between the neutron beam and the detector array will have ``x=0``.
    center_y: float
        Move the center of the detector to this Y-coordinate. If :py:obj:`None`, the
        detector will be moved such that the Y-coordinate of the intersection
        point between the neutron beam and the detector array will have ``y=0``.
    dark_current: ~mantid.dataobjects.Workspace2D
        histogram workspace containing the dark current measurement
    flux_method: str
        Method for flux normalization. Either 'monitor', or 'time'.
    monitor_fail_switch: bool
        Resort to 'time' normalization if 'monitor' was selected but no monitor counts are available
    mask_ws: ~mantid.dataobjects.Workspace2D
        Mask workspace
    mask_panel: str
        Either 'front' or 'back' to mask whole front or back panel.
    mask_btp: dict
        Additional properties to Mantid's MaskBTP algorithm
    solid_angle: bool
        Apply the solid angle correction
    sensitivity_workspace: str, ~mantid.api.MatrixWorkspace
        workspace containing previously calculated sensitivity correction. This
        overrides the sensitivity_filename if both are provided.
    output_workspace_name: str
        The output workspace name. If None will create data.name()+output_suffix
    debug: bool
        Flag for debugging output

    Returns
    -------
    ~mantid.dataobjects.Workspace2D
        Reference to the processed workspace
    """
    if not data:
        raise RuntimeError("Input workspace of prepare data cannot be None!")

    if not output_workspace_name:
        output_workspace_name = str(data)
        output_workspace_name = (
            output_workspace_name.replace("_raw_histo", "") + "_processed_histo"
        )

    mtd[str(data)].clone(
        OutputWorkspace=output_workspace_name
    )  # name gets into workspace

    if center_x is not None and center_y is not None:
        center_detector(output_workspace_name, center_x=center_x, center_y=center_y)

    if debug:
        SaveNexusProcessed(
            InputWorkspace=output_workspace_name,
            Filename=f"{output_workspace_name}_after_center.nxs",
        )

    # Dark current
    if dark_current is not None:
        subtract_dark_current(output_workspace_name, dark_current)

    # Normalization
    if str(flux_method).lower() == "monitor":
        try:
            normalize_by_monitor(output_workspace_name)
        except RuntimeError as e:
            if monitor_fail_switch:
                logger.warning(f"{e}. Resorting to normalization by time")
                normalize_by_time(output_workspace_name)
            else:
                msg = (
                    '. Setting configuration "normalizationResortToTime": True will cause the'
                    " reduction to normalize by time if monitor counts are not available"
                )
                raise RuntimeError(str(e) + msg)
    elif str(flux_method).lower() == "time":
        normalize_by_time(output_workspace_name)
    else:
        logger.notice("No time or monitor normalization is carried out")

    # Additional masks
    if mask_btp is None:
        mask_btp = dict()
    if debug:
        # output masking information
        logger.notice(
            f"mask panel: {mask_panel}\n"
            f"mask ws   : {str(mask_ws)}\n"
            f"mask btp  : {mask_btp}"
        )
        if mask_ws is not None:
            SaveNexusProcessed(
                InputWorkspace=mask_ws,
                Filename=f"{output_workspace_name}_{str(mask_ws)}.nxs",
            )

    apply_mask(output_workspace_name, panel=mask_panel, mask=mask_ws, **mask_btp)
    if debug:
        SaveNexusProcessed(
            InputWorkspace=output_workspace_name,
            Filename=f"{output_workspace_name}_after_mask.nxs",
        )

    # Solid angle
    if solid_angle:
        solid_angle_correction(output_workspace_name)

    if debug:
        SaveNexusProcessed(
            InputWorkspace=output_workspace_name,
            Filename=f"{output_workspace_name}_before_sens.nxs",
        )

    # Sensitivity
    if sensitivity_workspace is not None:
        apply_sensitivity_correction(
            output_workspace_name, sensitivity_workspace=sensitivity_workspace
        )

    if debug:
        out_ws = mtd[output_workspace_name]
        plot_detector(
            input_workspace=str(out_ws),
            filename=f"prepared{form_output_name(out_ws)}",
            backend="mpl",
            imshow_kwargs={"norm": LogNorm(vmin=1)},
        )

    return mtd[output_workspace_name]


def process_single_configuration(
    sample_ws_raw,
    sample_trans_ws=None,
    sample_trans_value=None,
    bkg_ws_raw=None,
    bkg_trans_ws=None,
    bkg_trans_value=None,
    blocked_ws_raw=None,
    theta_deppendent_transmission=True,
    center_x=None,
    center_y=None,
    dark_current=None,
    flux_method=None,  # normalization (time/monitor)
    monitor_fail_switch=False,
    mask_ws=None,  # apply a custom mask from workspace
    mask_panel=None,  # mask back or front panel
    mask_btp=None,  # mask bank/tube/pixel
    solid_angle=True,
    sensitivity_workspace=None,
    output_workspace=None,
    output_suffix="",
    thickness=1.0,
    absolute_scale_method="standard",
    empty_beam_ws=None,
    beam_radius=None,
    absolute_scale=1.0,
    keep_processed_workspaces=True,
    debug=False,
):
    r"""
    This function provides full data processing for a single experimental configuration,
    starting from workspaces (no data loading is happening inside this function)

    Parameters
    ----------
    sample_ws_raw: ~mantid.dataobjects.Workspace2D
        raw data histogram workspace
    sample_trans_ws: ~mantid.dataobjects.Workspace2D
        optional histogram workspace for sample transmission
    sample_trans_value: float
        optional value for sample transmission
    bkg_ws_raw: ~mantid.dataobjects.Workspace2D
        optional raw histogram workspace for background
    bkg_trans_ws: ~mantid.dataobjects.Workspace2D
        optional histogram workspace for background transmission
    bkg_trans_value: float
        optional value for background transmission
    blocked_ws_raw: ~mantid.dataobjects.Workspace2D
        optional histogram workspace for blocked beam
    theta_deppendent_transmission: bool
        flag to apply angle dependent transmission
    center_x: float
        x center for the beam
    center_y: float
        y center for the beam
    dark_current: ~mantid.dataobjects.Workspace2D
        dark current workspace
    flux_method: str
        normalization by time or monitor
    monitor_fail_switch: bool
        resort to 'time' normalization if 'monitor' was selected but no monitor counts are available
    mask_ws: ~mantid.dataobjects.Workspace2D
        user defined mask
    mask_panel: str
        mask fron or back panel
    mask_btp: dict
        optional bank, tube, pixel to mask
    solid_angle: bool
        flag to apply solid angle
    sensitivity_workspace: ~mantid.dataobjects.Workspace2D
        workspace containing sensitivity
    output_workspace: str
        output workspace name
    output_suffix:str
        suffix for output workspace
    thickness: float
        sample thickness (cm)
    absolute_scale_method: str
        method to do absolute scaling (standard or direct_beam)
    empty_beam_ws: ~mantid.dataobjects.Workspace2D
        empty beam workspace for absolute scaling
    beam_radius: float
        beam radius for absolute scaling
    absolute_scale: float
        absolute scaling value for standard method
    keep_processed_workspaces: bool
        flag to keep the processed blocked beam and background workspaces
    debug: bool
        flag to do some debugging output

    Returns
    -------
    ~mantid.dataobjects.Workspace2D
        Reference to the processed workspace
    """
    if debug:
        SaveNexusProcessed(
            InputWorkspace=sample_ws_raw, Filename="sample_raw", Title="Raw"
        )

    if not output_workspace:
        output_workspace = output_suffix + "_sample.nxs"

    # create a common configuration for prepare data
    prepare_data_conf = {
        "center_x": center_x,
        "center_y": center_y,
        "dark_current": dark_current,
        "flux_method": flux_method,
        "monitor_fail_switch": monitor_fail_switch,
        "mask_ws": mask_ws,
        "mask_panel": mask_panel,
        "mask_btp": mask_btp,
        "solid_angle": solid_angle,
        "sensitivity_workspace": sensitivity_workspace,
    }

    # process blocked
    if blocked_ws_raw:
        blocked_ws_name = output_suffix + "_blocked"
        if not registered_workspace(blocked_ws_name):
            blocked_ws = prepare_data_workspaces(
                blocked_ws_raw,
                output_workspace_name=blocked_ws_name,
                debug=debug,
                **prepare_data_conf,
            )
        else:
            blocked_ws = mtd[blocked_ws_name]

    # process sample
    sample_ws = prepare_data_workspaces(
        sample_ws_raw,
        output_workspace_name=output_workspace,
        debug=debug,
        **prepare_data_conf,
    )

    if debug:
        SaveNexusProcessed(
            InputWorkspace=sample_ws,
            Filename="sample_prepared.nxs",
            Title=f"Prepared Data Workspace: From {str(sample_ws_raw)} to {str(sample_ws)}",
        )

    # raise NotImplementedError('DEBUG STOP')

    if blocked_ws_raw:
        sample_ws = subtract_background(sample_ws, blocked_ws)
        if debug:
            SaveNexusProcessed(
                InputWorkspace=sample_ws,
                Filename="sample_block_subtracted",
                Title="Block run subtracted",
            )

    # apply transmission to the sample
    if sample_trans_ws or sample_trans_value:
        sample_ws = apply_transmission_correction(
            sample_ws,
            trans_workspace=sample_trans_ws,
            trans_value=sample_trans_value,
            theta_dependent=theta_deppendent_transmission,
            output_workspace=output_workspace,
        )
        if debug:
            SaveNexusProcessed(
                InputWorkspace=sample_ws,
                Filename="sample_trans_correction",
                Title="Transmission corrected",
            )

    # process background, if not already processed
    if bkg_ws_raw:
        bkgd_ws_name = output_suffix + "_background"
        if not registered_workspace(bkgd_ws_name):
            bkgd_ws = prepare_data_workspaces(
                bkg_ws_raw, output_workspace_name=bkgd_ws_name, **prepare_data_conf
            )
            if blocked_ws_raw:
                bkgd_ws = subtract_background(bkgd_ws, blocked_ws)
            # apply transmission to bkgd
            if bkg_trans_ws or bkg_trans_value:
                bkgd_ws = apply_transmission_correction(
                    bkgd_ws,
                    trans_workspace=bkg_trans_ws,
                    trans_value=bkg_trans_value,
                    theta_dependent=theta_deppendent_transmission,
                    output_workspace=bkgd_ws_name,
                )
        else:
            bkgd_ws = mtd[bkgd_ws_name]
        # subtract background
        sample_ws = subtract_background(sample_ws, bkgd_ws)

        if not keep_processed_workspaces:
            bkgd_ws.delete()

    if blocked_ws_raw and not keep_processed_workspaces:
        blocked_ws.delete()

    # finalize with absolute scale and thickness
    sample_ws = normalize_by_thickness(sample_ws, thickness)

    # standard method assumes absolute scale from outside
    if absolute_scale_method == "direct_beam":
        try:
            empty = mtd[str(empty_beam_ws)]
        except KeyError:
            raise ValueError(f"Could not find empty beam {str(empty_beam_ws)}")

        ac, ace = attenuation_factor(empty)
        empty_beam_scaling(
            sample_ws,
            empty,
            beam_radius=beam_radius,
            unit="mm",
            attenuator_coefficient=ac,
            attenuator_error=ace,
            output_workspace=output_workspace,
        )
    else:
        sample_ws *= absolute_scale

    # Final debug output
    if debug:
        plot_detector(
            input_workspace=str(output_workspace),
            filename=f"final_processed_{form_output_name(output_workspace)}",
            backend="mpl",
            imshow_kwargs={"norm": LogNorm(vmin=1)},
        )
        SaveNexusProcessed(
            InputWorkspace=output_workspace,
            Filename=f"final_processed_{form_output_name(output_workspace)}.nxs",
        )

    return mtd[output_workspace]


def reduce_single_configuration(
    loaded_ws, reduction_input, prefix="", skip_nan=True, debug_output=False
):
    reduction_config = reduction_input["configuration"]

    flux_method = reduction_config["normalization"]
    monitor_fail_switch = reduction_config["normalizationResortToTime"]
    transmission_radius = reduction_config["mmRadiusForTransmission"]
    solid_angle = reduction_config["useSolidAngleCorrection"]
    sample_trans_value = reduction_input["sample"]["transmission"]["value"]
    bkg_trans_value = reduction_input["background"]["transmission"]["value"]
    theta_deppendent_transmission = reduction_config["useThetaDepTransCorrection"]
    mask_panel = None
    if reduction_config["useMaskBackTubes"]:
        mask_panel = "back"
    output_suffix = ""
    thickness = reduction_input["sample"]["thickness"]
    absolute_scale_method = reduction_config["absoluteScaleMethod"]
    beam_radius = reduction_config["DBScalingBeamRadius"]
    absolute_scale = reduction_config["StandardAbsoluteScale"]

    output_dir = reduction_config["outputDir"]

    nybins_main = nxbins_main = reduction_config["numQxQyBins"]
    bin1d_type = reduction_config["1DQbinType"]
    log_binning = reduction_config["QbinType"] == "log"
    # FIXME - NO MORE EVENT DECADES: even_decades = reduction_config.get("LogQBinsEvenDecade", False)
    decade_on_center = reduction_config.get("useLogQBinsDecadeCenter", False)
    nbins_main = reduction_config.get("numQBins")
    nbins_main_per_decade = reduction_config.get("LogQBinsPerDecade")
    outputFilename = reduction_input["outputFileName"]
    weighted_errors = reduction_config["useErrorWeighting"]
    qmin = reduction_config["Qmin"]
    qmax = reduction_config["Qmax"]
    annular_bin = reduction_config["AnnularAngleBin"]
    wedges_min = reduction_config["WedgeMinAngles"]
    wedges_max = reduction_config["WedgeMaxAngles"]
    wedges = (
        None
        if wedges_min is None or wedges_max is None
        else list(zip(wedges_min, wedges_max))
    )

    # automatically determine wedge binning if it wasn't explicitly set
    autoWedgeOpts = {}
    symmetric_wedges = True
    if bin1d_type == "wedge" and wedges_min is None:
        # the JSON validator "wedgesources" guarantees that the parameters to be collected are all non-empty
        autoWedgeOpts = {
            "q_min": reduction_config["autoWedgeQmin"],
            "q_delta": reduction_config["autoWedgeQdelta"],
            "q_max": reduction_config["autoWedgeQmax"],
            "azimuthal_delta": reduction_config["autoWedgeAzimuthalDelta"],
            "peak_width": reduction_config["autoWedgePeakWidth"],
            "background_width": reduction_config["autoWedgeBackgroundWidth"],
            "signal_to_noise_min": reduction_config["autoWedgeSignalToNoiseMin"],
            "peak_search_window_size_factor": reduction_config[
                "autoWedgePeakSearchWindowSizeFactor"
            ],
        }
        # auto-aniso returns all of the wedges
        symmetric_wedges = False
        logger.debug(
            f'Wedge peak search window size factor: {autoWedgeOpts["peak_search_window_size_factor"]}'
        )

    fbc_options = fbc_options_json(reduction_input)
    xc, yc, fit_results = find_beam_center(loaded_ws.center, **fbc_options)
    logger.notice(f"Find beam center = {xc}, {yc}")

    # process the center if using it in absolute scaling
    if absolute_scale_method == "direct_beam":
        processed_center_ws_name = f"{prefix}_processed_center"
        processed_center_ws = prepare_data_workspaces(
            loaded_ws.center,
            flux_method=flux_method,
            monitor_fail_switch=monitor_fail_switch,
            center_x=xc,
            center_y=yc,
            solid_angle=False,
            sensitivity_workspace=loaded_ws.sensitivity,
            output_workspace_name=processed_center_ws_name,
            debug=debug_output,
        )
    else:
        processed_center_ws = None

    # empty beam transmission workspace
    if loaded_ws.empty is not None:
        empty_trans_ws_name = f"{prefix}_empty"
        empty_trans_ws = prepare_data_workspaces(
            loaded_ws.empty,
            flux_method=flux_method,
            monitor_fail_switch=monitor_fail_switch,
            center_x=xc,
            center_y=yc,
            solid_angle=False,
            sensitivity_workspace=loaded_ws.sensitivity,
            output_workspace_name=empty_trans_ws_name,
        )
    else:
        empty_trans_ws = None

    # background transmission
    background_transmission_dict = {}
    if loaded_ws.background_transmission is not None and empty_trans_ws is not None:
        bkgd_trans_ws_name = f"{prefix}_bkgd_trans"
        bkgd_trans_ws_processed = prepare_data_workspaces(
            loaded_ws.background_transmission,
            flux_method=flux_method,
            monitor_fail_switch=monitor_fail_switch,
            center_x=xc,
            center_y=yc,
            solid_angle=False,
            sensitivity_workspace=loaded_ws.sensitivity,
            output_workspace_name=bkgd_trans_ws_name,
        )
        bkgd_trans_ws = calculate_transmission(
            bkgd_trans_ws_processed,
            empty_trans_ws,
            radius=transmission_radius,
            radius_unit="mm",
        )
        logger.notice(f"Background transmission = {bkgd_trans_ws.extractY()[0, 0]}")
        background_transmission_dict = {
            "value": bkgd_trans_ws.extractY(),
            "error": bkgd_trans_ws.extractE(),
        }
    else:
        bkgd_trans_ws = None

    # sample transmission
    sample_transmission_dict = {}
    if loaded_ws.sample_transmission is not None and empty_trans_ws is not None:
        sample_trans_ws_name = f"{prefix}_sample_trans"
        sample_trans_ws_processed = prepare_data_workspaces(
            loaded_ws.sample_transmission,
            flux_method=flux_method,
            monitor_fail_switch=monitor_fail_switch,
            center_x=xc,
            center_y=yc,
            solid_angle=False,
            sensitivity_workspace=loaded_ws.sensitivity,
            output_workspace_name=sample_trans_ws_name,
        )
        sample_trans_ws = calculate_transmission(
            sample_trans_ws_processed,
            empty_trans_ws,
            radius=transmission_radius,
            radius_unit="mm",
        )
        logger.notice(f"Sample transmission = {sample_trans_ws.extractY()[0, 0]}")
        sample_transmission_dict = {
            "value": sample_trans_ws.extractY(),
            "error": sample_trans_ws.extractE(),
        }
    else:
        sample_trans_ws = None

    output = []
    detectordata = {}
    for i, raw_sample_ws in enumerate(loaded_ws.sample):
        name = "_slice_{}".format(i + 1)
        if len(loaded_ws.sample) > 1:
            output_suffix = f"_{i}"
        processed_data_main = process_single_configuration(
            raw_sample_ws,
            sample_trans_ws=sample_trans_ws,
            sample_trans_value=sample_trans_value,
            bkg_ws_raw=loaded_ws.background,
            bkg_trans_ws=bkgd_trans_ws,
            bkg_trans_value=bkg_trans_value,
            blocked_ws_raw=loaded_ws.blocked_beam,
            theta_deppendent_transmission=theta_deppendent_transmission,
            center_x=xc,
            center_y=yc,
            dark_current=loaded_ws.dark_current,
            flux_method=flux_method,
            monitor_fail_switch=monitor_fail_switch,
            mask_ws=loaded_ws.mask,
            mask_panel=mask_panel,
            solid_angle=solid_angle,
            sensitivity_workspace=loaded_ws.sensitivity,
            output_workspace="processed_data_main",
            output_suffix=output_suffix,
            thickness=thickness,
            absolute_scale_method=absolute_scale_method,
            empty_beam_ws=processed_center_ws,
            beam_radius=beam_radius,
            absolute_scale=absolute_scale,
            keep_processed_workspaces=False,
            debug=debug_output,
        )
        # binning
        subpixel_kwargs = dict()
        if reduction_config["useSubpixels"] is True:
            subpixel_kwargs = {
                "n_horizontal": reduction_config["subpixelsX"],
                "n_vertical": reduction_config["subpixelsY"],
            }
        iq1d_main_in = convert_to_q(
            processed_data_main, mode="scalar", **subpixel_kwargs
        )
        iq2d_main_in = convert_to_q(
            processed_data_main, mode="azimuthal", **subpixel_kwargs
        )
        if bool(autoWedgeOpts):  # determine wedges automatically
            logger.notice(f"Auto wedge options: {autoWedgeOpts}")
            autoWedgeOpts["debug_dir"] = output_dir
            wedges = getWedgeSelection(iq2d_main_in, **autoWedgeOpts)
            logger.notice(
                f"found wedge angles:\n"
                f"              peak: {wedges[0]}\n"
                f"        background: {wedges[1]}"
            )
            # sanity check
            assert len(wedges) == 2, f"Auto-wedges {wedges} shall have 2 2-tuples"

        # set the found wedge values to the reduction input, this will allow correct plotting
        reduction_config["wedges"] = wedges
        reduction_config["symmetric_wedges"] = symmetric_wedges

        iq2d_main_out, iq1d_main_out = bin_all(
            iq2d_main_in,
            iq1d_main_in,
            nxbins_main,
            nybins_main,
            n1dbins=nbins_main,
            n1dbins_per_decade=nbins_main_per_decade,
            decade_on_center=decade_on_center,
            bin1d_type=bin1d_type,
            log_scale=log_binning,
            qmin=qmin,
            qmax=qmax,
            annular_angle_bin=annular_bin,
            wedges=wedges,
            symmetric_wedges=symmetric_wedges,
            error_weighted=weighted_errors,
        )

        # save ASCII files
        filename = os.path.join(
            output_dir, "2D", f"{outputFilename}{output_suffix}_2D.dat"
        )
        save_ascii_binned_2D(filename, "I(Qx,Qy)", iq2d_main_out)

        for j in range(len(iq1d_main_out)):
            add_suffix = ""
            if len(iq1d_main_out) > 1:
                add_suffix = f"_wedge_{j}"
            ascii_1D_filename = os.path.join(
                output_dir, "1D", f"{outputFilename}{output_suffix}_1D{add_suffix}.txt"
            )
            save_iqmod(iq1d_main_out[j], ascii_1D_filename, skip_nan=skip_nan)

        IofQ_output = namedtuple("IofQ_output", ["I2D_main", "I1D_main"])
        current_output = IofQ_output(I2D_main=iq2d_main_out, I1D_main=iq1d_main_out)
        output.append(current_output)

        detectordata[name] = {"main": {"iq": iq1d_main_out, "iqxqy": iq2d_main_out}}

    # save reduction log

    filename = os.path.join(
        reduction_config["outputDir"],
        outputFilename + f"_reduction_log{output_suffix}.hdf",
    )
    starttime = datetime.now().isoformat()
    # try:
    #     pythonfile = __file__
    # except NameError:
    #     pythonfile = "Launched from notebook"
    reductionparams = {"data": copy.deepcopy(reduction_input)}
    specialparameters = {
        "beam_center": {"x": xc, "y": yc},
        "fit_results": fit_results,
        "sample_transmission": sample_transmission_dict,
        "background_transmission": background_transmission_dict,
    }
    samplelogs = {"main": SampleLogs(processed_data_main)}
    logslice_data_dict = reduction_input["logslice_data"]

    savereductionlog(
        filename=filename,
        detectordata=detectordata,
        reductionparams=reductionparams,
        # pythonfile=pythonfile,
        starttime=starttime,
        specialparameters=specialparameters,
        logslicedata=logslice_data_dict,
        samplelogs=samplelogs,
    )

    # change permissions to all files to allow overwrite
    allow_overwrite(reduction_config["outputDir"])
    allow_overwrite(os.path.join(reduction_config["outputDir"], "1D"))
    allow_overwrite(os.path.join(reduction_config["outputDir"], "2D"))

    return output


def form_output_name(workspace):
    workspace_name = str(workspace)
    file_name = workspace_name.split("/")[-1].split(".")[0]
    return f"{file_name}.png"


def plot_reduction_output(
    reduction_output,
    reduction_input,
    loglog=True,
    imshow_kwargs=None,
    close_figures=False,
):
    reduction_config = reduction_input["configuration"]
    output_dir = reduction_config["outputDir"]
    outputFilename = reduction_input["outputFileName"]
    output_suffix = ""

    bin1d_type = reduction_config["1DQbinType"]

    if imshow_kwargs is None:
        imshow_kwargs = {}
    for i, out in enumerate(reduction_output):
        if len(reduction_output) > 1:
            output_suffix = f"_{i}"
        filename = os.path.join(
            output_dir, "2D", f"{outputFilename}{output_suffix}_2D.png"
        )

        wedges = reduction_config["wedges"] if bin1d_type == "wedge" else None
        symmetric_wedges = reduction_config.get("symmetric_wedges", True)

        qmin = reduction_config["Qmin"]
        qmax = reduction_config["Qmax"]

        plot_IQazimuthal(
            out.I2D_main,
            filename,
            backend="mpl",
            imshow_kwargs=imshow_kwargs,
            title="Main",
            wedges=wedges,
            symmetric_wedges=symmetric_wedges,
            qmin=qmin,
            qmax=qmax,
        )
        if close_figures:
            plt.clf()
        for j in range(len(out.I1D_main)):
            add_suffix = ""
            if len(out.I1D_main) > 1:
                add_suffix = f"_wedge_{j}"
            filename = os.path.join(
                output_dir, "1D", f"{outputFilename}{output_suffix}_1D{add_suffix}.png"
            )
            plot_IQmod(
                [out.I1D_main[j]],
                filename,
                loglog=loglog,
                backend="mpl",
                errorbar_kwargs={"label": "main"},
            )
            if close_figures:
                plt.clf()
        if close_figures:
            plt.close()
    # allow overwrite
    allow_overwrite(os.path.join(output_dir, '1D'))
    allow_overwrite(os.path.join(output_dir, '2D'))


def adjust_back_panels_to_effective_position(workspace):
    """
    As the back panels are shadowed by its neighboring front panels when the
    incident beam is hitting at an angle, the effective position of the back
    panels needs to be adjusted to the center of the gap between its two
    neighboring front panels.

    :param workspace: intput workspace

    @NOTE: this adjustment must be done right before convert to q space
    """
    MoveInstrumentComponent(Workspace=workspace,
                            ComponentName="detector1/back-panel",
                            Z=-0.0082,
                            RelativePosition=True)
