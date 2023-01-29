""" Top-level API for EQSANS """
from collections import namedtuple
import copy
from datetime import datetime
import os
import matplotlib.pyplot as plt
from mantid.simpleapi import (
    mtd,
    logger,
    RebinToWorkspace,
    SaveNexus,
)  # noqa E402

# Import rolled up to complete a single top-level API
import drtsans  # noqa E402
from drtsans import (
    apply_sensitivity_correction,
    getWedgeSelection,
    load_sensitivity_workspace,
    solid_angle_correction,
)  # noqa E402
from drtsans import subtract_background  # noqa E402
from drtsans.settings import namedtuplefy  # noqa E402
from drtsans.process_uncertainties import set_init_uncertainties  # noqa E402
from drtsans.save_ascii import (
    save_ascii_1D,
    save_xml_1D,
    save_ascii_binned_2D,
)  # noqa E402
from drtsans.save_2d import save_nist_dat, save_nexus  # noqa E402
from drtsans.transmission import apply_transmission_correction  # noqa E402
from drtsans.tof.eqsans.transmission import calculate_transmission  # noqa E402
from drtsans.thickness_normalization import normalize_by_thickness  # noqa E402
from drtsans.beam_finder import find_beam_center, fbc_options_json  # noqa E402
from drtsans.instruments import extract_run_number  # noqa E402
from drtsans.path import abspath, abspaths, registered_workspace  # noqa E402
from drtsans.tof.eqsans.load import (
    load_events,
    load_events_and_histogram,
    load_and_split,
)  # noqa E402
from drtsans.tof.eqsans.dark_current import subtract_dark_current  # noqa E402
from drtsans.tof.eqsans.cfg import load_config  # noqa E402
from drtsans.samplelogs import SampleLogs  # noqa E402
from drtsans.mask_utils import apply_mask, load_mask  # noqa E402
from drtsans.tof.eqsans.normalization import normalize_by_flux  # noqa E402
from drtsans.tof.eqsans.meta_data import set_meta_data  # noqa E402
from drtsans.tof.eqsans.momentum_transfer import (
    convert_to_q,
    split_by_frame,
)  # noqa E402
from drtsans.plots import plot_IQmod, plot_IQazimuthal  # noqa E402
from drtsans.iq import bin_all  # noqa E402
from drtsans.dataobjects import save_iqmod  # noqa E402
from drtsans.path import allow_overwrite  # noqa E402
from drtsans.tof.eqsans.reduction_api import (
    prepare_data_workspaces,
    process_transmission,
    bin_i_with_correction,
)
from drtsans.tof.eqsans.correction_api import (
    parse_correction_config,
    CorrectionConfiguration,
)
from typing import Dict, Tuple, List


__all__ = [
    "apply_solid_angle_correction",
    "subtract_background",
    "prepare_data",
    "save_ascii_1D",
    "save_xml_1D",
    "save_nist_dat",
    "save_nexus",
    "set_init_uncertainties",
    "load_all_files",
    "prepare_data_workspaces",
    "pre_process_single_configuration",
    "reduce_single_configuration",
    "plot_reduction_output",
]

IofQ_output = namedtuple("IofQ_output", ["I2D_main", "I1D_main"])


def _get_configuration_file_parameters(sample_run):
    try:
        configuration_file_parameters = load_config(source=sample_run)
    except RuntimeError as e:
        logger.error(e)
        logger.warning("Not using previous configuration")
        configuration_file_parameters = {}
    return configuration_file_parameters


@namedtuplefy
def load_all_files(reduction_input, prefix="", load_params=None):
    r"""
    overwrites metadata for sample workspace

    Workflow:
    1. parse reduction_input
    2. remove existing related workspaces with same run numbers
    3. process beam center
    -  output: load_params, reduction_input
    4. adjust pixel heights and widths
    -  output: load_params, reduction_input
    5. load and optionally slice sample runs
    6. load other runs: bkgd, empty, sample_trans, bkgd_trans

    Returned namedtuple:
        - sample, background, empty, sample_transmission, background_transmission: namedtuple(data[ws], monitor[ws])
        - dark_current, sensitivity, mask: workspace

    Returns
    -------
    namedtuple
        Named tuple including all loaded workspaces
    """
    reduction_config = reduction_input[
        "configuration"
    ]  # a handy shortcut to the configuration parameters dictionary

    instrument_name = reduction_input["instrumentName"]
    ipts = reduction_input["iptsNumber"]
    sample = reduction_input["sample"]["runNumber"]
    sample_trans = reduction_input["sample"]["transmission"]["runNumber"]
    bkgd = reduction_input["background"]["runNumber"]
    bkgd_trans = reduction_input["background"]["transmission"]["runNumber"]
    empty = reduction_input["emptyTransmission"]["runNumber"]
    center = reduction_input["beamCenter"]["runNumber"]
    # elastic reference and background: incoherence correction
    elastic_ref_run = reduction_config["elasticReference"].get("runNumber")
    elastic_ref_bkgd_run = reduction_config["elasticReferenceBkgd"].get("runNumber")

    from drtsans.tof.eqsans.reduction_api import remove_workspaces

    remove_workspaces(
        reduction_config,
        instrument_name,
        prefix,
        sample,
        center,
        extra_run_numbers=[
            sample,
            bkgd,
            empty,
            sample_trans,
            bkgd_trans,
            elastic_ref_run,
            elastic_ref_bkgd_run,
        ],
    )

    filenames = set()
    default_mask = None
    if reduction_config["useDefaultMask"]:
        configuration_file_parameters = _get_configuration_file_parameters(
            sample.split(",")[0].strip()
        )
        default_mask = configuration_file_parameters["combined mask"]

    load_params = set_beam_center(
        center,
        prefix,
        instrument_name,
        ipts,
        filenames,
        reduction_config,
        reduction_input,
        default_mask,
        load_params,
    )

    # Adjust pixel heights and widths
    load_params["pixel_calibration"] = reduction_config.get(
        "usePixelCalibration", False
    )

    if reduction_config["detectorOffset"] is not None:
        load_params["detector_offset"] = reduction_config["detectorOffset"]
    if reduction_config["sampleOffset"] is not None:
        load_params["sample_offset"] = reduction_config["sampleOffset"]
    load_params["low_tof_clip"] = reduction_config["cutTOFmin"]
    load_params["high_tof_clip"] = reduction_config["cutTOFmax"]
    if reduction_config["wavelengthStep"] is not None:
        # account for wavelengthStepType
        step_type = 1
        if reduction_config["wavelengthStepType"] == "constant Delta lambda/lambda":
            step_type = -1
        load_params["bin_width"] = step_type * reduction_config["wavelengthStep"]
    load_params["monitors"] = reduction_config["normalization"] == "Monitor"

    # FIXME the issues with the monitor on EQSANS has not been fixed. Enable normalization by monitor (issue #538)
    if load_params["monitors"]:
        raise RuntimeError(
            "Normalization by monitor option will be enabled in a later drt-sans release"
        )

    # ----- END OF SETUP of load_params -----

    # check for time/log slicing
    timeslice, logslice = (
        reduction_config["useTimeSlice"],
        reduction_config["useLogSlice"],
    )
    if timeslice or logslice:
        if len(sample.split(",")) > 1:
            raise ValueError("Can't do slicing on summed data sets")

    # Load (and optionally slice) sample runs
    # special loading case for sample to allow the slicing options
    logslice_data_dict = {}
    if timeslice or logslice:
        ws_name = f"{prefix}_{instrument_name}_{sample}_raw_histo_slice_group"
        if not registered_workspace(ws_name):
            filename = abspath(sample.strip(), instrument=instrument_name, ipts=ipts)
            print(f"Loading filename {filename}")
            if timeslice:
                timesliceinterval = reduction_config["timeSliceInterval"]
                logslicename = logsliceinterval = None
            elif logslice:
                timesliceinterval = None
                logslicename, logsliceinterval = (
                    reduction_config["logSliceName"],
                    reduction_config["logSliceInterval"],
                )
            else:
                raise RuntimeError("There is no 3rd option besides time and log slice")
            filenames.add(filename)
            load_and_split(
                filename,
                output_workspace=ws_name,
                time_interval=timesliceinterval,
                log_name=logslicename,
                log_value_interval=logsliceinterval,
                **load_params,
            )
            for _w in mtd[ws_name]:
                if default_mask:
                    apply_mask(_w, mask=default_mask)

            if logslicename is not None:
                for n in range(mtd[ws_name].getNumberOfEntries()):
                    samplelogs = SampleLogs(mtd[ws_name].getItem(n))
                    logslice_data_dict[str(n)] = {
                        "data": list(samplelogs[logslicename].value),
                        "units": samplelogs[logslicename].units,
                        "name": logslicename,
                    }
        sample_bands = None
    else:
        # load Nexus file or files without splitting
        ws_name = f"{prefix}_{instrument_name}_{sample}_raw_histo"
        if not registered_workspace(ws_name):
            filename = abspaths(sample.strip(), instrument=instrument_name, ipts=ipts)
            print(f"Loading filename {filename}")
            filenames.add(filename)
            loaded_sample_tup = load_events_and_histogram(
                filename, output_workspace=ws_name, **load_params
            )
            sample_bands = loaded_sample_tup.bands
            if default_mask:
                apply_mask(ws_name, mask=default_mask)
        else:
            sample_bands = None

    reduction_input["logslice_data"] = logslice_data_dict

    # Load all other files without further processing
    # background, empty, sample transmission, background transmission
    other_ws_list = list()
    for irun, run_number in enumerate(
        [bkgd, empty, sample_trans, bkgd_trans, elastic_ref_run, elastic_ref_bkgd_run]
    ):
        if run_number:
            # run number is given
            ws_name = f"{prefix}_{instrument_name}_{run_number}_raw_histo"
            if not registered_workspace(ws_name):
                filename = abspaths(
                    run_number.strip(), instrument=instrument_name, ipts=ipts
                )
                print(f"Loading filename {filename}")
                filenames.add(filename)
                if irun in [4, 5]:
                    # elastic reference run and background run, the bands must be same as sample's
                    load_events_and_histogram(
                        filename,
                        output_workspace=ws_name,
                        sample_bands=sample_bands,
                        **load_params,
                    )
                else:
                    load_events_and_histogram(
                        filename, output_workspace=ws_name, **load_params
                    )
                if default_mask:
                    apply_mask(ws_name, mask=default_mask)
            other_ws_list.append(mtd[ws_name])
        else:
            # run number is not given
            other_ws_list.append(None)

    # elastic and elastic background reference run
    elastic_ref_ws = other_ws_list[4]
    elastic_ref_bkgd_ws = other_ws_list[5]

    # dark run (aka dark current run)
    dark_current_ws = None
    dark_current_mon_ws = None
    dark_current_file = reduction_config["darkFileName"]
    if dark_current_file is not None:
        run_number = extract_run_number(dark_current_file)
        ws_name = f"{prefix}_{instrument_name}_{run_number}_raw_histo"
        if not registered_workspace(ws_name):
            dark_current_file = abspath(dark_current_file)
            print(f"Loading filename {dark_current_file}")
            filenames.add(dark_current_file)
            loaded_dark = load_events_and_histogram(
                dark_current_file, output_workspace=ws_name, **load_params
            )
            dark_current_ws = loaded_dark.data
            if default_mask:
                apply_mask(ws_name, mask=default_mask)
        else:
            dark_current_ws = mtd[ws_name]

    if registered_workspace(
        f"{prefix}_{instrument_name}_{sample}_raw_histo_slice_group"
    ):
        sample_ws = mtd[f"{prefix}_{instrument_name}_{sample}_raw_histo_slice_group"]
        sample_ws_list = [w for w in sample_ws]
    else:
        sample_ws_list = [mtd[f"{prefix}_{instrument_name}_{sample}_raw_histo"]]
    background_ws = (
        mtd[f"{prefix}_{instrument_name}_{bkgd}_raw_histo"] if bkgd else None
    )
    empty_ws = mtd[f"{prefix}_{instrument_name}_{empty}_raw_histo"] if empty else None
    sample_transmission_ws = (
        mtd[f"{prefix}_{instrument_name}_{sample_trans}_raw_histo"]
        if sample_trans
        else None
    )
    background_transmission_ws = (
        mtd[f"{prefix}_{instrument_name}_{bkgd_trans}_raw_histo"]
        if bkgd_trans
        else None
    )
    if load_params["monitors"]:
        sample_mon_ws = mtd[f"{prefix}_{instrument_name}_{sample}_raw_histo_monitors"]
        background_mon_ws = (
            mtd[f"{prefix}_{instrument_name}_{bkgd}_raw_histo_monitors"]
            if bkgd
            else None
        )
        empty_mon_ws = (
            mtd[f"{prefix}_{instrument_name}_{empty}_raw_histo_monitors"]
            if empty
            else None
        )
        sample_transmission_mon_ws = (
            mtd[f"{prefix}_{instrument_name}_{sample_trans}" + "_raw_histo_monitors"]
            if sample_trans
            else None
        )
        background_transmission_mon_ws = (
            mtd[f"{prefix}_{instrument_name}_{bkgd_trans}" + "_raw_histo_monitors"]
            if bkgd_trans
            else None
        )
    else:
        sample_mon_ws = None
        background_mon_ws = None
        empty_mon_ws = None
        sample_transmission_mon_ws = None
        background_transmission_mon_ws = None

    # load required processed_files
    sensitivity_ws = None
    flood_file = reduction_input["configuration"]["sensitivityFileName"]
    if flood_file:
        sensitivity_ws_name = f"{prefix}_sensitivity"
        if not registered_workspace(sensitivity_ws_name):
            flood_file = abspath(flood_file)
            print(f"Loading filename {flood_file}")
            filenames.add(flood_file)
            load_sensitivity_workspace(flood_file, output_workspace=sensitivity_ws_name)
        sensitivity_ws = mtd[sensitivity_ws_name]

    mask_ws = None
    custom_mask_file = reduction_config["maskFileName"]
    if custom_mask_file is not None:
        mask_ws_name = f"{prefix}_mask"
        if not registered_workspace(mask_ws_name):
            custom_mask_file = abspath(custom_mask_file)
            print(f"Loading filename {custom_mask_file}")
            filenames.add(custom_mask_file)
            mask_ws = load_mask(custom_mask_file, output_workspace=mask_ws_name)
        else:
            mask_ws = mtd[mask_ws_name]

    # TODO load these files only once
    # beam_flux_ws = None
    # monitor_flux_ratio_ws = None

    sample_aperture_diameter = reduction_config["sampleApertureSize"]
    sample_thickness = reduction_input["sample"]["thickness"]
    smearing_pixel_size_x = reduction_config["smearingPixelSizeX"]
    smearing_pixel_size_y = reduction_config["smearingPixelSizeY"]

    # Sample workspace: set meta data
    for ws in sample_ws_list:
        set_meta_data(
            ws,
            wave_length=None,
            wavelength_spread=None,
            sample_offset=load_params["sample_offset"],
            sample_aperture_diameter=sample_aperture_diameter,
            sample_thickness=sample_thickness,
            source_aperture_diameter=None,
            smearing_pixel_size_x=smearing_pixel_size_x,
            smearing_pixel_size_y=smearing_pixel_size_y,
        )

    # Set  meta data to elastic reference run optionally
    if elastic_ref_run:
        reference_thickness = reduction_config["elasticReference"].get("thickness")
        set_meta_data(
            elastic_ref_ws,
            wave_length=None,
            wavelength_spread=None,
            sample_offset=load_params["sample_offset"],
            sample_aperture_diameter=sample_aperture_diameter,
            sample_thickness=reference_thickness,
            source_aperture_diameter=None,
            smearing_pixel_size_x=smearing_pixel_size_x,
            smearing_pixel_size_y=smearing_pixel_size_y,
        )
    # There is not extra setup for elastic reference background following typical background run

    print("FILE PATH, FILE SIZE:")
    total_size = 0
    for comma_separated_names in filenames:
        for name in comma_separated_names.split(","):
            try:
                file_size = os.path.getsize(name)
            except FileNotFoundError:
                hint = "EQSANS_{}".format(drtsans.instruments.extract_run_number(name))
                name = drtsans.path.abspath(hint, instrument="EQSANS")
                file_size = os.path.getsize(name)
            total_size += file_size
            print(name + ",", "{:.2f} MiB".format(file_size / 1024 ** 2))
    print("TOTAL: ", "{:.2f} MB".format(total_size / 1024 ** 2))

    ws_mon_pair = namedtuple("ws_mon_pair", ["data", "monitor"])

    loaded_ws_dict = dict(
        sample=[ws_mon_pair(data=ws, monitor=sample_mon_ws) for ws in sample_ws_list],
        background=ws_mon_pair(data=background_ws, monitor=background_mon_ws),
        empty=ws_mon_pair(data=empty_ws, monitor=empty_mon_ws),
        sample_transmission=ws_mon_pair(
            data=sample_transmission_ws, monitor=sample_transmission_mon_ws
        ),
        background_transmission=ws_mon_pair(
            data=background_transmission_ws, monitor=background_transmission_mon_ws
        ),
        dark_current=ws_mon_pair(data=dark_current_ws, monitor=dark_current_mon_ws),
        sensitivity=sensitivity_ws,
        mask=mask_ws,
        elastic_reference=ws_mon_pair(elastic_ref_ws, None),
        elastic_reference_background=ws_mon_pair(elastic_ref_bkgd_ws, None),
    )

    return loaded_ws_dict


def pre_process_single_configuration(
    sample_ws_raw: namedtuple,
    sample_trans_ws=None,
    sample_trans_value=None,
    bkg_ws_raw=None,
    bkg_trans_ws=None,
    bkg_trans_value=None,
    theta_dependent_transmission=True,
    dark_current=None,
    flux_method=None,  # normalization (time/monitor/proton charge)
    flux=None,  # file for flux
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
):
    r"""
    This function provides full data processing for a single experimental configuration,
    starting from workspaces (no data loading is happening inside this function)

    Parameters
    ----------
    sample_ws_raw: namedtuple
        (~mantid.dataobjects.Workspace2D, ~mantid.dataobjects.Workspace2D)
        raw data histogram workspace and monitor
    sample_trans_ws:  ~mantid.dataobjects.Workspace2D
        optional histogram workspace for sample transmission (already prepared)
    sample_trans_value: float
        optional value for sample transmission
    bkg_ws_raw: ~mantid.dataobjects.Workspace2D
        optional raw histogram workspace for background
    bkg_trans_ws: ~mantid.dataobjects.Workspace2D
        optional histogram workspace for background transmission
    bkg_trans_value: float
        optional value for background transmission
    theta_dependent_transmission: bool
        flag to apply angle dependent transmission
    dark_current: ~mantid.dataobjects.Workspace2D
        dark current workspace
    flux_method: str
        normalization by time or monitor
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
        flag to keep the processed background workspace

    Returns
    -------
    ~mantid.dataobjects.Workspace2D
        Reference to the processed workspace
    """
    if not output_workspace:
        output_workspace = output_suffix + "_sample"

    # create a common configuration for prepare data
    prepare_data_conf = {
        "dark_current": dark_current,
        "flux_method": flux_method,
        "flux": flux,
        "mask_ws": mask_ws,
        "mask_panel": mask_panel,
        "mask_btp": mask_btp,
        "solid_angle": solid_angle,
        "sensitivity_workspace": sensitivity_workspace,
    }

    # process sample
    sample_ws = prepare_data_workspaces(
        sample_ws_raw, output_workspace=output_workspace, **prepare_data_conf
    )
    # apply transmission to the sample
    if sample_trans_ws or sample_trans_value:
        if sample_trans_ws:
            RebinToWorkspace(
                WorkspaceToRebin=sample_trans_ws,
                WorkspaceToMatch=sample_ws,
                OutputWorkspace=sample_trans_ws,
            )
        sample_ws = apply_transmission_correction(
            sample_ws,
            trans_workspace=sample_trans_ws,
            trans_value=sample_trans_value,
            theta_dependent=theta_dependent_transmission,
            output_workspace=output_workspace,
        )

    # process background, if not already processed
    if bkg_ws_raw.data:
        # process background run
        bkgd_ws_name = output_suffix + "_background"
        if not registered_workspace(bkgd_ws_name):
            bkgd_ws = prepare_data_workspaces(
                bkg_ws_raw, output_workspace=bkgd_ws_name, **prepare_data_conf
            )
            # apply transmission to background
            if bkg_trans_ws or bkg_trans_value:
                if bkg_trans_ws:
                    RebinToWorkspace(
                        WorkspaceToRebin=bkg_trans_ws,
                        WorkspaceToMatch=bkgd_ws,
                        OutputWorkspace=bkg_trans_ws,
                    )
                bkgd_ws = apply_transmission_correction(
                    bkgd_ws,
                    trans_workspace=bkg_trans_ws,
                    trans_value=bkg_trans_value,
                    theta_dependent=theta_dependent_transmission,
                    output_workspace=bkgd_ws_name,
                )
        else:
            bkgd_ws = mtd[bkgd_ws_name]

        # subtract background
        sample_ws = subtract_background(sample_ws, bkgd_ws)

        if not keep_processed_workspaces:
            bkgd_ws.delete()

    # finalize with absolute scale and thickness
    sample_ws = normalize_by_thickness(sample_ws, thickness)

    # standard method assumes absolute scale from outside
    if absolute_scale_method == "direct_beam":
        raise NotImplementedError("This feature is not yet implemented for EQSANS")
    else:
        sample_ws *= absolute_scale

    return mtd[output_workspace]


def reduce_single_configuration(
    loaded_ws: namedtuple,
    reduction_input,
    prefix="",
    skip_nan=True,
    not_apply_incoherence_correction: bool = False,
):
    """Reduce samples from raw workspaces including
    1. prepare data
    1.

    This is the main entry point of reduction

    Input loaded workspaces as namedtuple:
      - sample, background, empty, sample_transmission, background_transmission: namedtuple(data[ws], monitor[ws])
      - dark_current, sensitivity, mask: workspace

    Parameters
    ----------
    loaded_ws: namedtuple
        loaded workspaces
    reduction_input: dict
        reduction configuration
    prefix
    skip_nan
    not_apply_incoherence_correction: bool
        If true, then no incoherence scattering correction will be applied to reduction overriding JSON

    Returns
    -------
    ~list
        list of IofQ_output: ['I2D_main', 'I1D_main']

    """
    # Process reduction input: configuration and etc.
    reduction_config = reduction_input["configuration"]

    # Process inelastic/incoherent scattering correction configuration if user does not specify
    assert isinstance(
        not_apply_incoherence_correction, bool
    ), "Only boolean for not_apply flag is allowed"
    if not_apply_incoherence_correction is True:
        # allow user to override JSON setup
        incoherence_correction_setup = CorrectionConfiguration(do_correction=False)
    else:
        # parse JSON for correction setup
        incoherence_correction_setup = parse_correction_config(reduction_input)

    # process: flux, monitor, proton charge, ...
    flux_method_translator = {
        "Monitor": "monitor",
        "Total charge": "proton charge",
        "Time": "time",
    }
    flux_method = flux_method_translator.get(reduction_config["normalization"], None)

    flux_translator = {
        "Monitor": reduction_config["fluxMonitorRatioFile"],
        "Total charge": reduction_config["beamFluxFileName"],
        "Time": "duration",
    }
    flux = flux_translator.get(reduction_config["normalization"], None)

    solid_angle = reduction_config["useSolidAngleCorrection"]
    transmission_radius = reduction_config["mmRadiusForTransmission"]
    sample_trans_value = reduction_input["sample"]["transmission"]["value"]
    bkg_trans_value = reduction_input["background"]["transmission"]["value"]
    theta_dependent_transmission = reduction_config["useThetaDepTransCorrection"]
    mask_panel = "back" if reduction_config["useMaskBackTubes"] is True else None
    output_suffix = ""

    thickness = reduction_input["sample"]["thickness"]
    absolute_scale_method = reduction_config["absoluteScaleMethod"]
    beam_radius = None  # EQSANS doesn't use keyword DBScalingBeamRadius
    absolute_scale = reduction_config["StandardAbsoluteScale"]
    output_dir = reduction_config["outputDir"]
    # process binning
    # Note: option {even_decades = reduction_config["useLogQBinsEvenDecade"]} is removed
    nybins_main = nxbins_main = reduction_config["numQxQyBins"]
    bin1d_type = reduction_config["1DQbinType"]
    log_binning = (
        reduction_config["QbinType"] == "log"
    )  # FIXME - note: fixed to log binning
    decade_on_center = reduction_config["useLogQBinsDecadeCenter"]
    nbins_main = reduction_config["numQBins"]
    nbins_main_per_decade = reduction_config["LogQBinsPerDecade"]
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
    # set the found wedge values to the reduction input, this will allow correct plotting
    reduction_config["wedges"] = wedges
    reduction_config["symmetric_wedges"] = True

    # automatically determine wedge binning if it wasn't explicitly set
    autoWedgeOpts, symmetric_wedges = parse_auto_wedge_setup(
        reduction_config, bin1d_type, wedges_min
    )

    # Prepare empty beam transmission workspace
    if loaded_ws.empty.data is not None:
        empty_trans_ws_name = f"{prefix}_empty"
        empty_trans_ws = prepare_data_workspaces(
            loaded_ws.empty,
            flux_method=flux_method,
            flux=flux,
            solid_angle=False,
            sensitivity_workspace=loaded_ws.sensitivity,
            output_workspace=empty_trans_ws_name,
        )
    else:
        empty_trans_ws = None

    # Background transmission
    # TODO 781 - to test and check for sanity
    # specific output filename (base) for background trans
    if loaded_ws.background_transmission.data:
        bkgd_trans = reduction_input["background"]["transmission"]["runNumber"].strip()
        base_out_name = f"{outputFilename}_bkgd_{bkgd_trans}"

        # process transmission
        bkgd_returned = process_transmission(
            loaded_ws.background_transmission,
            empty_trans_ws,
            transmission_radius,
            loaded_ws.sensitivity,
            flux_method,
            flux,
            prefix,
            "bkgd",
            output_dir,
            base_out_name,
        )
        (
            bkgd_trans_ws,
            background_transmission_dict,
            background_transmission_raw_dict,
        ) = bkgd_returned
    else:
        # no background transmission
        bkgd_trans_ws = (
            background_transmission_dict
        ) = background_transmission_raw_dict = None

    # sample transmission
    sample_returned = process_transmission(
        loaded_ws.sample_transmission,
        empty_trans_ws,
        transmission_radius,
        loaded_ws.sensitivity,
        flux_method,
        flux,
        prefix,
        "sample",
        output_dir,
        outputFilename,
    )

    (
        sample_trans_ws,
        sample_transmission_dict,
        sample_transmission_raw_dict,
    ) = sample_returned

    # set up subpixel binning options  FIXME - it does not seem to work
    subpixel_kwargs = dict()
    if reduction_config["useSubpixels"]:
        subpixel_kwargs = {
            "n_horizontal": reduction_config["subpixelsX"],
            "n_vertical": reduction_config["subpixelsY"],
        }

    # process elastic run
    if (
        incoherence_correction_setup.do_correction
        and incoherence_correction_setup.elastic_reference
    ):
        # sanity check
        assert loaded_ws.elastic_reference.data, (
            f"Reference run is not loaded: "
            f"{incoherence_correction_setup.elastic_reference}"
        )
        elastic_ref = incoherence_correction_setup.elastic_reference
        processed_elastic_ref = pre_process_single_configuration(
            loaded_ws.elastic_reference,
            sample_trans_ws=elastic_ref.transmission_run_number,
            sample_trans_value=elastic_ref.transmission_value,
            bkg_ws_raw=loaded_ws.elastic_reference_background,
            bkg_trans_ws=elastic_ref.background_transmission_run_number,  # noqa E502
            bkg_trans_value=elastic_ref.background_transmission_value,  # noqa E502
            theta_dependent_transmission=theta_dependent_transmission,  # noqa E502
            dark_current=loaded_ws.dark_current,
            flux_method=flux_method,
            flux=flux,
            mask_ws=loaded_ws.mask,
            mask_panel=mask_panel,
            solid_angle=solid_angle,
            sensitivity_workspace=loaded_ws.sensitivity,
            output_workspace="processed_elastic_ref",
            output_suffix=output_suffix,
            thickness=elastic_ref.thickness,
            absolute_scale_method=absolute_scale_method,
            empty_beam_ws=empty_trans_ws,
            beam_radius=beam_radius,
            absolute_scale=absolute_scale,
            keep_processed_workspaces=False,
        )
        # convert to I(Q)
        iq1d_elastic_ref = convert_to_q(
            processed_elastic_ref, mode="scalar", **subpixel_kwargs
        )
        iq2d_elastic_ref = convert_to_q(
            processed_elastic_ref, mode="azimuthal", **subpixel_kwargs
        )
        # split to frames
        iq1d_elastic_ref_frames = split_by_frame(
            processed_elastic_ref, iq1d_elastic_ref, verbose=True
        )
        iq2d_elastic_ref_frames = split_by_frame(
            processed_elastic_ref, iq2d_elastic_ref, verbose=True
        )

    else:
        iq1d_elastic_ref_frames = iq2d_elastic_ref_frames = None

    # Define output data structure
    output = []
    detectordata = {}
    processed_data_main = None
    for i, raw_sample_ws in enumerate(loaded_ws.sample):
        name = "slice_{}".format(i + 1)
        if len(loaded_ws.sample) > 1:
            output_suffix = f"_{i}"
        raw_name = f"EQSANS_{raw_sample_ws.data.getRunNumber()}"

        # process data without correction
        processed_data_main = pre_process_single_configuration(
            raw_sample_ws,
            sample_trans_ws=sample_trans_ws,
            sample_trans_value=sample_trans_value,
            bkg_ws_raw=loaded_ws.background,
            bkg_trans_ws=bkgd_trans_ws,
            bkg_trans_value=bkg_trans_value,
            theta_dependent_transmission=theta_dependent_transmission,  # noqa E502
            dark_current=loaded_ws.dark_current,
            flux_method=flux_method,
            flux=flux,
            mask_ws=loaded_ws.mask,
            mask_panel=mask_panel,
            solid_angle=solid_angle,
            sensitivity_workspace=loaded_ws.sensitivity,
            output_workspace="processed_data_main",
            output_suffix=output_suffix,
            thickness=thickness,
            absolute_scale_method=absolute_scale_method,
            empty_beam_ws=empty_trans_ws,
            beam_radius=beam_radius,
            absolute_scale=absolute_scale,
            keep_processed_workspaces=False,
        )

        # convert to Q
        iq1d_main_in = convert_to_q(
            processed_data_main, mode="scalar", **subpixel_kwargs
        )
        iq2d_main_in = convert_to_q(
            processed_data_main, mode="azimuthal", **subpixel_kwargs
        )

        # split to frames
        iq1d_main_in_fr = split_by_frame(
            processed_data_main, iq1d_main_in, verbose=True
        )
        iq2d_main_in_fr = split_by_frame(
            processed_data_main, iq2d_main_in, verbose=True
        )

        # Save nexus processed
        filename = os.path.join(output_dir, f"{outputFilename}{output_suffix}.nxs")
        SaveNexus(processed_data_main, Filename=filename)
        print(f"SaveNexus to {filename}")

        # Work with wedges
        if bool(
            autoWedgeOpts
        ):  # determine wedges automatically from the main detectora
            wedges = process_auto_wedge(
                autoWedgeOpts,
                iq2d_main_in,
                output_dir,
                reduction_config,
                symmetric_wedges,
            )

        n_wl_frames = len(iq2d_main_in_fr)
        _inside_detectordata = {}

        # Process each frame separately
        for wl_frame in range(n_wl_frames):
            if n_wl_frames > 1:
                fr_log_label = f"_frame_{wl_frame}"
                fr_label = fr_log_label
            else:
                fr_log_label = "frame"
                fr_label = ""

            assert (
                iq1d_main_in_fr[wl_frame] is not None
            ), "Input I(Q)      main input cannot be None."
            assert (
                iq2d_main_in_fr[wl_frame] is not None
            ), "Input I(qx, qy) main input cannot be None."

            iq2d_main_out, iq1d_main_out = bin_i_with_correction(
                iq1d_main_in_fr,
                iq2d_main_in_fr,
                wl_frame,
                weighted_errors,
                qmin,
                qmax,
                nxbins_main,
                nybins_main,
                nbins_main,
                nbins_main_per_decade,
                decade_on_center,
                bin1d_type,
                log_binning,
                annular_bin,
                wedges,
                symmetric_wedges,
                incoherence_correction_setup,
                iq1d_elastic_ref_frames,
                iq2d_elastic_ref_frames,
                raw_name,
                output_dir,
                outputFilename,
            )

            _inside_detectordata[fr_log_label] = {
                "iq": iq1d_main_out,
                "iqxqy": iq2d_main_out,
            }

            # save ASCII files
            filename = os.path.join(
                output_dir, f"{outputFilename}{output_suffix}{fr_label}_Iqxqy.dat"
            )
            if iq2d_main_out:
                save_ascii_binned_2D(filename, "I(Qx,Qy)", iq2d_main_out)

            for j in range(len(iq1d_main_out)):
                add_suffix = ""
                if len(iq1d_main_out) > 1:
                    add_suffix = f"_wedge_{j}"
                add_suffix += fr_label
                ascii_1D_filename = os.path.join(
                    output_dir, f"{outputFilename}{output_suffix}{add_suffix}_Iq.dat"
                )
                save_iqmod(iq1d_main_out[j], ascii_1D_filename, skip_nan=skip_nan)

            current_output = IofQ_output(I2D_main=iq2d_main_out, I1D_main=iq1d_main_out)
            output.append(current_output)
        # END binning loop over frame

        detectordata[name] = _inside_detectordata
    # END reduction loop over sample workspaces

    # Save reduction log
    save_reduction_log(
        reduction_input,
        outputFilename,
        processed_data_main,
        sample_transmission_dict,
        sample_transmission_raw_dict,
        background_transmission_dict,
        background_transmission_raw_dict,
        detectordata,
        output_dir,
    )

    return output


def save_reduction_log(
    reduction_input: Dict,
    output_file_name: str,
    processed_data_main,
    sample_transmission_dict: Dict,
    sample_transmission_raw_dict: Dict,
    background_transmission_dict: Dict,
    background_transmission_raw_dict,
    detector_data,
    output_dir: str,
):
    """Save reduction log to an HDF5 file"""
    # create reduction log
    filename = os.path.join(
        reduction_input["configuration"]["outputDir"],
        output_file_name + "_reduction_log.hdf",
    )
    starttime = datetime.now().isoformat()
    # try:
    #     pythonfile = __file__
    # except NameError:
    #     pythonfile = "Launched from notebook"
    reductionparams = {"data": copy.deepcopy(reduction_input)}
    beam_center_dict = reduction_input["beam_center"]
    specialparameters = {
        "beam_center": {
            "x": beam_center_dict["x"],
            "y": beam_center_dict["y"],
            "type": beam_center_dict["type"],
        },
        "fit_results": beam_center_dict["fit_results"],
        "sample_transmission": sample_transmission_dict,
        "sample_transmission_raw": sample_transmission_raw_dict,
        "background_transmission": background_transmission_dict,
        "background_transmission_raw": background_transmission_raw_dict,
    }

    # [#689] TODO FIXME - Reincarnate this section!
    # FIXME - check original code.  processed_data_main outside a loop is a BUG!
    #  The correction workflow does not output processed data workspace yet!
    assert processed_data_main is not None
    samplelogs = {"main": SampleLogs(processed_data_main)}
    logslice_data_dict = reduction_input["logslice_data"]

    drtsans.savereductionlog(
        filename=filename,
        detectordata=detector_data,
        reductionparams=reductionparams,
        starttime=starttime,
        specialparameters=specialparameters,
        logslicedata=logslice_data_dict,
        samplelogs=samplelogs,
    )

    # change permissions to all files to allow overwrite
    allow_overwrite(output_dir)


def process_auto_wedge(
    auto_wedge_setup: Dict,
    iq2d_input,
    output_dir: str,
    reduction_config: Dict,
    symmetric_wedges,
) -> List:
    """Process and set up auto wedge

    Returns
    -------
    ~list
      list containing 2 lists each contains 2 2-tuples
      as ``[[(peak1_min, peak1_max), (peak2_min, peak2_max)], [(..., ...), (..., ...)]]``
    """
    logger.notice(f"Auto wedge options: {auto_wedge_setup}")
    auto_wedge_setup["debug_dir"] = output_dir
    wedges = getWedgeSelection(iq2d_input, **auto_wedge_setup)
    logger.notice(
        f"found wedge angles:\n"
        f"              peak: {wedges[0]}\n"
        f"        background: {wedges[1]}"
    )
    # sanity check
    assert len(wedges) == 2, f"Auto-wedges {wedges} shall have 2 2-tuples"
    # set automated wedge to reduction configuration for correct plotting.
    # reduction_config is an in/out function argument
    reduction_config["wedges"] = wedges
    reduction_config["symmetric_wedges"] = symmetric_wedges

    return wedges


def parse_auto_wedge_setup(
    reduction_config: Dict, bin1d_type: str, wedges_min
) -> Tuple[Dict, bool]:
    """Parse JSON input for automatic wedge setup"""
    autoWedgeOpts = {}
    symmetric_wedges = True
    if bin1d_type == "wedge" and len(wedges_min) == 0:
        # the JSON validator "wedgesources" guarantees that the parameters to be collected are all non-empty
        autoWedgeOpts = {
            "q_min": reduction_config["autoWedgeQmin"],
            "q_delta": reduction_config["autoWedgeQdelta"],
            "q_max": reduction_config["autoWedgeQmax"],
            "azimuthal_delta": reduction_config["autoWedgeAzimuthalDelta"],
            "peak_width": reduction_config["autoWedgePeakWidth"],
            "background_width": reduction_config["autoWedgeBackgroundWidth"],
            "signal_to_noise_min": reduction_config["autoWedgeSignalToNoiseMin"],
        }
        # auto-aniso returns all of the wedges
        symmetric_wedges = False

    return autoWedgeOpts, symmetric_wedges


def plot_reduction_output(reduction_output, reduction_input, imshow_kwargs=None):
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

        wedges = reduction_config["wedges"] if bin1d_type == "wedge" else None
        symmetric_wedges = reduction_config.get("symmetric_wedges", True)

        qmin = reduction_config["Qmin"]
        qmax = reduction_config["Qmax"]

        filename = os.path.join(
            output_dir, f"{outputFilename}{output_suffix}_Iqxqy.png"
        )
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
        plt.clf()
        for j in range(len(out.I1D_main)):
            add_suffix = ""
            if len(out.I1D_main) > 1:
                add_suffix = f"_wedge_{j}"
            filename = os.path.join(
                output_dir, f"{outputFilename}{output_suffix}{add_suffix}_Iq.png"
            )
            plot_IQmod(
                [out.I1D_main[j]],
                filename,
                loglog=True,
                backend="mpl",
                errorbar_kwargs={"label": "main"},
            )
            plt.clf()
    plt.close()
    # change permissions to all files to allow overwrite
    allow_overwrite(output_dir)


def apply_solid_angle_correction(input_workspace):
    """Apply solid angle correction. This uses :func:`drtsans.solid_angle_correction`."""
    return solid_angle_correction(input_workspace, detector_type="VerticalTube")


def set_beam_center(
    center,
    prefix,
    instrument_name,
    ipts,
    filenames,
    reduction_config,
    reduction_input,
    default_mask,
    load_params,
):
    """Helping method to set beam center"""
    # find the center first
    if center != "":
        # calculate beam center from center workspace
        center_ws_name = f"{prefix}_{instrument_name}_{center}_raw_events"
        if not registered_workspace(center_ws_name):
            center_filename = abspath(center, instrument=instrument_name, ipts=ipts)
            filenames.add(center_filename)
            load_events(
                center_filename,
                pixel_calibration=reduction_config.get("usePixelCalibration", False),
                output_workspace=center_ws_name,
            )
            if reduction_config["useDefaultMask"]:
                apply_mask(center_ws_name, mask=default_mask)
        fbc_options = fbc_options_json(reduction_input)
        center_x, center_y, fit_results = find_beam_center(
            center_ws_name, **fbc_options
        )
        logger.notice(f"calculated center ({center_x}, {center_y})")
        beam_center_type = "calculated"
    else:
        # use default EQSANS center
        # TODO - it is better to have these hard code value defined out side of this method
        center_x = 0.025239
        center_y = 0.0170801
        logger.notice(f"use default center ({center_x}, {center_y})")
        beam_center_type = "default"
        fit_results = None

    # set beam center to reduction configuration
    reduction_input["beam_center"] = {
        "type": beam_center_type,
        "x": center_x,
        "y": center_y,
        "fit_results": fit_results,
    }
    # update to 'load_params'
    if load_params is None:
        load_params = dict(center_x=center_x, center_y=center_y, keep_events=False)
    elif isinstance(load_params, dict):
        load_params["center_x"] = center_x
        load_params["center_y"] = center_y
        load_params["keep_events"] = False
    else:
        raise RuntimeError(f"load_param of type {type(load_params)} is not allowed.")

    return load_params


def prepare_data(
    data,
    pixel_calibration=False,
    detector_offset=0,
    sample_offset=0,
    bin_width=0.1,
    low_tof_clip=500,
    high_tof_clip=2000,
    center_x=None,
    center_y=None,
    dark_current=None,
    flux_method=None,
    flux=None,
    mask=None,
    mask_panel=None,
    btp=dict(),
    solid_angle=True,
    sensitivity_file_path=None,
    sensitivity_workspace=None,
    sample_aperture_diameter=None,
    sample_thickness=None,
    source_aperture_diameter=None,
    smearing_pixel_size_x=None,
    smearing_pixel_size_y=None,
    output_workspace=None,
    output_suffix="",
):
    r"""
    Load an EQSANS data file and bring the data to a point where it can be used. This includes applying basic
    corrections that are always applied regardless of whether the data is background or scattering data.

    Parameters
    ----------
    data: int, str, ~mantid.api.IEventWorkspace
        Run number as int or str, file path, :py:obj:`~mantid.api.IEventWorkspace`
    pixel_calibration: bool
        Adjust pixel heights and widths according to barscan and tube-width calibrations.
    detector_offset: float
        Additional translation of the detector along Z-axis, in mili-meters.
    sample_offset: float
        Additional translation of the sample along the Z-axis, in mili-meters.
    bin_width: float
        Bin width for the output workspace, in Angstroms.
    low_tof_clip: float
        Ignore events with a time-of-flight (TOF) smaller than the minimal
        TOF plus this quantity.
    high_tof_clip: float
        Ignore events with a time-of-flight (TOF) bigger than the maximal
        TOF minus this quantity.
    center_x: float
        Move the center of the detector to this X-coordinate. If :py:obj:`None`, the
        detector will be moved such that the X-coordinate of the intersection
        point between the neutron beam and the detector array will have ``x=0``.
    center_y: float
        Move the center of the detector to this X-coordinate. If :py:obj:`None`, the
        detector will be moved such that the X-coordinate of the intersection
        point between the neutron beam and the detector array will have ``y=0``.
    dark_current: int, str, ~mantid.api.IEventWorkspace
        Run number as int or str, file path, :py:obj:`~mantid.api.IEventWorkspace`
    flux_method: str
        Method for flux normalization. Either 'proton charge',
        'monitor', or 'time'.
    flux: str
        if ``flux_method`` is proton charge, then path to file containing the
        wavelength distribution of the neutron flux. If ``flux method`` is
        monitor, then path to file containing the flux-to-monitor ratios.
        if ``flux_method`` is time, then pass one log entry name such
        as ``duration`` or leave it as :py:obj:`None` for automatic log search.
    panel: str
        Either 'front' or 'back' to mask a whole panel
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
    sample_aperture_diameter: float, None
        sample aperture diameter in unit mm
    sample_thickness: None, float
        sample thickness in unit cm
    source_aperture_diameter: float, None
        source aperture diameter in unit meter
    smearing_pixel_size_x: float, None
        pixel size in x direction in unit as meter, only for Q-resolution calculation
    smearing_pixel_size_y: float, None
        pixel size in Y direction in unit as meter, only for Q-resolutio calculation

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
    # First, load the event stream data into a workspace
    # The output_workspace name is for the Mantid workspace
    workspaces = load_events_and_histogram(
        data,
        pixel_calibration=pixel_calibration,
        detector_offset=detector_offset,
        sample_offset=sample_offset,
        output_workspace=output_workspace,
        output_suffix=output_suffix,
        center_x=center_x,
        center_y=center_y,
        bin_width=bin_width,
        low_tof_clip=low_tof_clip,
        high_tof_clip=high_tof_clip,
        keep_events=(dark_current is None),
        monitors=(flux_method == "monitor"),
    )

    output_workspace = workspaces.data

    # Next, we subtract dark current, if it exists.
    # Note that the function handles the normalization internally.
    if dark_current is not None:
        output_workspace = subtract_dark_current(output_workspace, dark_current)

    # The solid angle is corrected for next
    if solid_angle:
        if solid_angle is True:
            output_workspace = apply_solid_angle_correction(output_workspace)
        else:  # assume the solid_angle parameter is a workspace
            output_workspace = apply_solid_angle_correction(
                output_workspace, solid_angle_ws=solid_angle
            )

    # Interestingly, this is the only use of the btp dictionary.
    # The BTP stands for banks, tubes and pixels - it is a Mantid thing.
    apply_mask(output_workspace, panel=mask_panel, mask=mask, **btp)  # returns the mask

    # Correct for the detector sensitivity (the per pixel relative response)
    if sensitivity_file_path is not None or sensitivity_workspace is not None:
        kw = dict(
            sensitivity_filename=sensitivity_file_path,
            sensitivity_workspace=sensitivity_workspace,
        )
        output_workspace = apply_sensitivity_correction(output_workspace, **kw)

    # We can perform the desired normalization here.
    if flux_method is not None:
        kw = dict(method=flux_method)
        if flux_method == "monitor":
            kw["monitor_workspace"] = str(workspaces.monitor)
        output_workspace = normalize_by_flux(output_workspace, flux, **kw)

    # Overwrite meta data
    set_meta_data(
        output_workspace,
        None,
        None,
        sample_offset,
        sample_aperture_diameter,
        sample_thickness,
        source_aperture_diameter,
        smearing_pixel_size_x,
        smearing_pixel_size_y,
    )

    if isinstance(output_workspace, str):
        return mtd[output_workspace]  # shouldn't happen
    else:
        return output_workspace
