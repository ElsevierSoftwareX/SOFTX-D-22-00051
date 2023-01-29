import os
from typing import List, Tuple, Union
import numpy as np
import time
from mantid.simpleapi import mtd
from drtsans.mono.spice_data import map_to_nexus
from drtsans.mono.biosans import (
    load_all_files,
    reduce_single_configuration,
    plot_reduction_output,
    reduction_parameters,
    update_reduction_parameters,
)
import warnings

warnings.filterwarnings("ignore")

CG3 = "CG3"


def clear_buffer():
    """Clear memory buffer: clear mantid workspaces"""
    mtd.clear()


def reduce_biosans_nexus(
    ipts_number: int,
    experiment_number: int,
    sample_names: List[str],
    sample_runs: List[Tuple[int, int]],
    background_runs: List[Union[None, Tuple[int, int]]],
    sample_transmission_runs: List[Tuple[int, int]],
    background_transmission_runs: List[Union[None, Tuple[int, int]]],
    empty_transmission_run: Tuple[int, int],
    beam_center_runs: Tuple[int, int],
    sample_thickness_list: List[Union[str, float]],
    sample_identifier: str,
    overwrite_reduced_data: bool,
    main_detector_dark_run: Tuple[int, int],
    wing_detector_dark_run: Tuple[int, int],
    base_output_directory: str,
    main_detector_sensitivities_file: str,
    wing_detector_sensitivities_file: str,
    scale_factor: Union[float, str, None],
    scaling_beam_radius: Union[float, str, None],
    number_linear_q1d_bins_main_detector: Union[int, str],
    number_linear_q1d_bins_wing_detector: Union[int, str],
    number_linear_q2d_bins_main_detector: Union[int, str],
    number_linear_q2d_bins_wing_detector: Union[int, str],
    plot_type: str,
    plot_binning: str,
    number_log_bins_per_decade_main_detector: int,
    number_log_bins_per_decade_wing_detector: int,
    q_range_main: Union[List[float], Tuple[float, float]],
    q_range_wing: Union[List[float], Tuple[float, float]],
    overlap_stitch_range: Union[List[float], Tuple[float, float]],
    flexible_pixel_sizes: bool,
    wedge_min_angles: Union[List[float], None],
    wedge_max_angles: Union[List[float], None],
    auto_wedge_qmin: float,
    auto_wedge_qmax: float,
    auto_wedge_delta_q: float,
    auto_wedge_peak_width: float,
    auto_wedge_delta_azimuthal_angle: float,
    auto_wedge_background_width: float,
    auto_wedge_minimum_signal_to_noise_ratio: float,
    q_range_main_wedge0: Union[Tuple[float, float], List[float]],
    q_range_wing_wedge0: Union[Tuple[float, float], List[float]],
    OL_range_wedge0: Union[Tuple[float, float], List[float]],
    q_range_main_wedge1: Union[Tuple[float, float], List[float]],
    q_range_wing_wedge1: Union[Tuple[float, float], List[float]],
    OL_range_wedge1: Union[Tuple[float, float], List[float]],
    refresh_cycle: int,
    nexus_dir: Union[str, None],
):

    # Convert SPICE scan-pt tuple to NeXus files
    sample_runs = map_to_nexus(
        CG3, ipts_number, experiment_number, sample_runs, nexus_dir=nexus_dir
    )
    sample_transmission_runs = map_to_nexus(
        CG3,
        ipts_number,
        experiment_number,
        sample_transmission_runs,
        nexus_dir=nexus_dir,
    )
    background_runs = map_to_nexus(
        CG3, ipts_number, experiment_number, background_runs, nexus_dir=nexus_dir
    )
    background_transmission_runs = map_to_nexus(
        CG3,
        ipts_number,
        experiment_number,
        background_transmission_runs,
        nexus_dir=nexus_dir,
    )
    beam_center_runs = map_to_nexus(
        CG3, ipts_number, experiment_number, [beam_center_runs], nexus_dir=nexus_dir
    )[0]
    empty_transmission_run = map_to_nexus(
        CG3,
        ipts_number,
        experiment_number,
        [empty_transmission_run],
        nexus_dir=nexus_dir,
    )[0]
    main_detector_dark_run = map_to_nexus(
        CG3,
        ipts_number,
        experiment_number,
        [main_detector_dark_run],
        nexus_dir=nexus_dir,
    )[0]
    wing_detector_dark_run = map_to_nexus(
        CG3,
        ipts_number,
        experiment_number,
        [wing_detector_dark_run],
        nexus_dir=nexus_dir,
    )[0]

    # reduction parameters common to all the reduction runs to be carried out in this notebook
    common_configuration = {
        "iptsNumber": ipts_number,
        "beamCenter": {"runNumber": beam_center_runs},
        "emptyTransmission": {"runNumber": empty_transmission_run},
        "configuration": {
            "outputDir": base_output_directory,
            "darkMainFileName": main_detector_dark_run,
            "darkWingFileName": wing_detector_dark_run,
            "sensitivityMainFileName": main_detector_sensitivities_file,
            "sensitivityWingFileName": wing_detector_sensitivities_file,
            "defaultMask": [
                {"Pixel": "1-18,239-256"},
                {"Bank": "18-24,42-48"},
                {"Bank": "49", "Tube": "1"},
            ],
            "StandardAbsoluteScale": scale_factor,
            "DBScalingBeamRadius": scaling_beam_radius,
            "mmRadiusForTransmission": "",
            "absoluteScaleMethod": "standard",
            "numMainQBins": number_linear_q1d_bins_main_detector,
            "numWingQBins": number_linear_q1d_bins_wing_detector,
            "numMainQxQyBins": number_linear_q2d_bins_main_detector,
            "numWingQxQyBins": number_linear_q2d_bins_wing_detector,
            "1DQbinType": plot_type,
            "QbinType": plot_binning,
            "LogQBinsPerDecadeMain": number_log_bins_per_decade_main_detector,
            "LogQBinsPerDecadeWing": number_log_bins_per_decade_wing_detector,
            "useLogQBinsDecadeCenter": False,
            "useLogQBinsEvenDecade": False,
            "sampleApertureSize": 14,
            "QminMain": q_range_main[0],
            "QmaxMain": q_range_main[1],
            "QminWing": q_range_wing[0],
            "QmaxWing": q_range_wing[1],
            "overlapStitchQmin": overlap_stitch_range[0],
            "overlapStitchQmax": overlap_stitch_range[1],
            "usePixelCalibration": flexible_pixel_sizes,
            "useTimeSlice": False,
            "timeSliceInterval": 60,
            "WedgeMinAngles": wedge_min_angles,
            "WedgeMaxAngles": wedge_max_angles,
            "autoWedgeQmin": auto_wedge_qmin,
            "autoWedgeQmax": auto_wedge_qmax,
            "autoWedgeQdelta": auto_wedge_delta_q,
            "autoWedgePeakWidth": auto_wedge_peak_width,
            "autoWedgeAzimuthalDelta": auto_wedge_delta_azimuthal_angle,
            "autoWedgeBackgroundWidth": auto_wedge_background_width,
            "autoWedgeSignalToNoiseMin": auto_wedge_minimum_signal_to_noise_ratio,
            "wedge1QminMain": q_range_main_wedge0[0],
            "wedge1QmaxMain": q_range_main_wedge0[1],
            "wedge1QminWing": q_range_wing_wedge0[0],
            "wedge1QmaxWing": q_range_wing_wedge0[1],
            "wedge1overlapStitchQmin": OL_range_wedge0[0],
            "wedge1overlapStitchQmax": OL_range_wedge0[1],
            "wedge2QminMain": q_range_main_wedge1[0],
            "wedge2QmaxMain": q_range_main_wedge1[1],
            "wedge2QminWing": q_range_wing_wedge1[0],
            "wedge2QmaxWing": q_range_wing_wedge1[1],
            "wedge2overlapStitchQmin": OL_range_wedge1[0],
            "wedge2overlapStitchQmax": OL_range_wedge1[1],
        },
    }

    common_configuration_full = reduction_parameters(
        common_configuration, "BIOSANS", validate=False
    )
    # pretty_print(common_configuration_full)

    if len(background_runs) == 1 and len(sample_runs) > len(background_runs):
        background_runs = background_runs * len(sample_runs)
    if len(background_transmission_runs) == 1 and len(sample_transmission_runs) > len(
        background_transmission_runs
    ):
        background_transmission_runs = background_transmission_runs * len(
            sample_transmission_runs
        )
    if len(sample_thickness_list) == 1 and len(sample_runs) > len(
        sample_thickness_list
    ):
        sample_thickness_list = sample_thickness_list * len(sample_runs)

    # Checking if output directory exists, if it doesn't, creates the folder
    # Also, if do not overwrite, then makes sure the directory does not exists.
    output_dir = base_output_directory
    if not overwrite_reduced_data:
        suffix = 0
        while os.path.exists(output_dir):
            output_dir = (
                base_output_directory[0, len(base_output_directory) - 2]
                + "_"
                + str(suffix)
                + "/"
            )
            suffix += 1

    if sample_identifier != "":
        if sample_identifier != "":
            output_dir = base_output_directory + str(sample_identifier) + "/"
            change_outputdir = {
                "configuration": {
                    "outputDir": output_dir,
                },
            }
            common_configuration_full = update_reduction_parameters(
                common_configuration_full, change_outputdir, validate=False
            )
        for subfolder in ["1D", "2D"]:
            output_folder = os.path.join(output_dir, subfolder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

    start_time = time.time()
    # Loop for samples
    for i, sample_name in enumerate(sample_names):
        start_time_loop = time.time()

        # form base output file name
        output_file_name = generate_output_base_name(sample_runs[i], sample_names[i])

        run_data = {
            "sample": {
                "runNumber": sample_runs[i],
                "thickness": sample_thickness_list[i],
                "transmission": {"runNumber": sample_transmission_runs[i]},
            },
            "background": {
                "runNumber": background_runs[i],
                "transmission": {"runNumber": background_transmission_runs[i]},
            },
            "outputFileName": output_file_name,
        }

        # Update our common settings with the particulars of the current reduction
        reduction_input = update_reduction_parameters(
            common_configuration_full, run_data, validate=True
        )
        # pretty_print(reduction_input)
        reduction_input["configuration"]["WedgeMinAngles"] = wedge_min_angles
        reduction_input["configuration"]["WedgeMaxAngles"] = wedge_max_angles

        # Load all files
        loaded = load_all_files(reduction_input, use_nexus_idf=True)

        # Reduced from workspaces loaded from NeXus files
        out = reduce_single_configuration(loaded, reduction_input)

        plot_reduction_output(out, reduction_input)

        print("\nloop_" + str(i + 1) + ": ", time.time() - start_time_loop)

        if np.remainder(i, refresh_cycle) == 0 and i > 0:
            # mtd.clear()
            clear_buffer()

    print("Total Time : ", time.time() - start_time)

    # samples shall be list of file names
    test_log_files = list()
    for i_s, sample in enumerate(sample_runs):
        output_file_name = generate_output_base_name(sample, sample_names[i_s])
        test_log_file = generate_output_log_file(
            base_output_directory, output_file_name, ""
        )
        assert os.path.exists(
            test_log_file
        ), f"Output log file {test_log_file} cannot be found."
        test_log_files.append(test_log_file)

    return test_log_files


def generate_output_base_name(sample_run_i: Union[str, int], sample_name_i: str) -> str:
    """Generate base name for output files according to sample run (Nexus file or run number)
    and sample name
    """
    if isinstance(sample_run_i, str) and os.path.exists(sample_run_i):
        # a full path to NeXus is given
        part_1 = os.path.basename(sample_run_i).split(".")[0]
    else:
        part_1 = sample_run_i
    output_file_name_base = f"r{part_1}_{sample_name_i}"

    return output_file_name_base


def generate_output_log_file(
    output_directory: str, file_base_name: str, file_suffix: str
) -> str:
    """Generate output log file name as the standard drtsans/biosans convention"""
    filename = os.path.join(
        output_directory, f"{file_base_name}_reduction_log{file_suffix}.hdf"
    )
    return filename
