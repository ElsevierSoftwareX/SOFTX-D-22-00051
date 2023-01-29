from typing import List, Tuple, Union, Any
import json
import os
from drtsans.mono.gpsans import (
    load_all_files,
    reduce_single_configuration,
    plot_reduction_output,
    reduction_parameters,
    update_reduction_parameters,
)
from matplotlib.colors import LogNorm
from drtsans.mono.spice_data import map_to_nexus

CG2 = "CG2"


def reduce_gpsans_nexus(
    ipts_number: int,
    exp_number: int,
    samples: List[Tuple[int, int]],
    sample_thick: List[float],
    sample_names: List[str],
    bkgd: List[Tuple[int, int]],
    samples_trans: List[Tuple[int, int]],
    bkgd_trans: List[Tuple[int, int]],
    block_beam: Tuple[int, int],
    empty_trans: List[Tuple[int, int]],
    beam_center: List[Tuple[int, int]],
    nexus_dir: str,
    mask_file_name: str,  # '' for no mask file
    dark_file_name: str,  # '' for no dark file
    use_log_1d: bool,
    use_log_2d_binning: bool,
    common_configuration: Any,
    q_range: Union[List[Union[None, int]], Tuple[Union[int, None], Union[int, None]]],
    use_mask_back_tubes: bool,
    debug_output: bool = False,
):

    # Never touch!  drtsans specific

    # convert SPICE to Nexus
    samples = map_to_nexus(CG2, ipts_number, exp_number, samples, nexus_dir)
    samples_trans = map_to_nexus(CG2, ipts_number, exp_number, samples_trans, nexus_dir)
    bkgd = map_to_nexus(CG2, ipts_number, exp_number, bkgd, nexus_dir)
    bkgd_trans = map_to_nexus(CG2, ipts_number, exp_number, bkgd_trans, nexus_dir)
    block_beam = map_to_nexus(CG2, ipts_number, exp_number, [block_beam], nexus_dir)[0]

    # empty transmission
    empty_trans = map_to_nexus(CG2, ipts_number, exp_number, empty_trans, nexus_dir)
    common_configuration["emptyTransmission"]["runNumber"] = empty_trans
    # beam center
    beam_center = map_to_nexus(CG2, ipts_number, exp_number, beam_center, nexus_dir)
    common_configuration["beamCenter"]["runNumber"] = beam_center
    # mask file and dark file to "" if not specified
    mask_file_name = "" if mask_file_name is None else mask_file_name
    dark_file_name = "" if dark_file_name is None else dark_file_name

    if use_log_2d_binning:
        log_flag = {"norm": LogNorm()}
    else:
        log_flag = {"vmin": 0, "vmax": 100}

    # Add on the other reduction parameters with their default values (most will be empty)
    common_configuration_full = reduction_parameters(
        common_configuration, "GPSANS", validate=False
    )

    # Create output directory
    output_dir = common_configuration_full["configuration"]["outputDir"]
    for subfolder in ["1D", "2D"]:
        output_folder = os.path.join(output_dir, subfolder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for i in range(len(samples)):
        # Settings particular to each reduction session
        run_data = {
            "sample": {
                "runNumber": samples[i],
                "thickness": sample_thick[i],
                "transmission": {"runNumber": samples_trans[i]},
            },
            "background": {
                "runNumber": bkgd[i],
                "transmission": {"runNumber": bkgd_trans[i]},
            },
            "outputFileName": sample_names[i],
            "configuration": {
                "Qmin": q_range[0],
                "Qmax": q_range[1],
                "useMaskBackTubes": use_mask_back_tubes,
                "blockedBeamRunNumber": block_beam,
                "maskFileName": mask_file_name,
                "darkFileName": dark_file_name,
            },
        }

        # Update our common settings with the particulars of the current reduction
        reduction_input = update_reduction_parameters(
            common_configuration_full, run_data, validate=True
        )

        # Begin reduction. Be sure to validate the parameters before.
        # Load files
        loaded = load_all_files(
            reduction_input,
            path=f"/HFIR/CG2/IPTS-{ipts_number}/shared/Exp{exp_number}",
            use_nexus_idf=True,
            debug_output=debug_output,
        )
        # Reduce data from workspaces
        out = reduce_single_configuration(
            loaded, reduction_input, debug_output=debug_output
        )
        # Output
        plot_reduction_output(
            out, reduction_input, loglog=use_log_1d, imshow_kwargs=log_flag
        )

        # Save the reduction parameters of each reduction session to a JSON file
        output_dir = reduction_input["configuration"]["outputDir"]
        output_json_file = os.path.join(
            output_dir, f"{sample_names[i]}.json"
        )  # full path to the JSON file
        with open(output_json_file, "w") as file_handle:
            json.dump(reduction_input, file_handle, indent=2)

        return output_dir
