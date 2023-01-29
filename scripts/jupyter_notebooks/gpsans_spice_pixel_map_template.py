from matplotlib import pyplot as plt
import numpy as np
import os

# Mantid imports
from mantid.api import mtd
from mantid.simpleapi import (
    CreateWorkspace,
    LoadEventNexus,
    LoadNexus,
    HFIRSANS2Wavelength,
)

# drtsans imports
from drtsans.mono.gpsans import apply_calibrations, plot_detector
from drtsans.mono.bar_scan_pixel_calibration import generate_spice_pixel_map
from drtsans.tubecollection import TubeCollection

root_dir = "/HFIR/CG2/"
ipts = 828
exp_number = 280
scan_number = 5
first_pt = 1
last_pt = 111  # 111 shall be

flood_ipts = 828
flood_exp = 280
flood_scan = 4
flood_pt = 1

root_save_dir = f"/HFIR/CG2/IPTS-{ipts}/shared/pixel_calibration"
mask_file = "/HFIR/CG2/shared/calibration/spice/mask_pixel_map.nxs"

# --------------------------------------------------------------------------------------------------
# NO TOUCH ZONE
# ---------------------------------------------------------------------------------------------------


def show_bar_scan_files(bar_scan_histogram_files, pt_numbers):
    print(
        "#####\n\nWe inspect a few of the bar scans by looking at the intensity pattern",
        "on a detector with default (uncalibrated) detector heights and positions",
    )
    delta = len(bar_scan_histogram_files) // 4
    # Move to notebook only
    # Load every 4 bar scan for visualization
    # FIXME - consider to remove this for-section
    for index, pt_number in enumerate(pt_numbers):
        if index % delta != 0:
            continue
        output_workspace = f"Demo_Pt{pt_number:04}"
        LoadNexus(
            Filename=bar_scan_histogram_files[index], OutputWorkspace=output_workspace
        )
        plot_workspace(output_workspace)
    plt.show()
    plt.savefig("until_and_center.png")


def show_calibration_stage0(
    barscan_dataset, calibration, pt_numbers, flood_nexus_file, database_file
):
    # Report calibration
    report_tilt(calibration.positions)

    print("#####\n\nComparison before and after applying the calibration")
    middle_run = (pt_numbers[0] + pt_numbers[-1]) // 2
    middle_workspace = "CG2_Exp{}_Scan{}_Pt{}".format(
        exp_number, scan_number, middle_run
    )
    print(f"Middle Pt workspace: {middle_workspace}")
    if middle_workspace not in mtd:
        # Load middle Pt data if it is not loaded
        LoadNexus(
            Filename=barscan_dataset[middle_run], OutputWorkspace=middle_workspace
        )
    # LoadNexus(Filename=os.path.join(save_dir, middle_workspace + '.nxs'),
    #           OutputWorkspace=middle_workspace)
    middle_workspace_calibrated = middle_workspace + "_calibrated"
    calibration.apply(middle_workspace, output_workspace=middle_workspace_calibrated)
    plot_workspace(
        middle_workspace, axes_mode="xy", prefix="before_calibration_"
    )  # before calibration
    plot_workspace(
        middle_workspace_calibrated, axes_mode="xy", prefix="after_calibration"
    )  # calibrated

    # Load flood file, plot raw and first stage calibration
    flood_base_name = os.path.basename(flood_nexus_file).split(".")[0]
    raw_flood_ws_name = f"demo_raw_flood_{flood_base_name}"
    LoadEventNexus(Filename=flood_nexus_file, OutputWorkspace=raw_flood_ws_name)
    HFIRSANS2Wavelength(
        InputWorkspace=raw_flood_ws_name, OutputWorkspace=raw_flood_ws_name
    )
    plot_workspace(raw_flood_ws_name, axes_mode="xy", prefix="Raw Flood")
    # apply calibration and plot again
    calib_flood_ws_name = f"demo_calibrated0_flood_{flood_base_name}"
    apply_calibrations(
        raw_flood_ws_name, output_workspace=calib_flood_ws_name, database=database_file
    )
    plot_workspace(
        calib_flood_ws_name, axes_mode="xy", prefix="Calibrated (Step1 0) Flood"
    )

    # show
    plt.show()

    return raw_flood_ws_name


def show_calibration_stage1(raw_flood_ws_name, database_file):
    # Calibration calculation is over ... Starting testing

    print("#####\n\nApply the barscan and tube width calibration to the flood run")

    # Plot flood workspace raw and calibrated
    print("#####\n\nCompare applying the calibration to flood (stage 1)")

    calibrated_flood_ws_name = (
        f'demo_calibrated1_flood_{raw_flood_ws_name.split("flood_")[1]}'
    )
    apply_calibrations(
        raw_flood_ws_name,
        output_workspace=calibrated_flood_ws_name,
        database=database_file,
    )
    plot_workspace(
        calibrated_flood_ws_name, axes_mode="xy", prefix="Calibrated (Step 2) Flood"
    )

    print(
        "#####\n\nPlot the linear densities of the tubes before and after calibration.",
        "Suppresion of the oslillating intensities indicates the tube-width calibration is correct",
    )
    uncalibrated_densities = linear_density(raw_flood_ws_name)
    calibrated_densities = linear_density(calibrated_flood_ws_name)

    number_tubes = len(uncalibrated_densities)
    CreateWorkspace(
        DataX=range(number_tubes),
        DataY=np.array([uncalibrated_densities, calibrated_densities]),
        NSpec=2,  # two histograms
        Outputworkspace="linear_densities",
    )
    plot_histograms(
        "linear_densities",
        legend=["no calibration", "calibrated"],
        xlabel="Tube Index",
        ylabel="Intensity",
        linewidths=[3, 1],
    )

    plt.show()


def plot_histograms(
    input_workspace,
    legend=[],
    xlabel="X-axis",
    ylabel="Y-axis",
    title="",
    linewidths=[],
):
    r"""Line plot for the histograms of a workspace"""
    workspace = mtd[str(input_workspace)]
    number_histograms = workspace.getNumberHistograms()
    if len(legend) != number_histograms:
        legend = [str(i) for i in range(number_histograms)]
    if len(linewidths) != number_histograms:
        linewidths = [1] * number_histograms
    fig, ax = plt.subplots(subplot_kw={"projection": "mantid"})
    for workspace_index in range(number_histograms):
        ax.plot(
            workspace,
            wkspIndex=workspace_index,
            label=legend[workspace_index],
            linewidth=linewidths[workspace_index],
        )
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="out")
    ax.grid(True)
    fig.show()


def linear_density(workspace):
    r"""
    Tube total intensity per non-masked pixel and per unit length of tube width

    Integrate the total intensity per tube and divide by the number of non-masked pixels in the tube,
    and by the tube width. Front end tubes collect more intentity than the back tubes.
    Similarly, front end tubes have a larger apparent tube width than back tubes.
    The ratio of total intensity to width should be similar for front and end tubes after the calibration.
    """
    collection = TubeCollection(workspace, "detector1").sorted(view="fbfb")
    intensities = np.array([np.sum(tube.readY) for tube in collection])
    widths = np.array([tube.width for tube in collection])
    number_pixels_not_masked = np.array([np.sum(~tube.isMasked) for tube in collection])
    return list(intensities / (number_pixels_not_masked * widths))


def plot_workspace(input_workspace, axes_mode="tube-pixel", prefix=""):
    return plot_detector(
        input_workspace,
        filename=f"{prefix}{str(input_workspace)}.png",
        backend="mpl",
        axes_mode=axes_mode,
        imshow_kwargs={},
    )


def report_tilt(pixel_positions):
    r"""
    Variation in the position of the top and bottom pixels as a function of tube index.
    We perform a linear regression of this variation.
    """
    # Create a 2D array of pixel heights, dimensions are (number_tubes x pixels_in_tube)
    pixel_in_tube_count = 256
    tube_count = int(len(pixel_positions) / pixel_in_tube_count)
    positions = np.array(pixel_positions).reshape((tube_count, pixel_in_tube_count))

    def fit(tube_tip_positions):
        r"""This function will fit the bottom or top pixels against the tube index"""
        tube_indexes = np.arange(tube_count)  # heights as function of tube index
        coeffs = np.polyfit(tube_indexes, tube_tip_positions, 1)
        fitted = np.poly1d(coeffs)(tube_indexes)  # fitted positions of the tube tip
        return coeffs, fitted

    for location, tip_positions in (
        ["top", positions[:, -1]],
        ["bottom", positions[:, 0]],
    ):
        coeffs, fitted = fit(tip_positions)  # fit against tube index
        # Plot the raw positions and the fitted positions
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(tube_count), tip_positions)
        ax.plot(np.arange(tube_count), fitted)
        ax.set_title(f"{location} pixels")
        # Print a few representative properties of the tilt
        print(location, " pixels:")
        print(f"    slope = {1000 * coeffs[0]:.3f} mili-meters / tube")
        print(
            f"    position difference between last and first tube = {1000 * (fitted[-1] - fitted[0]):.3f} mili-meters"
        )
        # save figure
        plt.savefig(f"{location}.png")


# Calculate pixel calibration file
pt_number_list = range(first_pt, last_pt + 1)
calibration_returns = generate_spice_pixel_map(
    ipts,
    exp_number,
    scan_number,
    pt_number_list,
    flood_ipts,
    flood_exp,
    flood_scan,
    flood_pt,
    root_dir,
    root_save_dir,
    mask_file,
)

# Get all the return in the various stages
bar_scan_dataset = None
flood_nexus = None
demo_flood_raw_ws = None
calib_db_file = None

for index, returned in enumerate(calibration_returns):
    if index == 0:
        # show the bar scans
        bar_scan_dataset, flood_nexus = returned
        show_bar_scan_files(bar_scan_dataset, pt_number_list)

    elif index == 1:
        # first stage of the calibration
        calibration_stage0, calib_db_file = returned
        demo_flood_raw_ws = show_calibration_stage0(
            bar_scan_dataset,
            calibration_stage0,
            pt_number_list,
            flood_nexus,
            calib_db_file,
        )

    elif index == 2:
        # second stage of the calibration
        calibration_stage1, flood_ws_name = returned
        assert calibration_stage1.state_flag == 2
        show_calibration_stage1(demo_flood_raw_ws, calib_db_file)

    elif index == 3:
        calibration_table_file = returned
    else:
        raise RuntimeError(f"Index = {index} is not defined")
