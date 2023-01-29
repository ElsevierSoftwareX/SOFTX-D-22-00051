import pytest
import os
import numpy as np
import time
from drtsans.mono.bar_scan_pixel_calibration import generate_spice_pixel_map
from mantid.simpleapi import LoadNexusProcessed
from mantid.simpleapi import DeleteWorkspace


def test_pixel_calibration(reference_dir, generatecleanfile):
    """

    Parameters
    ----------
    reference_dir

    Returns
    -------

    """
    # Set and clean output
    test_output_dir = generatecleanfile("test_bar_scan")

    # First and last pt for the barscan: Set by user
    # -------------------------------------------------------------------------------------------------------
    # IPTS 828 Exp 280.  (/HFIR/CG2/IPTS-828/exp280/Datafiles)
    root_dir = os.path.join(reference_dir.new.gpsans, "calibrations")
    ipts = 828
    exp_number = 280
    scan_number = 5
    first_pt = 1
    last_pt = 12

    flood_ipts = 828
    flood_exp = 280
    flood_scan = 4
    flood_pt = 1

    mask_file = os.path.join(
        reference_dir.new.gpsans, "calibrations/mask_pixel_map.nxs"
    )
    assert os.path.exists(mask_file), f"Mask file {mask_file} does not exist"

    # Calculate pixel calibration file
    calibration_results = generate_spice_pixel_map(
        ipts,
        exp_number,
        scan_number,
        range(first_pt, last_pt + 1),
        flood_ipts,
        flood_exp,
        flood_scan,
        flood_pt,
        root_dir,
        test_output_dir,
        mask_file,
    )
    calibration_table_file = None
    for index, returned in enumerate(calibration_results):
        if index == 0:
            bar_scan_dataset, flood_file = returned
            assert len(bar_scan_dataset) == last_pt - first_pt + 1
            time.sleep(1)
        elif index == 1:
            calibration_stage0, db_file = returned
            time.sleep(1)
            assert calibration_stage0.state_flag == 1
        elif index == 2:
            calibration_stage1, flood_ws_name = returned
            assert calibration_stage1.state_flag == 2
        elif index == 3:
            calibration_table_file = returned
        else:
            raise RuntimeError(f"Index = {index} is not defined")

    print(
        f"Calibraton file {calibration_table_file} of type {type(calibration_table_file)}"
    )
    assert os.path.exists(calibration_table_file)

    # Get expected data file
    expected_calib_nexus = os.path.join(
        reference_dir.new.gpsans,
        f"calibrations/CG2_Pixel_Calibration_Expected_{last_pt - first_pt + 1}.nxs",
    )
    assert os.path.exists(
        expected_calib_nexus
    ), f"Gold result (file) {expected_calib_nexus} cannot be found."

    # Compare 2 NeXus file
    compare_pixel_calibration_files(calibration_table_file, expected_calib_nexus)

    # clean up
    # mysterious leftover workspace due to the design of generate_spice_pixel_map
    # barscan_GPSANS_detector1_20180220:	0.393216 MB
    # flood_run:	1.181425 MB
    # tubewidth_GPSANS_detector1_20180220:	0.393216 MB
    DeleteWorkspace("barscan_GPSANS_detector1_20180220")
    DeleteWorkspace("flood_run")
    DeleteWorkspace("tubewidth_GPSANS_detector1_20180220")


def compare_pixel_calibration_files(test_file_name, gold_file_name):
    """Compare 2 calibration file by the pixel calibration value

    Algorithm:
    1. Load both processed NeXus files
    2. Compare workspaces' Y value

    Parameters
    ----------
    test_file_name
    gold_file_name

    Returns
    -------

    """
    # Load calibration file to Mantid workspace
    test_calib_ws = LoadNexusProcessed(Filename=test_file_name)
    gold_calib_ws = LoadNexusProcessed(Filename=gold_file_name)

    # Get calibration values
    test_cal_values = np.array(test_calib_ws.column(1)).flatten()
    gold_cal_values = np.array(gold_calib_ws.column(1)).flatten()

    np.testing.assert_allclose(test_cal_values, gold_cal_values)

    # clean up
    DeleteWorkspace(test_calib_ws)
    DeleteWorkspace(gold_calib_ws)


if __name__ == "__main__":
    pytest.main(__file__)
