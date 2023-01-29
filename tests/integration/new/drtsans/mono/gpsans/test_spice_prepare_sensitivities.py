import pytest
import os
import numpy as np
from mantid.simpleapi import mtd
from mantid.simpleapi import DeleteWorkspace
from mantid.simpleapi import LoadNexusProcessed
from drtsans.mono.gpsans.prepare_sensitivities_correction import (
    prepare_spice_sensitivities_correction,
    SpiceRun,
)


def test_sensitivities_with_bar(reference_dir, generatecleanfile):
    """Test preparing sensitivities from converted SPICE files with bar scan calibration

    Returns
    -------

    """
    # Experiment setup
    experiment_number = 280
    # Input Flood Runs
    flood_scan_pt_list = (
        (38, 1),
        (40, 1),
        (42, 1),
    )  # list of 2 tuple as (scan number, pt number)

    # Direct beam / transmission
    direct_beam_scan_pt_list = (
        (37, 1),
        (39, 1),
        (41, 1),
    )  # list of 2 tuple as (scan number, pt number)
    # Beam center size
    beam_center_mask_radius_mm = 140  # mm

    # Default mask to detector
    pixels_to_mask = "1-8,249-256"

    # Pixel calibration
    # PIXEL_CALIBRATION = None
    pixel_calib_file = os.path.join(
        reference_dir.new.gpsans, "calibrations/pixel_calibration_gold_sens.json"
    )

    # Corrections
    do_solid_angle_correction = True

    # Use detectors in 3 positions for sensitivities
    moving_detector_method = True

    # THRESHOLD
    min_threshold = 0.5
    max_threshold = 1.5

    # Output directory
    output_dir = generatecleanfile("gpsans_sensitivity_test")

    # Mask
    mask_xml_file = None  # 'Mask.XML'

    # Set the directory for already converted SPICE files
    nexus_dir = os.path.join(reference_dir.new.gpsans, f"Exp{experiment_number}")
    # Check
    if not os.path.exists(nexus_dir):
        raise RuntimeError(
            f"[ERROR] Converted NeXus-SPICE directory {nexus_dir} does not exist"
        )

    # Convert flood runs
    CG2 = "CG2"
    flood_spice_runs = [
        SpiceRun(CG2, -1, experiment_number, scan_i, pt_i)
        for scan_i, pt_i in flood_scan_pt_list
    ]
    if direct_beam_scan_pt_list is None:
        transmission_spice_runs = None
    else:
        transmission_spice_runs = [
            SpiceRun(CG2, -1, experiment_number, scan_i, pt_i)
            for scan_i, pt_i in direct_beam_scan_pt_list
        ]

    # Correction
    test_sens_nxs = prepare_spice_sensitivities_correction(
        flood_spice_runs,
        transmission_spice_runs,
        moving_detector_method,
        min_threshold,
        max_threshold,
        nexus_dir,
        mask_xml_file,
        pixels_to_mask,
        beam_center_mask_radius_mm,
        solid_angle_correction=do_solid_angle_correction,
        pixel_calibration_file=pixel_calib_file,
        output_dir=output_dir,
        file_suffix="spice",
    )

    # Verify
    verify_results(test_sens_nxs, reference_dir)

    # NOTE:
    # leftover workspaces in memory
    # barscan_GPSANS_detector1_20180220:	0.393216 MB
    # BC_CG2_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000370001.nxs.h5:
    #   1.182257 MB
    # BC_CG2_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000390001.nxs.h5:
    #   1.182257 MB
    # BC_CG2_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000410001.nxs.h5:
    #   1.182257 MB
    # GPSANS_28000380001:	22.415297 MB
    # GPSANS_28000380001_processed_histo:	1.182257 MB
    # GPSANS_28000400001:	1.182257 MB
    # GPSANS_28000400001_processed_histo:	1.182257 MB
    # GPSANS_28000420001:	1.182257 MB
    # GPSANS_28000420001_processed_histo:	1.182257 MB
    # sensitivities:	1.179928 MB
    # sensitivities_new:	1.181633 MB
    # tubewidth_GPSANS_detector1_20180220:	0.393216 MB
    DeleteWorkspace("barscan_GPSANS_detector1_20180220")
    DeleteWorkspace("sensitivities")
    DeleteWorkspace("sensitivities_new")
    DeleteWorkspace("tubewidth_GPSANS_detector1_20180220")
    for ws in mtd.getObjectNames():
        if str(ws).startswith("BC_CG2_/SNS"):
            DeleteWorkspace(ws)
        if str(ws).startswith("GPSANS_28000"):
            DeleteWorkspace(ws)


def verify_results(test_sensitivities_file, reference_dir):
    """

    Parameters
    ----------
    test_sensitivities_file
    reference_dir: tests.conftest.rett

    Returns
    -------

    """
    # Get gold file
    gold_sens_file = os.path.join(
        reference_dir.new.gpsans, "calibrations/sens_CG2_spice_bar.nxs"
    )
    if not os.path.exists(gold_sens_file):
        raise RuntimeError(
            f"Expected (gold) sensitivities cannot be found at {gold_sens_file}"
        )

    # Compare sensitivities
    gold_sens_ws = LoadNexusProcessed(Filename=gold_sens_file)
    test_sens_ws = LoadNexusProcessed(Filename=test_sensitivities_file)
    np.testing.assert_allclose(test_sens_ws.extractY(), gold_sens_ws.extractY())

    # Clean up
    DeleteWorkspace(gold_sens_ws)
    DeleteWorkspace(test_sens_ws)


if __name__ == "__main__":
    pytest.main([__file__])
