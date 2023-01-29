"""
    Test EASANS sensitivities preparation algorithm
"""
import pytest
import numpy as np
import os
from drtsans.prepare_sensivities_correction import PrepareSensitivityCorrection
from mantid.simpleapi import LoadNexusProcessed
from mantid.simpleapi import DeleteWorkspace
from tempfile import mktemp


def verify_sensitivities_file(test_sens_file, gold_sens_file, atol=None):
    """ """
    if atol is None:
        atol = 3e-5

    # Load processed NeXus files from tests and gold result
    test_sens_ws = LoadNexusProcessed(Filename=test_sens_file)
    gold_sens_ws = LoadNexusProcessed(Filename=gold_sens_file)

    # Compare number of spectra
    assert test_sens_ws.getNumberHistograms() == gold_sens_ws.getNumberHistograms()

    # Verify sensitivity value
    test_y = test_sens_ws.extractY().flatten()
    gold_y = gold_sens_ws.extractY().flatten()
    np.testing.assert_allclose(gold_y, test_y, atol=atol, equal_nan=True)

    # Verify sensitivity error
    test_e = test_sens_ws.extractE().flatten()
    gold_e = gold_sens_ws.extractE().flatten()
    np.testing.assert_allclose(gold_e, test_e, atol=atol, equal_nan=True)


def test_eqsans_prepare_sensitivities(reference_dir, cleanfile):
    """Integration test on algorithm to prepare EQSANS' sensitivities

    Returns
    -------

    """
    # INSTRUMENT = 'CG2'  # 'CG2'  # From 'EQSANS', 'CG3'
    INSTRUMENT = "EQSANS"  # Main

    # Check whether the test shall be skipped
    if not os.path.exists("/SNS/EQSANS/IPTS-24648/nexus/EQSANS_111030.nxs.h5"):
        pytest.skip("Test files cannot be accessed.")

    # Input Flood Runs
    FLOOD_RUNS = os.path.join(reference_dir.new.eqsans, "EQSANS_111030.nxs.h5")

    # Beam center
    DIRECT_BEAM_RUNS = os.path.join(
        reference_dir.new.eqsans, "EQSANS_111042.nxs.h5"
    )  # 111042

    # Beam center size
    MASK_BEAM_CENTER_RADIUS = 65  # mm

    # Dark current: No mask, no solid angle
    DARK_CURRENT_RUNS = os.path.join(
        reference_dir.new.eqsans, "EQSANS_108764.nxs.h5"
    )  # 108764

    MASKED_PIXELS = "1-18,239-256"

    # Corrections
    SOLID_ANGLE_CORRECTION = True

    # If it is GPSANS or BIOSANS there could be 2 options to calculate detector efficiencies
    MOVING_DETECTORS = False

    # THRESHOLD
    MIN_THRESHOLD = 0.5
    MAX_THRESHOLD = 2.0

    preparer = PrepareSensitivityCorrection(INSTRUMENT, False)

    # Load flood runs
    preparer.set_flood_runs(FLOOD_RUNS)

    # Process beam center runs
    preparer.set_direct_beam_runs(DIRECT_BEAM_RUNS)

    # Set extra masks
    preparer.set_masks(None, MASKED_PIXELS)

    # Set beam center radius
    preparer.set_beam_center_radius(MASK_BEAM_CENTER_RADIUS)

    # Dark runs
    preparer.set_dark_current_runs(DARK_CURRENT_RUNS)

    # Solid angle
    preparer.set_solid_angle_correction_flag(SOLID_ANGLE_CORRECTION)

    # Run
    # Absolute path overrides saving to the default output directory selected by the developer in Mantid's preferences.
    output_sens_file = mktemp(suffix="nxs", prefix="meta_overwrite_test1")
    print("[DEBUG] Output file: {}".format(output_sens_file))
    cleanfile(output_sens_file)
    # output_sens_file = '/tmp/IntegrateTest_EQSANS_Sens.nxs'
    preparer.execute(
        MOVING_DETECTORS,
        MIN_THRESHOLD,
        MAX_THRESHOLD,
        output_nexus_name=output_sens_file,
    )

    # Verify file existence
    assert os.path.exists(
        output_sens_file
    ), "Output sensitivity file {} cannot be found".format(output_sens_file)

    # Verify value
    gold_eq_file = os.path.join(
        reference_dir.new.sans, "sensitivities", "EQSANS_sens_patched_20200602.nxs"
    )

    verify_sensitivities_file(output_sens_file, gold_eq_file)

    # Clean
    os.remove(output_sens_file)

    # NOTE:
    # mysterious leftover workspace from this test
    # BC_EQSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/sns/eqsans/EQSANS_111042.nxs.h5:	37.114117 MB
    # EQSANS_111030:	43.589589 MB
    # EQSANS_111030_sensitivity:	1.179928 MB
    # EQSANS_111030_sensitivity_new:	22.355925 MB
    # gold_sens_ws:	22.355492 MB
    # test_sens_ws:	22.355925 MB
    DeleteWorkspace(
        "BC_EQSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/sns/eqsans/EQSANS_111042.nxs.h5"
    )
    DeleteWorkspace("EQSANS_111030")
    DeleteWorkspace("EQSANS_111030_sensitivity")
    DeleteWorkspace("EQSANS_111030_sensitivity_new")
    DeleteWorkspace("gold_sens_ws")
    DeleteWorkspace("test_sens_ws")


@pytest.mark.skip(reason="This test is too large to run on build server")
def test_cg3_main_prepare_sensitivities():
    """Integration test on algorithms to prepare sensitivities for BIOSANS's main detector

    Returns
    -------

    """
    # Check whether the test shall be skipped
    if not os.path.exists("/HFIR/CG3/IPTS-23782/nexus/CG3_4829.nxs.h5"):
        pytest.skip("Test files of CG3 cannot be accessed.")

    INSTRUMENT = "CG3"  # Main

    # CG3: Main
    FLOOD_RUNS = 4829

    # About Masks
    # CG3 Main:
    DIRECT_BEAM_RUNS = 4827

    # Transmission run
    TRANSMISSION_RUNS = 4828  # GG3 main
    # Transmission flood run
    TRANSMISSION_FLOOD_RUNS = 4829

    # Default mask to detector
    # CG3:
    MASKED_PIXELS = "1-18,239-256"  # CG3
    # Mask angle: must 2 values as min and max or None
    MAIN_DET_MASK_ANGLE = 1.5
    WING_DET_MASK_ANGLE = 57.0
    BEAM_TRAP_SIZE_FACTOR = 2

    # Corrections
    SOLID_ANGLE_CORRECTION = True
    # Flag to do dependent correction with transmission correction
    THETA_DEPENDENT_CORRECTION = True

    # THRESHOLD
    MIN_THRESHOLD = 0.5
    MAX_THRESHOLD = 2.0

    preparer = PrepareSensitivityCorrection(INSTRUMENT, is_wing_detector=False)
    # Load flood runs
    preparer.set_flood_runs(FLOOD_RUNS)

    # Process beam center runs
    if DIRECT_BEAM_RUNS is not None:
        preparer.set_direct_beam_runs(DIRECT_BEAM_RUNS)

    # Set extra masks
    preparer.set_masks(
        None,
        MASKED_PIXELS,
        wing_det_mask_angle=WING_DET_MASK_ANGLE,
        main_det_mask_angle=MAIN_DET_MASK_ANGLE,
    )

    # Transmission
    preparer.set_transmission_correction(
        transmission_flood_runs=TRANSMISSION_FLOOD_RUNS,
        transmission_reference_runs=TRANSMISSION_RUNS,
        beam_trap_factor=BEAM_TRAP_SIZE_FACTOR,
    )
    preparer.set_theta_dependent_correction_flag(THETA_DEPENDENT_CORRECTION)

    # Solid angle
    preparer.set_solid_angle_correction_flag(SOLID_ANGLE_CORRECTION)

    # Run
    output_sens_file = "IntegrateTest_CG3_Main_Sens.nxs"
    preparer.execute(
        False, MIN_THRESHOLD, MAX_THRESHOLD, output_nexus_name=output_sens_file
    )

    # Verify file existence
    assert os.path.exists(output_sens_file)

    # Verify value
    gold_eq_file = (
        "/SNS/EQSANS/shared/sans-backend/data/new/ornl"
        "/sans/sensitivities/CG3_Sens_Main.nxs"
    )

    verify_sensitivities_file(output_sens_file, gold_eq_file)

    # Clean
    os.remove(output_sens_file)


def test_cg3_wing_prepare_sensitivities():
    """Integration test on algorithms to prepare sensitivities for BIOSANS's wing detector

    Returns
    -------

    """
    # Check whether the test shall be skipped
    if not os.path.exists("/HFIR/CG3/IPTS-23782/nexus/CG3_4835.nxs.h5"):
        pytest.skip("Test files of CG3 cannot be accessed.")

    INSTRUMENT = "CG3"  # Main

    # CG3: Wing
    FLOOD_RUNS = 4835
    # BIO-SANS detector
    WING_DETECTOR = True  # this is main detector

    # About Masks
    # CG3 Main:
    DIRECT_BEAM_RUNS = 4830

    # Transmission run
    TRANSMISSION_RUNS = 4831  # GG3 main
    # Transmission flood run
    TRANSMISSION_FLOOD_RUNS = 4835

    # CG3:
    MASKED_PIXELS = "1-18,239-256"  # CG3
    # Mask angle: must 2 values as min and max or None
    MAIN_DET_MASK_ANGLE = 0.75
    WING_DET_MASK_ANGLE = 57.0
    BEAM_TRAP_SIZE_FACTOR = 2

    # Corrections
    SOLID_ANGLE_CORRECTION = True
    # Flag to do dependent correction with transmission correction
    THETA_DEPENDENT_CORRECTION = True

    # If it is GPSANS or BIOSANS there could be 2 options to calculate detector efficiencies
    MOVING_DETECTORS = False

    # THRESHOLD
    MIN_THRESHOLD = 0.5
    MAX_THRESHOLD = 2.0

    # Prepare data
    preparer = PrepareSensitivityCorrection(INSTRUMENT, WING_DETECTOR)
    # Load flood runs
    preparer.set_flood_runs(FLOOD_RUNS)

    # Process beam center runs
    preparer.set_direct_beam_runs(DIRECT_BEAM_RUNS)

    # Set extra masks
    preparer.set_masks(
        None,
        MASKED_PIXELS,
        wing_det_mask_angle=WING_DET_MASK_ANGLE,
        main_det_mask_angle=MAIN_DET_MASK_ANGLE,
    )

    # Transmission
    preparer.set_transmission_correction(
        transmission_flood_runs=TRANSMISSION_FLOOD_RUNS,
        transmission_reference_runs=TRANSMISSION_RUNS,
        beam_trap_factor=BEAM_TRAP_SIZE_FACTOR,
    )
    preparer.set_theta_dependent_correction_flag(THETA_DEPENDENT_CORRECTION)

    preparer.set_solid_angle_correction_flag(SOLID_ANGLE_CORRECTION)

    # Run
    output_sens_file = "IntegrateTest_CG3_Wing_Sens.nxs"
    preparer.execute(MOVING_DETECTORS, MIN_THRESHOLD, MAX_THRESHOLD, output_sens_file)

    # Verify file existence
    assert os.path.exists(output_sens_file)

    # Verify value
    gold_cg2_wing_file = (
        "/SNS/EQSANS/shared/sans-backend/data/new/ornl"
        "/sans/sensitivities/CG3_Sens_Wing.nxs"
    )

    verify_sensitivities_file(output_sens_file, gold_cg2_wing_file, atol=1e-7)

    # Clean
    os.remove(output_sens_file)

    # NOTE:
    # mysterious leftover workspaces in memory
    # BC_CG3_CG3_4830:	2.763785 MB
    # BIOSANS_4835: 44.614937 MB
    # BIOSANS_4835_sensitivity:	2.162968 MB
    # BIOSANS_4835_sensitivity_new:	5.686553 MB
    # gold_sens_ws:	5.686328 MB
    # test_sens_ws:	5.686553 MB
    # TRANS_CG3_4831:	2.762857 MB
    # TRANS_CG3_4835:	5.687177 MB
    DeleteWorkspace("BC_CG3_CG3_4830")
    DeleteWorkspace("BIOSANS_4835")
    DeleteWorkspace("BIOSANS_4835_sensitivity")
    DeleteWorkspace("BIOSANS_4835_sensitivity_new")
    DeleteWorkspace("gold_sens_ws")
    DeleteWorkspace("test_sens_ws")
    DeleteWorkspace("TRANS_CG3_4831")
    DeleteWorkspace("TRANS_CG3_4835")


def test_cg2_sensitivities():
    """Integration test on algorithms to prepare sensitivities for GPSANS's
    with moving detector method

    Returns
    -------

    """
    if not os.path.exists("/HFIR/CG2/IPTS-23801/nexus/CG2_7116.nxs.h5"):
        pytest.skip("Testing file for CG2 cannot be accessed")

    INSTRUMENT = "CG2"  # 'CG2'  # From 'EQSANS', 'CG3'

    # Input Flood Runs
    FLOOD_RUNS = 7116, 7118, 7120  # Single value integer or a list or tuple

    # About Masks
    # CG2:
    DIRECT_BEAM_RUNS = 7117, 7119, 7121
    MASK_BEAM_CENTER_RADIUS = 65  # mm

    # CG2:
    MASKED_PIXELS = "1-8,249-256"

    # If it is GPSANS or BIOSANS there could be 2 options to calculate detector efficiencies
    MOVING_DETECTORS = True

    SOLID_ANGLE_CORRECTION = True

    # THRESHOLD
    MIN_THRESHOLD = 0.5
    MAX_THRESHOLD = 2.0

    preparer = PrepareSensitivityCorrection(INSTRUMENT)
    # Load flood runs
    preparer.set_flood_runs(FLOOD_RUNS)

    # Process beam center runs
    preparer.set_direct_beam_runs(DIRECT_BEAM_RUNS)

    # Set extra masks
    preparer.set_masks(None, MASKED_PIXELS)

    # Set beam center radius
    preparer.set_beam_center_radius(MASK_BEAM_CENTER_RADIUS)

    preparer.set_solid_angle_correction_flag(SOLID_ANGLE_CORRECTION)

    # Run
    output_sens_file = "IntegrateTest_CG2_MovingDet.nxs"
    preparer.execute(MOVING_DETECTORS, MIN_THRESHOLD, MAX_THRESHOLD, output_sens_file)

    # Verify file existence
    assert os.path.exists(output_sens_file)

    # Verify value
    gold_gp_file = (
        "/SNS/EQSANS/shared/sans-backend/data/new/ornl"
        "/sans/sensitivities/CG2_Sens_Moving_Dets.nxs"
    )

    verify_sensitivities_file(output_sens_file, gold_gp_file, atol=1e-7)

    # Clean
    os.remove(output_sens_file)

    # NOTE:
    # mysterious leftover workspaces in memory
    # BC_CG2_CG2_7117:	1.434937 MB
    # BC_CG2_CG2_7119:	1.448089 MB
    # BC_CG2_CG2_7121:	1.442393 MB
    # gold_sens_ws:	12.333224 MB
    # GPSANS_7116:	33.567113 MB
    # GPSANS_7116_processed_histo:	12.334073 MB
    # GPSANS_7118:	12.118841 MB
    # GPSANS_7118_processed_histo:	12.118841 MB
    # GPSANS_7120:	12.078681 MB
    # GPSANS_7120_processed_histo:	12.078681 MB
    # sensitivities:	1.179928 MB
    # sensitivities_new:	12.333449 MB
    # test_sens_ws:	12.333449 MB
    DeleteWorkspace("BC_CG2_CG2_7117")
    DeleteWorkspace("BC_CG2_CG2_7119")
    DeleteWorkspace("BC_CG2_CG2_7121")
    DeleteWorkspace("gold_sens_ws")
    DeleteWorkspace("GPSANS_7116")
    DeleteWorkspace("GPSANS_7116_processed_histo")
    DeleteWorkspace("GPSANS_7118")
    DeleteWorkspace("GPSANS_7118_processed_histo")
    DeleteWorkspace("GPSANS_7120")
    DeleteWorkspace("GPSANS_7120_processed_histo")
    DeleteWorkspace("sensitivities")
    DeleteWorkspace("sensitivities_new")
    DeleteWorkspace("test_sens_ws")


if __name__ == "__main__":
    pytest.main([__file__])
