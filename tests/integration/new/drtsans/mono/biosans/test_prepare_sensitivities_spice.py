import pytest
import os
import numpy as np
import warnings
from mantid.api import AnalysisDataService
from mantid.simpleapi import DeleteWorkspace
from drtsans.mono.spice_data import SpiceRun
from drtsans.mono.biosans.prepare_sensitivities_correction import (
    prepare_spice_sensitivities_correction,
)
from mantid.simpleapi import LoadNexusProcessed

warnings.simplefilter(action="ignore", category=FutureWarning)

workspaces = [
    "BC_CG3_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/biosans/CG3_054900200001.nxs.h5"
    "BIOSANS_54900200001",
    "BIOSANS_54900200001_sensitivity",
    "BIOSANS_54900200001_sensitivity_new",
    "BIOSANS_54900220001",
    "TRANS_CG3_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/biosans/CG3_054900160001.nxs.h5",
    "TRANS_CG3_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/biosans/CG3_054900200001.nxs.h5",
    "BC_CG3_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/biosans/CG3_054900200001.nxs.h5",
    "BIOSANS_54900200001",
]


@pytest.mark.skipif(True, reason="Job is too large to run on build server")
def test_main_detector(reference_dir, generatecleanfile, clean_workspace):
    """Test case for CG3 main detector

    This test is skipped

    Flood for main detector at 7m and wing detector at 12.2°-
    /HFIR/CG3/IPTS-17241/exp549/Datafiles/BioSANS_exp549_scan0009_0001.xml

    Empty Beam for Transmission Reference at 7m and 12.2° -
    /HFIR/CG3/IPTS-17241/exp549/Datafiles/BioSANS_exp549_scan0010_0001.xml

    Dark Current for all configurations above -
    /HFIR/CG3/IPTS-17241/exp549/Datafiles/BioSANS_exp549_scan0022_0001.xml
    """
    # output testing directory
    output_dir = generatecleanfile()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    CG3 = "CG3"  # Main

    # IPTS
    IPTS = 17241

    EXPERIMENT = 549

    # CG3: Main
    FLOOD_RUN = (9, 1)  # BIO-SANS detector
    WING_DETECTOR = False  # this is main detector

    # About Masks
    # CG3 Main beam center file/Empty beam.  It is allowed to be left blank
    DIRECT_BEAM_RUN = None
    # Beam center size
    MASK_BEAM_CENTER_RADIUS = 65  # mm

    # Dark current
    DARK_CURRENT_RUN = (22, 1)

    # Transmission empty beam
    OPEN_BEAM_TRANSMISSION = (10, 1)
    # Transmission flood run
    TRANSMISSION_FLOOD_RUN = FLOOD_RUN

    # Default mask to detector
    UNIVERSAL_MASK = None  # 'Mask.XML'
    # CG3:
    MASKED_PIXELS = "1-18,249-256"  # CG3
    # Mask angle: must 2 values as min and max or None
    MAIN_DET_MASK_ANGLE = 2.0  # 0.75# 1.5 #0.75
    WING_DET_MASK_ANGLE = 57.0
    BEAM_TRAP_SIZE_FACTOR = 1.0  # 2

    # Corrections
    SOLID_ANGLE_CORRECTION = True
    # Flag to do dependent correction with transmission correction
    THETA_DEPENDENT_CORRECTION = True

    # THRESHOLD
    MIN_THRESHOLD = 0.5
    MAX_THRESHOLD = 2.0

    # Output
    FILE_SURFIX = "wing" if WING_DETECTOR else "main"
    SENSITIVITY_FILE = os.path.join(
        output_dir, f"{CG3}_sens_{FILE_SURFIX}{FLOOD_RUN}sac_tdc7m.nxs"
    )

    # Convert SPICE file to NeXus file
    flood_run = SpiceRun(CG3, IPTS, EXPERIMENT, FLOOD_RUN[0], FLOOD_RUN[1])
    direct_beam_run = (
        SpiceRun(CG3, IPTS, EXPERIMENT, DIRECT_BEAM_RUN[0], DIRECT_BEAM_RUN[1])
        if DIRECT_BEAM_RUN
        else None
    )
    open_beam_transmission = (
        SpiceRun(
            CG3, IPTS, EXPERIMENT, OPEN_BEAM_TRANSMISSION[0], OPEN_BEAM_TRANSMISSION[1]
        )
        if OPEN_BEAM_TRANSMISSION
        else None
    )
    transmission_flood_run = SpiceRun(
        CG3, IPTS, EXPERIMENT, TRANSMISSION_FLOOD_RUN[0], TRANSMISSION_FLOOD_RUN[1]
    )
    dark_current_run = (
        SpiceRun(CG3, IPTS, EXPERIMENT, DARK_CURRENT_RUN[0], DARK_CURRENT_RUN[1])
        if DARK_CURRENT_RUN
        else None
    )

    prepare_spice_sensitivities_correction(
        WING_DETECTOR,
        flood_run,
        direct_beam_run,
        dark_current_run,
        SOLID_ANGLE_CORRECTION,
        transmission_flood_run,
        open_beam_transmission,
        BEAM_TRAP_SIZE_FACTOR,
        THETA_DEPENDENT_CORRECTION,
        UNIVERSAL_MASK,
        MASKED_PIXELS,
        MASK_BEAM_CENTER_RADIUS,
        MAIN_DET_MASK_ANGLE,
        WING_DET_MASK_ANGLE,
        MIN_THRESHOLD,
        MAX_THRESHOLD,
        SENSITIVITY_FILE,
        nexus_dir=reference_dir.new.biosans,
    )

    # Verify
    gold_sens_file = os.path.join(
        reference_dir.new.biosans, "CG3_sens_main_exp549_scan9.nxs"
    )
    assert os.path.exists(gold_sens_file)
    verify_results(SENSITIVITY_FILE, gold_sens_file, clean_workspace)


def test_wing_detector(reference_dir, generatecleanfile, clean_workspace):
    """Test case for CG3 wing detector

    Flood for wing detector at 1.4° -
    /HFIR/CG3/IPTS-17241/exp549/Datafiles/BioSANS_exp549_scan0020_0001.xml

    Empty Beam for Transmission Reference for wing detector at 1.4° -
    /HFIR/CG3/IPTS-17241/exp549/Datafiles/BioSANS_exp549_scan0016_0001.xml

    Dark Current for all configurations above -
    /HFIR/CG3/IPTS-17241/exp549/Datafiles/BioSANS_exp549_scan0022_0001.xml
    """
    # output testing directory
    output_dir = generatecleanfile()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    CG3 = "CG3"  # Main

    # IPTS
    IPTS = 17241

    EXPERIMENT = 549

    # CG3: Main
    FLOOD_RUN = (20, 1)  # BIO-SANS detector
    WING_DETECTOR = True  # this is main detector

    # About Masks
    # CG3 Main beam center file/Empty beam.  It is allowed to be left blank
    DIRECT_BEAM_RUN = None
    # Beam center size
    MASK_BEAM_CENTER_RADIUS = 65  # mm

    # Transmission empty beam
    OPEN_BEAM_TRANSMISSION = (16, 1)
    # Transmission flood run
    TRANSMISSION_FLOOD_RUN = FLOOD_RUN

    # Dark current
    DARK_CURRENT_RUN = (22, 1)

    # Default mask to detector
    UNIVERSAL_MASK = None  # 'Mask.XML'
    # CG3:
    MASKED_PIXELS = "1-18,249-256"  # CG3
    # Mask angle: must 2 values as min and max or None
    MAIN_DET_MASK_ANGLE = 2.0  # 0.75# 1.5 #0.75
    WING_DET_MASK_ANGLE = 57.0
    BEAM_TRAP_SIZE_FACTOR = 1.0  # 2

    # Corrections
    SOLID_ANGLE_CORRECTION = True
    # Flag to do dependent correction with transmission correction
    THETA_DEPENDENT_CORRECTION = True

    # THRESHOLD
    MIN_THRESHOLD = 0.5
    MAX_THRESHOLD = 2.0

    # Output
    FILE_SURFIX = "wing" if WING_DETECTOR else "main"
    SENSITIVITY_FILE = os.path.join(
        output_dir, f"{CG3}_sens_{FILE_SURFIX}{FLOOD_RUN}sac_tdc7m.nxs"
    )

    # Convert SPICE file to NeXus file
    flood_run = SpiceRun(CG3, IPTS, EXPERIMENT, FLOOD_RUN[0], FLOOD_RUN[1])
    direct_beam_run = (
        SpiceRun(CG3, IPTS, EXPERIMENT, DIRECT_BEAM_RUN[0], DIRECT_BEAM_RUN[1])
        if DIRECT_BEAM_RUN
        else None
    )
    open_beam_transmission = (
        SpiceRun(
            CG3, IPTS, EXPERIMENT, OPEN_BEAM_TRANSMISSION[0], OPEN_BEAM_TRANSMISSION[1]
        )
        if OPEN_BEAM_TRANSMISSION
        else None
    )
    transmission_flood_run = SpiceRun(
        CG3, IPTS, EXPERIMENT, TRANSMISSION_FLOOD_RUN[0], TRANSMISSION_FLOOD_RUN[1]
    )
    dark_current_run = (
        SpiceRun(CG3, IPTS, EXPERIMENT, DARK_CURRENT_RUN[0], DARK_CURRENT_RUN[1])
        if DARK_CURRENT_RUN
        else None
    )

    prepare_spice_sensitivities_correction(
        WING_DETECTOR,
        flood_run,
        direct_beam_run,
        dark_current_run,
        SOLID_ANGLE_CORRECTION,
        transmission_flood_run,
        open_beam_transmission,
        BEAM_TRAP_SIZE_FACTOR,
        THETA_DEPENDENT_CORRECTION,
        UNIVERSAL_MASK,
        MASKED_PIXELS,
        MASK_BEAM_CENTER_RADIUS,
        MAIN_DET_MASK_ANGLE,
        WING_DET_MASK_ANGLE,
        MIN_THRESHOLD,
        MAX_THRESHOLD,
        SENSITIVITY_FILE,
        nexus_dir=reference_dir.new.biosans,
    )
    # Verify
    gold_sens_file = os.path.join(
        reference_dir.new.biosans, "CG3_sens_wing_exp549_scan20.nxs"
    )
    assert os.path.exists(gold_sens_file)
    verify_results(SENSITIVITY_FILE, gold_sens_file, clean_workspace)


def verify_results(test_sensitivities_file: str, gold_sens_file: str, clean_workspace):
    """Verify sensitivities of tested result from gold file"""
    # Get gold file
    # gold_sens_file = os.path.join(reference_dir.new.gpsans, 'calibrations/sens_CG2_spice_bar.nxs')
    if not os.path.exists(gold_sens_file):
        raise RuntimeError(
            f"Expected (gold) sensitivities cannot be found at {gold_sens_file}"
        )

    # Compare sensitivities
    gold_sens_ws = LoadNexusProcessed(Filename=gold_sens_file)
    test_sens_ws = LoadNexusProcessed(Filename=test_sensitivities_file)
    clean_workspace(gold_sens_ws)
    clean_workspace(test_sens_ws)
    np.testing.assert_allclose(test_sens_ws.extractY(), gold_sens_ws.extractY())
    clean_all_ws()


def clean_all_ws():
    for workspace in workspaces:
        remove_ws(workspace)


def remove_ws(workspace):
    if AnalysisDataService.doesExist(workspace):
        DeleteWorkspace(workspace)


if __name__ == "__main__":
    pytest.main([__file__])
