"""
    SANS sensitivities preparation script

    # goal
    1. implement a universal mask_beam_center(flood_ws, beam_center_mask=None, beam_center_ws=None)
       for 3 types of mask
    2. add option for wing/main detector for BIOSANS:w


"""
import os
import warnings
from drtsans.mono.gpsans.prepare_sensitivities_correction import SpiceRun
from drtsans.mono.gpsans.prepare_sensitivities_correction import (
    prepare_spice_sensitivities_correction,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


IPTS = 828
EXPERIMENT = 280
# Input Flood Runs
FLOOD_RUNS = (38, 1), (40, 1), (42, 1)  # list of 2 tuple as (scan number, pt number)

# Direct beam / transmission
DIRECT_BEAM_RUNS = (
    (37, 1),
    (39, 1),
    (41, 1),
)  # list of 2 tuple as (scan number, pt number)
# Beam center size
MASK_BEAM_CENTER_RADIUS = 140  # mm

# Default mask to detector
MASKED_PIXELS = "1-8,249-256"

# Pixel calibration
# Pixel calibration: False/True (default database)/user specified calibration database
# PIXEL_CALIBRATION = None
PIXEL_CALIBRATION = (
    "/HFIR/CG2/IPTS-828/shared/pixel_calibration/runs_1_111/pixel_calibration.json"
)

# Corrections
SOLID_ANGLE_CORRECTION = True  # shall be on!

# If it is GPSANS or BIOSANS there could be 2 options to calculate detector efficiencies
MOVING_DETECTORS = True

# THRESHOLD
MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 1.5

# Output directory
OUTPUT_DIR = (
    "/tmp/"  # use None for default: f'/HFIR/{MY_BEAM_LINE}/shared/drt_sensitivity/'
)

# ----------------- Some options not used so far ---------------------
UNIVERSAL_MASK = None  # 'Mask.XML'

# ----------------- NO ON TOUCH ANYTHING BELOW -----------------------
# Set the directory for already converted SPICE files
NEXUS_DIR = os.path.join(f"/HFIR/CG2/IPTS-{IPTS}/shared", f"Exp{EXPERIMENT}")
# Check
if not os.path.exists(NEXUS_DIR):
    print(f"[ERROR] Converted NeXus-SPICE directory {NEXUS_DIR} does not exist")

# Convert flood runs
CG2 = "CG2"
flood_spice_runs = [
    SpiceRun(CG2, IPTS, EXPERIMENT, scan_i, pt_i) for scan_i, pt_i in FLOOD_RUNS
]
if DIRECT_BEAM_RUNS is None:
    transmission_spice_runs = None
else:
    transmission_spice_runs = [
        SpiceRun(CG2, IPTS, EXPERIMENT, scan_i, pt_i)
        for scan_i, pt_i in DIRECT_BEAM_RUNS
    ]

# Correction
prepare_spice_sensitivities_correction(
    flood_spice_runs,
    transmission_spice_runs,
    MOVING_DETECTORS,
    MIN_THRESHOLD,
    MAX_THRESHOLD,
    NEXUS_DIR,
    UNIVERSAL_MASK,
    MASKED_PIXELS,
    MASK_BEAM_CENTER_RADIUS,
    solid_angle_correction=SOLID_ANGLE_CORRECTION,
    pixel_calibration_file=PIXEL_CALIBRATION,
    output_dir=OUTPUT_DIR,
    file_suffix="spice",
)
