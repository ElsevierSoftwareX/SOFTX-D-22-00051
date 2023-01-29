import os
import warnings
from drtsans.mono.spice_data import SpiceRun
from drtsans.mono.biosans.prepare_sensitivities_correction import (
    prepare_spice_sensitivities_correction,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
CG3 = "CG3"  # Main

# ------------------   USER INPUT ------------------------------------
# IPTS number
IPTS = 17241

# Experiment number
EXPERIMENT = 549

# Main detector or wing detector
WING_DETECTOR = False

# Set flood run: scan, pt
FLOOD_RUN = (9, 1)

# Masks
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
# Flag to do dependent correction with transmission correction
THETA_DEPENDENT_CORRECTION = True

# Default mask (XML file) to detector for all runs
UNIVERSAL_MASK = None  # 'Mask.XML'
# Pixels to mask
MASKED_PIXELS = "1-18,249-256"
# Mask angle: must 2 values as min and max or None
MAIN_DET_MASK_ANGLE = 2.0  # 0.75# 1.5 #0.75
WING_DET_MASK_ANGLE = 57.0
BEAM_TRAP_SIZE_FACTOR = 1.0  # 2

# Corrections
SOLID_ANGLE_CORRECTION = True

# THRESHOLD
MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 2.0

# Output: surfix as wing or main
FILE_SURFIX = "wing" if WING_DETECTOR else "main"
output_dir = os.getcwd()  # current directory
SENSITIVITY_FILE = os.path.join(
    output_dir, f"{CG3}_sens_{FILE_SURFIX}{FLOOD_RUN[0]}.nxs"
)

# where to find converted NeXus file
nexus_dir = None  # as default synchronized with default output directory of SPICE-NeXus converter

# --------------  END OF USER INPUTS --------------

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

# prepare data
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
    nexus_dir=nexus_dir,
)
