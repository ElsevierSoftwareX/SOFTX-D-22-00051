# Modified from Volker's RC488_IPTS24666_VolkerTemplate.py

"""
Example:

Sample - /HFIR/CG3/IPTS-17240/exp318/Datafiles/BioSANS_exp318_scan0217_0001.xml (Scattering/Transmission)
Empty Beam - /HFIR/CG3/IPTS-17240/exp318/Datafiles/BioSANS_exp318_scan0220_0001.xml (For Transmission)
Beam Center - /HFIR/CG3/IPTS-17240/exp318/Datafiles/BioSANS_exp318_scan0220_0001.xml (Transmission Measurement)
Dark - /HFIR/CG3/IPTS-17240/exp318/Datafiles/BioSANS_exp318_scan0044_0001.xml (for both main and wing detectors)
"""

# IPTS and experiment
IPTS_Number = 17240
EXPERIMENT_NUMBER = 318

# Non-TimeSlice Single Configuration-
sample_identifier = ""  # DO NOT CHANGE IF Non-TimeSlice Experiments
sample_names = ["Spice_318_217"]  # DO NOT LEAVE BLANK
sample_thick = ["0.1"]  # Do not repeat if sample for ALL samples
samples = [(217, 1)]  # Enter the list of runs for 'samples'
samples_trans = samples  # Enter its own list if different from 'samples' list
backgrounds = [None]  # Do not repeat multiple times if SAME for ALL samples
backgrounds_trans = (
    backgrounds  # Enter its own list if different from 'backgrounds' list
)

# Change if reducing a subset of 'samples' list
start_index = 1  # Default start index is 1; DO NOT START FROM 'ZERO'
end_index = len(samples)  # Default is 'len(samples)'

# Setup once at the beginning of the experiment
User3LetInitial = (
    "whoever"  # 3-Letter initials that identifies you in the output directory
)
overWrite = True  # Option to overwrite existing data or create another folder (Default is 'False')

# ## Instrument Scientist or Local contact input below (And Expert Users)

# Advanced Settings for Data Reduction--
# Buffer clearing frequency
clearBuffer = False
refreshCycle = 25  # loops... depending on the activities.

# Common setting for all options
scalefac = "4.05e-9"
beam_center = (220, 1)
empty_trans = (220, 1)
dark_mfname = (44, 1)
dark_wfname = (44, 1)
sens_mfname = "/HFIR/CG3/shared/Cycle488/Sens_f6368m4p0_bsSVP.nxs"
sens_wfname = "/HFIR/CG3/shared/Cycle488/Sens_f6380w1p4_bsSVP.nxs"

# Plotting range--
q_range_main = [0.003, 0.045]  # Q-range for isotropic data
q_range_wing = [0.03, 0.9]
OL_range = [0.0325, 0.0425]

# Miscellaneous settings--
base_output_directory = f"/HFIR/CG3/IPTS-{IPTS_Number}/shared/{User3LetInitial}/"
scaling_beam_radius = None
flexible_pixelsizes = True  # 'True'- if use barscan/flood information for flexible pixel sizes, else 'False'
# Make sure the barscan used sensitivity file used above if 'True'

# Plotting Options--
Plot_type = (
    "scalar"  # 'scalar' for isotropic and 'wedge' for anisotropic (manual or auto)
)
Plot_binning = "log"  # 'log' or 'linear' Q-binning
# LINEAR BINNING
Lin1DQbins_Main = (
    ""  # No. of bins for linear binning of 1D Main Detector, Default is 100;
)
# If per decade is used default is ''
Lin1DQbins_Wing = (
    ""  # No. of bins for linear binning of 1D Wing Detector, Default is 100;
)
# If per decade is used default is ''
Lin2DQxy_Main = (
    100  # No. of bins for linear binning of 2D Main Detector, Default is 100
)
Lin2DQxy_Wing = (
    100  # No. of bins for linear binning of 2D Main Detector, Default is 100
)

# LOGARITHMIC BINNING
LogQbinsPerDecade_Main = 25  # No. of bins per decade of 1D Main Detector, Default is 33
LogQbinsPerDecade_Wing = 25  # No. of bins per decade of 1D Main Detector, Default is 33

# If time Slicing--
timeSliceExpt = False  # 'True' if time slice experiment, else 'False'
timeSliceDuration = 60  # Units - seconds and irrelevant if above is 'False'

# ANISOTROPIC DATA REDUCTION--
# Wedge_0...
q_range_main_wedge0 = [0.003, 0.0425]  # Q-range for anisotropic data -- wedge0
q_range_wing_wedge0 = [0.02, 0.45]
OL_range_wedge0 = [0.025, 0.04]

# Wedge_1...
q_range_main_wedge1 = [0.003, 0.0425]  # Q-range for anisotropic data -- wedge1
q_range_wing_wedge1 = [0.03, 0.45]
OL_range_wedge1 = [0.03, 0.04]

# If Manual Wedges--
wedge_min_angles = None  # If AUTOWEDGE reduction type 'None'; Irrelevant for ISOTROPIC [Wedge0_min,Wedge1_min]
wedge_max_angles = None  # If AUTOWEDGE reduction type 'None'; Irrelevant for ISOTROPIC [Wedge0_max,Wedge1_max]

# If automatic determination of wedge angles-- (To determine Wedges-TDW)
Qmin_TDW = 0.003  # Minimum Q of the Main Detector
Qmax_TDW = 0.04  # Maximum Q of the Main Detector
Qdelta_TDW = 0.01  # Q step-size (or annual ring widths); Default is 0.01. Too fine will fail autodetection.
PeakWidth_TDW = 0.5  # Wedge0 opening angle based of peak width (Default is 0.5--- 50%)
AziDelta_TDW = 1.0  # Azimuthal Angle, Phi step-size
BkgWidth_TDW = 1.0  # Wedge1 opening angle based of peak width (Default is 1.5--- 150%)
MinSigtoNoise_TDW = 2.0  # The intensity ratio between peak and background to detect a peak (Default is 2.0)

# ----------  NO TOUCH UNDER THIS LINE ----------------------

# imports
import os  # noqa: E401
from drtsans.mono.biosans.reduce_spice import (
    clear_buffer,
    reduce_biosans_nexus,
)  # noqa: E401

if clearBuffer:
    clear_buffer()
if not os.path.exists(base_output_directory):
    os.makedirs(base_output_directory)
# Reduce data
test_log_files = reduce_biosans_nexus(
    IPTS_Number,
    EXPERIMENT_NUMBER,
    sample_names,
    samples,
    backgrounds,
    samples_trans,
    backgrounds_trans,
    empty_trans,
    beam_center,
    sample_thick,
    sample_identifier,
    overWrite,
    dark_mfname,
    dark_wfname,
    base_output_directory,
    sens_mfname,
    sens_wfname,
    scalefac,
    scaling_beam_radius,
    Lin1DQbins_Main,
    Lin1DQbins_Wing,
    Lin2DQxy_Main,
    Lin2DQxy_Wing,
    Plot_type,
    Plot_binning,
    LogQbinsPerDecade_Main,
    LogQbinsPerDecade_Wing,
    q_range_main,
    q_range_wing,
    OL_range,
    flexible_pixelsizes,
    wedge_min_angles,
    wedge_max_angles,
    Qmin_TDW,
    Qmax_TDW,
    Qdelta_TDW,
    PeakWidth_TDW,
    AziDelta_TDW,
    BkgWidth_TDW,
    MinSigtoNoise_TDW,
    q_range_main_wedge0,
    q_range_wing_wedge0,
    OL_range_wedge0,
    q_range_main_wedge1,
    q_range_wing_wedge1,
    OL_range_wedge1,
    refreshCycle,
    nexus_dir=None,
)
