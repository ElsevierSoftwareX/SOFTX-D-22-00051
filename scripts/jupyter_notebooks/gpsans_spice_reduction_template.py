# USER INPUT
ipts_number = 828
exp_number = 280

# single sample
samples = [(35, 1)]
samples_trans = [(27, 1)]
sample_thick = [0.1]
sample_names = ["Porasil_B"]
bkgd = [(34, 1)]
bkgd_trans = [(26, 1)]

empty_trans = [(28, 1)]
beam_center = [(20, 1)]

# output directory
output_directory = f"/HFIR/CG2/IPTS-{ipts_number}/shared/reduction_exp{exp_number}"

# q range to use to clean 1D curve of each configuration
q_range = [None, None]

# STAFF INPUT
use_mask_file = True
mask_file_name = "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/calibrations/mask_pixel_map.nxs"
use_dark_file = False
dark_file_name = ""
block_beam = (9, 1)
use_mask_back_tubes = False
wavelength = None
wavelength_spread = None
wedge_min_angles = None
wedge_max_angles = None

sensitivity_file = "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/calibrations/sens_CG2_spice_bar.nxs"
use_log_2d_binning = False
use_log_1d = True
common_configuration = {
    "iptsNumber": ipts_number,
    "emptyTransmission": {"runNumber": empty_trans},
    "beamCenter": {"runNumber": beam_center},
    "configuration": {
        "outputDir": output_directory,
        "darkFileName": dark_file_name,
        "sensitivityFileName": sensitivity_file,
        "DBScalingBeamRadius": 40,
        "sampleApertureSize": 8,
        "mmRadiusForTransmission": 40,
        "absoluteScaleMethod": "direct_beam",
        "numQxQyBins": 256,
        "1DQbinType": "scalar",
        "QbinType": "log",
        "numQBins": "",
        "LogQBinsPerDecade": 33,
        "useLogQBinsDecadeCenter": True,
        "useLogQBinsEvenDecade": False,
        "wavelength": wavelength,
        "wavelengthSpread": wavelength_spread,
        "blockedBeamRunNumber": "Whatever",
        "maskFileName": mask_file_name,
        "WedgeMinAngles": wedge_min_angles,
        "WedgeMaxAngles": wedge_max_angles,
        "AnnularAngleBin": 2.0,
        "Qmin": 0.0028,
        "Qmax": 0.0035,
        "useSubpixels": False,
        "subpixelsX": 5,
        "subpixelsY": 5,
        "useTimeSlice": False,
        "useLogSlice": False,
        "logSliceName": "",
        "logSliceInterval": "",
    },
}

# ---------- DO NOT TOUCH BELOW THIS LINE ----------

import os  # noqa: E401
from drtsans.mono.gpsans.reduce_spice import reduce_gpsans_nexus  # noqa: E401

# set path to nexus files converted from SPICE files
nexus_dir = os.path.join(f"/HFIR/CG2/IPTS-{ipts_number}/shared/", f"Exp{exp_number}")
# create output directory if it does not exist
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Reduce
output_dir = reduce_gpsans_nexus(
    ipts_number,
    exp_number,
    samples,
    sample_thick,
    sample_names,
    bkgd,
    samples_trans,
    bkgd_trans,
    block_beam,
    empty_trans,
    beam_center,
    nexus_dir,
    mask_file_name=mask_file_name if use_mask_file else "",
    dark_file_name=dark_file_name if use_dark_file else "",
    use_log_1d=use_log_1d,
    use_log_2d_binning=use_log_2d_binning,
    common_configuration=common_configuration,
    q_range=q_range,
    use_mask_back_tubes=use_mask_back_tubes,
    debug_output=False,
)
