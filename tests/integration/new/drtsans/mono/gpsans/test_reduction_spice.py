import pytest
import os
from drtsans.files.log_h5_reader import verify_cg2_reduction_results
from drtsans.mono.gpsans.reduce_spice import reduce_gpsans_nexus
import warnings
from mantid.simpleapi import mtd, DeleteWorkspace

warnings.filterwarnings("ignore")

""" Test data information

Data files
----------
                    scan/pt     SDD         detector trans
sample              35          1152.0      0.999998
background          34          1152.0      0.999998
sample trans        27          10852.0     0.999998
background trans    26          10852.0     0.999998
empty trans         28          10852.0     0.999998
beam center         20          1152.0      0.999998
block beam           9          1000.0      399.999962

other files
-----------
- sensitivities: reference_dir/new/gpsans/calibrations/sens_CG2_spice_bar.nxs
    from GPSANS SPICE data from sensitivities test

- mask: reference_dir/new/gpsans/calibrations/mask_pixel_map.nxs
        JUST FOR DEMO PURPOSE
"""


def test_reduction_spice(reference_dir, generatecleanfile):
    """
    Test reduction from SPICE-converted Nexus file

    """
    nexus_dir = os.path.join(reference_dir.new.gpsans, "Exp280")

    # Set output (temp) directory
    output_directory = generatecleanfile(prefix="cg2_spice_reduction")

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

    # q range to use to clean 1D curve of each configuration
    q_range = [None, None]

    # STAFF INPUT
    use_mask_file = True
    mask_file_name = os.path.join(
        reference_dir.new.gpsans, "calibrations/mask_pixel_map.nxs"
    )
    use_dark_file = False
    dark_file_name = ""
    block_beam = (9, 1)
    use_mask_back_tubes = False
    wavelength = None
    wavelength_spread = None
    wedge_min_angles = None
    wedge_max_angles = None

    sensitivity_file = os.path.join(
        reference_dir.new.gpsans, "calibrations/sens_CG2_spice_bar.nxs"
    )
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

    # verify
    expected_data_dir = os.path.join(
        reference_dir.new.gpsans, "spice_reduction/exp280_normal_bin"
    )
    verify_cg2_reduction_results(
        sample_names, output_dir, expected_data_dir, "SPICE reduction", prefix=""
    )

    # clean up
    # mysterious leftover workspaces
    # _bkgd_trans:	1.182593 MB
    # _empty:	1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000090001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000200001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000260001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000270001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000280001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000340001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000350001.nxs.h5_raw_histo:
    #   1.182969 MB
    # _mask:	1.203736 MB
    # _processed_center:	1.182593 MB
    # _sample_trans:	1.182593 MB
    # _sensitivity:	1.181408 MB
    # chi:	9.6e-05 MB
    # processed_data_main:	1.182969 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_mask")
    DeleteWorkspace("_processed_center")
    DeleteWorkspace("_sample_trans")
    DeleteWorkspace("_sensitivity")
    DeleteWorkspace("chi")
    DeleteWorkspace("processed_data_main")
    for ws in mtd.getObjectNames():
        if str(ws).startswith("_GPSANS_"):
            DeleteWorkspace(ws)


def test_reduction_spice_subpixel(reference_dir, generatecleanfile):
    """
    Test reduction from SPICE-converted Nexus file
    """
    nexus_dir = os.path.join(reference_dir.new.gpsans, "Exp280")

    # Set output (temp) directory
    output_directory = generatecleanfile(prefix="cg2_spice_reduction_subpixel")

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

    # q range to use to clean 1D curve of each configuration
    q_range = [None, None]

    # STAFF INPUT
    use_mask_file = True
    mask_file_name = os.path.join(
        reference_dir.new.gpsans, "calibrations/mask_pixel_map.nxs"
    )
    use_dark_file = False
    dark_file_name = ""
    block_beam = (9, 1)
    use_mask_back_tubes = False
    wavelength = None
    wavelength_spread = None
    wedge_min_angles = None
    wedge_max_angles = None

    # sensitivity_file = '/HFIR/CG2/shared/drt_sensitivity/sens_CG2_spice_bar.nxs'
    sensitivity_file = os.path.join(
        reference_dir.new.gpsans, "calibrations/sens_CG2_spice_bar.nxs"
    )
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
            "useSubpixels": True,
            "subpixelsX": 5,
            "subpixelsY": 5,
            "useTimeSlice": False,
            "useLogSlice": False,
            "logSliceName": "",
            "logSliceInterval": "",
        },
    }

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

    # verify
    expected_data_dir = os.path.join(
        reference_dir.new.gpsans, "spice_reduction/exp280_subpixel_bin/"
    )
    verify_cg2_reduction_results(
        sample_names, output_dir, expected_data_dir, "SPICE reduction", prefix=""
    )

    # clean up
    # mysterious leftover workspaces
    # _bkgd_trans:	1.182593 MB
    # _empty:	1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000090001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000200001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000260001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000270001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000280001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000340001.nxs.h5_raw_histo:
    #   1.182593 MB
    # _GPSANS_/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/Exp280/CG2_028000350001.nxs.h5_raw_histo:
    #   1.182969 MB
    # _mask:	1.203736 MB
    # _processed_center:	1.182593 MB
    # _sample_trans:	1.182593 MB
    # _sensitivity:	1.181408 MB
    # chi:	9.6e-05 MB
    # processed_data_main:	1.182969 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_mask")
    DeleteWorkspace("_processed_center")
    DeleteWorkspace("_sample_trans")
    DeleteWorkspace("_sensitivity")
    DeleteWorkspace("chi")
    DeleteWorkspace("processed_data_main")
    for ws in mtd.getObjectNames():
        if str(ws).startswith("_GPSANS_"):
            DeleteWorkspace(ws)


if __name__ == "__main__":
    pytest.main([__file__])
