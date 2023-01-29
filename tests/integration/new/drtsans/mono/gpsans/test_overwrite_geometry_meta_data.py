# Integration test for overwriting instrument geometry related meta data for GP-SANS
import pytest
import time
import os
from drtsans.files.log_h5_reader import verify_cg2_reduction_results

from drtsans.mono.gpsans import (
    load_all_files,
    plot_reduction_output,
    reduce_single_configuration,
    reduction_parameters,
    update_reduction_parameters,
)
from mantid.simpleapi import DeleteWorkspace, mtd


def reduce_gpsans_data(data_dir, reduction_input_common, output_dir, prefix):
    """Standard reduction workflow

    Parameters
    ----------'
    data_dir
    reduction_input_common: dict
        reduction parameters common to all samples
    output_dir
    prefix: str
        prefix for all the workspaces loaded

    Returns
    -------

    """
    # USER Input here with scan numbers etc.
    samples = ["9166", "9167", "9176"]
    samples_trans = ["9178", "9179", "9188"]
    sample_thick = ["0.1"] * 3
    bkgd = ["9165", "9165", "9165"]
    bkgd_trans = ["9177", "9177", "9177"]

    # Sample names for output
    sample_names = ["Al4", "PorasilC3", "PTMA-15"]

    # set output directory
    reduction_input_common["configuration"]["outputDir"] = output_dir
    # create output directory
    for subfolder in ["1D", "2D"]:
        output_folder = os.path.join(output_dir, subfolder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    start_time = time.time()
    for i in range(len(samples)):
        specs = {
            "dataDirectories": data_dir,
            "sample": {
                "runNumber": samples[i],
                "thickness": sample_thick[i],
                "transmission": {"runNumber": samples_trans[i]},
            },
            "background": {
                "runNumber": bkgd[i],
                "transmission": {"runNumber": bkgd_trans[i]},
            },
            "outputFileName": sample_names[i],
        }
        reduction_input = update_reduction_parameters(
            reduction_input_common, specs, validate=True
        )
        loaded = load_all_files(reduction_input, path=data_dir, prefix=prefix)
        out = reduce_single_configuration(loaded, reduction_input)
        plot_reduction_output(out, reduction_input, loglog=False, close_figures=True)

    end_time = time.time()
    print("Execution Time: {}".format(end_time - start_time))


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Debeer-Schmitt, Lisa M. debeerschmlm@ornl.gov, He, Lilin <hel3@ornl.gov>
def test_no_overwrite(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data overwriting neither SampleToSi (distance) nor SampleDetectorDistance.

    This test case is provided by Lisa and verified by Lilin
    Location of original test: /HFIR/CG2/shared/UserAcceptance/overwrite_meta/
    Test json:  /HFIR/CG2/shared/UserAcceptance/overwrite_meta/gpsans_reduction_test1.json
    Verified result: /HFIR/CG2/shared/UserAcceptance/overwrite_meta/test1/

    Returns
    -------

    """
    sensitivity_file = os.path.join(
        reference_dir.new.gpsans, "overwrite_gold_04282020/sens_c486_noBar.nxs"
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_test1")
    specs = {
        "iptsNumber": 21981,
        "beamCenter": {"runNumber": 9177},
        "emptyTransmission": {"runNumber": 9177},
        "configuration": {
            "outputDir": output_dir,
            "useDefaultMask": True,
            "defaultMask": ["{'Pixel':'1-10,247-256'}"],
            "sensitivityFileName": sensitivity_file,
            "absoluteScaleMethod": "direct_beam",
            "DBScalingBeamRadius": 40,
            "mmRadiusForTransmission": 40,
            "numQxQyBins": 180,
            "1DQbinType": "scalar",
            "QbinType": "linear",
            "numQBins": 180,
            "LogQBinsPerDecade": None,
            "useLogQBinsEvenDecade": False,
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "usePixelCalibration": False,
            "useSubpixels": False,
        },
    }
    reduction_input = reduction_parameters(
        specs, "GPSANS", validate=False
    )  # add defaults and defer validation
    reduce_gpsans_data(
        reference_dir.new.gpsans, reduction_input, output_dir, prefix="CG2MetaRaw"
    )

    # Get result files
    sample_names = ["Al4", "PorasilC3", "PTMA-15"]
    gold_path = os.path.join(reference_dir.new.gpsans, "overwrite_gold_04282020/test1/")

    # Verify results
    verify_cg2_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="Raw (No Overwriting)",
        prefix="CG2MetaRaw",
    )

    # cleanup
    # leftover workspaces due to the design of load_all_files
    # _bkgd_trans:	1.439529 MB
    # _empty:	1.439529 MB
    # _processed_center:	1.439529 MB
    # _sample_trans:	1.449257 MB
    # CG2MetaRaw_GPSANS_9165_raw_histo:	1.799657 MB
    # CG2MetaRaw_GPSANS_9166_raw_histo:	1.796017 MB
    # CG2MetaRaw_GPSANS_9167_raw_histo:	1.815313 MB
    # CG2MetaRaw_GPSANS_9176_raw_histo:	16.279889 MB
    # CG2MetaRaw_GPSANS_9177_raw_histo:	1.439529 MB
    # CG2MetaRaw_GPSANS_9178_raw_histo:	1.439337 MB
    # CG2MetaRaw_GPSANS_9179_raw_histo:	1.439241 MB
    # CG2MetaRaw_GPSANS_9188_raw_histo:	1.449257 MB
    # CG2MetaRaw_sensitivity:	16.933112 MB
    # chi:	9.6e-05 MB
    # processed_data_main:	16.279889 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_processed_center")
    DeleteWorkspace("_sample_trans")
    DeleteWorkspace("chi")
    DeleteWorkspace("processed_data_main")
    for me in mtd.getObjectNames():
        if str(me).startswith("CG2MetaRaw"):
            DeleteWorkspace(me)


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Debeer-Schmitt, Lisa M. debeerschmlm@ornl.gov, He, Lilin <hel3@ornl.gov>
def test_overwrite_sample2si(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data overwriting SampleToSi (distance) but not SampleDetectorDistance.
    Sample to detector distance will be changed accordingly.

    - Overwrite SampleToSi (distance) to 94 mm.

    This test case is provided by Lisa and verified by Lilin
    Location of original test: /HFIR/CG2/shared/UserAcceptance/overwrite_meta/
    Test json:  /HFIR/CG2/shared/UserAcceptance/overwrite_meta/gpsans_reduction_test2.json
    Verified result: /HFIR/CG2/shared/UserAcceptance/overwrite_meta/test2/

    Returns
    -------

    """
    sensitivity_file = os.path.join(
        reference_dir.new.gpsans, "overwrite_gold_04282020/sens_c486_noBar.nxs"
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_test2")
    specs = {
        "iptsNumber": 21981,
        "beamCenter": {"runNumber": 9177},
        "emptyTransmission": {"runNumber": 9177},
        "configuration": {
            "outputDir": output_dir,
            "sampleToSi": 94,
            "useDefaultMask": True,
            "defaultMask": ["{'Pixel':'1-10,247-256'}"],
            "sensitivityFileName": sensitivity_file,
            "absoluteScaleMethod": "direct_beam",
            "DBScalingBeamRadius": 40,
            "mmRadiusForTransmission": 40,
            "numQxQyBins": 150,
            "1DQbinType": "scalar",
            "QbinType": "linear",
            "numQBins": 150,
            "LogQBinsPerDecade": None,
            "useLogQBinsEvenDecade": False,
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "usePixelCalibration": False,
            "useSubpixels": False,
        },
    }
    reduction_input = reduction_parameters(
        specs, "GPSANS", validate=False
    )  # add defaults and defer validation
    reduce_gpsans_data(
        reference_dir.new.gpsans, reduction_input, output_dir, "CG2MetaSWD"
    )

    # Get result files
    sample_names = ["Al4", "PorasilC3", "PTMA-15"]

    # Verify results
    gold_path = os.path.join(reference_dir.new.gpsans, "overwrite_gold_04282020/test2/")
    verify_cg2_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="Overwrite SampleToSi to 94mm",
        prefix="CG2MetaSWD",
    )

    # cleanup
    # leftover workspaces due to the design of load_all_files
    # _bkgd_trans:	1.439529 MB
    # _empty:	1.439529 MB
    # _processed_center:	1.439529 MB
    # _sample_trans:	1.449257 MB
    # CG2MetaSWD_GPSANS_9165_raw_histo:	1.799657 MB
    # CG2MetaSWD_GPSANS_9166_raw_histo:	1.796017 MB
    # CG2MetaSWD_GPSANS_9167_raw_histo:	1.815313 MB
    # CG2MetaSWD_GPSANS_9176_raw_histo:	16.279889 MB
    # CG2MetaSWD_GPSANS_9177_raw_histo:	1.439529 MB
    # CG2MetaSWD_GPSANS_9178_raw_histo:	1.439337 MB
    # CG2MetaSWD_GPSANS_9179_raw_histo:	1.439241 MB
    # CG2MetaSWD_GPSANS_9188_raw_histo:	1.449257 MB
    # CG2MetaSWD_sensitivity:	16.933112 MB
    # chi:	9.6e-05 MB
    # processed_data_main:	16.279889 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_processed_center")
    DeleteWorkspace("_sample_trans")
    DeleteWorkspace("chi")
    DeleteWorkspace("processed_data_main")
    for me in mtd.getObjectNames():
        if str(me).startswith("CG2MetaSWD"):
            DeleteWorkspace(me)


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Debeer-Schmitt, Lisa M. debeerschmlm@ornl.gov, He, Lilin <hel3@ornl.gov>
def test_overwrite_sdd(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data overwriting SampleDetectorDistance but not SampleDetectorDistance

    - Overwrite DetectorToSample (distance) to 40 meter

    This test case is provided by Lisa and verified by Lilin
    Location of original test: /HFIR/CG2/shared/UserAcceptance/overwrite_meta/
    Test json:  /HFIR/CG2/shared/UserAcceptance/overwrite_meta/gpsans_reduction_test3.json
    Verified result: /HFIR/CG2/shared/UserAcceptance/overwrite_meta/test3/

    Returns
    -------

    """
    # Set test and run: sample to detector distance is changed to 40 meter
    sensitivity_file = os.path.join(
        reference_dir.new.gpsans, "overwrite_gold_04282020/sens_c486_noBar.nxs"
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_test3")

    # Set up reduction
    specs = {
        "iptsNumber": 21981,
        "beamCenter": {"runNumber": 9177},
        "emptyTransmission": {"runNumber": 9177},
        "configuration": {
            "outputDir": output_dir,
            "sampleDetectorDistance": 40,
            "useDefaultMask": True,
            "defaultMask": ["{'Pixel':'1-10,247-256'}"],
            "sensitivityFileName": sensitivity_file,
            "absoluteScaleMethod": "direct_beam",
            "DBScalingBeamRadius": 40,
            "mmRadiusForTransmission": 40,
            "numQxQyBins": 150,
            "1DQbinType": "scalar",
            "QbinType": "linear",
            "numQBins": 150,
            "LogQBinsPerDecade": None,
            "useLogQBinsEvenDecade": False,
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "usePixelCalibration": False,
            "useSubpixels": False,
        },
    }
    reduction_input = reduction_parameters(
        specs, "GPSANS", validate=False
    )  # add defaults and defer validation
    reduce_gpsans_data(
        reference_dir.new.gpsans, reduction_input, output_dir, "CG2MetaSDD"
    )

    # Get result files
    sample_names = ["Al4", "PorasilC3", "PTMA-15"]
    output_log_files = [
        os.path.join(output_dir, "{}_reduction_log.hdf".format(sn))
        for sn in sample_names
    ]
    for output_file_path in output_log_files:
        assert os.path.exists(output_file_path), "Output {} cannot be found".format(
            output_file_path
        )

    # Verify results
    gold_path = os.path.join(reference_dir.new.gpsans, "overwrite_gold_20201027/test3/")
    verify_cg2_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="Overwrite DetectorSampleDistance to 40 meter",
        prefix="CG2MetaSDD",
    )

    # cleanup
    # leftover workspaces due to the design of load_all_files
    # _bkgd_trans:	1.439529 MB
    # _empty:	1.439529 MB
    # _processed_center:	1.439529 MB
    # _sample_trans:	1.449257 MB
    # CG2MetaSDD_GPSANS_9165_raw_histo:	1.799657 MB
    # CG2MetaSDD_GPSANS_9166_raw_histo:	1.796017 MB
    # CG2MetaSDD_GPSANS_9167_raw_histo:	1.815313 MB
    # CG2MetaSDD_GPSANS_9176_raw_histo:	16.279889 MB
    # CG2MetaSDD_GPSANS_9177_raw_histo:	1.439529 MB
    # CG2MetaSDD_GPSANS_9178_raw_histo:	1.439337 MB
    # CG2MetaSDD_GPSANS_9179_raw_histo:	1.439241 MB
    # CG2MetaSDD_GPSANS_9188_raw_histo:	1.449257 MB
    # CG2MetaSDD_sensitivity:	16.933112 MB
    # chi:	9.6e-05 MB
    # processed_data_main:	16.279889 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_processed_center")
    DeleteWorkspace("_sample_trans")
    DeleteWorkspace("chi")
    DeleteWorkspace("processed_data_main")
    for me in mtd.getObjectNames():
        if str(me).startswith("CG2MetaSDD"):
            DeleteWorkspace(me)


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Debeer-Schmitt, Lisa M. debeerschmlm@ornl.gov, He, Lilin <hel3@ornl.gov>
def test_overwrite_both(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data overwriting both SampleToSi (distance) and SampleDetectorDistance

    - Overwrite SampleToSi (distance) to 200 mm.
    - Overwrite DetectorToSample (distance) to 30 meter

    This test case is provided by Lisa and verified by Lilin
    Location of original test: /HFIR/CG2/shared/UserAcceptance/overwrite_meta/
    Test json:  /HFIR/CG2/shared/UserAcceptance/overwrite_meta/gpsans_reduction_test4.json
    Verified result: /HFIR/CG2/shared/UserAcceptance/overwrite_meta/test4/

    Returns
    -------

    """
    # Set test and run: sample to silicon window to 94 mm and sample to detector distance to 15 meter
    sensitivity_file = os.path.join(
        reference_dir.new.gpsans, "overwrite_gold_04282020/sens_c486_noBar.nxs"
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_test4")
    specs = {
        "iptsNumber": 21981,
        "beamCenter": {"runNumber": 9177},
        "emptyTransmission": {"runNumber": 9177},
        "configuration": {
            "outputDir": output_dir,
            "sampleToSi": 200,
            "sampleDetectorDistance": 30,
            "useDefaultMask": True,
            "defaultMask": ["{'Pixel':'1-10,247-256'}"],
            "sensitivityFileName": sensitivity_file,
            "absoluteScaleMethod": "direct_beam",
            "DBScalingBeamRadius": 40,
            "mmRadiusForTransmission": 40,
            "numQxQyBins": 150,
            "1DQbinType": "scalar",
            "QbinType": "linear",
            "numQBins": 150,
            "LogQBinsPerDecade": None,
            "useLogQBinsEvenDecade": False,
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "usePixelCalibration": False,
            "useSubpixels": False,
        },
    }
    reduction_input = reduction_parameters(
        specs, "GPSANS", validate=False
    )  # add defaults and defer validation
    reduce_gpsans_data(
        reference_dir.new.gpsans, reduction_input, output_dir, "CG2MetaBoth"
    )

    # Get result files
    sample_names = ["Al4", "PorasilC3", "PTMA-15"]
    output_log_files = [
        os.path.join(output_dir, "{}_reduction_log.hdf".format(sn))
        for sn in sample_names
    ]
    for output_file_path in output_log_files:
        assert os.path.exists(output_file_path), "Output {} cannot be found".format(
            output_file_path
        )

    # Verify results
    gold_path = os.path.join(reference_dir.new.gpsans, "overwrite_gold_20201027/test4/")
    verify_cg2_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="Overwrite DetectorSampleDistance to 30 meter, SampleToSi to 200 mm",
        prefix="CG2MetaBoth",
    )

    # cleanup
    # leftover workspaces due to the design of load_all_files
    # _bkgd_trans:	1.439529 MB
    # _empty:	1.439529 MB
    # _processed_center:	1.439529 MB
    # _sample_trans:	1.449257 MB
    # CG2MetaBoth_GPSANS_9165_raw_histo:	1.799657 MB
    # CG2MetaBoth_GPSANS_9166_raw_histo:	1.796017 MB
    # CG2MetaBoth_GPSANS_9167_raw_histo:	1.815313 MB
    # CG2MetaBoth_GPSANS_9176_raw_histo:	16.279889 MB
    # CG2MetaBoth_GPSANS_9177_raw_histo:	1.439529 MB
    # CG2MetaBoth_GPSANS_9178_raw_histo:	1.439337 MB
    # CG2MetaBoth_GPSANS_9179_raw_histo:	1.439241 MB
    # CG2MetaBoth_GPSANS_9188_raw_histo:	1.449257 MB
    # CG2MetaBoth_sensitivity:	16.933112 MB
    # chi:	9.6e-05 MB
    # processed_data_main:	16.279889 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_processed_center")
    DeleteWorkspace("_sample_trans")
    DeleteWorkspace("chi")
    DeleteWorkspace("processed_data_main")
    for me in mtd.getObjectNames():
        if str(me).startswith("CG2MetaBoth"):
            DeleteWorkspace(me)


if __name__ == "__main__":
    pytest.main([__file__])
