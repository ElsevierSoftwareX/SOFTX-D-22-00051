# Integration test for overwriting instrument geometry related meta data for BIO-SANS
# (imports alphabetically ordered)
import h5py
import json
import numpy as np
import os
import pytest
import time

# drtsans imports
from drtsans.mono.biosans import (
    load_all_files,
    plot_reduction_output,
    reduce_single_configuration,
    reduction_parameters,
    validate_reduction_parameters,
)


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Shuo Qian <qians@ornl.gov>
def crash_worker_test_no_overwrite(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data without overwriting either sampleToSi or sampleDetectorDistance

    This integration test is from a test from and verified by Shuo Qian.
    Location of testing scirpts and results verified: /HFIR/CG3/shared/UserAcceptance/override_round3/
    Test script: /HFIR/CG3/shared/UserAcceptance/override_round3/test_reduce_cg3_4822_test1.py
    Verified result: /HFIR/CG3/shared/UserAcceptance/override_round3/test1/

    Returns
    -------

    """
    # Set up test: specify sensitivity file
    json_str = generate_testing_json(
        os.path.join(reference_dir.new.biosans, "overwrite_gold_04282020"), None, None
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_bio_test1")

    # Run
    reduce_biosans_data(
        reference_dir.new.biosans, json_str, output_dir, prefix="BioMetaRaw"
    )

    # Get result files
    sample_names = ["csmb_ecoli1h_n2", "insect1hTime_n2"]
    gold_path = os.path.join(
        reference_dir.new.biosans, "overwrite_gold_20200815/test1/"
    )

    # Verify
    verify_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="Raw (no overwriting)",
        prefix="test1",
    )


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Shuo Qian <qians@ornl.gov>
def crash_worker_test_overwrite_both_minor(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data overwriting both sampleToSi and sampleDetectorDistance
        with minor change.
        - Overwrite sampleToSi (distance) to 61 mm.
        - Overwrite DetectorToSample (distance) to 6.9 meter

        This integration test is from a test from and verified by Shuo Qian.
        Location of testing scirpts and results verified: /HFIR/CG3/shared/UserAcceptance/override_round3/
        Test script: /HFIR/CG3/shared/UserAcceptance/override_round3/test_reduce_cg3_4822_test1a.py
        Verified result: /HFIR/CG3/shared/UserAcceptance/override_round3/test1a/
    `
        Returns
        -------

    """
    # Set up test
    json_file = generate_testing_json(
        os.path.join(reference_dir.new.biosans, "overwrite_gold_04282020"), 61, 6.9
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_bio_test1a")

    # Run
    reduce_biosans_data(
        reference_dir.new.biosans, json_file, output_dir, prefix="BioMetaMinor"
    )

    # Get result files
    sample_names = ["csmb_ecoli1h_n2", "insect1hTime_n2"]
    gold_path = os.path.join(
        reference_dir.new.biosans, "overwrite_gold_20200815/test1a/"
    )
    # Verify
    verify_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="SampleToSi -> 61 mm, SampleDetectorDistance -> 6.9 meter",
        prefix="test1a",
    )


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Shuo Qian <qians@ornl.gov>
def Failed_test_overwrite_both_major(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data overwriting both sampleToSi and sampleDetectorDistance
    with significant changes.
    - Overwrite sampleToSi (distance) to 200 mm.
    - Overwrite DetectorToSample (distance) to 14 meter

    This integration test is from a test from and verified by Shuo Qian.
    Location of testing scirpts and results verified: /HFIR/CG3/shared/UserAcceptance/override_round3/
    Test script: /HFIR/CG3/shared/UserAcceptance/override_round3/test_reduce_cg3_4822_test4.py
    Verified result: /HFIR/CG3/shared/UserAcceptance/override_round3/test4/

    Returns
    -------

    """
    # Set up test
    json_file = generate_testing_json(
        os.path.join(reference_dir.new.biosans, "overwrite_gold_04282020"), 200, 14
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_bio_test4")

    # Run
    reduce_biosans_data(
        reference_dir.new.biosans, json_file, output_dir, prefix="CG3MetaMajor2"
    )

    # Get result files
    sample_names = ["csmb_ecoli1h_n2", "insect1hTime_n2"]
    gold_path = os.path.join(
        reference_dir.new.biosans, "overwrite_gold_04282020/test4/"
    )
    # Verify
    verify_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="SampleToSi -> 200 mm, SampleDetectorDistance -> 14 meter",
        prefix="test4",
    )


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Shuo Qian <qians@ornl.gov>
@pytest.mark.skip(
    reason="Underlying geometry algorithm is incorrect but can be tolerated in real experiment"
    ". Refer to MR #784."
)
def skip_test_overwrite_sample_to_si(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data overwriting sampleToSi but not sampleDetectorDistance
    Sample to detector distance will be modified accordingly with the move of sample relative to nominal point.

    - Overwrite sampleToSi (distance) to 500 mm.

    This integration test is from a test from and verified by Shuo Qian.
    Location of testing scirpts and results verified: /HFIR/CG3/shared/UserAcceptance/override_round3/
    Test script: /HFIR/CG3/shared/UserAcceptance/override_round3/test_reduce_cg3_4822_test2.py
    Verified result: /HFIR/CG3/shared/UserAcceptance/override_round3/test2/

    Returns
    -------

    """
    pytest.skip("")
    # Set up test
    json_file = generate_testing_json(
        os.path.join(reference_dir.new.biosans, "overwrite_gold_04282020"), 500, None
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_bio_test2")

    # Run
    reduce_biosans_data(
        reference_dir.new.biosans, json_file, output_dir, prefix="BioMetaSWD"
    )

    # Get result files
    sample_names = ["csmb_ecoli1h_n2", "insect1hTime_n2"]
    gold_path = os.path.join(
        reference_dir.new.biosans, "overwrite_gold_04282020/test2/"
    )
    # Verify
    verify_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="SampleToSi -> 500 mm",
        prefix="test2",
    )


# dev - Wenduo Zhou <wzz@ornl.gov>
# SME - Shuo Qian <qians@ornl.gov>
# Test is skipped because it occasionally crashes gw0 on drt-sans pipeline.
# There is no readable information from build server that can be retrieved.
def crash_gw0_test_overwrite_sample_to_detector(reference_dir, generatecleanfile):
    """Test reduce 3 sets of data overwriting sampleToSi but not sampleDetectorDistance.

    - Overwrite DetectorToSample (distance) to 14 meter

    This integration test is from a test from and verified by Shuo Qian.
    Location of testing scirpts and results verified: /HFIR/CG3/shared/UserAcceptance/override_round3/
    Test script: /HFIR/CG3/shared/UserAcceptance/override_round3/test_reduce_cg3_4822_test3.py
    Verified result: /HFIR/CG3/shared/UserAcceptance/override_round3/test3/

    Returns
    -------

    """
    # Set up test
    json_file = generate_testing_json(
        os.path.join(reference_dir.new.biosans, "overwrite_gold_04282020"), None, 14
    )
    output_dir = generatecleanfile(prefix="meta_overwrite_bio_test3")

    # Run
    reduce_biosans_data(
        reference_dir.new.biosans, json_file, output_dir, prefix="CG3MetaSDD"
    )

    # Get result files
    sample_names = ["csmb_ecoli1h_n2", "insect1hTime_n2"]
    gold_path = os.path.join(
        reference_dir.new.biosans, "overwrite_gold_20200815/test3/"
    )
    # Verify
    verify_reduction_results(
        sample_names,
        output_dir,
        gold_path,
        title="SampleDetectorDistance -> 14 meter",
        prefix="test3",
    )


def reduce_biosans_data(nexus_dir, json_str, output_dir, prefix):
    """Reduce BIOSANS runs

    Parameters
    ----------
    nexus_dir: str
        path to NeXus files
    json_str: str
        configuration json
    output_dir: str
        output directory
    prefix: str
        prefix for output workspaces' names

    Returns
    -------

    """
    # Set up (testing) runs
    sample_names = ["csmb_ecoli1h_n2", "insect1hTime_n2"]
    samples = ["5709", "5712"]
    samples_trans = samples
    backgrounds = ["5715", "5715"]
    backgrounds_trans = backgrounds

    # checking if output directory exists, if it doesn't, creates the folder
    for subfolder in ["1D", "2D"]:
        output_folder = os.path.join(output_dir, subfolder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    start_time = time.time()
    for i in range(len(samples)):
        # Load JSON for configuration
        # start with a fresh set of reduction parameters for every sample, because the reduction "pollutes"
        # the reduction parameters dictionary with additions not allowed by the schema
        reduction_input = json.loads(json_str)
        reduction_input["dataDirectories"] = nexus_dir
        reduction_input["configuration"]["outputDir"] = output_dir
        reduction_input["sample"]["runNumber"] = samples[i]
        reduction_input["sample"]["transmission"]["runNumber"] = samples_trans[i]
        reduction_input["background"]["runNumber"] = backgrounds[i]
        reduction_input["background"]["transmission"]["runNumber"] = backgrounds_trans[
            i
        ]
        reduction_input["outputFileName"] = sample_names[i]
        # always check after updating the parameters
        reduction_input = validate_reduction_parameters(reduction_input)
        loaded = load_all_files(reduction_input, path=nexus_dir, prefix=prefix)
        out = reduce_single_configuration(loaded, reduction_input)
        plot_reduction_output(out, reduction_input, loglog=False)

    end_time = time.time()
    print(end_time - start_time)


def generate_testing_json(
    sens_nxs_dir, sample_to_si_window_distance, sample_to_detector_distance
):
    """Generating testing JSON

    Parameters
    ----------
    sens_nxs_dir: str
        directory path to sensitivity files
    sample_to_si_window_distance: float or None
        sample to silicon window distance to overwrite the EPICS value with unit millimeter
    sample_to_detector_distance: float or None
        sample to silicon window distance to overwrite the EPICS value with unit meter

    Returns
    -------
    str
        JSON string
    """
    specs = {
        "iptsNumber": "23782",
        "sample": {
            "runNumber": "4822",
            "thickness": "0.1",
            "transmission": {"runNumber": "4822"},
        },
        "outputFileName": "CG3_4822",
        "background": {"runNumber": "4821", "transmission": {"runNumber": "4821"}},
        "beamCenter": {"runNumber": "1322"},
        "emptyTransmission": {"runNumber": "5705"},
        "configuration": {
            "outputDir": "/HFIR/CG3/shared/UserAcceptance/override/test1",
            "sampleApertureSize": "14",
            "useDefaultMask": True,
            "defaultMask": ["{'Pixel':'1-12,244-256'}", "{'Bank':'21-24,45-48'}"],
            "darkMainFileName": "CG3_1383.nxs.h5",
            "darkWingFileName": "CG3_1383.nxs.h5",
            "sensitivityMainFileName": "/HFIR/CG3/shared/Cycle486/sens_f4829m7p0_TDC_SAC.h5",
            "sensitivityWingFileName": "/HFIR/CG3/shared/Cycle486/sens_f4835w3p2_TDC_SAC.h5",
            "absoluteScaleMethod": "standard",
            "StandardAbsoluteScale": "0.0055e-8",
            "numMainQxQyBins": "100",
            "numWingQxQyBins": "100",
            "1DQbinType": "scalar",
            "QbinType": "log",
            "useLogQBinsEvenDecade": False,
            "LogQBinsPerDecadeMain": 20,
            "LogQBinsPerDecadeWing": 25,
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "AnnularAngleBin": "1",
            "QminMain": "0.003",
            "QminWing": "0.003",
            "overlapStitchQmin": "0.075",
            "overlapStitchQmax": "0.095",
            "useTimeSlice": False,
            "timeSliceInterval": "200",
            "usePixelCalibration": False,
            "useSubpixels": False,
        },
    }
    reduction_input = reduction_parameters(
        specs, "BIOSANS", validate=False
    )  # add defaults and defer validation
    reduction_config = reduction_input["configuration"]  # a handy shortcut

    #  '/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/meta_overwrite/biosans/sens_f4829m7p0_TDC_SAC.h5'
    main_sens = os.path.join(sens_nxs_dir, "sens_f4829m7p0_TDC_SAC.h5")
    reduction_config["sensitivityMainFileName"] = main_sens

    # '/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/meta_overwrite/biosans/sens_f4835w3p2_TDC_SAC.h5'
    wing_sens = os.path.join(sens_nxs_dir, "sens_f4835w3p2_TDC_SAC.h5")
    reduction_config["sensitivityWingFileName"] = wing_sens

    if sample_to_si_window_distance is not None:
        reduction_config["sampleToSi"] = sample_to_si_window_distance

    if sample_to_detector_distance is not None:
        reduction_config["sampleDetectorDistance"] = sample_to_detector_distance

    return json.dumps(reduction_input)  # return a string representation


def verify_reduction_results(sample_names, output_dir, gold_path, title, prefix):

    # Over all message
    unmatched_errors = ""

    for sample_name in sample_names:
        # output log file name
        output_log_file = os.path.join(
            output_dir, "{}_reduction_log.hdf".format(sample_name)
        )
        assert os.path.exists(output_log_file), "Output {} cannot be found".format(
            output_log_file
        )
        # gold file
        gold_log_file = os.path.join(
            gold_path, "{}_{}_reduction_log.hdf".format(sample_name, prefix)
        )
        assert os.path.exists(gold_path), "Gold file {} cannot be found".format(
            gold_log_file
        )
        # compare
        title_i = "{}: {}".format(sample_name, title)

        # compare
        try:
            compare_reduced_iq(output_log_file, gold_log_file, title_i, prefix)
        except AssertionError as unmatched_error:
            unmatched_errors += (
                "Testing output {} does not match gold result {}:\n{}\n"
                "".format(output_log_file, gold_log_file, unmatched_error)
            )
    # END-FOR

    # raise error for all
    if unmatched_errors != "":
        print("[VERIFICATION ERROR MESSAGE] {}".format(unmatched_errors))
        raise AssertionError(unmatched_errors)


def compare_reduced_iq(test_log_file, gold_log_file, title, prefix):
    """

    Parameters
    ----------
    test_log_file: str
        Absolute
    gold_log_file: str
    title: str
        plot title
    prefix: str
        file name prefix

    Returns
    -------

    """
    log_errors = list()

    for is_main_detector in [True, False]:
        vec_q_a, vec_i_a = get_iq1d(test_log_file, is_main=is_main_detector)
        vec_q_b, vec_i_b = get_iq1d(gold_log_file, is_main=is_main_detector)

        try:
            np.testing.assert_allclose(vec_q_a, vec_q_b)
            np.testing.assert_allclose(vec_i_a, vec_i_b)
            log_errors.append(None)
        except AssertionError as assert_err:
            log_errors.append(assert_err)
            from matplotlib import pyplot as plt

            if is_main_detector:
                flag = "Main_detector"
            else:
                flag = "Wing_detector"
            plt.cla()
            plt.plot(
                vec_q_a,
                vec_i_a,
                color="red",
                label="{} Test Data.     Q in {:.5f}, {:.5f}".format(
                    flag, vec_q_a[0], vec_q_a[-1]
                ),
            )
            plt.plot(
                vec_q_b,
                vec_i_b,
                color="black",
                label="{} Expected Data. Q in {:.5f}, {:.5f}".format(
                    flag, vec_q_b[0], vec_q_b[-1]
                ),
            )
            plt.yscale("log")
            plt.title(title)
            plt.legend()

            # defaults
            if prefix is None:
                prefix = "compare"
            if test_log_file is None:
                test_log_file = "iq"
            out_name = (
                prefix
                + "_"
                + os.path.basename(test_log_file).split(".")[0]
                + "_{}.png".format(flag)
            )
            plt.savefig(out_name)
    # END-FOR

    # Report
    if not (log_errors[0] is None and log_errors[1] is None):
        error_message = "Main: {}; Wing: {}".format(log_errors[0], log_errors[1])
        raise AssertionError(error_message)


def get_iq1d(log_file_name, is_main=True):
    """

    Parameters
    ----------
    log_file_name: str
        output log file's name
    is_main: bool
        for main or wing

    Returns
    -------

    """
    # Open file and entry
    log_h5 = h5py.File(log_file_name, "r")
    try:
        if is_main:
            iq1d_entry = log_h5["main_0"]["I(Q)"]
        else:
            iq1d_entry = log_h5["wing_0"]["I(Q)"]
    except KeyError:
        if is_main:
            iq1d_entry = log_h5["_slice_1"]["main_0"]["I(Q)"]
        else:
            iq1d_entry = log_h5["_slice_1"]["wing_0"]["I(Q)"]

    # Get data with a copy
    vec_q = np.copy(iq1d_entry["Q"][()])
    vec_i = np.copy(iq1d_entry["I"][()])

    # close file
    log_h5.close()

    return vec_q, vec_i


if __name__ == "__main__":
    pytest.main(__file__)
