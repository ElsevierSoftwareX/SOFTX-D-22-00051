import pytest
import os
from jsonschema.exceptions import ValidationError
from drtsans.tof.eqsans import reduction_parameters
from drtsans.tof.eqsans.api import (
    load_all_files,
    reduce_single_configuration,
)  # noqa E402
from drtsans.dataobjects import _Testing
from drtsans.settings import amend_config
import json
from typing import Tuple, Dict
from drtsans.dataobjects import load_iq1d_from_h5, load_iq2d_from_h5
from mantid.simpleapi import mtd, DeleteWorkspace


def test_parse_json(reference_dir):
    """Test the JSON to dictionary"""
    elastic_reference_run = "124680"
    elastic_reference_bkgd_run = ""
    # Specify JSON input
    reduction_input = {
        "instrumentName": "EQSANS",
        "iptsNumber": "26015",
        "sample": {"runNumber": "115363", "thickness": "1.0"},
        "background": {
            "runNumber": "115361",
            "transmission": {"runNumber": "115357", "value": ""},
        },
        "emptyTransmission": {"runNumber": "115356", "value": ""},
        "beamCenter": {"runNumber": "115356"},
        "outputFileName": "test_wavelength_step",
        "configuration": {
            "outputDir": "/path/to/nowhere",
            "cutTOFmax": "1500",
            "wavelengthStepType": "constant Delta lambda/lambda",
            "sampleApertureSize": "10",
            "fluxMonitorRatioFile": (
                "/SNS/EQSANS/" "IPTS-24769/shared/EQSANS_110943.out"
            ),
            "sensitivityFileName": (
                "/SNS/EQSANS/"
                "shared/NeXusFiles/EQSANS/"
                "2020A_mp/Sensitivity_patched_thinPMMA_2o5m_113514_mantid.nxs"
            ),
            "numQBins": "100",
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "AnnularAngleBin": "5",
            "useSliceIDxAsSuffix": True,
            "fitInelasticIncoh": True,
            "elasticReference": {
                "runNumber": elastic_reference_run,
                "thickness": "1.0",
                "transmission": {"runNumber": None, "value": "0.89"},
            },
            "elasticReferenceBkgd": {
                "runNumber": elastic_reference_bkgd_run,
                "transmission": {"runNumber": "", "value": "0.9"},
            },
            "selectMinIncoh": True,
        },
    }

    # Validate
    with amend_config(data_dir=reference_dir.new.eqsans):
        input_config = reduction_parameters(reduction_input)

    # Check that inelastic incoherence config items were parsed
    assert input_config["configuration"].get("fitInelasticIncoh")
    assert (
        input_config["configuration"]["elasticReference"].get("runNumber")
        == elastic_reference_run
    )
    assert input_config["configuration"].get("selectMinIncoh")

    # Parse
    from drtsans.tof.eqsans.correction_api import parse_correction_config

    correction = parse_correction_config(input_config)
    assert correction.do_correction
    assert correction.elastic_reference
    assert correction.elastic_reference.run_number == "124680"
    assert correction.elastic_reference.thickness == 1.0
    assert correction.elastic_reference.transmission_value == 0.89
    assert correction.elastic_reference.background_run_number is None


def test_parse_invalid_json():
    """Test the JSON to dictionary"""
    invalid_run_num = "260159121"
    valid_run_num = "115363"
    # Specify JSON input
    reduction_input = {
        "instrumentName": "EQSANS",
        "iptsNumber": "26015",
        "sample": {"runNumber": "115363", "thickness": "1.0"},
        "background": {
            "runNumber": "115361",
            "transmission": {"runNumber": "115357", "value": ""},
        },
        "emptyTransmission": {"runNumber": "115356", "value": ""},
        "beamCenter": {"runNumber": "115356"},
        "outputFileName": "test_wavelength_step",
        "configuration": {
            "outputDir": "/path/to/nowhere",
            "cutTOFmax": "1500",
            "wavelengthStepType": "constant Delta lambda/lambda",
            "sampleApertureSize": "10",
            "fluxMonitorRatioFile": (
                "/SNS/EQSANS/" "IPTS-24769/shared/EQSANS_110943.out"
            ),
            "sensitivityFileName": (
                "/SNS/EQSANS/"
                "shared/NeXusFiles/EQSANS/"
                "2020A_mp/Sensitivity_patched_thinPMMA_2o5m_113514_mantid.nxs"
            ),
            "numQBins": "100",
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "AnnularAngleBin": "5",
            "useSliceIDxAsSuffix": True,
            "fitInelasticIncoh": True,
            "elasticReference": {
                "runNumber": invalid_run_num,
                "thickness": "1.0",
                "transmission": {"runNumber": valid_run_num, "value": "0.9"},
            },
            "elasticReferenceBkgd": {
                "runNumber": valid_run_num,
                "transmission": {"runNumber": valid_run_num, "value": "0.9"},
            },
            "selectMinIncoh": True,
        },
    }

    # Validate
    with pytest.raises(ValidationError):
        # expect to fail as elastic reference run 260159121 does not exist
        reduction_parameters(reduction_input)

    # Respecify to use a valid run
    # json_str.replace('260159121', '26015')
    reduction_input["configuration"]["elasticReference"]["runNumber"] = valid_run_num
    # Defaults and Validate
    input_config = reduction_parameters(reduction_input)

    # Check that inelastic incoherence config items were parsed
    assert input_config["configuration"].get("fitInelasticIncoh")
    assert (
        input_config["configuration"]["elasticReference"].get("runNumber")
        == valid_run_num
    )
    assert input_config["configuration"].get("selectMinIncoh")


def generate_configuration_with_correction(output_dir: str = "/tmp/") -> Dict:
    """Generate configuration dictionary (JSON) from test 2 in issue 689

    Source: https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/issues/689  *Test 2*

    Returns
    -------
    ~dict
        Reduction donfiguration JSON

    """
    reduction_configuration = {
        "schemaStamp": "2020-05-21T17:18:11.528041",
        "instrumentName": "EQSANS",
        "iptsNumber": "24876",
        "sample": {
            "runNumber": "113915",
            "thickness": 1,
            "transmission": {"runNumber": "113914", "value": ""},
        },
        "background": {
            "runNumber": "113919",
            "transmission": {"runNumber": "113918", "value": ""},
        },
        "emptyTransmission": {"runNumber": "113682", "value": None},
        "beamCenter": {"runNumber": "113682"},
        "outputFileName": "water65D_2o5m2o5a_full",
        "configuration": {
            "outputDir": f"{output_dir}",
            "useTimeSlice": False,
            "timeSliceInterval": 300,
            "useLogSlice": False,
            "logSliceName": None,
            "logSliceInterval": 10,
            "cutTOFmin": 500,
            "cutTOFmax": 2000,
            "wavelengthStep": 0.1,
            "wavelengthStepType": "constant Delta lambda",
            "sampleOffset": 314.5,
            "useDetectorOffset": True,
            "detectorOffset": 80,
            "sampleApertureSize": 10,
            "sourceApertureDiameter": None,
            "usePixelCalibration": None,
            "maskFileName": "/SNS/EQSANS/shared/NeXusFiles/EQSANS/2020A_mp/mask_4m_extended.nxs",
            "useDefaultMask": True,
            "defaultMask": None,
            "useMaskBackTubes": False,
            "darkFileName": "/SNS/EQSANS/shared/NeXusFiles/EQSANS/2020A_mp/EQSANS_113569.nxs.h5",
            "normalization": "Total charge",
            "fluxMonitorRatioFile": None,
            "beamFluxFileName": "/SNS/EQSANS/shared/instrument_configuration/bl6_flux_at_sample",
            "sensitivityFileName": "/SNS/EQSANS/shared/NeXusFiles/EQSANS/2020A_mp/"
            "Sensitivity_patched_thinPMMA_2o5m_113514_mantid.nxs",
            "useSolidAngleCorrection": True,
            "useThetaDepTransCorrection": True,
            "mmRadiusForTransmission": 25,
            "absoluteScaleMethod": "standard",
            "StandardAbsoluteScale": 1.0,
            "numQxQyBins": 80,
            "1DQbinType": "scalar",
            "QbinType": "linear",
            "numQBins": 100,
            "LogQBinsPerDecade": None,
            "useLogQBinsDecadeCenter": False,
            "useLogQBinsEvenDecade": False,
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "autoWedgeQmin": None,
            "autoWedgeQmax": None,
            "autoWedgeQdelta": None,
            "autoWedgeAzimuthalDelta": None,
            "autoWedgePeakWidth": None,
            "autoWedgeBackgroundWidth": None,
            "autoWedgeSignalToNoiseMin": None,
            "AnnularAngleBin": 5.0,
            "Qmin": None,
            "Qmax": None,
            "useErrorWeighting": True,
            "smearingPixelSizeX": None,
            "smearingPixelSizeY": None,
            "useSubpixels": None,
            "subpixelsX": None,
            "subpixelsY": None,
            "useSliceIDxAsSuffix": None,
            # inelastic/incoherent correction
            "fitInelasticIncoh": True,
            "elasticReference": {
                "runNumber": None,
                "thickness": "1.0",
                "transmission": {"runNumber": None, "value": "0.9"},
            },
            "elasticReferenceBkgd": {
                "runNumber": None,
                "transmission": {"runNumber": None, "value": "0.9"},
            },
            "selectMinIncoh": True,
        },
    }

    return reduction_configuration


@pytest.mark.skipif(
    reason="The test is either incorrect or using wrong ref values",
)
def test_incoherence_correction_step4only(reference_dir, generatecleanfile):
    """Test incoherence correction without elastic correction"""
    # Set up the configuration dict
    configuration = generate_configuration_with_correction()

    # Create temp output directory
    test_dir = generatecleanfile()
    base_name = "EQSANS_113915_Incoh_1d"

    assert os.path.exists(test_dir), f"Output dir {test_dir} does not exit"
    configuration["configuration"]["outputDir"] = test_dir
    configuration["outputFileName"] = base_name
    configuration["dataDirectories"] = test_dir

    # validate and clean configuration
    input_config = reduction_parameters(configuration)
    loaded = load_all_files(input_config)

    # Reduce
    reduction_output = reduce_single_configuration(
        loaded, input_config, not_apply_incoherence_correction=False
    )

    # Gold data directory
    gold_dir = os.path.join(
        reference_dir.new.eqsans, "gold_data/Incoherence_Corrected_113915/"
    )
    assert os.path.exists(
        gold_dir
    ), f"Gold/expected data directory {gold_dir} does not exist"

    # Verify with gold data
    gold_file_dict = dict()
    for frame_index in range(1):
        iq1d_h5_name = os.path.join(gold_dir, f"EQSANS_11395iq1d_{frame_index}_0.h5")
        gold_file_dict[1, frame_index, 0] = iq1d_h5_name
        iq2d_h5_name = os.path.join(gold_dir, f"EQSANS_11395iq2d_{frame_index}.h5")
        gold_file_dict[2, frame_index] = iq2d_h5_name
        assert os.path.exists(iq1d_h5_name) and os.path.exists(iq2d_h5_name), (
            f"{iq1d_h5_name} and/or {iq2d_h5_name}" f"do not exist"
        )

    # Verify
    verify_binned_iq(gold_file_dict, reduction_output)


def test_incoherence_correction_elastic_normalization(reference_dir, generatecleanfile):
    """Test incoherence correction with elastic correction"""
    # Set up the configuration dict
    config_json_file = os.path.join(
        reference_dir.new.eqsans, "test_incoherence_correction/agbe_125707_test1.json"
    )
    assert os.path.exists(
        config_json_file
    ), f"Test JSON file {config_json_file} does not exist."
    with open(config_json_file, "r") as config_json:
        configuration = json.load(config_json)
    assert isinstance(configuration, dict)

    # Create temp output directory
    test_dir = generatecleanfile()
    base_name = "EQSANS_125707_"

    assert os.path.exists(test_dir), f"Output dir {test_dir} does not exit"
    configuration["configuration"]["outputDir"] = test_dir
    configuration["outputFileName"] = base_name
    configuration["dataDirectories"] = test_dir

    # validate and clean configuration
    input_config = reduction_parameters(configuration)
    loaded = load_all_files(input_config)

    # check loaded JSON file
    assert loaded.elastic_reference.data
    assert loaded.elastic_reference_background.data is None

    # Reduce
    reduction_output = reduce_single_configuration(
        loaded, input_config, not_apply_incoherence_correction=False
    )
    assert reduction_output
    print(f"Output directory: {test_dir}")

    # Check output result
    iq1d_base_name = "EQSANS_125707__Iq.dat"
    test_iq1d_file = os.path.join(test_dir, iq1d_base_name)
    # FIXME: The gold data are not stored inside the repository so when
    # gold data are changed a version prefix is added with date and developer
    # information. The old data will be kept as it is.
    version = "20220321_rys_"
    assert os.path.exists(
        test_iq1d_file
    ), f"Expected test result {test_iq1d_file} does not exist"
    gold_iq1d_file = os.path.join(
        reference_dir.new.eqsans, "test_incoherence_correction", version + iq1d_base_name
    )
    assert os.path.exists(
        gold_iq1d_file
    ), f"Expected gold file {gold_iq1d_file} does not exist"
    # compare
    import filecmp

    print(f"TEST DEBUT: {filecmp.cmp(test_iq1d_file, gold_iq1d_file)}")
    assert filecmp.cmp(test_iq1d_file, gold_iq1d_file)

    # cleanup
    # NOTE: loaded is not a dict that is iterable, so we have to delete the
    #       leftover workspace explicitly
    # _empty:	37.277037 MB
    # _EQSANS_124667_raw_histo:	106.769917 MB
    # _EQSANS_124680_raw_histo:	40.050957 MB
    # _EQSANS_125701_raw_events:	41.005101 MB
    # _EQSANS_125701_raw_histo:	37.276829 MB
    # _EQSANS_125707_raw_histo:	38.326685 MB
    # _mask:	27.092797 MB
    # _sensitivity:	30.03614 MB
    # processed_data_main:	38.327517 MB
    # processed_elastic_ref:	40.051789 MB
    DeleteWorkspace("_empty")
    DeleteWorkspace("_mask")
    DeleteWorkspace("_sensitivity")
    DeleteWorkspace("processed_data_main")
    DeleteWorkspace("processed_elastic_ref")
    for ws in mtd.getObjectNames():
        if str(ws).startswith("_EQSANS_"):
            DeleteWorkspace(ws)


def test_incoherence_correction_elastic_normalization_weighted(reference_dir, generatecleanfile):
    """Test incoherence correction with elastic correction"""
    import filecmp

    # Set up the configuration dict
    config_json_file = os.path.join(
        reference_dir.new.eqsans, "test_incoherence_correction/porsil_29024_abs_inel.json"
    )
    assert os.path.exists(
        config_json_file
    ), f"Test JSON file {config_json_file} does not exist."
    with open(config_json_file, "r") as config_json:
        configuration = json.load(config_json)
    assert isinstance(configuration, dict)

    # Create temp output directory
    test_dir = generatecleanfile()

    def run_reduction_and_compare(config, expected_result_filename):
        with amend_config(data_dir=reference_dir.new.eqsans):
            # validate and clean configuration
            input_config = reduction_parameters(config)
            loaded = load_all_files(input_config)

            # Reduce
            reduction_output = reduce_single_configuration(
                loaded, input_config, not_apply_incoherence_correction=False
            )
        assert reduction_output

        test_iq1d_file = os.path.join(test_dir, config["outputFileName"] + "_Iq.dat")
        gold_iq1d_file = os.path.join(
            reference_dir.new.eqsans, "test_incoherence_correction", expected_result_filename
        )
        # compare
        assert filecmp.cmp(test_iq1d_file, gold_iq1d_file)

        DeleteWorkspace("_empty")
        DeleteWorkspace("_mask")
        DeleteWorkspace("_sensitivity")
        DeleteWorkspace("processed_data_main")
        for ws in mtd.getObjectNames():
            if str(ws).startswith("_EQSANS_"):
                DeleteWorkspace(ws)

    # Run without intensity weighted correction
    base_name = "EQSANS_132078"
    assert os.path.exists(test_dir), f"Output dir {test_dir} does not exit"
    configuration["configuration"]["outputDir"] = test_dir
    configuration["outputFileName"] = base_name
    configuration["dataDirectories"] = reference_dir.new.eqsans
    run_reduction_and_compare(configuration, "EQSANS_132078_Iq.dat")

    # Run with weighted
    base_name = "EQSANS_132078_weighted"
    configuration["outputFileName"] = base_name
    configuration["configuration"]["incohfit_intensityweighted"] = True
    configuration["configuration"]["incohfit_factor"] = None
    configuration["configuration"]["incohfit_qmin"] = None
    configuration["configuration"]["incohfit_qmax"] = None
    run_reduction_and_compare(configuration, "EQSANS_132078_weighted_Iq.dat")

    # Run with weighted and factor
    base_name = "EQSANS_132078_weighted_factor"
    configuration["outputFileName"] = base_name
    configuration["configuration"]["incohfit_intensityweighted"] = True
    configuration["configuration"]["incohfit_factor"] = 10
    configuration["configuration"]["incohfit_qmin"] = None
    configuration["configuration"]["incohfit_qmax"] = None
    run_reduction_and_compare(configuration, "EQSANS_132078_weighted_factor_Iq.dat")

    # Run with weighted and manual q range
    # q-range is set to be the same as what the factor calculation finds
    base_name = "EQSANS_132078_weighted_qrange"
    configuration["outputFileName"] = base_name
    configuration["configuration"]["incohfit_intensityweighted"] = True
    configuration["configuration"]["incohfit_factor"] = None
    configuration["configuration"]["incohfit_qmin"] = 0.085
    configuration["configuration"]["incohfit_qmax"] = 0.224
    run_reduction_and_compare(configuration, "EQSANS_132078_weighted_factor_Iq.dat")

    print(f"Output directory: {test_dir}")


def verify_binned_iq(gold_file_dict: Dict[Tuple, str], reduction_output):
    """Verify reduced I(Q1D) and I(qx, qy) by expected/gold data

    Parameters
    ----------
    gold_file_dict: ~dict
        dictionary for gold files
    reduction_output: ~list
        list of binned I(Q1D) and I(qx, qy)

    """
    num_frames_gold = len(gold_file_dict) // 2
    assert num_frames_gold == len(
        reduction_output
    ), f"Frame numbers are different: gold = {len(gold_file_dict) // 2}; test = {len(reduction_output)}"

    for frame_index in range(num_frames_gold):
        # 1D
        iq1d_h5_name = gold_file_dict[1, frame_index, 0]
        gold_iq1d = load_iq1d_from_h5(iq1d_h5_name)
        _Testing.assert_allclose(reduction_output[frame_index].I1D_main[0], gold_iq1d)

        # 2D
        iq2d_h5_name = gold_file_dict[2, frame_index]
        gold_iq2d = load_iq2d_from_h5(iq2d_h5_name)
        _Testing.assert_allclose(reduction_output[frame_index].I2D_main, gold_iq2d)


if __name__ == "__main__":
    pytest.main([__file__])
