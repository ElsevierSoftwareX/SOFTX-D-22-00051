import pytest
import json
import drtsans.mono.meta_data as mono_meta_data
from drtsans.mono.meta_data import parse_json_meta_data


def test_parse_json_geometry_current():
    """Test parse JSON string with geometry related parameters such as SDD and SWD overwritten.

    Ths JSON is the current agreed format as 2020.04.21

    Returns
    -------

    """
    test_json = """ {
    "configuration": {
        "outputDir": "/HFIR/CG3/shared/UserAcceptance/override/test1",
        "sampleApertureSize": "14",
        "sourceApertureDiameter": "",
        "maskFileName": "",
        "useMaskFileName": false,
        "useDefaultMask": true,
        "DefaultMask":["{'Pixel':'1-12,244-256'}", "{'Bank':'21-24,45-48'}"],
        "useBlockedBeam": false,
        "BlockBeamFileName":"",
        "useDarkFileName": true,
        "darkMainFileName": "CG3_1383.nxs",
        "darkWingFileName": "CG3_1383.nxs",
        "useSensitivityFileName": true,
        "sensitivityMainFileName": "/HFIR/CG3/shared/Cycle486/sens_f4829m7p0_TDC_SAC.h5",
        "sensitivityWingFileName": "/HFIR/CG3/shared/Cycle486/sens_f4835w3p2_TDC_SAC.h5",
        "UseBarScan": false,
        "BarScanMainFileName":"",
        "BarScanWingFileName":"",
        "absoluteScaleMethod":"standard",
        "DBScalingBeamRadius": "",
        "StandardAbsoluteScale": "0.0055e-8",
        "normalization": "Monitor",
        "sampleOffset": "",
        "useSampleOffset": false,
        "useDetectorTubeType": true,
        "useSolidAngleCorrection": true,
        "useThetaDepTransCorrection": true,
        "mmRadiusForTransmission": "",
        "numMainQxQyBins": "100",
        "numWingQxQyBins": "100",
        "1DQbinType": "scalar",
        "QbinType": "log",
        "LogQBinsEvenDecade": false,
        "LogQBinsPerDecadeMain":20,
        "LogQBinsPerDecadeWing": 25,
        "WedgeMinAngles": "-30, 60",
        "WedgeMaxAngles": "30, 120",
        "numMainQBins": "",
        "numWingQBins": "",
        "AnnularAngleBin": "1",
        "Qmin": "0.003",
        "Qmax": "",
        "useErrorWeighting": false,
        "useMaskBackTubes": false,
        "wavelength": "",
        "wavelengthSpread": "",
        "overlapStitchQmin": "0.075",
        "overlapStitchQmax": "0.095",
        "timeslice": false,
        "timesliceinterval": "200",
        "logslicename": "",
        "logslice": false,
        "logsliceinterval": "",
        "SampleToSi": "234.56",
        "SampleDetectorDistance": "32.11"
        }
    }"""

    # convert from JSON to inputs
    input_dict = json.loads(test_json)

    # parse JSON for sample to si window distance with instrument preferred default
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/542
    sample_to_si_dict = parse_json_meta_data(
        input_dict,
        meta_name="SampleToSi",
        unit_conversion_factor=1e-3,
        beam_center_run=True,
        background_run=True,
        empty_transmission_run=True,
        transmission_run=True,
        background_transmission=True,
        block_beam_run=True,
        dark_current_run=False,
    )

    # parse JSON for sample to detector distance with instrument preferred default
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/542
    sample_to_detector_dict = parse_json_meta_data(
        input_dict,
        meta_name="SampleDetectorDistance",
        unit_conversion_factor=1.0,
        beam_center_run=True,
        background_run=True,
        empty_transmission_run=False,
        transmission_run=False,
        background_transmission=False,
        block_beam_run=True,
        dark_current_run=False,
    )

    # verify SampleToSi
    for run_type in [
        mono_meta_data.SAMPLE,
        mono_meta_data.BACKGROUND,
        mono_meta_data.BEAM_CENTER,
        mono_meta_data.EMPTY_TRANSMISSION,
        mono_meta_data.TRANSMISSION,
        mono_meta_data.TRANSMISSION_BACKGROUND,
        mono_meta_data.BLOCK_BEAM,
    ]:
        assert sample_to_si_dict[run_type] == pytest.approx(0.23456, 0.000004)
    assert sample_to_si_dict[mono_meta_data.DARK_CURRENT] is None

    # verify SampleDetectorDistance
    for run_type in [
        mono_meta_data.SAMPLE,
        mono_meta_data.BACKGROUND,
        mono_meta_data.BEAM_CENTER,
        mono_meta_data.BLOCK_BEAM,
    ]:
        assert sample_to_detector_dict[run_type] == pytest.approx(32.11, 0.004)
    for run_type in [
        mono_meta_data.TRANSMISSION,
        mono_meta_data.TRANSMISSION_BACKGROUND,
        mono_meta_data.EMPTY_TRANSMISSION,
        mono_meta_data.DARK_CURRENT,
    ]:
        assert sample_to_detector_dict[run_type] is None
