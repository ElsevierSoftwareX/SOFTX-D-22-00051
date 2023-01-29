"""
Test methods to determine the correct sample and detector positions from meta data and overwriting
"""
import os
import pytest
from mantid.simpleapi import LoadEmptyInstrument
from drtsans.samplelogs import SampleLogs
from drtsans.mono.meta_data import get_sample_detector_offset
from drtsans.geometry import sample_detector_distance


@pytest.mark.parametrize(
    "generic_IDF",
    [{"Nx": 4, "Ny": 4, "dx": 0.006, "dy": 0.004, "zc": 1.25, "l1": -15.0}],
    indirect=True,
)
def test_zero_offsets(generic_IDF):
    """Test instrument without offset

    Returns
    -------

    """
    # Generate a generic SANS instrument with detector dimension stated in
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/178
    idf_name = r"/tmp/GenericHFIRSANS_Definition.xml"
    with open(idf_name, "w") as tmp:
        tmp.write(generic_IDF)
        tmp.close()
    ws = LoadEmptyInstrument(
        Filename=tmp.name,
        InstrumentName="GenericSANS",
        OutputWorkspace="test_integration_roi",
    )
    # clean
    os.remove(idf_name)

    print(
        "TestZeroOffsets: sample detector distance [0] = {}".format(
            sample_detector_distance(ws, unit="m")
        )
    )

    # Add sample log
    sample_logs = SampleLogs(ws)
    sample_logs.insert("sample-detector-distance", 1.25 * 1e3, unit="mm")
    sample_logs.insert("Generic:CS:SampleToSi", 71, unit="mm")

    # Test method
    sample_offset, detector_offset = get_sample_detector_offset(
        ws, "Generic:CS:SampleToSi", 0.071
    )

    # Verify: sample_offset = detector_offset = 0. as expecgted
    assert sample_offset == pytest.approx(0, 1e-12)
    assert detector_offset == pytest.approx(0, 1e-12)


@pytest.mark.parametrize(
    "generic_IDF",
    [{"Nx": 4, "Ny": 4, "dx": 0.006, "dy": 0.004, "zc": 1.25, "l1": -15.0}],
    indirect=True,
)
def test_non_zero_offsets(generic_IDF):
    """Test instrument with offset between sampleToSi and its default value

    Returns
    -------

    """
    # Generate a generic SANS instrument with detector dimension stated in
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/178
    idf_name = r"/tmp/GenericHFIRSANS_Definition.xml"
    with open(idf_name, "w") as tmp:
        tmp.write(generic_IDF)
        tmp.close()
    ws = LoadEmptyInstrument(
        Filename=tmp.name,
        InstrumentName="GenericSANS",
        OutputWorkspace="test_integration_roi",
    )
    # clean
    os.remove(idf_name)

    # Set up logs
    sdd = sample_detector_distance(ws)
    print("TestNonZeroOffsets: sample detector distance [1] = {}".format(sdd))

    # Add sample log
    sample_logs = SampleLogs(ws)
    sample_logs.insert("sample-detector-distance", sdd * 1e3, unit="mm")
    sample_logs.insert("Generic:CS:SampleToSi", 75.32, unit="mm")

    # Test method
    sample_offset, detector_offset = get_sample_detector_offset(
        ws, "Generic:CS:SampleToSi", 0.071
    )

    # Verify
    assert sample_offset == pytest.approx(-4.32 * 1e-3, 1e-12)
    assert detector_offset == pytest.approx(-4.32 * 1e-3, 1e-12)


@pytest.mark.parametrize(
    "generic_IDF",
    [{"Nx": 4, "Ny": 4, "dx": 0.006, "dy": 0.004, "zc": 1.25, "l1": -5.0}],
    indirect=True,
)
def test_overwrite_sample_si_distance(generic_IDF):
    """Test instrument with a user-overwriting sampleToSi distance

    Returns
    -------

    """
    # Generate a generic SANS instrument with detector dimension stated in
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/178
    idf_name = r"/tmp/GenericHFIRSANS_Definition.xml"
    with open(idf_name, "w") as tmp:
        tmp.write(generic_IDF)
        tmp.close()
    ws = LoadEmptyInstrument(
        Filename=tmp.name,
        InstrumentName="GenericSANS",
        OutputWorkspace="test_integration_roi",
    )
    # clean
    os.remove(idf_name)

    # Set up sample logs
    sdd = sample_detector_distance(ws, unit="m")
    print(
        "TestOverwriteSampleSiDistance: sample detector distance [1] = {} meter".format(
            sdd
        )
    )

    # Add sample log
    sample_logs = SampleLogs(ws)
    sample_logs.insert("sample-detector-distance", sdd * 1e3, unit="mm")
    # shift 1.23 mm from default
    default_sample_si_distance = 0.071  # meter, i.e., 71 mm
    sample_logs.insert("Generic:CS:SampleToSi", 72.23, unit="mm")

    # Test method
    sample_offset, detector_offset = get_sample_detector_offset(
        ws,
        "Generic:CS:SampleToSi",
        default_sample_si_distance,
        overwrite_sample_si_distance=75.32 * 1e-3,
    )

    # Verify
    # 1. SampleToSi is 72.23 mm.  It results in a 1.23 mm shift (to source, i.e., -Y direction) on both
    #    sample position and detector position
    # 2. User-overwriting SampleToSi 75.32 results in a 4.32 mm shift of sample only from nominal position
    # 3. Original detector-sample position then will be changed to (1250 + (75.32 - 72.23)) mm
    assert sample_offset == pytest.approx(-4.32 * 1e-3, 1e-12)
    assert detector_offset == pytest.approx(-1.23 * 1e-3, 1e-12)


@pytest.mark.parametrize(
    "generic_IDF",
    [{"Nx": 4, "Ny": 4, "dx": 0.006, "dy": 0.004, "zc": 1.25, "l1": -5.0}],
    indirect=True,
)
def test_overwrite_sample_detector_distance(generic_IDF):
    """Test instrument with a user-overwriting sample to detector distance

    Returns
    -------

    """
    # Generate a generic SANS instrument with detector dimension stated in
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/178
    idf_name = r"/tmp/GenericHFIRSANS_Definition.xml"
    with open(idf_name, "w") as tmp:
        tmp.write(generic_IDF)
        tmp.close()
    ws = LoadEmptyInstrument(
        Filename=tmp.name,
        InstrumentName="GenericSANS",
        OutputWorkspace="test_integration_roi",
    )
    # clean
    os.remove(idf_name)

    # Set up sample logs
    sdd = sample_detector_distance(ws, unit="m")
    print(
        "TestOverwriteSampleDetectorDistance: sample detector distance [1] = {} meter".format(
            sdd
        )
    )

    # Add sample log
    sample_logs = SampleLogs(ws)
    sample_logs.insert("sample-detector-distance", sdd, unit="m")
    default_sample_si_distance = 0.071  # meter, i.e., 71 mm
    # shift sample 1.23 mm to source from silicon window
    sample_logs.insert("Generic:CS:SampleToSi", 72.23, unit="mm")

    # Test method
    sample_offset, detector_offset = get_sample_detector_offset(
        ws,
        "Generic:CS:SampleToSi",
        default_sample_si_distance,
        overwrite_sample_detector_distance=1.40,
    )

    # Verify:
    # 1. shift both sample and detector to -Y direction by 1.23 mm with SampleToSi value
    # 2. shift the detector position by overwrite-sample-detector distance, i.e., (-1.23 + (1400 - 1250) = 148.77 mm
    assert sample_offset == pytest.approx(-1.23 * 1e-3, 1e-12)
    assert detector_offset == pytest.approx(0.14877, 1e-12)


@pytest.mark.parametrize(
    "generic_IDF",
    [{"Nx": 4, "Ny": 4, "dx": 0.006, "dy": 0.004, "zc": 1.25, "l1": -5.0}],
    indirect=True,
)
def test_overwrite_both_distance(generic_IDF):
    """Test instrument with a user-overwriting both SampleToSi distance and sample to detector distance

    Returns
    -------

    """
    # Generate a generic SANS instrument with detector dimension stated in
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/178
    idf_name = r"/tmp/GenericHFIRSANS_Definition.xml"
    with open(idf_name, "w") as tmp:
        tmp.write(generic_IDF)
        tmp.close()
    ws = LoadEmptyInstrument(
        Filename=tmp.name,
        InstrumentName="GenericSANS",
        OutputWorkspace="test_integration_roi",
    )
    # clean
    os.remove(idf_name)

    # Set up sample logs
    sdd = sample_detector_distance(ws, unit="m")
    print(
        "TestOverwriteSampleDetectorDistance: sample detector distance [1] = {} meter".format(
            sdd
        )
    )

    # Add sample log
    sample_logs = SampleLogs(ws)
    sample_logs.insert("sample-detector-distance", sdd, unit="m")
    default_sample_si_distance = 0.071  # meter, i.e., 71 mm
    # shift sample 1.23 mm to source from silicon window
    sample_logs.insert("Generic:CS:SampleToSi", 72.23, unit="mm")

    # Test method
    sample_offset, detector_offset = get_sample_detector_offset(
        ws,
        "Generic:CS:SampleToSi",
        default_sample_si_distance,
        overwrite_sample_si_distance=75.32 * 1e-3,
        overwrite_sample_detector_distance=1.40,
    )

    # Verify:
    # 1. shift both sample and detector to -Y direction by 1.23 mm with SampleToSi value with real SDD = 1250 mm
    # 2. User-overwriting SampleToSi 75.32 results in a 4.32 mm shift of sample only from nominal position
    #    thus SDD is changed.  SDD = 1250 + (4.32 - 1.23) = 1252.09
    # 3. shift the detector position by overwrite-sample-detector distance, i.e., (-4.32 + (1400 - 1250) = 145.68 mm
    assert sample_offset == pytest.approx(-4.32 * 1e-3, 1e-12)
    assert detector_offset == pytest.approx(0.14568, 1e-12)
