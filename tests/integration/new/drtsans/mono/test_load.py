# Test load GPSANS and BIOSANS data
import pytest
import os
import numpy as np
from drtsans.mono.load import load_events
from drtsans.mono.meta_data import get_sample_detector_offset
from drtsans.samplelogs import SampleLogs
from drtsans.geometry import sample_detector_distance
from drtsans.load import move_instrument
from mantid.simpleapi import AddSampleLogMultiple, DeleteWorkspace


def test_load_gpsans():
    """Test load GPSANS data

    Returns
    -------

    """
    nexus_file_name = "/HFIR/CG2/IPTS-23801/nexus/CG2_7116.nxs.h5"
    if not os.path.exists(nexus_file_name):
        pytest.skip(
            "Skip due to NeXus file {} is not accessible.".format(nexus_file_name)
        )

    # Load data
    ws = load_events(
        nexus_file_name,
        output_workspace="gptest01",
        overwrite_instrument=True,
        detector_offset=0,
        sample_offset=0,
    )

    # Check current instrument setup and meta data (sample logs)
    logs = SampleLogs(ws)
    print(
        "[TEST INFO] SampleToSi = {} mm".format(
            logs.find_log_with_units("CG2:CS:SampleToSi", unit="mm")
        )
    )
    raw_sample_det_distance = sample_detector_distance(ws, unit="m", search_logs=False)
    print(
        "[TEST INFO] Sample to detector distance = {} /{} meter"
        "".format(
            raw_sample_det_distance,
            sample_detector_distance(
                ws, unit="m", log_key="sample_detector_distance", search_logs=True
            ),
        )
    )

    # sample and detector offsets can only be retrieved from a loaded workspace
    # This is a technical debt
    sample_offset, detector_offset = get_sample_detector_offset(
        ws, "CG2:CS:SampleToSi", 0.0
    )

    assert sample_offset == -0.088
    assert detector_offset == -0.088

    # Move instrument
    # Move sample and detector
    ws = move_instrument(
        ws,
        sample_offset,
        detector_offset,
        is_mono=True,
        sample_si_name="CG2:CS:SampleToSi",
        si_window_to_nominal_distance=0,
    )

    # Verify
    new_sample_det_distance = sample_detector_distance(ws, unit="m", search_logs=False)
    print(
        "[TEST INFO] Sample detector distance after moving = {} meter".format(
            new_sample_det_distance
        )
    )
    print(
        "[TEST INFO] Sample position = {}".format(
            ws.getInstrument().getSample().getPos()
        )
    )

    assert new_sample_det_distance == raw_sample_det_distance

    # cleanup
    DeleteWorkspace(ws)


@pytest.mark.skip(reason="Too large to run on build server")
def test_load_biosans():
    """Test load BIOSANS data

    Returns
    -------

    """
    # Decide to skip data or not
    nexus_file_name = "/HFIR/CG3/IPTS-23782/nexus/CG3_4829.nxs.h5"
    if not os.path.exists(nexus_file_name):
        pytest.skip(
            "Skip due to NeXus file {} is not accessible.".format(nexus_file_name)
        )

    # Load data
    ws = load_events(
        nexus_file_name,
        output_workspace="biotest01",
        overwrite_instrument=True,
        detector_offset=0,
        sample_offset=0,
    )

    # Check current instrument setup and meta data (sample logs)
    logs = SampleLogs(ws)
    print(
        "[TEST INFO] (Raw) sampleToSi = {} mm".format(
            logs.find_log_with_units("CG3:CS:SampleToSi", unit="mm")
        )
    )
    raw_sample_det_distance = sample_detector_distance(ws)
    print(
        "[TEST INFO] (Raw) sample to detector distance = {} /{} meter"
        "".format(
            raw_sample_det_distance,
            sample_detector_distance(
                ws, log_key="sample_detector_distance", search_logs=True
            ),
        )
    )

    # Calculate offset without any overwriting
    # sample and detector offsets can only be retrieved from a loaded workspace
    # This is a technical debt
    sample_offset, detector_offset = get_sample_detector_offset(
        ws, "CG3:CS:SampleToSi", 71.0 * 1e-3
    )
    print(
        "[TEST INFO] Sample offset = {}, Detector offset = {}"
        "".format(sample_offset, detector_offset)
    )

    # Verify: No sample offset from nominal position (origin)
    assert sample_offset == pytest.approx(0.0, 1e-12)
    # Verify: No sample offset
    assert detector_offset == pytest.approx(0.0, 1e-12)
    # Verify: sample position at (0., 0., 0.)
    sample_pos = np.array(ws.getInstrument().getSample().getPos())
    expected_sample_pos = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(sample_pos, expected_sample_pos, atol=1e-12)
    # Verify: sample detector distance is equal to 7.00000019 meter
    sample_det_distance_cal = sample_detector_distance(ws, unit="m", search_logs=False)
    sample_det_distance_meta = sample_detector_distance(ws, unit="mm", search_logs=True)
    assert sample_det_distance_cal == pytest.approx(
        sample_det_distance_meta * 1e-3, 1e-7
    )
    assert sample_det_distance_cal == pytest.approx(7.00000019, 1e-7)

    # cleanup
    DeleteWorkspace(ws)


@pytest.mark.skip(reason="Too large to run on build server")
def test_load_biosans_sample_off_nominal():
    """Test load BIOSANS data with sample position off nominal position

    Returns
    -------

    """
    # Decide to skip data or not
    nexus_file_name = "/HFIR/CG3/IPTS-23782/nexus/CG3_4829.nxs.h5"
    if not os.path.exists(nexus_file_name):
        pytest.skip(
            "Skip due to NeXus file {} is not accessible.".format(nexus_file_name)
        )

    # Load data
    ws = load_events(
        nexus_file_name,
        output_workspace="biotest01",
        overwrite_instrument=True,
        detector_offset=0,
        sample_offset=0,
    )

    # Verify: sample position at (0., 0., 0.)
    sample_pos = np.array(ws.getInstrument().getSample().getPos())
    expected_sample_pos = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(sample_pos, expected_sample_pos, atol=1e-12)
    # Verify: sample detector distance is equal to 7.00000019 meter
    sample_det_distance_cal = sample_detector_distance(ws, unit="m", search_logs=False)
    sample_det_distance_meta = sample_detector_distance(ws, unit="mm", search_logs=True)
    assert sample_det_distance_cal == pytest.approx(
        sample_det_distance_meta * 1e-3, 1e-7
    )
    assert sample_det_distance_cal == pytest.approx(7.00000019, 1e-7)

    # Second test on SampleToSi distance other than 71.00 mm
    # Simulate sample log CG3:CS:SampleToSi
    # Note: this is not overwriting SampleSiDistance
    test_sample_si_distance = 74.21
    AddSampleLogMultiple(
        Workspace=ws,
        LogNames="{}".format("CG3:CS:SampleToSi"),
        LogValues="{}".format(test_sample_si_distance),
        LogUnits="mm",
    )

    # Calculate offset without any overwriting
    sample_offset, detector_offset = get_sample_detector_offset(
        ws, "CG3:CS:SampleToSi", 71.0 * 1e-3
    )
    print(
        "[TEST INFO 2] Sample offset = {}, Detector offset = {}"
        "".format(sample_offset, detector_offset)
    )

    # Both sample and detector shall move toward souce (-Y direction) with (74.21 - 71.) = 3.21 mm
    assert sample_offset == pytest.approx(-0.00321, 1e-12)
    assert detector_offset == pytest.approx(-0.00321, 1e-12)

    # Move sample and detector
    ws = move_instrument(
        ws,
        sample_offset,
        detector_offset,
        is_mono=True,
        sample_si_name="CG3:CS:SampleToSi",
        si_window_to_nominal_distance=0,
    )

    # Verify: sample position at (0., 0., -0.00321)
    sample_pos = np.array(ws.getInstrument().getSample().getPos())
    expected_sample_pos = np.array([0.0, 0.0, -0.00321])
    np.testing.assert_allclose(sample_pos, expected_sample_pos, atol=1e-12)

    # Verify the sample detector distance which shall be same as raw meta data
    sample_det_distance_cal = sample_detector_distance(ws, unit="m", search_logs=False)
    sample_det_distance_meta = sample_detector_distance(ws, unit="mm", search_logs=True)
    assert sample_det_distance_cal == pytest.approx(
        sample_det_distance_meta * 1e-3, 1e-7
    )
    assert sample_det_distance_cal == pytest.approx(7.00000019, 1e-7)

    # cleanup
    DeleteWorkspace(ws)


@pytest.mark.skip(reason="Too large to run on build server")
def test_load_biosans_overwrite_swd():
    """Test load BIOSANS data with overwriting sample Si window distance

    Returns
    -------

    """
    # Decide to skip data or not
    nexus_file_name = "/HFIR/CG3/IPTS-23782/nexus/CG3_4829.nxs.h5"
    if not os.path.exists(nexus_file_name):
        pytest.skip(
            "Skip due to NeXus file {} is not accessible.".format(nexus_file_name)
        )

    # Load data
    ws = load_events(
        nexus_file_name,
        output_workspace="biotest02",
        overwrite_instrument=True,
        detector_offset=0,
        sample_offset=0,
    )

    # Calculate offset with overwriting to sample-detector-distance
    sample_offset, detector_offset = get_sample_detector_offset(
        ws, "CG3:CS:SampleToSi", 71.0 * 1e-3, overwrite_sample_si_distance=0.07421
    )
    print(
        "[TEST INFO] Sample offset = {}, Detector offset = {}"
        "".format(sample_offset, detector_offset)
    )

    # Move sample and detector
    ws = move_instrument(
        ws,
        sample_offset,
        detector_offset,
        is_mono=True,
        sample_si_name="CG3:CS:SampleToSi",
        si_window_to_nominal_distance=0.071,
    )

    # Verify: sample position at (0., 0., -0.00321) because SampleToSi is overwritten to 74.21 mm
    sample_pos = np.array(ws.getInstrument().getSample().getPos())
    expected_sample_pos = np.array([0.0, 0.0, -0.00321])
    np.testing.assert_allclose(sample_pos, expected_sample_pos, atol=1e-12)

    # Verify the sample detector distance shall be increased by 3.21mm due to the shift of sample position
    sample_det_distance_cal = sample_detector_distance(ws, unit="m", search_logs=False)
    assert sample_det_distance_cal == pytest.approx(7.00321, 1e-7)
    # verify the values from calculated and from meta data are identical
    sample_det_distance_meta = sample_detector_distance(ws, unit="mm", search_logs=True)
    assert sample_det_distance_cal == pytest.approx(
        sample_det_distance_meta * 1e-3, 1e-7
    )
    # verify that SampleToSi is overwritten to 74.21 mm
    logs = SampleLogs(ws)
    swd = logs.find_log_with_units("CG3:CS:SampleToSi", unit="mm")
    assert swd == pytest.approx(74.21, 1e-10)

    # cleanup
    DeleteWorkspace(ws)


@pytest.mark.skip(reason="Too large to run on build server")
def test_load_biosans_overwrite_sdd():
    """Test load BIOSANS data with overwriting sample detector distance related meta data

    Returns
    -------

    """
    # Decide to skip data or not
    nexus_file_name = "/HFIR/CG3/IPTS-23782/nexus/CG3_4829.nxs.h5"
    if not os.path.exists(nexus_file_name):
        pytest.skip(
            "Skip due to NeXus file {} is not accessible.".format(nexus_file_name)
        )

    # Load data
    ws = load_events(
        nexus_file_name,
        output_workspace="biotest02",
        overwrite_instrument=True,
        detector_offset=0,
        sample_offset=0,
    )

    # Check current instrument setup and meta data (sample logs)
    logs = SampleLogs(ws)
    print(
        "[TEST INFO] SampleToSi = {} mm".format(
            logs.find_log_with_units("CG3:CS:SampleToSi", unit="mm")
        )
    )
    raw_sample_det_distance = sample_detector_distance(ws)
    print(
        "[TEST INFO] Sample to detector distance = {} /{} meter"
        "".format(
            raw_sample_det_distance,
            sample_detector_distance(
                ws, log_key="sample_detector_distance", search_logs=True
            ),
        )
    )

    # Calculate offset with overwriting to sample-detector-distance
    sample_offset, detector_offset = get_sample_detector_offset(
        ws, "CG3:CS:SampleToSi", 71.0 * 1e-3, overwrite_sample_detector_distance=7.1234
    )
    print(
        "[TEST INFO] Sample offset = {}, Detector offset = {}"
        "".format(sample_offset, detector_offset)
    )

    # Move sample and detector
    ws = move_instrument(
        ws,
        sample_offset,
        detector_offset,
        is_mono=True,
        sample_si_name="CG3:CS:SampleToSi",
        si_window_to_nominal_distance=0,
    )

    # Verify: sample position at (0., 0., 0.) because SampleToSi == 71 mm and not overwritten
    sample_pos = np.array(ws.getInstrument().getSample().getPos())
    expected_sample_pos = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(sample_pos, expected_sample_pos, atol=1e-12)

    # Verify the sample detector distance which shall be same as raw meta data
    sample_det_distance_cal = sample_detector_distance(ws, unit="m", search_logs=False)
    assert sample_det_distance_cal == pytest.approx(7.1234, 1e-7)
    # verify the values from calculated and from meta data are identical
    sample_det_distance_meta = sample_detector_distance(ws, unit="mm", search_logs=True)
    assert sample_det_distance_cal == pytest.approx(
        sample_det_distance_meta * 1e-3, 1e-7
    )

    # cleanup
    DeleteWorkspace(ws)


if __name__ == "__main__":
    pytest.main([__file__])
