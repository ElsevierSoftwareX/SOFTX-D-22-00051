import pytest
import os
import numpy as np
import math
from mantid.simpleapi import LoadHFIRSANS, LoadEventNexus
from drtsans.mono.biosans.cg3_spice_to_nexus import convert_spice_to_nexus  # noqa: E401


def test_benchmark_spice(reference_dir):
    """Test the benchmark SPICE file that is created to expose and verify the pixel mapping issue
    between old BIOSANS IDF and new BIOSANS IDF
    """
    # Access the test spice file
    spice_name = os.path.join(
        reference_dir.new.biosans, "BioSANS_exp549_scan0020_0001_benchmark.xml"
    )
    assert os.path.exists(spice_name)

    # Load data
    spice_ws = LoadHFIRSANS(
        Filename=spice_name, OutputWorkspace="CG3_5490020001_Benchmark"
    )
    assert spice_ws

    # Test geometry
    shift_pixels = 127  # this is from the benchmark setup

    # main detector: 192 tubes
    main_det_tuple = list()
    for itube in range(192):
        det_id = 2 + itube * 256 + shift_pixels
        # get position X
        det_pos_x = spice_ws.getDetector(det_id).getPos().X()
        det_pos_y = spice_ws.getDetector(det_id).getPos().Y()
        # get count
        count = int(spice_ws.readY(det_id)[0])
        main_det_tuple.append((det_pos_x, det_pos_y, count))

    # Verify the positions
    # sort
    main_det_tuple.sort(reverse=True)
    # split
    pos_x_list, pos_y_list, count_list = zip(*main_det_tuple)

    # x position: shall be linear decreasing with constant step: from positive X to negative X
    pos_x_array = np.array(pos_x_list)
    pixel_distance_array = pos_x_array[1:] - pos_x_array[:-1]
    assert pixel_distance_array.mean() == pytest.approx(-0.0055)
    assert pixel_distance_array.std() < 5e-17

    # y position: shall be same
    pos_y_array = np.array(pos_y_list)
    assert pos_y_array.std() < 1e-17
    assert pos_y_array.mean() == pytest.approx(-0.00215)

    # counts
    assert np.allclose(np.array(count_list), np.arange(1, 192 + 1))

    # Wing detector: 160 tubes
    main_det_tuple = list()
    for itube in range(160):
        det_id = 2 + (192 + itube) * 256 + shift_pixels
        # get position X
        det_pos_x = spice_ws.getDetector(det_id).getPos().X()
        det_pos_y = spice_ws.getDetector(det_id).getPos().Y()
        # get count
        count = int(spice_ws.readY(det_id)[0])
        main_det_tuple.append((det_pos_x, det_pos_y, count))

    # Verify the positions
    # sort
    main_det_tuple.sort(reverse=True)
    # split
    pos_x_list, pos_y_list, count_list = zip(*main_det_tuple)

    # x position: shall be linear decreasing with constant step: from positive X to negative X
    pos_x_array = np.array(pos_x_list)
    pixel_distance_array = pos_x_array[1:] - pos_x_array[:-1]
    # always to the right but since it is curved not constant
    assert (len(pixel_distance_array[pixel_distance_array >= 0])) == 0

    # y position: shall be same
    pos_y_array = np.array(pos_y_list)
    assert pos_y_array.std() < 1e-17
    assert pos_y_array.mean() == pytest.approx(-0.00215)

    # counts
    assert np.allclose(np.array(count_list), np.arange(1, 160 + 1) * 1000)


def test_spice_conversion(reference_dir, cleanfile):
    """Test conversion from SPICE to NeXus with pixel ID mapping to new IDF"""
    # Access the test spice file
    spice_name = os.path.join(
        reference_dir.new.biosans, "BioSANS_exp549_scan0020_0001_benchmark.xml"
    )
    assert os.path.exists(spice_name)

    #
    template_event_nexus = "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/biosans/CG3_5705.nxs.h5"

    # output
    test_temp_dir = "/tmp/test_cg3_spice_geom"
    if os.path.exists(test_temp_dir) is False:
        os.mkdir(test_temp_dir)

    cleanfile(test_temp_dir)

    # Convert
    nexus = convert_spice_to_nexus(
        1,
        549,
        20,
        1,
        template_event_nexus,
        masked_detector_pixels=[],
        output_dir=test_temp_dir,
        spice_data=spice_name,
    )
    assert os.path.exists(nexus)

    # Test
    # Load data: must use new IDF
    nexus_ws = LoadEventNexus(
        Filename=nexus,
        OutputWorkspace="CG3_5490020001_NeXus",
        LoadNexusInstrumentXML=True,
        NumberOfBins=1,
    )
    assert nexus_ws
    assert nexus_ws.getNumberHistograms() == (192 + 160) * 256

    # Test geometry
    shift_pixels = 127  # this is from the benchmark setup

    # main detector: 192 tubes
    main_det_tuple = list()
    for itube in range(192):
        det_id = itube * 256 + shift_pixels
        # get position X
        det_pos_x = nexus_ws.getDetector(det_id).getPos().X()
        det_pos_y = nexus_ws.getDetector(det_id).getPos().Y()
        # get count
        count = int(nexus_ws.readY(det_id)[0])
        main_det_tuple.append((det_pos_x, det_pos_y, count, det_id))

    # Verify the positions
    # sort
    main_det_tuple.sort(reverse=True)
    # split
    pos_x_list, pos_y_list, count_list, det_id_list = zip(*main_det_tuple)

    # x position: shall be linear decreasing with constant step: from positive X to negative X
    pos_x_array = np.array(pos_x_list)
    pixel_distance_array = pos_x_array[1:] - pos_x_array[:-1]
    assert len(pixel_distance_array[pixel_distance_array >= 0]) == 0, f"{pos_x_array}"

    # y position: shall be same
    pos_y_array = np.array(pos_y_list)
    assert pos_y_array.std() < 1e-17
    # assert pos_y_array.mean() == pytest.approx(-0.00215)

    # counts shall start from 1, from left to right, to 192
    np.testing.assert_allclose(np.array(count_list), np.arange(1, 192 + 1))

    # Wing detector: 160 tubes
    main_det_tuple = list()
    for itube in range(160):
        det_id = (192 + itube) * 256 + shift_pixels
        # get 2theta and Y
        det_pos = nexus_ws.getDetector(det_id).getPos()
        pos_x = det_pos.X()
        pos_z = det_pos.Z()
        det_2theta = math.atan(abs(pos_x) / pos_z) * 180.0 / math.pi
        det_pos_y = det_pos.Y()
        # get count
        count = int(nexus_ws.readY(det_id)[0])
        # will sort by counts
        main_det_tuple.append((count, det_2theta, det_pos_y, det_id))

    # Verify the positions
    # sort by counts (presumably from left to right, i.e., from close to beam and rotate to -z direction)
    main_det_tuple.sort(reverse=False)
    # split
    count_list, pos_2theta_list, pos_y_list, det_id_list = zip(*main_det_tuple)

    # y position: shall be same
    pos_y_array = np.array(pos_y_list)
    assert pos_y_array.std() < 1e-17

    # 2theta position: shall increase
    two_theta_array = np.array(pos_2theta_list)
    delta_2theta_array = two_theta_array[1:] - two_theta_array[:-1]
    assert len(delta_2theta_array[delta_2theta_array <= 0]) == 0

    # counts
    test_counts = np.array(count_list)
    expected_counts = np.arange(1, 160 + 1) * 1000
    np.testing.assert_allclose(test_counts, expected_counts)


if __name__ == "__main__":
    pytest.main([__file__])
