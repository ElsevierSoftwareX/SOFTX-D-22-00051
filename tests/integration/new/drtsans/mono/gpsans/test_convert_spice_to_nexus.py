import pytest
import os
import numpy as np
from drtsans.mono.gpsans.cg2_spice_to_nexus import convert_spice_to_nexus
from mantid.simpleapi import DeleteWorkspace
from mantid.simpleapi import LoadEventNexus
from mantid.simpleapi import LoadHFIRSANS


def test_convert_spice(reference_dir, generatecleanfile):
    """
    Test converting GPSANS SPICE file to event Nexus
    """
    # Set file
    ipts = 828
    exp = 280
    scan_pt_list = [(12, 1)]

    # Create output directory
    output_dir = generatecleanfile(prefix="cg2spiceconverter")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    temp_event_nexus = (
        "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/CG2_9177.nxs.h5"
    )
    nexus_files = list()
    for scan_num, pt_num in scan_pt_list:
        fake_nexus = convert_spice_to_nexus(
            ipts,
            exp,
            scan_num,
            pt_num,
            temp_event_nexus,
            output_dir=output_dir,
            spice_dir=reference_dir.new.gpsans,
        )
        nexus_files.append(fake_nexus)

    # Verify result
    raw_spice = os.path.join(reference_dir.new.gpsans, "CG2_exp280_scan0012_0001.xml")
    verify_result(nexus_files[0], raw_spice)

    # NOTE:
    # mysterious leftover workspace
    # CG2_exp280_scan0012_0001:	1.20332 MB
    DeleteWorkspace("CG2_exp280_scan0012_0001")


def verify_result(test_nexus, raw_spice):
    # Load data
    test_ws = LoadEventNexus(
        Filename=test_nexus, OutputWorkspace="test2", NumberOfBins=1
    )
    raw_ws = LoadHFIRSANS(Filename=raw_spice, OutputWorkspace="raw")

    # Compare counts
    assert (
        test_ws.getNumberHistograms() + 2 == raw_ws.getNumberHistograms()
    ), "Spectra number unmatched"

    # Compare counts
    raw_y = raw_ws.extractY().flatten()
    test_y = test_ws.extractY().flatten()
    np.testing.assert_allclose(raw_y[2:], test_y)

    # Compare geometry
    for iws in range(0, test_ws.getNumberHistograms(), 20):
        nexus_det_pos = test_ws.getDetector(iws).getPos()
        spice_det_pos = raw_ws.getDetector(iws + 2).getPos()
        np.testing.assert_allclose(nexus_det_pos, spice_det_pos)

    # Cleanup
    DeleteWorkspace(test_ws)
    DeleteWorkspace(raw_ws)


if __name__ == "__main__":
    pytest.main([__file__])
