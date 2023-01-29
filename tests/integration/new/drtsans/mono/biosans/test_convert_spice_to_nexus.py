import pytest
import os
import numpy as np

from mantid.api import AnalysisDataService
from drtsans.mono.biosans.cg3_spice_to_nexus import convert_spice_to_nexus
from mantid.simpleapi import LoadEventNexus, LoadHFIRSANS, DeleteWorkspace


@pytest.mark.skipif(
    not os.path.exists("/HFIR/HB2B/shared/autoreduce/"), reason="On build server"
)
def test_convert_spice(reference_dir, generatecleanfile, clean_workspace):
    """
    Test converting BIOSANS SPICE file to event Nexus
    """
    # Set file
    ipts = 17241
    exp = 402
    scan_pt_list = [(6, 1)]

    # Create output directory
    output_dir = generatecleanfile(prefix="cg3spiceconverter")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    temp_event_nexus = "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/biosans/CG3_5705.nxs.h5"
    nexus_files = list()
    for scan_num, pt_num in scan_pt_list:
        fake_nexus = convert_spice_to_nexus(
            ipts,
            exp,
            scan_num,
            pt_num,
            temp_event_nexus,
            masked_detector_pixels=[70911],
            output_dir=output_dir,
            spice_dir=reference_dir.new.biosans,
        )
        nexus_files.append(fake_nexus)

    # Verify result
    raw_spice = os.path.join(
        reference_dir.new.biosans, "BioSANS_exp402_scan0006_0001.xml"
    )
    verify_result(nexus_files[0], raw_spice, [70911], clean_workspace)


def verify_result(test_nexus, raw_spice, masked_pixels, clean_workspace):
    # Load data
    test_ws = LoadEventNexus(
        Filename=test_nexus,
        OutputWorkspace="test2",
        NumberOfBins=1,
        LoadNexusInstrumentXML=True,
    )
    raw_ws = LoadHFIRSANS(Filename=raw_spice, OutputWorkspace="raw")

    clean_workspace(test_ws)
    clean_workspace(raw_ws)

    # Compare counts
    assert (
        test_ws.getNumberHistograms() + 2 == raw_ws.getNumberHistograms()
    ), "Spectra number unmatched"

    # Compare counts
    # NOTE:
    #   In NeXus, the first two spectra are monitor counts, hence we need
    #   to compare
    #   - nexus_spectrum[2:]
    #   - reference_spectrum[:]
    raw_y = raw_ws.extractY().flatten()
    test_y = test_ws.extractY().flatten()

    # check masked pixels
    for pid in masked_pixels:
        assert test_y[masked_pixels] == 0
        # reset back to original count
        test_y[masked_pixels] = raw_y[2 + pid]
    # check the rest pixels' counts
    # spice spectra v nexus spectra
    # tube 1 <--> tube 1 (first tube in the front)
    # tube 2 <--> tube 5 (first tube in the back)
    np.testing.assert_allclose(raw_y[2 : 256 + 2], test_y[:256])
    np.testing.assert_allclose(raw_y[256 + 2 : 512 + 2], test_y[4 * 256 : 5 * 256])

    if AnalysisDataService.doesExist("BioSANS_exp402_scan0006_0001"):
        DeleteWorkspace("BioSANS_exp402_scan0006_0001")


if __name__ == "__main__":
    pytest.main([__file__])
