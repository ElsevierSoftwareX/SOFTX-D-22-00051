#!/usr/bin/env python
import pytest
from mantid import mtd

# https://docs.mantidproject.org/nightly/algorithms/LoadHFIRSANS-v1.html
from mantid.simpleapi import LoadHFIRSANS
from drtsans.settings import unique_workspace_dundername
from drtsans.mono.gpsans import center_detector, find_beam_center


# the data for this test is defined in tests/conftest.py and is currently CG2_exp325_scan0020_0001.xml
def test_beam_finder(gpsans_f):
    ws = unique_workspace_dundername()
    LoadHFIRSANS(Filename=gpsans_f["beamcenter"], OutputWorkspace=ws)

    x, y, _ = find_beam_center(ws)
    print("Beam center found = ({:.3}, {:.3}) meters.".format(x, y))

    assert x == pytest.approx(0.02201, abs=1e-4)
    assert y == pytest.approx(-0.0193, abs=1e-4)

    # Let's center the instrument and get the new center: It should be 0 after
    # the re-centring
    center_detector(ws, x, y)
    x, y, _ = find_beam_center(ws)

    # Tolerance 1e-3 == milimeters
    assert x == pytest.approx(0.0, abs=1e-4)
    assert y == pytest.approx(0.0, abs=1e-4)

    mtd.remove(ws)


if __name__ == "__main__":
    pytest.main([__file__])
