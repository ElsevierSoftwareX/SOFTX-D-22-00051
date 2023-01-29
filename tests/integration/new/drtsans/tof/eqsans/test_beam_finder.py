from tempfile import NamedTemporaryFile
import pytest
from pytest import approx

# https://docs.mantidproject.org/nightly/algorithms/ClearMaskFlag-v1.html
# https://docs.mantidproject.org/nightly/algorithms/ExtractMaskMask-v1.html
# https://docs.mantidproject.org/nightly/algorithms/SaveMask-v1.html
from mantid.simpleapi import ClearMaskFlag, ExtractMask, SaveMask, DeleteWorkspace
from drtsans.settings import unique_workspace_dundername as uwd
from drtsans.tof.eqsans import (
    apply_mask,
    center_detector,
    find_beam_center,
    load_events,
)


# eqsans_f and eqsans_p are defined in tests/conftest.py. Currently  them beamcenter file is EQSANS_68183
def test_find_beam_center(eqsans_f, eqsans_p):
    r"""
    Integration test to find the location on the detector where
    the beam impinges

    1. Apply mask
    2. Find the beam center
    """
    ws = load_events(eqsans_f["beamcenter"], output_workspace=uwd())
    #
    # Find the beam center
    #
    assert find_beam_center(ws)[:-1] == approx((0.02651957, 0.01804375), abs=1e-04)
    #
    # Find the beam center with a mask workspace
    #
    apply_mask(ws, Tube=eqsans_p["tubes_to_mask"])
    x0, y0, _ = find_beam_center(ws)
    mask_ws = ExtractMask(ws, OutputWorkspace=uwd()).OutputWorkspace
    ClearMaskFlag(ws)
    assert find_beam_center(ws, mask=mask_ws)[:-1] == approx((x0, y0))
    #
    # Find the beam center with a mask file
    #
    ClearMaskFlag(ws)
    with NamedTemporaryFile(delete=True, suffix=".xml") as f:
        SaveMask(InputWorkspace=mask_ws, OutputFile=f.name)
        xy = find_beam_center(ws, mask=f.name)
        assert xy[:-1] == approx((x0, y0), abs=1e-04)
    #
    # Let's move the beam center to the intersection point between the Z-axis
    # and the detector. The new (x, y) coordinates for the beam center
    # should be (0, 0) now.
    #
    center_detector(ws, center_x=x0, center_y=y0)
    assert find_beam_center(ws)[:-1] == pytest.approx((0, 0), abs=1e-04)
    #
    DeleteWorkspace(ws)


if __name__ == "__main__":
    pytest.main([__file__])
