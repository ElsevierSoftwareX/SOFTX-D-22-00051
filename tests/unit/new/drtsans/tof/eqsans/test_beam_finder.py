import numpy as np
import pytest
from pytest import approx
from mantid.simpleapi import LoadEventNexus
from drtsans.settings import unique_workspace_dundername as uwd
from drtsans.settings import amend_config
from drtsans.tof.eqsans import center_detector, find_beam_center


def test_find_beam_center(reference_dir):
    r"""
    Test for finding the beam center for EQSANS on real data

    Functions to test: drtsans.tof.eqsans.beam_finder.find_beam_center

    Underlying Mantid algorithms:
        FindCenterOfMassPosition https://docs.mantidproject.org/nightly/algorithms/FindCenterOfMassPosition-v2.html
    """
    with amend_config(data_dir=reference_dir.new.eqsans):
        w = LoadEventNexus(Filename="EQSANS_92160.nxs.h5", OutputWorkspace=uwd())
    assert find_beam_center(w)[:-1] == approx((0.02997, 0.01379), abs=1e-3)


def test_center_detector(reference_dir):
    r"""
    Translate detector according to method center_of_mass

    Functions to test: drtsans.tof.eqsans.beam_finder.find_beam_center
                       drtsans.tof.eqsans.beam_finder.center_detector

    Underlying Mantid algorithms:
        FindCenterOfMassPosition https://docs.mantidproject.org/nightly/algorithms/FindCenterOfMassPosition-v2.html
        MoveInstrumentComponent https://docs.mantidproject.org/nightly/algorithms/MoveInstrumentComponent-v1.html
    """
    with amend_config(data_dir=reference_dir.new.eqsans):
        w = LoadEventNexus(Filename="EQSANS_92160.nxs.h5", OutputWorkspace=uwd())
    r = find_beam_center(w, method="center_of_mass")
    assert r[:-1] == approx((0.02997, 0.0138), abs=1e-3)
    pos_old = w.getInstrument().getDetector(0).getPos()
    expected = np.array([0.52164, -0.54785, -0.02559])
    assert pos_old == approx(expected, abs=1e-5)
    center_detector(w, r[0], r[1])
    pos_new = w.getInstrument().getDetector(0).getPos()
    assert pos_new == approx(expected + np.array([-r[0], -r[1], 0]), abs=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
