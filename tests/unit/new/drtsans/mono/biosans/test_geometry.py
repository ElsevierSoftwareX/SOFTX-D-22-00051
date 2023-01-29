import pytest
import numpy as np

# https://docs.mantidproject.org/nightly/algorithms/LoadHFIRSANS-v1.html
from mantid.simpleapi import LoadHFIRSANS
from drtsans.mono.biosans.beam_finder import center_detector


def test_api_geometry(biosans_f):

    ws = LoadHFIRSANS(Filename=biosans_f["beamcenter"])

    instrument = ws.getInstrument()
    pos_main = instrument.getComponentByName("detector1").getPos()
    pos_wing = instrument.getComponentByName("wing_detector").getPos()

    # Both detectors are at Y = 0
    assert pos_wing[1] == pos_main[1] == 0.0

    center_x = 0.0014
    center_y = -0.0243
    center_y_gravity = -0.0267

    center_detector(ws, center_x, center_y, center_y_gravity)

    instrument = ws.getInstrument()
    pos_main_2 = instrument.getComponentByName("detector1").getPos()
    pos_wing_2 = instrument.getComponentByName("wing_detector").getPos()

    np.testing.assert_allclose(abs(pos_main[0] - pos_main_2[0]), abs(center_x), rtol=1e-4)
    np.testing.assert_allclose(abs(pos_main[1] - pos_main_2[1]), abs(center_y), rtol=1e-4)
    np.testing.assert_allclose(abs(pos_wing[1] - pos_wing_2[1]), abs(center_y_gravity), rtol=1e-4)
    # Note that after the gravity correction the center Y of the wing detector
    # it's higher than the centre of the main detector
    assert pos_wing_2[1] > pos_main_2[1]


if __name__ == "__main__":
    pytest.main([__file__])
