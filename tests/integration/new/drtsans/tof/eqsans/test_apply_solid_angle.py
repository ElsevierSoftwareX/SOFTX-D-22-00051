import pytest
from drtsans.tof import eqsans
from mantid.simpleapi import SolidAngle
from mantid.simpleapi import DeleteWorkspace
from mantid.kernel import V3D
import numpy as np
from copy import deepcopy


@pytest.mark.parametrize(
    "generic_workspace",
    [{"Nx": 3, "Ny": 3, "dx": 0.00425, "dy": 0.0055, "xc": 0.32, "yc": -0.16}],
    indirect=True,
)
def test_solid_angle(generic_workspace):
    """Tests calculating and applying the correction for
    the solid angle of a pixel as described in the master
    document section 3.5
    dev - Steven Hahn <hahnse@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>
    """
    # generate a generic SANS instrument with a pixel of
    # the size and position specified in
    # sans-backend/documents/Master_document_022219.pdf
    ws = generic_workspace  # friendlier name to type

    # set intensity and error to match test document
    ws.dataY(4)[0] = 156.0
    ws.dataE(4)[0] = np.sqrt(156.0)

    assert ws.dataY(4)[0] == 156.0
    assert ws.dataE(4)[0] == np.sqrt(156.0)

    d_info = ws.detectorInfo()
    s_info = ws.spectrumInfo()

    # confirm geometry matches master document
    # beam center position c
    # c = [0., 0., 5.000]
    # sample position s
    s = s_info.samplePosition()
    assert s == V3D(0.0, 0.0, 0.0)
    # pixel position r
    r = d_info.position(4)
    assert r == V3D(0.320, -0.160, 5.0)
    # angle between beam center and pixel 2*theta
    cos_two_theta = np.cos(d_info.twoTheta(4))
    assert cos_two_theta == pytest.approx(0.9974497886)

    # create vector to the tube of the pixel, b
    b = deepcopy(r)
    b[1] = 0.0
    # angle between b and c
    cos_alpha = b.scalar_prod(r) / (b.norm() * r.norm())
    assert cos_alpha == pytest.approx(0.9994904783)

    # calculate solid angle with Mantid and verify result
    ws2 = SolidAngle(InputWorkspace=str(ws), Method="VerticalTube")
    assert ws2.dataY(4)[0] == pytest.approx(9.2976320958e-07)

    # calculate and apply solid angle correction to workspace
    # and verify result
    ws = eqsans.apply_solid_angle_correction(str(ws))

    assert ws.dataY(4)[0] == pytest.approx(167784655.70)
    assert ws.dataE(4)[0] == pytest.approx(13433523.577)

    # clean up
    DeleteWorkspace(ws)
    DeleteWorkspace(ws2)


if __name__ == "__main__":
    pytest.main([__file__])
