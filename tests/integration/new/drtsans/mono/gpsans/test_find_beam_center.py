"""
    Test EASANS sensitivities preparation algorithm
"""
import pytest
import os
import numpy as np
from drtsans.mono.gpsans import prepare_data, find_beam_center
from mantid.simpleapi import DeleteWorkspace


def test_gpsans_find_beam_center():
    """Integration test on algorithm to find beam center for GPSANS

    Returns
    -------

    """
    # Check data mount to decide to skip or not
    if not os.path.exists("/HFIR/CG2/IPTS-23801/nexus/CG2_8148.nxs.h5"):
        pytest.skip(
            "Testing file /HFIR/CG2/IPTS-23801/nexus/CG2_8148.nxs.h5 cannot be accessed"
        )

    # Load data
    beam_center_ws = prepare_data(
        data="/HFIR/CG2/IPTS-23801/nexus/CG2_8148.nxs.h5",
        btp={"Pixel": "1-8,249-256"},
        detector_offset=0,
        sample_offset=0,
        center_x=0,
        center_y=0,
        flux_method="monitor",
        solid_angle=False,
        output_workspace="BC_8148",
        sample_thickness=0.1,
    )

    # Find beam center
    beam_center = find_beam_center(beam_center_ws)

    # Get detector center
    instrument = beam_center_ws.getInstrument()
    det = instrument.getComponentByName("detector1")
    det_center = det.getPos()

    # Calculate shift:
    center_x, center_y, _ = beam_center
    beam_center_shift = np.sqrt(
        (center_x - det_center[0]) ** 2 + (center_y - det_center[1]) ** 2
    )

    assert beam_center_shift == pytest.approx(
        0.400, abs=0.007
    ), "Beam center shift {} to {} is beyond" "0.4 +/- 7E-3".format(
        beam_center, det_center
    )

    # cleanup
    DeleteWorkspace(beam_center_ws)


if __name__ == "__main__":
    pytest.main([__file__])
