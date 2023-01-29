import pytest
import os
from os.path import join as path_join
from mantid.simpleapi import LoadEventNexus
from drtsans.settings import unique_workspace_dundername
from drtsans.geometry import sample_detector_distance


def test_translated_gpsans(reference_dir):
    """Test sample detector (plane) distance for GPSANS"""
    # Get test data
    test_nexus_file = path_join(
        reference_dir.new.gpsans, "Exp280/CG2_028000090001.nxs.h5"
    )
    assert os.path.join(test_nexus_file)

    # Load data
    workspace = LoadEventNexus(
        Filename=test_nexus_file,
        OutputWorkspace=unique_workspace_dundername(),
        LoadLogs=True,
    )
    assert workspace

    # Verify calculated SDD against SDD in meta data
    calculated_sdd = sample_detector_distance(
        workspace, unit="m", log_key=None, search_logs=False
    )
    das_sdd = sample_detector_distance(
        workspace, unit="m", log_key=None, search_logs=True, forbid_calculation=True
    )
    assert abs(calculated_sdd - das_sdd) == pytest.approx(
        0, rel=1e-7
    ), f"{das_sdd} vs {calculated_sdd}"

    # Verify calculated SDD against pixel positions
    ll_det_pos = workspace.getDetector(0).getPos()
    ur_det_pos = workspace.getDetector(192 * 256 - 1).getPos()
    expected_sdd = (ll_det_pos.Z() + ur_det_pos.Z()) * 0.5
    assert abs(calculated_sdd - expected_sdd) == pytest.approx(
        0, abs=1e-7
    ), f"{calculated_sdd}, {expected_sdd}"


if __name__ == "__main__":
    pytest.main([__file__])
