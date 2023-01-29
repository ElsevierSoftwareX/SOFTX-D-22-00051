import pytest
from pathlib import Path

from drtsans.load import load_events
from drtsans.settings import unique_workspace_dundername
from mantid.simpleapi import DeleteWorkspace


class TestLoadEvents:
    def test_pixel_calibration(self, reference_dir):
        r"""Check the pixel calibration is applied to a workspace upon loading"""
        file_name = str(
            Path(reference_dir.new.sans) / "pixel_calibration" / "CG2_8508.nxs.h5"
        )
        workspace = load_events(
            file_name,
            pixel_calibration=True,
            output_workspace=unique_workspace_dundername(),
        )
        component_info = workspace.componentInfo()
        component_info_index = 42  # identifies some detector pixel

        # Assert pixel width is calibrated
        nominal_pixel_width = (
            component_info.shape(component_info_index).getBoundingBox().width().X()
        )
        assert nominal_pixel_width == pytest.approx(
            0.00804, abs=1.0e-5
        )  # uncalibrated width
        pixel_width = component_info.scaleFactor(42).X() * nominal_pixel_width
        assert pixel_width == pytest.approx(0.00968, abs=1.0e-5)  # calibrated width

        # Assert pixel height is calibrated
        nominal_pixel_height = (
            component_info.shape(component_info_index).getBoundingBox().width().Y()
        )
        assert nominal_pixel_height == pytest.approx(
            0.00409, abs=1.0e-5
        )  # uncalibrated width
        pixel_height = component_info.scaleFactor(42).X() * nominal_pixel_height
        assert pixel_height == pytest.approx(0.00492, abs=1.0e-5)  # calibrated width

        # NOTE:
        # It is unclear where the following two workspaces are loaded/created and
        # whether they should be allowed to be persistent in memory.
        # For testing purpose, we are deleteing them.
        # barscan_GPSANS_detector1_20200103:	0.393216 MB
        # tubewidth_GPSANS_detector1_20200130:	0.393216 MB
        DeleteWorkspace("barscan_GPSANS_detector1_20200103")
        DeleteWorkspace("tubewidth_GPSANS_detector1_20200130")


if __name__ == "__main__":
    pytest.main([__file__])
