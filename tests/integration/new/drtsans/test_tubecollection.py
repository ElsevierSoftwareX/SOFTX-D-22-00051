import numpy as np
import pytest
import time


r""" Hyperlinks to mantid algorithms
LoadEmptyInstrument <https://docs.mantidproject.org/nightly/algorithms/LoadEmptyInstrument-v1.html>
"""
from mantid.simpleapi import LoadEmptyInstrument

r"""
Hyperlinks to drtsans functions
ElementComponentInfo, PixelInfo, TubeInfo, TubeCollection <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tubecollection.py>
"""  # noqa: E501
from drtsans.settings import namedtuplefy
from drtsans.tubecollection import TubeCollection
from mantid.simpleapi import DeleteWorkspace
from unittest import TestCase


@namedtuplefy
def collection():
    r"""BIOSANS instrument with a run containing few events"""
    workspace = LoadEmptyInstrument(InstrumentName="BIOSANS")
    return {
        "main": TubeCollection(workspace, "detector1"),
        "wing": TubeCollection(workspace, "wing_detector"),
    }


class TestTubeCollection(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collection = collection()

    @classmethod
    def tearDownClass(cls):
        DeleteWorkspace("workspace")

    def test_tubes(self):
        collection = self.collection
        assert len(collection.main) == 192
        assert len(collection.wing) == 160

    def test_getitem(self):
        collection = self.collection
        assert collection.main[0][0].position == pytest.approx(
            [0.53, -0.52, 0.00], abs=0.01
        )  # first pixel
        assert collection.main[-1][-1].position == pytest.approx(
            [-0.53, 0.52, 0.00], abs=0.01
        )  # last pixel

    def test_sorted(self):
        collection = self.collection
        # Sort by decreasing tube position along the X-axis
        sorted_tubes = collection.main.sorted(view="decreasing X")
        x_coords = [
            tube.position[0] for tube in sorted_tubes
        ]  # X coord for the center of each tube
        assert np.all(x_coords[1:] < x_coords[:-1])  # x_coords strictly decreasing
        # Sort by increasing spectrum index
        sorted_tubes = collection.main.sorted(view="workspace index")
        spectrum_info_indexes = [tube.spectrum_info_index[0] for tube in sorted_tubes]
        assert np.all(spectrum_info_indexes[1:] > spectrum_info_indexes[:-1])

    def test_detector_ids(self):
        collection = self.collection
        #
        tubes = collection.main.tubes
        start_time = time.time()
        detector_ids = [tube.detector_ids for tube in tubes]
        assert time.time() - start_time < 0.2  # below one tenth of a second
        assert (detector_ids[0][0], detector_ids[-1][-1]) == (0, 49151)

    def test_pixel_heights(self):
        collection = self.collection
        #
        tubes = collection.main.tubes
        start_time = time.time()
        heights = [tube.pixel_heights for tube in tubes]
        assert time.time() - start_time < 2.0  # below one second
        assert heights[0][0] == pytest.approx(0.00409, abs=1e-5)

    def test_pixel_widths(self):
        collection = self.collection
        #
        tubes = collection.main.tubes
        start_time = time.time()
        widths = [tube.pixel_widths for tube in tubes]
        assert time.time() - start_time < 2.0  # below one second
        assert widths[0][0] == pytest.approx(0.00804, abs=1e-5)

    def test_pixel_y(self):
        collection = self.collection
        #
        tubes = collection.main.tubes
        start_time = time.time()
        positions = [tube.pixel_y for tube in tubes]
        assert time.time() - start_time < 2.0  # below one second
        assert (positions[0][0], positions[-1][-1]) == pytest.approx(
            (-0.52096, 0.52096), abs=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__])
