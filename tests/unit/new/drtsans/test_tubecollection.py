import numpy as np
import pytest

r""" Hyperlinks to mantid algorithms
LoadEmptyInstrument <https://docs.mantidproject.org/nightly/algorithms/LoadEmptyInstrument-v1.html>
"""
from mantid.kernel import V3D

r"""
Hyperlinks to the tubecollection
<https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tubecollection.py>
"""  # noqa: E501
from drtsans import tubecollection


@pytest.fixture(scope="function")
def simple_tubes_panel(workspace_with_instrument):
    workspace = workspace_with_instrument(
        axis_values=[1.0, 2.0], intensities=np.arange(9).reshape((9, 1))
    )
    return dict(
        workspace=workspace,
        component_info=workspace.componentInfo(),
        detector_info=workspace.detectorInfo(),
        spectrum_info=workspace.spectrumInfo(),
    )


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 3,
            "n_pixels": 3,
            "diameter": 1.0,
            "height": 1.0,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,
        }
    ],
    indirect=True,
)
def test_decrement_arity(simple_tubes_panel):
    # Functions `foo` and `foo_two` for which inspect.signature works
    def foo(one):
        return one

    assert tubecollection._decrement_arity(foo, 1) == 1

    def foo_two(one, two):
        return one + two

    foo_one = tubecollection._decrement_arity(foo_two, 1)
    assert foo_one(1) == 2

    # Bound methods for which inspect.signature works
    class Foo:
        def __init__(self):
            pass

        def foo_one(self, one):
            return one

        def foo_two(self, one, two):
            return one and two

    assert tubecollection._decrement_arity(Foo().foo_one, True) is True
    foo_one = tubecollection._decrement_arity(Foo().foo_two, False)
    assert foo_one(True) is False

    # Bound methods for which inspect.signature doesn't work
    element = tubecollection.ElementComponentInfo(
        simple_tubes_panel["component_info"], 4
    )
    assert element.isDetector is True
    set_position = (
        element.setPosition
    )  # setPosition as arity == 2 (excluding parameter 'self')
    set_position(V3D(0.0, 0.0, 0.0))
    assert element.position == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 3,
            "n_pixels": 3,
            "diameter": 1.0,
            "height": 1.0,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,
        }
    ],
    indirect=True,
)
class TestSpectrumInfo:
    def test_init(self, simple_tubes_panel):
        assert tubecollection.SpectrumInfo(simple_tubes_panel["workspace"], 4)

    def test_some_methods(self, simple_tubes_panel):
        # Test for only one workspace index
        s = tubecollection.SpectrumInfo(simple_tubes_panel["workspace"], 4)
        assert s.l1 == pytest.approx(11.0)  # l1 doesn't take an index as first argument
        assert s.isMasked is False  # isMasked only takes one argument, the index
        s.setMasked(True)  # setMasked takes two arguments: an index and a bool
        assert s.isMasked is True
        assert s.twoTheta == pytest.approx(1.57, abs=0.01)
        assert s.readX == pytest.approx(
            [1.0, 2.0]
        )  # method borrowed from methods of the workspace object
        assert s.readY == pytest.approx([4.0])
        assert len(s) == 1
        # Test for more than one workspace index
        s = tubecollection.SpectrumInfo(simple_tubes_panel["workspace"], [1, 4, 7])
        assert s.l1 == pytest.approx([11.0, 11.0, 11.0])
        assert list(s.isMasked) == [
            False,
            True,
            False,
        ]  # remember we changed before the mask for workspace index 4
        assert s.setMasked(True)  # mask all spectra
        assert list(s.isMasked) == [True, True, True]
        assert s.readX == pytest.approx(np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))
        assert s.readY == pytest.approx(np.array([[1.0], [4.0], [7.0]]))
        assert len(s) == 3


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 3,
            "n_pixels": 3,
            "diameter": 1.0,
            "height": 1.0,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,
        }
    ],
    indirect=True,
)
@pytest.mark.usefixtures("simple_tubes_panel")
class TestElementComponentInfo:
    def test_init(self, simple_tubes_panel):
        collection = tubecollection.ElementComponentInfo(
            simple_tubes_panel["component_info"], 4
        )
        assert isinstance(collection, tubecollection.ElementComponentInfo)

    def test_some_methods(self, simple_tubes_panel):
        el = tubecollection.ElementComponentInfo(
            simple_tubes_panel["component_info"], 4
        )
        assert el.isDetector is True  # isDetector's arity is one
        assert el.position == pytest.approx([0.0, -0.5, 0.0])  # position's arity is one
        el.setPosition(V3D(0.0, 0.0, 0.0))  # setPosition's arity is two
        assert el.position == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 3,
            "n_pixels": 3,
            "diameter": 1.0,
            "height": 1.0,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,
        }
    ],
    indirect=True,
)
def test_resolve_indexes(simple_tubes_panel):
    workspace = simple_tubes_panel["workspace"]
    assert tubecollection._resolve_indexes(workspace, 4, 42) == (4, 42)
    with pytest.raises(RuntimeError):
        tubecollection._resolve_indexes(workspace, None, None)
    assert tubecollection._resolve_indexes(workspace, 4, None) == (
        4,
        4,
    )  # resolve the workspace index
    assert tubecollection._resolve_indexes(workspace, None, 4) == (
        4,
        4,
    )  # resolve the component info index


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 3,
            "n_pixels": 3,
            "diameter": 1.0,
            "height": 1.0,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,
        }
    ],
    indirect=True,
)
@pytest.mark.usefixtures("simple_tubes_panel")
class TestPixelSpectrum(object):
    def test_init(self, simple_tubes_panel):
        stp = simple_tubes_panel
        pixel = tubecollection.PixelSpectrum(stp["workspace"], component_info_index=4)
        assert pixel.spectrum_info_index == 4
        pixel = tubecollection.PixelSpectrum(stp["workspace"], workspace_index=4)
        assert pixel.component_info_index == 4

    def test_component_info(self, simple_tubes_panel):
        # test accessibility of componentInfo's methods
        pixel = tubecollection.PixelSpectrum(
            simple_tubes_panel["workspace"], workspace_index=4
        )
        assert pixel.isDetector is True
        assert pixel.scaleFactor == pytest.approx([1.0, 1.0, 1.0])
        pixel.setScaleFactor(V3D(0.98, 1.03, 1.11))
        assert pixel.scaleFactor == pytest.approx([0.98, 1.03, 1.11])

    def test_detector_info(self, simple_tubes_panel):
        # test accessibility of detectorInfo's methods
        pixel = tubecollection.PixelSpectrum(
            simple_tubes_panel["workspace"], workspace_index=4
        )
        assert pixel.l2 == pytest.approx(0.5, abs=0.01)
        assert pixel.twoTheta == pytest.approx(1.57, abs=0.01)
        assert pixel.isMasked is False
        pixel.setMasked(True)
        assert pixel.isMasked is True

    def test_spectrum_info(self, simple_tubes_panel):
        # test accessibility of spectrumInfo's methods
        pixel = tubecollection.PixelSpectrum(
            simple_tubes_panel["workspace"], workspace_index=4
        )
        assert pixel.hasUniqueDetector is True
        assert pixel.samplePosition == pytest.approx([0.0, 0.0, 0.0])
        assert pixel.signedTwoTheta == pytest.approx(1.57, abs=0.01)

    def test_properties(self, simple_tubes_panel):
        pixel = tubecollection.PixelSpectrum(
            simple_tubes_panel["workspace"], workspace_index=4
        )
        # Test property 'position'
        assert pixel.position == pytest.approx([0, -0.5, 0])
        pixel.position = ("y", 0.0)
        assert pixel.position == pytest.approx([0, 0, 0])
        pixel.position = (0.1, 0.2, 0.3)
        assert pixel.position == pytest.approx([0.1, 0.2, 0.3])
        # Test property 'width'
        assert pixel.width == pytest.approx(1.0)
        pixel.width = 1.42 * pixel.width
        assert pixel.width == pytest.approx(1.42)
        # Test property 'height'
        assert pixel.height == pytest.approx(1.0)
        pixel.height = 1.42 * pixel.height
        assert pixel.height == pytest.approx(1.42)
        # Test property 'area'
        pixel.height, pixel.width = 1.42, 1.42
        assert pixel.area == 1.42**2


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 3,
            "n_pixels": 3,
            "diameter": 1.0,
            "height": 1.0,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,
        }
    ],
    indirect=True,
)
@pytest.mark.usefixtures("simple_tubes_panel")
class TestTubeSpectrum(object):
    def test_is_valid_tube(self, simple_tubes_panel):
        workspace = simple_tubes_panel["workspace"]
        assert tubecollection.TubeSpectrum(
            workspace, 11, [0, 1, 2]
        )  # 11 is the first tube
        with pytest.raises(ValueError):
            tubecollection.TubeSpectrum(workspace, 10, [0, 1, 2])  # this is a detector
        with pytest.raises(ValueError):
            tubecollection.TubeSpectrum(
                workspace, 15, [0, 1, 2]
            )  # this is the whole panel

    def test_pixels(self, simple_tubes_panel):
        tube = tubecollection.TubeSpectrum(
            simple_tubes_panel["workspace"], 11, [0, 1, 2]
        )
        pixels = tube.pixels
        assert pixels[0].position == pytest.approx([1, -1.5, 0])
        assert pixels[2].position == pytest.approx([1, 0.5, 0])

    def test_getitem(self, simple_tubes_panel):
        tube = tubecollection.TubeSpectrum(
            simple_tubes_panel["workspace"], 11, [0, 1, 2]
        )
        assert tube[0].position == pytest.approx([1, -1.5, 0])
        assert tube[2].position == pytest.approx([1, 0.5, 0])

    def test_len(self, simple_tubes_panel):
        tube = tubecollection.TubeSpectrum(
            simple_tubes_panel["workspace"], 11, [0, 1, 2]
        )
        assert len(tube) == 3

    def test_spectrum_info(self, simple_tubes_panel):
        # test some of spectrum info's methods
        tube = tubecollection.TubeSpectrum(
            simple_tubes_panel["workspace"], 11, [0, 1, 2]
        )
        assert list(tube.hasUniqueDetector) == [True, True, True]
        tube[1].setMasked(True)
        assert list(tube.isMasked) == [False, True, False]
        assert tube.setMasked(True)  # mask all spectra
        assert list(tube.isMasked) == [True, True, True]
        assert tube.readX == pytest.approx(
            np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        )
        assert tube.readY == pytest.approx(np.array([[0.0], [1.0], [2.0]]))

    def test_component_info(self, simple_tubes_panel):
        # test accessibility of componentInfo's methods
        tube = tubecollection.TubeSpectrum(
            simple_tubes_panel["workspace"], 11, [0, 1, 2]
        )
        assert tube.isDetector is False
        assert tube.scaleFactor == pytest.approx([1.0, 1.0, 1.0])
        tube.setScaleFactor(V3D(0.98, 1.03, 1.11))
        assert tube.scaleFactor == pytest.approx([0.98, 1.03, 1.11])
        assert tube.samplePosition == pytest.approx(np.array([0.0, 0.0, 0.0]))
        assert tube.l1 == pytest.approx(11.0)
        assert tube.position == pytest.approx(np.array([1, 0, 0]))


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 3,
            "n_pixels": 3,
            "diameter": 1.0,
            "height": 1.0,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,
        }
    ],
    indirect=True,
)
@pytest.mark.usefixtures("simple_tubes_panel")
class TestTubeCollection(object):
    def test_init(self, simple_tubes_panel):
        collection = tubecollection.TubeCollection(
            simple_tubes_panel["workspace"], "detector1"
        )
        assert isinstance(collection, tubecollection.TubeCollection)

    def test_len(self, simple_tubes_panel):
        collection = tubecollection.TubeCollection(
            simple_tubes_panel["workspace"], "detector1"
        )
        assert len(collection) == 3

    def test_getitem(self, simple_tubes_panel):
        collection = tubecollection.TubeCollection(
            simple_tubes_panel["workspace"], "detector1"
        )
        assert collection[0][0].position == pytest.approx(
            [1.0, -1.5, 0.0]
        )  # first pixel in first tube
        assert collection[2][2].position == pytest.approx(
            [-1.0, 0.5, 0.0]
        )  # last pixel in last tube

    def test_sorted(self, simple_tubes_panel):
        collection = tubecollection.TubeCollection(
            simple_tubes_panel["workspace"], "detector1"
        )
        sorted_tubes = collection.sorted(view="decreasing X")
        assert (
            sorted_tubes == collection.tubes
        )  # true for the simple_tubes_panel instrument


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "rectangular detector",
            "Nx": 3,
            "Ny": 3,
            "dx": 1.0,
            "dy": 1.0,
            "xc": 0.0,
            "yc": 0.0,
            "zc": 0.0,
        }
    ],
    indirect=True,
)
def test_flat_panel(workspace_with_instrument):
    r"""Make sure a flat detector is pieced into 'tubes'"""
    workspace = workspace_with_instrument(
        axis_values=[1.0, 2.0], intensities=np.random.rand(9).reshape((9, 1))
    )
    collection = tubecollection.TubeCollection(workspace, "detector1")
    assert len(collection) == 3


if __name__ == "__main__":
    pytest.main([__file__])
