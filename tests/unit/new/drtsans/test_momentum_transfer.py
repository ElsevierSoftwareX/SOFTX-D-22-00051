import pytest
import numpy as np

r"""
Hyperlinks to mantid algorithms
MaskDetectors <https://docs.mantidproject.org/nightly/algorithms/MaskDetectors-v1.html>
"""
from mantid.simpleapi import MaskDetectors

r"""
Hyperlinks to drtsans functions
pixel_centers <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/geometry.py>
convert_to_q, _filter_and_replicate, subpixel_info <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/momentum_transfer.py>
namedtuplefy, unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
"""  # noqa: E501
from drtsans.geometry import pixel_centers
from drtsans.momentum_transfer import convert_to_q, _filter_and_replicate, subpixel_info
from drtsans.settings import namedtuplefy, unique_workspace_dundername


def fake_resolution1(*args, **kwargs):
    """
    resolution dependent only on q
    """
    mode = kwargs.get("mode")
    if mode == "scalar":
        q = args[0]
        return np.arange(len(q)).reshape(-1, 1)
    elif mode == "azimuthal":
        qx = args[0]
        qy = args[1]
        return qx + qy, qx - qy
    else:
        raise NotImplementedError("fake_resolution1 mode")


def fake_resolution2(*args, **kwargs):
    """
    resolution dependent on q and geometry
    """
    mode = kwargs.get("mode")
    info = kwargs.get("pixel_info", None)
    if mode == "scalar":
        q = args[0]
        if info:
            azi = info.azimuthal
            num_bins = q.shape[1]
            return np.repeat(azi, num_bins).reshape(-1, num_bins)
    else:
        raise NotImplementedError("fake_resolution1 mode")


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "intensities": [[10, 20], [30, 40]],
            "uncertainties": [[1, 2], [3, 4]],
            "axis_values": [5.9, 6.1],
            "name": "BIOSANS",
            "dx": 1,
            "dy": 1,
        }
    ],
    indirect=True,
)
def test_convert_to_mod_q(generic_workspace):
    r"""
    Calculate Q-values and associated intensities for every detector-pixel and every wavelength bin. We have only
    one wavelength bin, so we only have one Q-value per pixel detector. This Q-value is associated to the midpoint
    of the wavelength bin.

    The instrument is a flat detector with four pixels. All pixels subtend the same two_theta scattering angle with
    the direction of the incoming neutron beam (along the Z-axis). Thus, there's only one value of two_theta of
    relevance.
    """
    ws = generic_workspace
    # We will ignore intensities from the third spectrum (workspace index = 2). Thus, only three values for the
    # modulus of the momentum transfer will be obtained, one for each of the three valid  spectra.
    MaskDetectors(ws, WorkspaceIndexList=[2])
    # For every detector pixel, calculate the intensity, intensity error, Q-value, it's error, and the associated
    # vawelength.
    intensity, error, modq, dq, lam = convert_to_q(ws, mode="scalar")
    assert intensity == pytest.approx([10, 20, 40], abs=1e-5)
    assert error == pytest.approx([1, 2, 4], abs=1e-5)
    # we are not passing any resolution function as argument 'resolution_function' in function
    # 'convert_to_q'. Thus, we assume infinite precision and the error in Q is therefore zero.
    assert dq == pytest.approx([0, 0, 0], abs=1e-5)
    assert lam == pytest.approx(
        [6, 6, 6], abs=1e-5
    )  # 6 Angstroms is the middle point of the bin [5.9, 6.1]

    # All detector pixels subtend the same scattering angle. Detectors are located at coordinates
    # (x, y, z) = (+-0.5, +-0.5, 5.0)
    two_theta = np.arccos(
        5.0 / np.sqrt(5**2 + 0.5**2 + 0.5**2)
    )  # cos(two_theta) = z / sqrt(x^2 + y^2 + z^2)
    assert ws.spectrumInfo().twoTheta(0) == pytest.approx(two_theta, abs=1e-5)

    # assert the Q value for each detector pixel. Again, all Q-values are the same because all two_theta values
    # are the same for all three unmasked detector pixels
    q = 4.0 * np.pi * np.sin(two_theta * 0.5) / 6.0
    assert modq == pytest.approx([q, q, q], abs=1e-5)

    # test namedtuple result and resolution
    # Function fake_resolution1 returns the workspace index of the pixel detector as the error of
    # the Q modulus for that pixel detector
    result = convert_to_q(ws, mode="scalar", resolution_function=fake_resolution1)
    assert result.delta_mod_q == pytest.approx([0, 1, 3], abs=1e-5)

    # test geometry dependent resolution
    result = convert_to_q(ws, mode="scalar", resolution_function=fake_resolution2)
    # note that x axis is pointing to the left, so the azimuthal angles by
    # spectrum number are -45, 45, -135, 135
    # Function fake_resolution1 returns the azimuthal angle of the pixel detector as the error of
    #     # the Q modulus for that pixel detector
    assert np.degrees(result.delta_mod_q) == pytest.approx([-45, 45, 135], abs=1e-5)


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "intensities": [[10, 20], [30, 40]],
            "uncertainties": [[1, 2], [3, 4]],
            "axis_values": [5.9, 6.1],
            "name": "BIOSANS",
            "dx": 1,
            "dy": 1,
        }
    ],
    indirect=True,
)
def test_convert_q_azimuthal(generic_workspace):
    ws = generic_workspace
    MaskDetectors(ws, WorkspaceIndexList=[2])
    result = convert_to_q(ws, mode="azimuthal")
    assert result.intensity == pytest.approx([10, 20, 40], abs=1e-5)
    assert result.error == pytest.approx([1, 2, 4], abs=1e-5)
    assert result.delta_qx == pytest.approx([0, 0, 0], abs=1e-5)
    assert result.delta_qy == pytest.approx([0, 0, 0], abs=1e-5)
    assert result.wavelength == pytest.approx([6, 6, 6], abs=1e-5)
    two_theta = np.arccos(5.0 / np.sqrt(25.5))
    q = 4.0 * np.pi * np.sin(two_theta * 0.5) / 6.0
    assert result.qx * (-1) == pytest.approx(
        [q * np.sqrt(0.5), q * np.sqrt(0.5), -q * np.sqrt(0.5)], abs=1e-5
    )
    assert result.qy == pytest.approx(
        [-q * np.sqrt(0.5), q * np.sqrt(0.5), q * np.sqrt(0.5)], abs=1e-5
    )

    # test with simple resolution
    result = convert_to_q(ws, mode="azimuthal", resolution_function=fake_resolution1)
    assert result.delta_qx == pytest.approx(
        [-q * np.sqrt(2.0), 0, q * np.sqrt(2.0)], abs=1e-5
    )
    assert result.delta_qy == pytest.approx([0, -q * np.sqrt(2.0), 0], abs=1e-5)


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "intensities": [[10, 20], [30, 40]],
            "uncertainties": [[1, 2], [3, 4]],
            "axis_values": [5.9, 6.1],
            "name": "BIOSANS",
            "dx": 1,
            "dy": 1,
        }
    ],
    indirect=True,
)
def test_convert_q_crystal(generic_workspace):
    ws = generic_workspace
    MaskDetectors(ws, WorkspaceIndexList=[2])
    result = convert_to_q(ws, mode="crystallographic")
    assert result.intensity == pytest.approx([10, 20, 40], abs=1e-5)
    assert result.error == pytest.approx([1, 2, 4], abs=1e-5)
    assert result.delta_qx == pytest.approx([0, 0, 0], abs=1e-5)
    assert result.delta_qy == pytest.approx([0, 0, 0], abs=1e-5)
    assert result.delta_qz == pytest.approx([0, 0, 0], abs=1e-5)
    assert result.wavelength == pytest.approx([6, 6, 6], abs=1e-5)
    two_theta = np.arccos(5.0 / np.sqrt(25.5))
    kf = 2.0 * np.pi / 6.0
    assert result.qx == pytest.approx(
        [
            kf * np.sin(two_theta) * np.sqrt(0.5),
            kf * np.sin(two_theta) * np.sqrt(0.5),
            -kf * np.sin(two_theta) * np.sqrt(0.5),
        ],
        abs=1e-5,
    )
    assert result.qy == pytest.approx(
        [
            -kf * np.sin(two_theta) * np.sqrt(0.5),
            kf * np.sin(two_theta) * np.sqrt(0.5),
            kf * np.sin(two_theta) * np.sqrt(0.5),
        ],
        abs=1e-5,
    )
    assert result.qz == pytest.approx(
        [
            kf * (np.cos(two_theta) - 1),
            kf * (np.cos(two_theta) - 1),
            kf * (np.cos(two_theta) - 1),
        ],
        abs=1e-5,
    )

    # test with simple resolution - it should thow not implemented
    with pytest.raises(NotImplementedError):
        assert convert_to_q(
            ws, mode="crystallographic", resolution_function=fake_resolution1
        )


@pytest.fixture(scope="module")
@namedtuplefy
def data_subpixel_info():
    r"""Data to be used for `test_subpixel_info`"""
    return dict(
        number_pixels=4,  # number of pixels in the instrument
        wavelength_bin_boundaries=[2.0, 2.1],
        intensities=[[1.0, 2.0], [3.0, 2.0]],  # first tube  # second tube
        # Pixel (x, y) coordinates, in mili-meters
        pixel_positions=[[10, -20], [10, 0], [-10, -20], [-10, 0]],  # units are mm
        n_horizontal=2,  # number of subpixel along the X-axis
        n_vertical=4,  # number of subpixel along the Y-axis
        number_subpixels=8,  # just n_horizontal * n_vertical
        # subpixel (x, y) coordinates for a pixel centered at (x, y) == (0, 0)
        subpixel_positions=[
            [5, -7.5],
            [5, -2.5],
            [5, 2.5],
            [5, 7.5],  # units are mm
            [-5, -7.5],
            [-5, -2.5],
            [-5, 2.5],
            [-5, 7.5],
        ],
        sample_detector_distance=1000.0,  # all subpixels have same Z coordinate (units are mm)
    )


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 2,
            "n_pixels": 2,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 1.0,  # detector center
            "diameter": 0.02,
            "height": 0.02,
        }
    ],  # pixel dimensions
    indirect=True,
)
def test_subpixel_info(data_subpixel_info, workspace_with_instrument):
    r"""
    Test for correct generatio of subpixel polar coordinates in a detector array made up of two tubes,
    each with two pixels.
    There's no spacing between tubes.
    Detector is 1 meter away from the sample
    The shape of a detector pixel is a cylinder of 20mm diameter and 20mm in height.
    Each pixel is divided in 8 subpixels (n_horizontal=2, n_vertical=4)'

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    """
    data = data_subpixel_info  # handy shortcut
    input_workspace = unique_workspace_dundername()  # temporary workspace
    # A workspace containing data.intensities in an instrument made up of two tubes with two pixels per tube
    workspace = workspace_with_instrument(
        axis_values=data.wavelength_bin_boundaries,
        intensities=data.intensities,
        output_workspace=input_workspace,
    )
    workspace.detectorInfo().setMasked(
        2, True
    )  # Mask the third pixel. No subpixes will be calculated for this one

    # Compare the positions of the pixels in the workspace against data.pixel_positions
    spectrum_info = workspace.spectrumInfo()
    get_spectrum_definition = spectrum_info.getSpectrumDefinition
    # Find the detectorInfo() indexes starting from the workspace indexes. These indexes are neccessary to later
    # find out the (x, y) coordinates of the pixels
    info_indexes = [
        get_spectrum_definition(idx)[0][0]
        for idx in range(workspace.getNumberHistograms())
    ]
    pixel_positions = pixel_centers(input_workspace, info_indexes)
    assert 1.0e03 * pixel_positions[:, :-1] == pytest.approx(
        np.array(data.pixel_positions), abs=1e-6
    )

    # Find the position of the subpixels in polar coordinates, contained in data structure "info"
    info = subpixel_info(input_workspace, data.n_horizontal, data.n_vertical)
    # Transform from polar to cartesian coordinates
    x = info.l2 * np.sin(info.two_theta) * np.cos(info.azimuthal)
    y = info.l2 * np.sin(info.two_theta) * np.sin(info.azimuthal)
    z = info.l2 * np.cos(info.two_theta)
    # All subpixels have the same Z-coordinate. Test this (factor 1.e-03 to convert from mili-meters to meters)
    assert z == pytest.approx(
        1.0e-03
        * data.sample_detector_distance
        * np.ones(len(info.keep) * data.number_subpixels),
        abs=1e-6,
    )

    # Construct array `xy` which will contain the coordinates of the subpixels, in mili-meters
    # xy.shape = (number_pixels, number_subpixels, 2)
    # xy[0] contains the XY coordinates of the subpixel contained within the first pixel, xy[1] for the
    # second pixel, and so on.
    # We only retain coordinates for pixels that were not masked, thus we use info.keep array
    xy = np.column_stack((x, y)).reshape((len(info.keep), data.number_subpixels, 2))

    # We test the coordinates of the subpixels in each unmasked pixel, one pixel at a time.
    # The procedure is as follows:
    # 1. use info.keep to filter out those indexes corresponding to masked pixels. In our case, index 2.
    # 2. translate the subpixel coordinates such that the position of the parent pixel is at (x, y) = (0, 0).
    # 3. compare the subpixel coordinates to data.subpixel_positions. These are the expected subpixel coordinates
    #    for a parent pixel having its position at (x, y) = (0, 0).
    xy = 1000 * (xy - pixel_positions[info.keep, :-1][:, None, :])
    for subpixel_positions in xy:  # unmasked pixels
        assert subpixel_positions == pytest.approx(
            np.array(data.subpixel_positions), abs=1e-6
        )


def test_filter_and_replicate():
    def to_np(alist):
        r"""Given a python list of length L, return a numpy array with shape (L, 1)"""
        return np.array(alist).reshape((len(alist), 1))

    lam, intensity, error = (
        to_np([1.0, 2.0, 3.0]),
        to_np([100.0, 81.0, 64.0]),
        to_np([10.0, 9.0, 8.0]),
    )
    # lam is numpy array array([[1.0], [2.0], [3.0]])
    keep = np.array(
        [1, 2]
    )  # Keep only data with indexes 1 and 2 (i.e., discard the data with index 0)
    # Apply function _filter_and_replicate to arrays 'lam', 'intensity', and 'error'.
    # 1. discard the first item of array 'lam'. Array 'lam' becomes array([[2.0], [3.0]])
    # 2. replicate each item a total number of n_horizontal * n_vertical = 2. Array 'lam' becomes
    #    array([[2.0, 2.0], [3.0, 3.0]])
    # 3. flatten array 'lam' by dropping the second dimension. Array 'lam' becomes array([2., 2., 3., 3.])
    lam, intensity, error = _filter_and_replicate(
        (lam, intensity, error), keep, n_horizontal=2, n_vertical=1
    )
    assert lam == pytest.approx([2.0, 2.0, 3.0, 3.0])
    assert intensity == pytest.approx([81.0, 81.0, 64.0, 64.0])
    assert error == pytest.approx([9.0, 9.0, 8.0, 8.0])


if __name__ == "__main__":
    pytest.main([__file__])
