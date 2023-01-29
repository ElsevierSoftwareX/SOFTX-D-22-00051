import pytest
import numpy as np

from mantid.simpleapi import mtd

r""" Links to drtsans imports
center_detector <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/beam_finder.py>
unique_workspace_name <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
namedtuplefy <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
empty_beam_scaling <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/mono/absolute_units.py>
"""
from drtsans import center_detector
from drtsans.settings import unique_workspace_dundername, namedtuplefy
from drtsans.mono import empty_beam_scaling


@pytest.fixture(scope="module")
@namedtuplefy
def test_data_15b():
    r"""
    Data from test 15B, addressing master document section 12b.
    Link to the test:
    <https://www.dropbox.com/s/8xddym8iteozrhz/Calculate%20scale%20factor%20from%20the%20absolute%20intensity_He.xlsx>

    Mimic a detector array with 17 tubes and 15 pixels per tube. Included are the intensities for an empty beam run,
    with the maximum recorded intensity (2059.2) located at pixel with pixel coordinates (6, 7)

    Returns
    -------
    dict
    """
    return dict(
        intensities="""0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.8, 0, 0, 0
0, 1.8, 5.4, 3.6, 5.4, 3.6, 5.4, 9, 14.4, 3.6, 5.4, 3.6, 3.6, 1.8, 0, 0, 1.8
3.6, 18, 21.6, 54, 91.8, 84.6, 75.6, 91.8, 72, 46.8, 45, 30.6, 14.4, 3.6, 0, 0, 0
7.2, 117, 234, 370.8, 457.2, 549, 585, 563.4, 525.6, 462.6, 441, 367.2, 259.2, 145.8, 64.8, 32.4, 21.6
84.6, 221.4, 379.8, 487.8, 473.4, 545.4, 473.4, 453.6, 345.6, 331.2, 280.8, 205.2, 131.4, 59.4, 19.8, 5.4, 1.8
174.6, 657, 930.6, 1373.4, 1459.8, 1411.2, 1405.8, 1339.2, 1159.2, 1035, 950.4, 748.8, 487.8, 342, 169.2, 41.4, 18
82.8, 333, 487.8, 576, 925.2, 837, 907.2, 739.8, 817.2, 720, 563.4, 545.4, 379.8, 275.4, 147.6, 61.2, 23.4
127.8, 583.2, 1040.4, 1396.8, 1702.8, 1956.6, 2059.2, 1929.6, 1841.4, 1825.2, 1582.2, 1281.6, 1078.2, 658.8, 378, 185.4, 79.2
228.6, 567, 757.8, 923.4, 1065.6, 1006.2, 927, 761.4, 738, 590.4, 529.2, 387, 217.8, 136.8, 66.6, 10.8, 7.2
257.4, 734.4, 1096.2, 1450.8, 1542.6, 1629, 1587.6, 1427.4, 1305, 1054.8, 939.6, 730.8, 516.6, 288, 111.6, 48.6, 21.6
63, 221.4, 358.2, 495, 549, 608.4, 626.4, 547.2, 473.4, 424.8, 311.4, 284.4, 162, 81, 36, 28.8, 10.8
43.2, 214.2, 325.8, 478.8, 549, 552.6, 570.6, 586.8, 457.2, 403.2, 363.6, 207, 201.6, 82.8, 27, 18, 5.4
10.8, 19.8, 41.4, 70.2, 115.2, 95.4, 106.2, 79.2, 88.2, 55.8, 41.4, 18, 9, 3.6, 1.8, 1.8, 0
0, 1.8, 9, 14.4, 16.2, 25.2, 19.8, 25.2, 25.2, 10.8, 3.6, 5.4, 1.8, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0""",  # noqa: E501 empty beam intensities
        center_xy=(6, 7),  # pixel coordinates of the pixel with the highest intensity
        number_of_tubes=17,
        pixels_per_tube=15,  # number of pixels in any given tube
        pixel_size=0.01,  # assume the units are in meters, then the pixel size is 1 cm.
        number_of_pixels=255,
        wavelength_bin=[2.5, 3.0],  # some arbitrary wavelength bin
        beam_radius=60,  # radius of the beam impinging on the detector
        beam_radius_units="mm",
        attenuator_coefficient=1.0 / 30,
        attenuator_error=0.01 / 30,
        scaled_intensity=0.04229,  # integrated intensity after scaling
        scaled_uncertainty=0.00046,  # uncertainty of the integrated intensity
        precision=1.0e-05,  # precision when comparing test-data and computed scaling intensity and uncertainty
    )


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [dict(name="EQSANS", Nx=17, Ny=15, dx=0.01, dy=0.01, zc=1.0)],
    indirect=True,
)
def test_empty_beam_scaling(workspace_with_instrument, test_data_15b):
    r"""
    This test implements issue #179 and test 15B, addressing master document section 12b.

    dev - Jose Borreguero <borreguerojm@ornl.gov>
    SME - Lilin He <hel3@ornl.gov>

    **Instrument properties**:
    - 17 tubes, 15 pixels per tube.
    - pixel size is 1cm x 1cm
    - detector panel 1m away from the sample


    **drtsans functions used:**
    ~drtsans.tof.eqsans.center_detector,
    <https://scse.ornl.gov/docs/drt/sans/drtsans/tof/eqsans/index.html>
    ~drtsans.absolute_units.empty_beam_scaling,
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/absolute_units.py>
    """

    # save the intensities  of the empty beam in a numpy.ndarray.
    intensities = [
        float(x)
        for line in test_data_15b.intensities.split("\n")
        for x in line.split(",")
    ]
    intensities = np.array(intensities, dtype=float).reshape((15, 17, 1))
    number_of_pixels = intensities.size

    # Create a Mantid workspace with an embedded instrument and store the attenuated empty beam intensities
    empty_beam_workspace = unique_workspace_dundername()  # some random name
    workspace_with_instrument(
        axis_values=test_data_15b.wavelength_bin,
        intensities=intensities,
        view="array",
        output_workspace=empty_beam_workspace,
    )

    # Section to center the detector
    # The tests assumes the beam center is located at the center of the pixel with the highest intensity. Thus,
    # we need to obtain the coordinates of this pixel, in units of meters.
    (
        center_x,
        center_y,
    ) = (
        test_data_15b.center_xy
    )  # coordinates of the highest-intensity pixel, in pixel coords.
    center_pixel_id = (
        center_x * test_data_15b.pixels_per_tube + center_y
    )  # integer ID for the pixel
    center_x, center_y, _ = (
        mtd[empty_beam_workspace].spectrumInfo().position(center_pixel_id)
    )  # now in meters
    center_detector(empty_beam_workspace, center_x, center_y)
    # Verify that the XY coordinates of the pixel with the highest intensity are now at (0, 0)
    center_x, center_y, _ = (
        mtd[empty_beam_workspace].spectrumInfo().position(center_pixel_id)
    )
    assert (center_x, center_y) == pytest.approx(
        (
            0.0,
            0,
        ),
        abs=1e-06,
    )  # precision of 0.001 mili-meters

    # Create workspace for the sample run. The test provides a single value (100000) for the "Integrated scattering
    # of the sample", but we have detector with many pixels. We opt to assign all the intensity to the pixel
    # showing the highest intensity in the empty beam run. This assignment allows for comparison of both scaled
    # intensity and associated uncertainty against the test data.
    # Link to the test:
    # https://www.dropbox.com/s/8xddym8iteozrhz/Calculate%20scale%20factor%20from%20the%20absolute%20intensity_He.xlsx
    data_intensities = np.zeros(number_of_pixels).reshape((15, 17, 1))
    data_workspace = unique_workspace_dundername()  # some random name
    workspace_with_instrument(
        axis_values=test_data_15b.wavelength_bin,
        intensities=data_intensities,
        view="array",
        uncertainties=np.sqrt(data_intensities),
        output_workspace=data_workspace,
    )
    mtd[data_workspace].dataY(center_pixel_id)[0] = 100000
    mtd[data_workspace].dataE(center_pixel_id)[0] = np.sqrt(100000)

    # Use drtsans to scale the data with the empty beam method
    empty_beam_scaling(
        data_workspace,
        empty_beam_workspace,
        beam_radius=test_data_15b.beam_radius,
        unit=test_data_15b.beam_radius_units,
        attenuator_coefficient=test_data_15b.attenuator_coefficient,
        attenuator_error=test_data_15b.attenuator_error,
    )

    # Check computed intensity and error of the first pixel against test data
    assert mtd[data_workspace].readY(center_pixel_id)[0] == pytest.approx(
        test_data_15b.scaled_intensity, abs=test_data_15b.precision
    )
    assert mtd[data_workspace].readE(center_pixel_id)[0] == pytest.approx(
        test_data_15b.scaled_uncertainty, abs=test_data_15b.precision
    )
