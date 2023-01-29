import pytest
import numpy as np
from drtsans.mono.meta_data import set_meta_data as mono_set_meta_data
from drtsans.tof.eqsans.meta_data import set_meta_data as eqsans_set_meta_data
from drtsans.geometry import (
    logged_smearing_pixel_size,
    sample_aperture_diameter,
    source_aperture_diameter,
    sample_detector_distance,
    source_sample_distance,
)


@pytest.mark.parametrize(
    "workspace_with_instrument", [{"Nx": 3, "Ny": 3}], indirect=True
)
def test_set_mono_meta_data(workspace_with_instrument):
    """ """
    data = np.array([[7.0, 8.0, 12.0], [10.0, 17.0, 13.0], [10.0, 10.0, 9.0]])

    data_error = np.sqrt(data)

    # create workspaces
    data_ws = workspace_with_instrument(
        axis_values=[6.0],  # fake wavelength
        intensities=data,
        uncertainties=data_error,
        view="pixel",
    )

    # Add meta data
    mono_set_meta_data(
        data_ws,
        wave_length=10.0,
        wavelength_spread=1.5,
        source_aperture_diameter=2.29,
        sample_aperture_diameter=13,
        sample_thickness=1.23,
        sample_offset=4.5,
        smearing_pixel_size_x=0.0021,
        smearing_pixel_size_y=0.0022,
    )

    # verify
    test_sample_aperture_diameter = sample_aperture_diameter(data_ws, unit="m")
    assert test_sample_aperture_diameter == 0.013

    test_sdd = sample_detector_distance(data_ws, "m")
    assert test_sdd > 0

    # verify pixel size
    test_ps_x, test_ps_y = logged_smearing_pixel_size(data_ws)
    assert test_ps_x == 0.0021, "Expected: {}, Got: {}".format(0.0021, test_ps_x)
    assert test_ps_y == 0.0022, "Expected: {}, Got: {}".format(0.0022, test_ps_y)


@pytest.mark.parametrize(
    "workspace_with_instrument", [{"Nx": 3, "Ny": 3}], indirect=True
)
def test_set_eqsans_meta_data(workspace_with_instrument):
    """ """
    data = np.array([[7.0, 8.0, 12.0], [10.0, 17.0, 13.0], [10.0, 10.0, 9.0]])

    data_error = np.sqrt(data)

    # create workspaces
    data_ws = workspace_with_instrument(
        axis_values=[6.0],  # fake wavelength
        intensities=data,
        uncertainties=data_error,
        view="pixel",
    )

    assert data_ws is not None

    # Add meta data
    eqsans_set_meta_data(
        data_ws,
        source_aperture_diameter=2.29,
        sample_aperture_diameter=13,
        sample_thickness=1.23,
        sample_offset=4.5,
        smearing_pixel_size_x=0.0021,
        smearing_pixel_size_y=0.0022,
    )

    # verify
    test_source_aperture_diameter = source_aperture_diameter(data_ws, unit="m")
    assert test_source_aperture_diameter == 0.00229

    test_l1 = source_sample_distance(data_ws, "m")
    assert test_l1 > 0
    assert logged_smearing_pixel_size(data_ws) == pytest.approx((0.0021, 0.0022))
