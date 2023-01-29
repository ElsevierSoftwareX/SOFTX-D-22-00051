import numpy as np
import pytest
from collections import namedtuple

from drtsans.resolution import InstrumentSetupParameters, calculate_sigma_theta_geometry

PixelInfo = namedtuple("PixelInfo", "smearing_pixel_size_x smearing_pixel_size_y")


def test_calculate_sigma_theta_geometry():
    # duck typing of a full pixel info object
    pixel_info = PixelInfo(np.array([0.0082, 0.0079]), np.array([0.0055, 0.0050]))
    # define a minimal instrument parameters.
    instrument_parameters = InstrumentSetupParameters(
        15.0, 4.0, 0.0011, 0.0007, 1.1, 0.9
    )
    # invoke calculate_sigma_theta_geometry
    sigma = calculate_sigma_theta_geometry("scalar", pixel_info, instrument_parameters)
    assert sigma == pytest.approx([4.63e-06, 4.21e-06], abs=1.0e-8)


if __name__ == "__main__":
    pytest.main([__file__])
