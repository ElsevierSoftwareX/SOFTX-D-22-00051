"""
    Test EQSANS resolution
"""
import numpy as np
from collections import namedtuple
from scipy import constants
import pytest

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/convert_to_q.py
from drtsans.tof.eqsans.momentum_transfer import (
    eqsans_resolution,
    moderator_time_uncertainty,
)
import drtsans.resolution


def test_moderator_uncertainty():
    """Test moderator time uncertainty function using two wavelengths above and below 2 Angstroms
    and verify the output with expected results.
    dev - Jiao Lin <linjiao@ornl.gov>
        - Andrei Savici <saviciat@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>

    For details see https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/168
    """
    # the wavelengths to test
    wavelengths = [1.5, 9.3]
    # expected output
    expected = [214.74671875, 258.8954766]
    # calculate
    out = moderator_time_uncertainty(np.array(wavelengths))
    # verify
    assert np.allclose(out, expected)


def sigma_neutron(
    wave_length,
    delta_wave_length,
    Qx,
    Qy,
    theta,
    L1,
    L2,
    R1,
    R2,
    x3,
    y3,
    s2p,
    m2s,
    sig_emission,
):
    """
    Q resolution calculation from Wei-ren and modified by wzz
    Parameters
    ----------
    wave_length: float
        neutron wavelength (A)
    delta_wave_length: float
        wavelength width (A)
    Qx: float
        momentum transfer in x direction
    Qy: float
         momentum transfer in y direction
    theta: float
        scattering angle (half of 2theta of the pixel)
    L1: float
        source to sample
    L2: float
        sample to center of detector
    R1: float
        source aperture (m)
    R2: float
        sample aperture (m)
    x3: float
        detector pixel dimensions along x-axis
    y3: float
        detector pixel dimensions along y-axis
    s2p: float
        sample to pixel distance (m), which is slightly different from L2
    m2s: float
        moderator to sample distance
    sig_emission: float
        neutron emission time (second?)

    Returns
    -------
    float, float
        dQx, dQy

    """
    # For EQ-SANS

    # Define constants
    h = 6.62607004e-34  # scipy.const.h
    mn = 1.674929e-27  # scipy.const.neutron_mass
    g = constants.g  # 9.81 n/s^2

    # Calculate B
    B = 0.5 * g * mn**2 * L2 * (L1 + L2) / h**2
    B /= 10**20  # add a factor of 10^-2 to cancel out the A in the term using B
    # (dWL/WL)**2
    r = (delta_wave_length / wave_length) ** 2

    # dQx
    sigma_x = (
        2.0 * np.pi * np.cos(theta) * np.cos(2.0 * theta) ** 2 / wave_length / L2
    ) ** 2
    sigma_x = sigma_x * (
        (L2 / L1) ** 2 * R1**2 / 4 + (1 + L2 / L1) ** 2 * R2**2 / 4 + x3**2 / 12
    )  # geometry
    sigma_x = np.sqrt(
        sigma_x
        + Qx**2
        / 12
        * (r + (3.9560 * sig_emission) ** 2 / (1000 * wave_length * (s2p + m2s)) ** 2)
    )

    # dQy
    sigma_y = (
        2.0 * np.pi * np.cos(theta) * np.cos(2 * theta) ** 2 / wave_length / L2
    ) ** 2
    sigma_y = sigma_y * (
        (L2 / L1) ** 2 * R1**2 / 4
        + (1 + L2 / L1) ** 2 * R2**2 / 4
        + y3**2 / 12
        + B**2 * wave_length**4 * 2 / 3 * r
    )
    sigma_y = np.sqrt(
        sigma_y
        + Qy**2
        / 12
        * (r + (3.9560 * sig_emission) ** 2 / (1000 * wave_length * (s2p + m2s)) ** 2)
    )

    return sigma_x, sigma_y


def test_eqsans_resolution():
    """
    Test the full resolution calculation for EQSANS
    formula 10.5 and 10.6 in the master document
    function to test drtsans.tof.eqsans.convert_to_q.eqsans_resolution
    """
    l1 = 15.0
    l2 = 15.5
    r1 = 0.02  # source aperture
    r2 = 0.007  # sample aperture
    size_x = 0.0055  # pixelsize
    size_y = 0.0043  # pixel size

    instrument_params = drtsans.resolution.InstrumentSetupParameters(
        l1=l1,
        sample_det_center_dist=l2,
        source_aperture_radius=r1,
        sample_aperture_radius=r2,
    )
    qx = -0.000593411755
    qy = -0.000767944624
    wave_length = 6.0
    wl_resolution = 0.15
    two_theta = 0.00092676  # radian (corner pixel)
    sample_pixel_distance = l2 + 0.1
    emission_error = 248.89  # wave length = 3.5 A

    # Calculate Q resolution by Weiren's algorithm
    golden_dqx, golden_dqy = sigma_neutron(
        wave_length,
        wl_resolution,
        qx,
        qy,
        0.5 * two_theta,
        l1,
        l2,
        r1,
        r2,
        size_x,
        size_y,
        sample_pixel_distance,
        l1,
        emission_error,
    )
    # Calculate EQSANS resolution
    pixel_info = namedtuple(
        "pixel_info",
        [
            "two_theta",
            "azimuthal",
            "l2",
            "keep",
            "smearing_pixel_size_x",
            "smearing_pixel_size_y",
        ],
    )
    pix = pixel_info(
        np.array([two_theta]),
        np.array([np.arctan2(qy, qx)]),
        np.array([sample_pixel_distance]),
        np.array([True]),
        np.array([size_x]),
        np.array([size_y]),
    )
    q_x_res, q_y_res = eqsans_resolution(
        np.array([qx]),
        np.array([qy]),
        mode="azimuthal",
        instrument_parameters=instrument_params,
        pixel_info=pix,
        wavelength=wave_length,
        delta_wavelength=wl_resolution,
    )
    assert q_x_res == pytest.approx(golden_dqx, abs=1e-8)
    assert q_y_res == pytest.approx(golden_dqy, abs=1e-8)
    q_res = eqsans_resolution(
        np.array([qx]),
        np.array([qy]),
        mode="scalar",
        instrument_parameters=instrument_params,
        pixel_info=pix,
        wavelength=wave_length,
        delta_wavelength=wl_resolution,
    )
    assert q_res == pytest.approx(np.sqrt(golden_dqx**2 + golden_dqy**2), abs=1e-7)


if __name__ == "__main__":
    pytest.main([__file__])
