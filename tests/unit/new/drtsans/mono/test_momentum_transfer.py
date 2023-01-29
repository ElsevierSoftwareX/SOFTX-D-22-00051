import numpy as np
import pytest
import scipy
import scipy.constants
from collections import namedtuple
from mantid.simpleapi import AddSampleLog

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/mono/convert_to_q.py
from drtsans.mono.momentum_transfer import (
    convert_to_q,
    mono_resolution,
    retrieve_instrument_setup,
)

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/resolution.py
import drtsans.resolution


# This implements Issue #168: calculate dQx and dQy
# dev - Wenduo Zhou <wzz@ornl.gov>
#     - Andrei Savici <saviciat@ornl.gov>
# SME - William Heller <hellerwt@ornl.gov>, Wei-Ren Chen


def sigma_neutron(wavelength, delta_lambda, Qx, Qy, theta, L1, L2, R1, R2, x3, y3):
    """
    Function given by Wei-Ren

    input: sigma_neutron(7, 0.7, 0.03, 0.03, 0.1, 10, 20, 5, 10, 0.03, 0.03, 20, 50)


    For GP-SANS and Bio-SANS
    lambda: netron wavelength in angstrom
    delta_lambda: wavelength width in angstrom
    Qx,Qy: momentum transfer in x and y direction, respectively, with unit (1/angstrom)
    theta: scattering angle in radian
    L1,L2: the flight path lengths whose units are meter.
    R1, R2: sample and source apertures, respectively, in meter.
    x3,y3: detector pixel dimensions in meter

    """
    h = 6.62607004e-34
    # h = scipy.constants.h
    mn = 1.674929e-27
    # mn = scipy.constants.neutron_mass
    g = scipy.constants.g  # 6.67408e-11
    B = 0.5 * g * mn**2 * L2 * (L1 + L2) / h**2
    B = B / 10**20
    r = (delta_lambda / wavelength) ** 2
    sigma_x = (
        2 * np.pi * np.cos(theta) * np.cos(2 * theta) ** 2 / wavelength / L2
    ) ** 2
    sigma_x = sigma_x * (
        (L2 / L1) ** 2 * R1**2 / 4 + (1 + L2 / L1) ** 2 * R2**2 / 4 + x3**2 / 12
    )
    sigma_x = sigma_x + Qx**2 / 6 * r
    sigma_y = (
        2 * np.pi * np.cos(theta) * np.cos(2 * theta) ** 2 / wavelength / L2
    ) ** 2
    sigma_y = (
        sigma_y
        * (
            (L2 / L1) ** 2 * R1**2 / 4
            + (1 + L2 / L1) ** 2 * R2**2 / 4
            + y3**2 / 12
            + 2 * B**2 * wavelength**4 * r / 3
        )
        + Qy**2 / 6 * r
    )
    sigma_x = np.sqrt(sigma_x)
    sigma_y = np.sqrt(sigma_y)

    return sigma_x, sigma_y


def test_mono_resolution():
    """Test resolution"""
    R1 = 0.02
    R2 = 0.007

    x3 = 0.0055
    y3 = 0.0043
    L1 = 15
    L2 = 15.5

    wavelength = 6.0
    delta_lambda = 0.15

    Qx = -5.93411755e-04
    Qy = -7.67944624e-04

    two_theta = 0.00092676  # radian (corner pixel)
    theta = 0.00092676 * 0.5  # radian (corner pixel)

    dqx, dqy = sigma_neutron(
        wavelength, delta_lambda, Qx, Qy, theta, L1, L2, R1, R2, x3, y3
    )

    # Calculate by drtsans.mono method
    setup = drtsans.resolution.InstrumentSetupParameters(L1, L2, R1, R2)
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
        np.array([np.arctan2(Qy, Qx)]),
        np.array([L2]),
        np.array([True]),
        np.array([x3]),
        np.array([y3]),
    )
    dqx3, dqy3 = mono_resolution(
        np.array([Qx]),
        np.array([Qy]),
        mode="azimuthal",
        wavelength=wavelength,
        delta_wavelength=delta_lambda,
        instrument_parameters=setup,
        pixel_info=pix,
    )

    # Check
    assert dqx3 == pytest.approx(dqx, abs=1e-8)
    assert dqy3 == pytest.approx(dqy, abs=1e-8)


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "name": "GPSANS",
            "Nx": 5,
            "Ny": 5,
            "dx": 0.00425,
            "dy": 0.0055,
            "xc": 0.0,
            "yc": 0.0,
            "zc": 15.5,
            "l1": 15,
            "axis_values": [5.925, 6.075],
        }
    ],
    indirect=True,
)
def test_retrieve_instrument_setup(generic_workspace):
    workspace = generic_workspace

    # Insert logs
    names = (
        "wavelength wavelength-spread source_aperture_diameter sample_aperture_diameter "
        "smearingPixelSizeX smearingPixelSizeY source-sample-distance sample-detector-distance".split()
    )
    values = [6.0, 0.15, 0.02, 0.007, 0.0085, 0.011, 15.0, 15.5]
    units = "A A mm mm m m m m".split()
    for name, value, unit in zip(names, values, units):
        AddSampleLog(
            Workspace=workspace,
            LogName=name,
            LogText="{}".format(value),
            LogType="Number",
            LogUnit=unit,
        )

    params = retrieve_instrument_setup(workspace)
    assert [
        params.smearing_pixel_width_ratio,
        params.smearing_pixel_height_ratio,
    ] == pytest.approx([2.0, 2.0])


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "name": "GPSANS",
            "Nx": 5,
            "Ny": 5,
            "dx": 0.00425,
            "dy": 0.0055,
            "xc": 0.0,
            "yc": 0.0,
            "zc": 15.5,
            "l1": 15,
            "axis_values": [5.925, 6.075],
        }
    ],
    indirect=True,
)
def test_momentum_and_resolution(generic_workspace):
    """
    Test Q resolution method against Wei-ren and Ricardo's early implementation
    Parameters
    ----------
    generic_workspace : Workspace instance
        A generic workspace with 5 x 5 instrument

    Returns
    -------

    """
    # Define constants
    wavelength = 6
    delta_lambda = 0.15
    R1 = 0.02  # source aperture radius
    R2 = 0.007  # sample aperture radius
    x3 = 0.00425  # pixel X size (meter)
    y3 = 0.00550  # pixel Y size (meter)
    L1 = 15  # meter
    L2 = 15.5  # meter (sample to detetor center distance)

    # Get workspace and add sample logs as wave length, wave length spread,
    ws = generic_workspace
    AddSampleLog(
        Workspace=ws,
        LogName="wavelength",
        LogText="{}".format(wavelength),
        LogType="Number",
        LogUnit="A",
    )
    AddSampleLog(
        Workspace=ws,
        LogName="wavelength-spread",
        LogText="{}".format(delta_lambda),
        LogType="Number",
        LogUnit="A",
    )
    AddSampleLog(
        Workspace=ws,
        LogName="source_aperture_diameter",
        LogText="{}".format(R1 * 2.0 * 1000),
        LogType="Number",
        LogUnit="mm",
    )
    AddSampleLog(
        Workspace=ws,
        LogName="sample_aperture_diameter",
        LogText="{}".format(R2 * 2.0 * 1000),
        LogType="Number",
        LogUnit="mm",
    )

    # Calculate Q and dQ
    result = convert_to_q(ws, mode="azimuthal")
    qx = result.qx.reshape(5, 5)
    qy = result.qy.reshape(5, 5)
    dqx = result.delta_qx.reshape(5, 5)
    dqy = result.delta_qy.reshape(5, 5)
    lam = result.wavelength
    assert lam == pytest.approx(np.array([6] * 25), abs=1e-8)

    for i in range(5):
        for j in range(5):
            # positions
            x = x3 * (2 - i)
            y = y3 * (j - 2)
            tt = np.arctan2(np.sqrt(x**2 + y**2), L2)
            azi = np.arctan2(y, x)
            # expected momentum
            mod_q = 4 * np.pi * np.sin(0.5 * tt) / wavelength
            expected_qx = mod_q * np.cos(azi)
            expected_qy = mod_q * np.sin(azi)
            assert qx[i, j] * (-1) == pytest.approx(expected_qx, abs=1e-8)
            assert qy[i, j] == pytest.approx(expected_qy, abs=1e-8)
            # expected resolution
            sigma_x, sigma_y = sigma_neutron(
                wavelength,
                delta_lambda,
                qx[i, j],
                qy[i, j],
                0.5 * tt,
                L1,
                L2,
                R1,
                R2,
                x3,
                y3,
            )
            assert dqx[i, j] == pytest.approx(sigma_x, abs=1e-8)
            assert dqy[i, j] == pytest.approx(sigma_y, abs=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
