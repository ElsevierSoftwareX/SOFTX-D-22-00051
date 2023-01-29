import numpy as np
from scipy import constants


__all__ = [
    "InstrumentSetupParameters",
    "calculate_sigma_theta_prefactor",
    "calculate_sigma_geometry",
    "calculate_sigma_theta_geometry",
    "calculate_sigma_theta_gravity",
]


class InstrumentSetupParameters(object):
    """
    Data structure containing the parameters used to calculate Q resolution
    """

    def __init__(
        self,
        l1,
        sample_det_center_dist,
        source_aperture_radius,
        sample_aperture_radius,
        pixel_width_ratio=None,
        pixel_height_ratio=None,
    ):
        """
        Initialization to set all the parameters (6) to calculate momentrum transfer resolution

        Parameters
        ----------
        l1:
            source to sample distance
        sample_det_center_dist:
            sample detector (bank) center distance
        source_aperture_radius:
            source aperture radius (meter)
        sample_aperture_radius:
            sample aperture radius (meter)
        pixel_width_ratio: float
            custom pixel width ratio (relative to nominal width)to replace the nominal pixel width of the
            instrument pixel detectors. Only
            for the purpose of Q-resolution calculation.
        pixel_height_ratio: float
            custom pixel height ratio (relative to nominal height) to replace the nominal pixel height
            of the instrument pixel detectors. Only
            for the purpose of Q-resolution calculation.
        """
        self._l1 = l1
        self._sample_det_center_dist = sample_det_center_dist
        self._source_aperture = source_aperture_radius
        self._sample_aperture = sample_aperture_radius
        self.smearing_pixel_width_ratio, self.smearing_pixel_height_ratio = (
            pixel_width_ratio,
            pixel_height_ratio,
        )

    def __str__(self):
        """
        Nice output string
        :return:
        """
        out = "L1 = {} (m)\nSample-Detector-Center-Distance (L2)= {} (m)\n" "".format(
            self.l1, self._sample_det_center_dist
        )
        out += "Source aperture radius (R1) = {} (m)\n".format(self._source_aperture)
        out += "Sample aperture radius (R2) = {} (m)\n".format(self._sample_aperture)

        return out

    @property
    def l1(self):
        """
        Get L1 value
        :return: L1 (meter)
        """
        return self._l1

    @property
    def sample_det_center_distance(self):
        """
        Distance from sample to detector bank center,
        which is L2 in the SANS master document
        :return: sample detector center distance, aka SANS L2 (meter)
        """
        return self._sample_det_center_dist

    @property
    def source_aperture_radius(self):
        """
        Source aperture radius, which is R1 in SANS master document
        :return: source aperture radius (R1) in meter
        """
        return self._source_aperture

    @property
    def sample_aperture_radius(self):
        """
        Sample aperture radius, which is R2 in SANS master document
        :return: sample aperture radius (R2) in meter
        """
        return self._sample_aperture


def calculate_sigma_theta_prefactor(wavelength, pixel_info, instrument_parameters):
    r"""
    Calculates for every pixel and wavelength

    .. math::

       \left(\frac{2\pi\cos\theta\cos^2(2\theta)}{\lambda L_2}\right)^2


    Parameters
    ----------

    wavelength: ~np.array
        the array of wavelengths (same shape as momentum transfer)
    pixel_info: ~collections.namedtuple
        A namedtuple with fields for two_theta, azimuthal, l2, keep
    instrument_parameters: ~drtsans.resolution.InstrumentSetupParameters
        Data structure containing the parameters used to calculate Q resolution. In particular:
        1. distance from source aperture to sample,
        2. distance from sample to detector,
        3. source aperture radius,
        4. sample aperture radius,
        5. custom pixel width and height to replace nominal pixel width and height, only for Q-resolution calculation.

    Returns
    -------
    float
        The coefficient described above
    """
    two_theta = pixel_info.two_theta.reshape(-1, 1)
    L2 = instrument_parameters.sample_det_center_distance
    return np.square(
        2 * np.pi * np.cos(0.5 * two_theta) * np.cos(two_theta) ** 2 / wavelength / L2
    )


def calculate_sigma_theta_geometry(mode, pixel_info, instrument_parameters):
    r"""
    Calculates the effect of the geometry and wavelength uncertainty on the uncertainty in the value of Q.

    .. math::

       \left(\frac {L_2}{L_1}\right)^2\frac{R_1^2}{4}+\left(\frac {L_1+L_2}{L_1}\right)^2\frac{R_2^2}{4}+
       \frac {1}{12}(\Delta R)^2

    If mode is "scalar", :math:`((\Delta R)^2=(\Delta x)^2+(\Delta y)^2)/2`, else

    :math:`(\Delta R)^2=[(\Delta x)^2,(\Delta y)^2]`. The formula for scalar is consistent with
    the equations 10.3 and 10.4 in the master document. when you add the two together, the geometry
    part is twice the contribution of :math:`(\Delta R)^2` plus the gravity part.


    Parameters
    ----------
    mode: str
        One of "scalar", "azimuthal", "crystalographic"
    pixel_info: ~collections.namedtuple
        A namedtuple with fields for two_theta, azimuthal, l2, keep, smearing_pixel_size_x, smearing_pixel_size_y
    instrument_parameters: ~drtsans.resolution.InstrumentSetupParameters
        Data structure containing the parameters used to calculate Q resolution. In particular:
        1. distance from source aperture to sample,
        2. distance from sample to detector,
        3. source aperture radius,
        4. sample aperture radius,
        5. custom pixel width and height to replace nominal pixel width and height, only for Q-resolution calculation.

    Returns
    -------
    float or list
        The coefficient described above
    """
    L1 = instrument_parameters.l1
    L2 = instrument_parameters.sample_det_center_distance
    R1 = instrument_parameters.source_aperture_radius
    R2 = instrument_parameters.sample_aperture_radius
    dx = pixel_info.smearing_pixel_size_x
    dy = pixel_info.smearing_pixel_size_y

    # Rescale pixel dimensions if custom pixel dimensions are present in the instrument parameters
    if instrument_parameters.smearing_pixel_width_ratio is not None:
        dx *= instrument_parameters.smearing_pixel_width_ratio
    if instrument_parameters.smearing_pixel_height_ratio is not None:
        dy *= instrument_parameters.smearing_pixel_height_ratio

    dx2, dy2 = np.square(dx), np.square(dy)

    if mode == "scalar":
        pixel_size2 = 0.5 * (dx2 + dy2)
    elif mode == "azimuthal":
        pixel_size2 = np.array([dx2, dy2])
    return (
        0.25 * np.square(L2 / L1 * R1)
        + 0.25 * np.square((L1 + L2) / L1 * R2)
        + pixel_size2 / 12.0
    )


def calculate_sigma_theta_gravity(wavelength, delta_wavelength, instrument_parameters):
    r"""
    Calculates

    .. math::

       \frac 23 B^2\lambda^2(\Delta\lambda)^2

    where :math:`B=g m_N^2L_2(L_1+L_2)/(2h^2)`

    Parameters
    ----------
    wavelength: ~np.array
        the array of wavelengths
    delta_wavelength: ~np.array
        the array of wavelength spreads
    instrument_parameters: ~drtsans.resolution.InstrumentSetupParameters
        Data structure containing the parameters used to calculate Q resolution. In particular:
        1. distance from source aperture to sample,
        2. distance from sample to detector,
        3. source aperture radius,
        4. sample aperture radius,
        5. custom pixel width and height to replace nominal pixel width and height, only for Q-resolution calculation.

    Returns
    -------
    ~np.array
        The formula above
    """
    # derived constant where:
    # h = 6.626e-34    # m^2 kg s^-1
    # m_n = 1.675e-27  # kg
    # g = 9.8          # m s^-2
    G_MN2_OVER_H2 = constants.g * np.square(
        constants.neutron_mass / constants.h
    )  # Unit as m, s, Kg
    L1 = instrument_parameters.l1
    L2 = instrument_parameters.sample_det_center_distance
    B = 0.5 * G_MN2_OVER_H2 * L2 * (L1 + L2) * 1.0e-20
    return 2.0 * np.square(B * wavelength * delta_wavelength) / 3.0


def calculate_sigma_geometry(
    mode, wavelength, delta_wavelength, pixel_info, instrument_parameters
):
    r"""
    Calculates the Q independent part of the resolution, the common parts in formula 10.3 - 10.6

    Parameters
    ----------
    mode: str
        One of "scalar", "azimuthal", "crystalographic"
    wavelength: ~np.array
        the array of wavelengths (same shape as momentum transfer)
    delta_wavelength: ~np.array
        the array of wavelength widths (same shape as momentum transfer)
    pixel_info: ~collections.namedtuple
        A namedtuple with fields for two_theta, azimuthal, l2, keep
    instrument_parameters: ~drtsans.resolution.InstrumentSetupParameters
        Data structure containing the parameters used to calculate Q resolution. In particular:
        1. distance from source aperture to sample,
        2. distance from sample to detector,
        3. source aperture radius,
        4. sample aperture radius,
        5. custom pixel width and height to replace nominal pixel width and height, only for Q-resolution calculation.

    Returns
    =======
    ~np.array
    """
    factor = calculate_sigma_theta_prefactor(
        wavelength, pixel_info, instrument_parameters
    )
    geometry_part = calculate_sigma_theta_geometry(
        mode, pixel_info, instrument_parameters
    )
    gravity_part = calculate_sigma_theta_gravity(
        wavelength, delta_wavelength, instrument_parameters
    )

    if mode == "scalar":
        return factor * (geometry_part[:, np.newaxis] * 2 + gravity_part)
    if mode == "azimuthal":
        return [
            factor * geometry_part[0][:, np.newaxis],
            factor * (geometry_part[1][:, np.newaxis] + gravity_part),
        ]
