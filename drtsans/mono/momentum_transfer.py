import numpy as np

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/momentum_transfer.py
import drtsans.momentum_transfer

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/resolution.py
import drtsans.resolution

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/geometry.py
from drtsans.geometry import (
    logged_smearing_pixel_size,
    sample_aperture_diameter,
    sample_detector_distance,
    source_aperture_diameter,
    source_sample_distance,
    nominal_pixel_size,
)

__all__ = ["convert_to_q"]


def mono_resolution(*args, **kwargs):
    r"""
    Function to compute the resolution for GPSANS and BIOSANS. Some parts are calculated in drtsans.resolution

    Parameters
    ----------
    args:
        Either |Q| or Qx and Qy
    kwargs: dict
        The following parameters are required

        - mode: one of 'scalar', 'azimuthal'
        - instrument_parameters: an InstrumentSetupParameters object containg information about L1, apertures, etc.
        - pixel_info: dict containing arrays for two_theta, azimuthal, L2 (per pixel)
        - wavelength: array of wavenegths (same shape as Q)
        - delta_wavelength: array of wavenegth widths (same shape as Q)

    Returns
    -------
    ~np.array or list of arrays
        Resolution along the components of momentum transfer from the input
    """
    mode = kwargs.get("mode")
    instrument_parameters = kwargs.get("instrument_parameters")
    pixel_info = kwargs.get("pixel_info")
    wavelength = kwargs.get("wavelength")
    delta_wavelength = kwargs.get("delta_wavelength")

    # get the general sigma_geom^2 (not mode/ instrument dependent)
    sigma_geom = drtsans.resolution.calculate_sigma_geometry(
        mode, wavelength, delta_wavelength, pixel_info, instrument_parameters
    )

    # return resolution according to formulas 10.5 and 10.6 in the master document
    if mode == "scalar":
        q = args[0]
        return np.sqrt(
            sigma_geom + np.square(q) * np.square(delta_wavelength / wavelength) / 6.0
        )
    if mode == "azimuthal":
        qx = args[0]
        qy = args[1]
        return [
            np.sqrt(
                sigma_geom[0]
                + np.square(qx) * np.square(delta_wavelength / wavelength) / 6.0
            ),
            np.sqrt(
                sigma_geom[1]
                + np.square(qy) * np.square(delta_wavelength / wavelength) / 6.0
            ),
        ]

    # should not get here
    raise NotImplementedError("The mode you selected is not yet implemented")


def convert_to_q(ws, mode, resolution_function=mono_resolution, **kwargs):
    r"""
    Convert a workspace with units of wavelength into a
    series of arrays: intensity, error, q (or q components),
    delta q (or delta q components), and wavelength

    Using the scattering angle as :math:`2\theta` and azimuthan angle as
    :math:`\phi`,the calculaion of momentum transfer is:

    - 'scalar' mode:

    .. math:: |Q| = \frac{4\pi}{\lambda}\sin\theta

    - 'azimuthal' mode:

    .. math::

       Q_x=\frac{4\pi}{\lambda}\sin\theta\cos\phi

       Q_y=\frac{4\pi}{\lambda}\sin\theta\sin\phi

    - 'crystallographic' mode:

    .. math::

       Q_x=\frac{2\pi}{\lambda}\sin(2\theta)\cos\phi

       Q_y=\frac{2\pi}{\lambda}\sin(2\theta)\sin\phi

       Qz_=\frac{2\pi}{\lambda}(\cos(2\theta)-1)


    It calls drtsans.momentum_transfer.convert_to_q

    Parameters
    ----------

    ws:  str, ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Workspace in units of wavelength
    mode: str
        Available options are 'scalar', 'azimuthal', and 'crystallographic'
    resolution_function:
        Function to calculate resolution
    kwargs:
        Parameters to be passed to the resolution function

    Returns
    -------
    ~collections.namedtuple
       A namedtuple with fields for

      - intensity
      - error
      - mod_q (:math:`|Q|`) or qx, qy (:math:`Q_x, Q_y`) or qx, qy, qz (:math:`Q_x, Q_y, Q_z`) (depending on the mode)
      - delta_q or delta_qx, delta_qy or delta_qx, delta_qy, delta_qz - the resolution along the q components
      - wavelength

    """
    # get the InstrumentSetupParameters
    instrument_setup = retrieve_instrument_setup(ws)
    return drtsans.momentum_transfer.convert_to_q(
        ws, mode, resolution_function, instrument_parameters=instrument_setup, **kwargs
    )


def retrieve_instrument_setup(input_workspace):
    """
    Collect instrument parameters to be used in the calculation of Q-resolution.

    Parameters collected are:
    - L1
    - L2
    - distance between the source aperture and the sample
    - distance between sample and the detector
    - diameter of the source aperture
    - diameter of the sample aperture
    - custom pixel width and height

    Parameters
    ----------
    input_workspace:  str, ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace

    Returns
    -------
    ~drtsans.resolution.InstrumentSetupParameters
    """
    l1 = source_sample_distance(input_workspace, unit="m")
    l2 = sample_detector_distance(
        input_workspace, unit="m", search_logs=False
    )  # enforce to use calculated SDD/L2
    r1 = source_aperture_diameter(input_workspace, unit="m") / 2.0
    r2 = sample_aperture_diameter(input_workspace, unit="m") / 2.0
    pixel_width, pixel_height = logged_smearing_pixel_size(input_workspace)

    nominal_pixel = nominal_pixel_size(input_workspace)
    pixel_width_ratio = None
    pixel_height_ratio = None
    if pixel_width is not None:
        pixel_width_ratio = pixel_width / nominal_pixel.width
    if pixel_height is not None:
        pixel_height_ratio = pixel_height / nominal_pixel.height

    setup_params = drtsans.resolution.InstrumentSetupParameters(
        l1=l1,
        sample_det_center_dist=l2,
        source_aperture_radius=r1,
        sample_aperture_radius=r2,
        pixel_width_ratio=pixel_width_ratio,
        pixel_height_ratio=pixel_height_ratio,
    )
    return setup_params
