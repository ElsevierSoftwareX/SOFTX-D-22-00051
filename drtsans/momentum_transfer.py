import numpy as np

from mantid.simpleapi import mtd

r"""
Hyperlinks to drtsans functions
IQazimuthal, IQcrystal, IQmod <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/dataobjects.py>
pixel_centers <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/pixel_centers.py>
namedtuplefy, unpack_v3d <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
"""  # noqa: E501
from drtsans.dataobjects import IQazimuthal, IQcrystal, IQmod
from drtsans.geometry import pixel_centers
from drtsans.settings import namedtuplefy, unpack_v3d

__all__ = ["convert_to_q"]


def convert_to_q(ws, mode, resolution_function=None, **kwargs):
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

       Q_x=-\frac{4\pi}{\lambda}\sin\theta\cos\phi

       Q_y=\frac{4\pi}{\lambda}\sin\theta\sin\phi

    - 'crystallographic' mode:

    .. math::

       Q_x=\frac{2\pi}{\lambda}\sin(2\theta)\cos\phi

       Q_y=\frac{2\pi}{\lambda}\sin(2\theta)\sin\phi

       Qz_=\frac{2\pi}{\lambda}(\cos(2\theta)-1)

    Note the minus sign in :math:`Q_x` in the azimuthal mode, so it increases
    to the right when looking at the detector.

    devs - Andrei Savici <saviciat@ornl.gov>

    Parameters
    ----------

    ws:  str, ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Workspace in units of wavelength
    mode: str
        Available options are 'scalar', 'azimuthal', and 'crystallographic'
    resolution_function:
        Function to calculate resolution. If :py:obj:`None`, then we assume an infinite preccission in Q and the
        associated error is thus zero.
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

    # check that the workspace is in units of wavelength
    if not ws:
        raise RuntimeError("Workspace cannot be None")
    wsh = mtd[str(ws)]
    if wsh.getAxis(0).getUnit().unitID() != "Wavelength":
        raise RuntimeError(
            "Input workspace {} for calculate Q and resolution must be in unit Wavelength but not {}"
            "".format(wsh, wsh.getAxis(0).getUnit().unitID())
        )

    # switch according to mode
    if mode == "scalar":
        return _convert_to_q_scalar(wsh, resolution_function, **kwargs)
    if mode == "azimuthal":
        return _convert_to_q_azimuthal(wsh, resolution_function, **kwargs)
    if mode == "crystallographic":
        return _convert_to_q_crystal(wsh, resolution_function, **kwargs)
    raise NotImplementedError("The mode you selected is not yet implemented")


def _convert_to_q_scalar(ws, resolution_function, **kwargs):
    r"""
    Convert to scalar momentum transfer

    **Mantid algorithms used:**
        :ref:`ConvertUnits <algm-ConvertUnits-v1>`,
        :ref:`ConvertToPointData <algm-ConvertToPointData-v1>`,
        :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`

    devs - Andrei Savici <saviciat@ornl.gov>
           Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------

    ws:  ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Workspace in units of wavelength
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
      - mod_q
      - delta_q
      - wavelength

    """
    # get the data
    lam = ws.extractX()  # wavelength boundaries
    delta_lam = lam[:, 1:] - lam[:, :-1]  # wavelength bins
    lam = (lam[:, 1:] + lam[:, :-1]) * 0.5  # wavelengths at the bin centers
    intensity = ws.extractY()
    error = ws.extractE()

    # get geometry info from the original workspace for resolution
    info = pixel_info(ws)  # polar coordinates for each pixel
    number_of_bins = lam.shape[1]
    two_theta = np.repeat(info.two_theta, number_of_bins).reshape(-1, number_of_bins)
    mod_q = 4.0 * np.pi * np.sin(two_theta * 0.5) / lam

    # calculate the  resolution for each pixel
    if resolution_function is not None:
        delta_q = resolution_function(
            mod_q,
            mode="scalar",
            pixel_info=info,
            wavelength=lam,
            delta_wavelength=delta_lam,
            **kwargs
        )
    else:
        delta_q = mod_q * 0.0

    # Has the user requested subpixels?
    keep = info.keep.astype(bool)  # valid spectra indexes (unmasked, not monitor)
    n_horizontal, n_vertical = kwargs.get("n_horizontal", 1), kwargs.get(
        "n_vertical", 1
    )
    # We filter out the bad pixels and then for each pixel we replicate the list of wavelength bin centers,
    # as many times as the number of subpixels in the pixel.
    # `lam` becomes an 1D array of length (number_of_good_pixels * number_of_bins * n_horizontal * n_vertical)
    [lam,] = _filter_and_replicate(
        [
            lam,
        ],
        keep,
        n_horizontal,
        n_vertical,
    )
    if n_horizontal * n_vertical > 1:
        # Calculate modulus Q for each subpixel
        subpixel_polar_coords = subpixel_info(ws, n_horizontal, n_vertical)
        # We need to replicate the two_theta of a subpixel as many times as wavelength bin centers
        # `two_theta` is a 2D array of shape (number_of_good_pixels * n_horizontal * n_vertical, number_of_bins)
        two_theta = np.repeat(subpixel_polar_coords.two_theta, number_of_bins).reshape(
            -1, number_of_bins
        )
        # For a given subpixel, we have as many Q-moduli as wavelength bin centers, i.e, number_of_bins
        # We need to reshape `lam` to have the same shape as `two_theta`. After that, we flatten the resulting
        # mod_q array into a 1D array of length (number_of_good_pixels * number_of_bins * n_horizontal * n_vertical)
        mod_q = (
            4.0
            * np.pi
            * np.sin(two_theta * 0.5)
            / lam.reshape(len(two_theta), number_of_bins)
        ).ravel()
        # Scale the error by square root of (n_horizontal * n_vertical)
        error *= np.sqrt(n_horizontal * n_vertical)
    else:
        # retain only those pixels that are unmasked or not monitor
        [mod_q,] = _filter_and_replicate(
            [
                mod_q,
            ],
            keep,
            n_horizontal=1,
            n_vertical=1,
        )

    # retain only those pixels that are unmasked or not monitor
    # if user requested subpixels, then replicate pixel quantities as many times as the number of subpixels
    # in the pixel
    intensity, error, delta_q = _filter_and_replicate(
        [intensity, error, delta_q], keep, n_horizontal, n_vertical
    )

    return IQmod(
        intensity=intensity,
        error=error,
        mod_q=mod_q,
        delta_mod_q=delta_q,
        wavelength=lam,
    )


def _convert_to_q_azimuthal(ws, resolution_function, **kwargs):
    r"""
    Convert to 2D momentum transfer in azimuthal convention

    **Mantid algorithms used:**
        :ref:`ConvertUnits <algm-ConvertUnits-v1>`,
        :ref:`ConvertToPointData <algm-ConvertToPointData-v1>`,
        :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`

    devs - Andrei Savici <saviciat@ornl.gov>
           Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------

    ws:  ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Workspace in units of wavelength
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
      - qx
      - qy
      - delta_qx
      - delta_qy
      - wavelength

    """
    # get the data
    lam = ws.extractX()  # wavelength boundaries
    delta_lam = lam[:, 1:] - lam[:, :-1]  # wavelength bins
    lam = (lam[:, 1:] + lam[:, :-1]) * 0.5  # wavelengths at the bin centers
    intensity = ws.extractY()
    error = ws.extractE()

    # get geometry info from the original workspace for resolution
    info = pixel_info(ws)  # polar coordinates for each pixel
    number_of_bins = lam.shape[1]
    two_theta = np.repeat(info.two_theta, number_of_bins).reshape(-1, number_of_bins)
    mod_q = 4.0 * np.pi * np.sin(two_theta * 0.5) / lam
    azimuthal = np.repeat(info.azimuthal, number_of_bins).reshape(-1, number_of_bins)
    qx = -mod_q * np.cos(
        azimuthal
    )  # note the convention for the left handed reference frame
    qy = mod_q * np.sin(azimuthal)

    # calculate the resolution for pixels
    if resolution_function is not None:
        delta_qx, delta_qy = resolution_function(
            qx,
            qy,
            mode="azimuthal",
            pixel_info=info,
            wavelength=lam,
            delta_wavelength=delta_lam,
            **kwargs
        )
    else:
        delta_qx = mod_q * 0.0
        delta_qy = delta_qx

    # Has the user requested subpixels?
    keep = info.keep.astype(bool)  # valid spectra indexes (unmasked, not monitor)
    n_horizontal, n_vertical = kwargs.get("n_horizontal", 1), kwargs.get(
        "n_vertical", 1
    )
    # We filter out the bad pixels and then for each pixel we replicate the list of wavelength bin centers,
    # as many times as the number of subpixels in the pixel.
    # `lam` becomes an 1D array of length (number_of_good_pixels * number_of_bins * n_horizontal * n_vertical)
    [lam,] = _filter_and_replicate(
        [
            lam,
        ],
        keep,
        n_horizontal,
        n_vertical,
    )
    if n_horizontal * n_vertical > 1:
        # Calculate modulus Q for each subpixel
        subpixel_polar_coords = subpixel_info(ws, n_horizontal, n_vertical)
        # We need to replicate the two_theta of a subpixel as many times as wavelength bin centers
        # `two_theta` is a 2D array of shape (number_of_good_pixels * n_horizontal * n_vertical, number_of_bins)
        two_theta = np.repeat(subpixel_polar_coords.two_theta, number_of_bins).reshape(
            -1, number_of_bins
        )
        # For a given subpixel, we have as many Q-modules as wavelength bin centers, i.e, number_of_bins
        # We need to reshape `lam` to have the same shape as `two_theta`. After that, we flatten the resulting
        # mod_q array into a 1D array of length (number_of_good_pixels * number_of_bins * n_horizontal * n_vertical)
        mod_q = (
            4.0
            * np.pi
            * np.sin(two_theta * 0.5)
            / lam.reshape(len(two_theta), number_of_bins)
        ).ravel()
        azimuthal = np.repeat(subpixel_polar_coords.azimuthal, number_of_bins)
        qx = -mod_q * np.cos(
            azimuthal
        )  # note the convention for the left handed reference frame
        qy = mod_q * np.sin(azimuthal)
        # Scale the error by square root of (n_horizontal * n_vertical)
        error *= np.sqrt(n_horizontal * n_vertical)
    else:
        # retain only those pixels that are unmasked or not monitor
        qx, qy = _filter_and_replicate([qx, qy], keep, n_horizontal=1, n_vertical=1)

    # retain only those pixels that are unmasked or not monitor
    # if user requested subpixels, then replicate pixel quantities as many times as the number of subpixels
    # in the pixel
    intensity, error, delta_qx, delta_qy = _filter_and_replicate(
        [intensity, error, delta_qx, delta_qy], keep, n_horizontal, n_vertical
    )

    return IQazimuthal(
        intensity=intensity,
        error=error,
        qx=qx,
        qy=qy,
        delta_qx=delta_qx,
        delta_qy=delta_qy,
        wavelength=lam,
    )


def _convert_to_q_crystal(ws, resolution_function, **kwargs):
    r"""
    Convert to 3D momentum transfer in crystallographic convention

    **Mantid algorithms used:**
        :ref:`ConvertUnits <algm-ConvertUnits-v1>`,
        :ref:`ConvertToPointData <algm-ConvertToPointData-v1>`,
        :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`

    devs - Andrei Savici <saviciat@ornl.gov>
           Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------

    ws:  ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Workspace in units of wavelength
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
      - qx
      - qy
      - qz
      - delta_qx
      - delta_qy
      - delta_qz
      - wavelength

    """
    # get the data
    lam = ws.extractX()  # wavelength boundaries
    delta_lam = lam[:, 1:] - lam[:, :-1]  # wavelength bins
    lam = (lam[:, 1:] + lam[:, :-1]) * 0.5  # wavelengths at the bin centers
    intensity = ws.extractY()
    error = ws.extractE()

    # get geometry info from the original workspace for resolution
    info = pixel_info(ws)  # polar coordinates for each pixel
    number_of_bins = lam.shape[1]
    two_theta = np.repeat(info.two_theta, number_of_bins).reshape(-1, number_of_bins)
    azimuthal = np.repeat(info.azimuthal, number_of_bins).reshape(-1, number_of_bins)
    temp = 2.0 * np.pi / lam
    qx = temp * np.sin(two_theta) * np.cos(azimuthal)
    qy = temp * np.sin(two_theta) * np.sin(azimuthal)
    qz = temp * (np.cos(two_theta) - 1.0)

    # calculate the  resolution for each pixel
    if resolution_function is not None:
        delta_qx, delta_qy, delta_qz = resolution_function(
            qx,
            qy,
            qz,
            mode="crystallographic",
            pixel_info=info,
            wavelength=lam,
            delta_wavelength=delta_lam,
            **kwargs
        )
    else:
        delta_qx = lam * 0.0
        delta_qy = delta_qx
        delta_qz = delta_qx

    # Has the user requested subpixels?
    keep = info.keep.astype(bool)  # valid spectra indexes (unmasked, not monitor)
    n_horizontal, n_vertical = kwargs.get("n_horizontal", 1), kwargs.get(
        "n_vertical", 1
    )
    # We filter out the bad pixels and then for each pixel we replicate the list of wavelength bin centers,
    # as many times as the number of subpixels in the pixel.
    # lam becomes an 1D array of length (number_of_good_pixels * number_of_bins * n_horizontal * n_vertical)
    [lam,] = _filter_and_replicate(
        [
            lam,
        ],
        keep,
        n_horizontal,
        n_vertical,
    )
    if n_horizontal * n_vertical > 1:
        # Calculate modulus Q for each subpixel
        subpixel_polar_coords = subpixel_info(ws, n_horizontal, n_vertical)
        # We need to replicate the two_theta of a subpixel as many times as wavelength bin centers.
        # `two_theta` is a 2D array of shape (number_of_good_pixels * n_horizontal * n_vertical, number_of_bins)
        two_theta = np.repeat(subpixel_polar_coords.two_theta, number_of_bins).reshape(
            -1, number_of_bins
        )
        azimuthal = np.repeat(subpixel_polar_coords.azimuthal, number_of_bins).reshape(
            -1, number_of_bins
        )
        # We need to reshape `lam` to have the same shape as `two_theta`.
        temp = 2.0 * np.pi / lam.reshape(len(two_theta), number_of_bins)
        qx = (temp * np.sin(two_theta) * np.cos(azimuthal)).ravel()
        qy = (temp * np.sin(two_theta) * np.sin(azimuthal)).ravel()
        qz = (temp * (np.cos(two_theta) - 1.0)).ravel()
        # Scale the error by square root of (n_horizontal * n_vertical)
        error *= np.sqrt(n_horizontal * n_vertical)
    else:
        # retain only those pixels that are unmasked or not monitor
        [qx, qy, qz] = _filter_and_replicate(
            [qx, qy, qz], keep, n_horizontal=1, n_vertical=1
        )

    # retain only those pixels that are unmasked or not monitor
    # if user requested subpixels, then replicate pixel quantities as many times as the number of subpixels
    # in the pixel
    intensity, error = _filter_and_replicate(
        [intensity, error], keep, n_horizontal, n_vertical
    )
    delta_qx, delta_qy, delta_qz = _filter_and_replicate(
        [delta_qx, delta_qy, delta_qz], keep, n_horizontal, n_vertical
    )

    return IQcrystal(
        intensity=intensity,
        error=error,
        qx=qx,
        qy=qy,
        qz=qz,
        delta_qx=delta_qx,
        delta_qy=delta_qy,
        delta_qz=delta_qz,
        wavelength=lam,
    )


def _masked_or_monitor(spec_info, idx):
    r"""
    Helper function to check if a spectra is valid

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    spec_info: ~mantid.api.SpectrumInfo
        SpectrumInfo from a workspace
    idx: int
        index

    Returns
    -------
    bool
        True if spectrum has no detectors, the detector is a monitor, or the spectrum is masked
        False otherwise
    """
    return (
        spec_info.isMonitor(idx)
        or spec_info.isMasked(idx)
        or not spec_info.hasDetectors(idx)
    )


@namedtuplefy
def pixel_info(input_workspace):
    r"""
    Helper function to extract: two theta angle, azimuthal angle, l2, smearing_pixel_size_x, smearing_pixel_size_y,
    and a "keep" flag for unmasked pixel detectors.

    devs - Andrei Savici <saviciat@ornl.gov>
           Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    input_workspace: str, ~mantid.api.IEventWorkspace, ~maFntid.api.MatrixWorkspace
        Name or reference to a Mantid workspace

    Returns
    -------
    ~collections.namedtuple
        A namedtuple with fields for two_theta, azimuthal, l2, keep
    """
    ws = mtd[str(input_workspace)]
    spectrum_info = ws.spectrumInfo()
    number_spectra = ws.getNumberHistograms()

    info = [
        [np.nan, np.nan, np.nan, False]
        if _masked_or_monitor(spectrum_info, i)
        else [
            spectrum_info.twoTheta(i),
            spectrum_info.azimuthal(i),
            spectrum_info.l2(i),
            True,
        ]
        for i in range(number_spectra)
    ]
    info = np.array(info)
    info_map = dict(
        two_theta=info[:, 0], azimuthal=info[:, 1], l2=info[:, 2], keep=info[:, 3]
    )

    # Find out pixel dimensions
    component_info = ws.componentInfo()
    # Find the componentInfo indexes corresponding to the workspace indexes
    get_spectrum_definition = spectrum_info.getSpectrumDefinition
    info_indexes = [get_spectrum_definition(idx)[0][0] for idx in range(number_spectra)]
    # Find the dimensions of each pixel. Due to barscan and tube-width calibrations, each pixel will have its own size
    last_info_index = info_indexes[-1]  # index of the last valid pixel
    # The nominal pixel dimensions are the same for all pixels. We find the nominal dimensions of the last pixel.
    # The `nominal_pixel_dimensions` for EQSANS are (0.00804, 0.00409, 0.00804) along X, Y, and Z, respectively.
    nominal_pixel_dimensions = np.array(
        component_info.shape(last_info_index).getBoundingBox().width()
    )
    # Each pixel has a specific (x, y, z) scale factor. For instance, pixel 42 may have scale factor
    # (0.79, 1.02, 1.00). Thus, the final dimension of this pixel is
    # (0.00804 * 0.79, 0.00409 * 1.02, 0.00804 * 1.00) == (0.00635, 0.00417, 0.00804)
    scale_factors = np.array(
        [unpack_v3d(component_info.scaleFactor, i) for i in info_indexes]
    )
    # The product of each specific scale factor and the nominal pixel dimension gives us the specific pixel dimensions
    pixel_dimensions = (
        scale_factors * nominal_pixel_dimensions
    )  # shape = (number_pixels, 3)

    info_map.update(
        {
            "smearing_pixel_size_x": pixel_dimensions[:, 0],
            "smearing_pixel_size_y": pixel_dimensions[:, 1],
        }
    )

    return info_map


@namedtuplefy
def subpixel_info(input_workspace, n_horizontal, n_vertical):
    r"""
    Calculate the polar coordinates (two theta angle, azimuthal angle, l2) for the subpixels of each
    unmasked pixel detector.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    input_workspace: str, ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Name or reference to a Mantid workspace
    n_horizontal: int
        Number of subpixels along the horizontal direction (on the XZ plane)
    n_vertical: int
        Number of subpixels along the vertical direction (along the Y axis)

    Returns
    -------
    ~collections.namedtuple
        A namedtuple with the following fields:
        two_theta, numpy.ndarray of scattering angles for each subpixel of each valid pixel detector.
        azimuthal, numpy.ndarray of angles on the XY plane  for each subpixel of each valid pixel detector.
        l2, numpy.ndarray for the distance between the sample and each subpixel of each valid pixel detector.
        keep, numpy.ndarray for the workspace indexes of the valid pixel detectors.
    """
    workspace = mtd[str(input_workspace)]
    number_spectra = workspace.getNumberHistograms()
    spectrum_info = workspace.spectrumInfo()
    component_info = workspace.componentInfo()

    # Find workspace indexes not masked, and associated to a detector that is not a monitor (a pixel detector).
    def valid_index(idx):
        return not (
            spectrum_info.isMonitor(idx)
            or spectrum_info.isMasked(idx)
            or not spectrum_info.hasDetectors(idx)
        )

    valid_indexes = [idx for idx in range(number_spectra) if valid_index(idx) is True]

    # Find the componentInfo indexes corresponding to the valid workspace indexes
    get_spectrum_definition = spectrum_info.getSpectrumDefinition
    info_indexes = [get_spectrum_definition(idx)[0][0] for idx in valid_indexes]

    # Find the position of the pixel centers in the frame of reference of the sample
    pixel_positions = pixel_centers(input_workspace, info_indexes)
    sample_position = component_info.samplePosition()
    pixel_positions -= sample_position

    # Fractional coordinates of subpixels, which are the same for all parent pixels.
    # The frame of reference is located at the center of a pixel of width==1 and height==1
    x_fractional = np.linspace(0.5, -0.5, n_horizontal, endpoint=False) - 1 / (
        2 * n_horizontal
    )
    y_fractional = np.linspace(-0.5, 0.5, n_vertical, endpoint=False) + 1 / (
        2 * n_vertical
    )
    z_fractional = np.array([0])
    # xyz_fractional.shape = (number_subpixels, 3)
    xyz_fractional = np.array(
        np.meshgrid(x_fractional, y_fractional, z_fractional)
    ).T.reshape(-1, 3)

    # Find the dimensions of each pixel. Due to barscan and tube-width calibrations, each pixel will have its own size
    last_info_index = info_indexes[-1]  # index of the last valid pixel
    # The nominal pixel dimensions are the same for all pixels. We find the nominal dimensions of the last pixel.
    # The `nominal_pixel_dimensions` for EQSANS are (0.00804, 0.00409, 0.00804) along X, Y, and Z, respectively.
    nominal_pixel_dimensions = np.array(
        component_info.shape(last_info_index).getBoundingBox().width()
    )
    # Each pixel has a specific (x, y, z) scale factor. For instance, pixel 42 may have scale factor
    # (0.79, 1.02, 1.00). Thus, the final dimension of this pixel is
    # (0.00804 * 0.79, 0.00409 * 1.02, 0.00804 * 1.00) == (0.00635, 0.00417, 0.00804)
    scale_factors = np.array(
        [unpack_v3d(component_info.scaleFactor, i) for i in info_indexes]
    )
    # The product of each specific scale factor and the nominal pixel dimension gives us the specific pixel dimensions
    pixel_dimensions = (
        scale_factors * nominal_pixel_dimensions
    )  # shape = (number_pixels, 3)

    # Internal coordinates of subpixels are different for each pixel, because each pixel has its own size
    # xyz_internal.shape = (number_pixels, number_subpixels, 3)
    # We have a frame of reference for each pixel, located at the center of the pixel
    xyz_internal = pixel_dimensions[:, None, :] * xyz_fractional[None, :, :]

    # We obtain coordinates of subpixels in the sample's frame of reference
    # by translating the internal coordinates of the subpixels by a vector equal to the position of the pixel centers
    # with respect to the sample's position.
    xyz = (
        xyz_internal + pixel_positions[:, None, :]
    )  # shape = (number_pixels, number_subpixels, 3)
    xyz = xyz.reshape((len(info_indexes) * n_horizontal * n_vertical, 3))

    # Calculate polar coordinates from the cartesian xyz coordinates
    l2 = np.linalg.norm(xyz, axis=1)  # shape = (number_subpixels * number_pixels, )
    two_theta = np.arccos(xyz[:, 2] / l2)  # cos(two_theta) = z / l2
    azimuthal = np.arctan2(
        xyz[:, 1], xyz[:, 0]
    )  # numpy.arctan2(y, x) is quadrant-aware

    return dict(two_theta=two_theta, azimuthal=azimuthal, l2=l2, keep=valid_indexes)


def _filter_and_replicate(arrays, unmasked_indexes, n_horizontal=1, n_vertical=1):
    r"""
    Retain only items in the array corresponding to unmasked pixels, and replicate
    these values for each subpixel if subpixels are requested. Finally, flatten the array
    to one-dimension.

    For every array in ```arrays```, the same filter and replication steps are applied.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    arrays: list
        List of arrays to be filtered.
    unmasked_indexes: numpy.ndarray
        List of array indexes corresponding to spectra with unmasked pixels.
    n_horizontal: int
        Number of subpixels along the horizontal direction (on the XZ plane)
    n_vertical: int
        Number of subpixels along the vertical direction (along the Y axis)

    Returns
    -------
    list
        List of filtered and replicated arrays
    """
    # It's assumed that arrays are of the shape (...,1) so that reshape(-1) will eliminate the last dimension
    return [
        np.repeat(
            array[unmasked_indexes, :], n_horizontal * n_vertical, axis=0
        ).reshape(-1)
        for array in arrays
    ]
