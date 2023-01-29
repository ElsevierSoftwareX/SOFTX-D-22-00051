# Main method in this module implement step 2 of
# wavelength dependent inelastic incoherent scattering correction
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/issues/689
from drtsans.dataobjects import verify_same_q_bins, IQmod, IQazimuthal
import numpy as np


__all__ = [
    "normalize_by_elastic_reference",
    "determine_reference_wavelength_q1d_mesh",
    "reshape_q_wavelength_matrix",
    "build_i_of_q1d",
    "determine_common_mod_q_range_mesh",
]


# TODO - make it dataclass when python is upgraded to 3.7
# TODO - @dataclass
class ReferenceWavelengths:
    """
    Class for keeping track of reference wavelength for each momentum transfer Q (1D)
    """

    # TODO - make the following as definition of dataclass when python is upgraded to 3.7
    # q_vec: np.ndarray
    # ref_wl_vec: np.ndarray
    # intensity_vec: np.ndarray
    # error_vec: np.ndarray

    def __init__(self, q_values, ref_wavelengths, intensities, errors):
        """Initialization

        Parameters
        ----------
        q_values: ~numpy.ndarray
            vector for Q
        ref_wavelengths: ~numpy.ndarray
            vector for reference wavelength vector for each Q
        intensities: ~numpy.ndarray
            vector for intensities of (Q, reference wavelength)
        errors: ~numpy.ndarray
            vector for errors of (Q, reference wavelength)
        """
        self.q_vec = q_values
        self.ref_wl_vec = ref_wavelengths
        self.intensity_vec = intensities
        self.error_vec = errors


class ReferenceWavelengths2D:
    """
    Class for keeping track of reference wavelength for each momentum transfer Q (1D)
    """
    def __init__(self, qx_values, qy_values, ref_wavelengths, intensities, errors):
        """Initialization

        Parameters
        ----------
        qx_values: ~numpy.ndarray (1, qx_values_size * num_wavelengths)
            2D vector for Qx
        qy_values: ~numpy.ndarray (1, qy_values_size * num_wavelengths)
            2D vector for Qy
        ref_wavelength: int
            wavelength value of ref wavelength
        intensities: ~numpy.ndarray (qx_values_size, qy_values_size, 1)
            3D vector for intensities of (Qx, Qy, 1)
        errors: ~numpy.ndarray
            3D vector for errors of (Qx, Qy, 1)
        """
        self.qx = qx_values
        self.qy = qy_values
        self.ref_wl_vec = ref_wavelengths
        self.intensity_vec = intensities
        self.error_vec = errors


def reshape_q_wavelength_matrix(i_of_q):
    """Reshape I(Q) into a mesh grid of (Q, wavelength) and limit Q into q_min and q_max

    Parameters
    ----------
    i_of_q: ~drtsans.dataobjects.IQmod
        Input I(Q, wavelength) to find common Q range from

    Returns
    -------
    tuple
        wavelength vector, q vector,  intensity (2D), error (2D), dq array (2D) or None

    """
    # Retrieve unique wave length in ascending order
    wavelength_vec = np.unique(i_of_q.wavelength)
    assert len(wavelength_vec.shape) == 1
    wavelength_vec.sort()

    q_vec = np.unique(i_of_q.mod_q)
    assert len(q_vec.shape) == 1
    q_vec.sort()

    # Create a matrix for q, wavelength, intensity and error
    if i_of_q.delta_mod_q is None:
        i_q_wl_matrix = np.array(
            [i_of_q.mod_q, i_of_q.wavelength, i_of_q.intensity, i_of_q.error]
        )
    else:
        i_q_wl_matrix = np.array(
            [
                i_of_q.mod_q,
                i_of_q.wavelength,
                i_of_q.intensity,
                i_of_q.error,
                i_of_q.delta_mod_q,
            ]
        )
    i_q_wl_matrix = i_q_wl_matrix.transpose()

    # Order by wavelength and momentum transfer (Q)
    i_q_wl_matrix = i_q_wl_matrix[
        np.lexsort((i_q_wl_matrix[:, 1], i_q_wl_matrix[:, 0]))
    ]

    # Unique wavelength and unique momentum transfer
    wl_vector = np.unique(i_of_q.wavelength)
    q_vector = np.unique(i_of_q.mod_q)
    # verify whether (q, wl) are on mesh grid by checking unique Q and wavelength
    assert wl_vector.shape[0] * q_vector.shape[0] == i_of_q.intensity.shape[0]

    # Reformat
    intensity_array = i_q_wl_matrix[:, 2].reshape(
        (q_vector.shape[0], wl_vector.shape[0])
    )
    error_array = i_q_wl_matrix[:, 3].reshape((q_vector.shape[0], wl_vector.shape[0]))
    if i_of_q.delta_mod_q is not None:
        dq_array = i_q_wl_matrix[:, 4].reshape((q_vector.shape[0], wl_vector.shape[0]))
    else:
        dq_array = None

    return wl_vector, q_vector, intensity_array, error_array, dq_array


def normalize_by_elastic_reference(i_of_q, ref_i_of_q):
    """Normalize I(Q1D) by elastic reference run

    Parameters
    ----------
    i_of_q: ~drtsans.dataobjects.IQmod
        Input I(Q, wavelength) to normalize
    ref_i_of_q: ~drtsans.dataobjects.IQmod
        Input I(Q, wavelength) as elastic reference run

    Returns
    -------
    tuple
        normalized Q(1D), K vector and delta K vector

    """
    # check i_of_q and ref_i_of_q shall have same binning
    if not verify_same_q_bins(i_of_q, ref_i_of_q):
        raise RuntimeError(
            "Input I(Q) and elastic reference I(Q) have different Q and wavelength binning"
        )

    # Reshape Q, wavelength, intensities and errors to unique 1D array or 2D array
    wl_vec, q_vec, i_array, error_array, dq_array = reshape_q_wavelength_matrix(i_of_q)
    try:
        # in some case, I(Q) and ref I(Q) are the same
        np.testing.assert_allclose(i_of_q.mod_q, ref_i_of_q.mod_q)
        np.testing.assert_allclose(
            i_of_q.intensity, ref_i_of_q.intensity, equal_nan=True
        )
        ref_i_array, ref_error_array = i_array, error_array
    except AssertionError:
        (
            ref_wl_vec,
            ref_q_vec,
            ref_i_array,
            ref_error_array,
            ref_dq_array,
        ) = reshape_q_wavelength_matrix(ref_i_of_q)

    # Calculate Qmin and Qmax
    qmin_index, qmax_index = determine_common_mod_q_range_mesh(q_vec, ref_i_array)

    # Calculate reference
    ref_wl_ie = determine_reference_wavelength_q1d_mesh(
        wl_vec, q_vec, ref_i_array, ref_error_array, qmin_index, qmax_index
    )

    # Calculate scale factor
    k_vec, k_error_vec, p_vec, s_vec = calculate_scale_factor_mesh_grid(
        wl_vec, ref_i_array, ref_error_array, ref_wl_ie, qmin_index, qmax_index
    )

    # Normalize
    data_ref_wl_ie = determine_reference_wavelength_q1d_mesh(
        wl_vec, q_vec, i_array, error_array, qmin_index, qmax_index
    )
    normalized = normalize_intensity_q1d(
        wl_vec,
        q_vec,
        i_array,
        error_array,
        data_ref_wl_ie,
        k_vec,
        p_vec,
        s_vec,
        qmin_index,
        qmax_index,
    )

    # Convert normalized intensities and errors to IModQ
    normalized_i_of_q = build_i_of_q1d(
        wl_vec, q_vec, normalized[0], normalized[1], dq_array
    )

    return normalized_i_of_q, k_vec, k_error_vec


def normalize_by_elastic_reference2D(i_of_q, ref_i_of_q):
    """Normalize I(Q2D) by elastic reference run

    Parameters
    ----------
    i_of_q: ~drtsans.dataobjects.IQmod
        Input I(Q, wavelength) to normalize
    ref_i_of_q: ~drtsans.dataobjects.IQmod
        Input I(Q, wavelength) as elastic reference run

    Returns
    -------
    tuple
        normalized Q(2D), K vector and delta K vector

    """
    # check i_of_q and ref_i_of_q shall have same binning
    if not verify_same_q_bins(i_of_q, ref_i_of_q):
        raise RuntimeError(
            "Input I(Q) and elastic reference I(Q) have different Q and wavelength binning"
        )

    # skip reshape step of the 1D case?

    k_vec, k_error2_vec, p_vec, s_vec = calculate_K_2d(i_of_q)

    # common grouping mask
    mask = determine_common_mod_q2d_range_mesh(i_of_q.qx, i_of_q.qy, i_of_q.wavelength, i_of_q.intensity)

    ref_wavelength_vec = determine_reference_wavelength_q2d(i_of_q, mask)

    normalized_intensity_array, normalized_error2_array = normalize_intensity_q2d(
        i_of_q.wavelength,
        i_of_q.qx,
        i_of_q.qy,
        i_of_q.intensity,
        i_of_q.error,
        ref_wavelength_vec,
        k_vec,
        p_vec,
        s_vec,
        mask
    )

    normalized_i_of_q = IQazimuthal(
        intensity=normalized_intensity_array,
        error=normalized_error2_array,
        qx=i_of_q.qx,
        qy=i_of_q.qy,
        wavelength=i_of_q.wavelength,
        delta_qx=i_of_q.delta_qx,
        delta_qy=i_of_q.delta_qy
    )

    return normalized_i_of_q, k_vec, k_error2_vec


def build_i_of_q1d(wl_vector, q_vector, intensity_array, error_array, delta_q_array):
    """From wavelength, Q, intensity, error and delta Q to build I(Q1D)

    This is the reversed operation to method reshape_q_wavelength_matrix

    Parameters
    ----------
    wl_vector: ~numpy.ndarray
        wavelength (1D)
    q_vector: ~numpy.ndarray
        Q (1D)
    intensity_array: ~numpy.ndarray
        intensities (2D)
    error_array: ~numpy.ndarray
        intensity errors (2D)
    delta_q_array: ~numpy.ndarray
        delta Q (1D) size = number wavelength * number Q

    Returns
    -------
    ~drtsans.dataobjects.IQmod
        constructed I(Q, wavelength)

    """
    # assume that intensity, error and delta q have the same as (num_q, num_wl)
    assert (
        intensity_array.shape[0] == q_vector.shape[0]
        and intensity_array.shape[1] == wl_vector.shape[0]
    )

    # tile wave length
    wl_array_1d = np.tile(wl_vector, q_vector.shape[0])
    q_array_1d = np.repeat(q_vector, wl_vector.shape[0])

    # flatten intensity, error and optionally delta q
    intensity_array = intensity_array.flatten()
    error_array = error_array.flatten()
    if delta_q_array is not None:
        delta_q_array = delta_q_array.flatten()

    return IQmod(
        intensity=intensity_array,
        error=error_array,
        mod_q=q_array_1d,
        wavelength=wl_array_1d,
        delta_mod_q=delta_q_array,
    )


def determine_common_mod_q_range_mesh(q_vec, intensity_array):
    """Determine the common Q1D range among all the wavelengths such that I(q, lambda) does exist.

    This method assumes that I(Q, wavelength) are on mesh grid of Q and wavelength

    Detailed requirement:
        Determine q_min and q_max  that exist in all I(q, lambda) for the fitting (minimization) process

    Parameters
    ----------
    q_vec: numpy.ndarray
        vector of sorted unique Q
    intensity_array: numpy.ndarray
        2D array of intensity.  Each row is of same wavelength

    Returns
    -------
    tuple
        index of qmin and qmax

    """
    # Find q min
    qmin_index = None
    qmax_index = None

    # Sanity check
    assert q_vec.shape[0] == intensity_array.shape[0], "Shape mismatch"

    num_q = q_vec.shape[0]
    for q_index in range(num_q):
        if len(np.where(np.isnan(intensity_array[q_index]))[0]) == 0:
            qmin_index = q_index
            break
    for q_index in range(num_q - 1, -1, -1):
        if len(np.where(np.isnan(intensity_array[q_index]))[0]) == 0:
            qmax_index = q_index
            break

    if qmin_index is None:
        raise RuntimeError("Unable to find common q range")

    return qmin_index, qmax_index


def calculate_scale_factor_mesh_grid(
    wl_vec, intensity_array, error_array, ref_wl_intensities, qmin_index, qmax_index
):
    """Same functionality as calculate_scale_factor but the algorithm is improved
    as I(Q, wavelength) are in meshgrid

    Parameters
    ----------
    wl_vec: numpy.array
        wavelength vector
    intensity_array: numpy.array
        intensity 2D array
    error_array: numpy.array
        error 2D array
    ref_wl_intensities: ReferenceWavelengths
        reference wavelength intensity/error
    qmin_index: int
        index of min Q in q vector
    qmax_index: int
        index of max Q in q vector

    Returns
    -------
    tuple
        K vector, K error vector, P vector, S vector
    """
    # Check input
    assert wl_vec.shape[0] == intensity_array.shape[1]

    # Calculate P(wl), S(wl)
    p_vec = np.zeros_like(wl_vec)
    s_vec = np.zeros_like(wl_vec)
    k_error2_vec = np.zeros_like(wl_vec)

    for i_wl, lambda_i in enumerate(wl_vec):
        # P(wl) = sum_q I(q, ref_wl) * I(q, wl)
        p_value = np.sum(
            ref_wl_intensities.intensity_vec[qmin_index : qmax_index + 1]
            * intensity_array[:, i_wl][qmin_index : qmax_index + 1]
        )
        # S(wl) = sum_q I(q, wl)**2
        s_value = np.sum(intensity_array[:, i_wl][qmin_index : qmax_index + 1] ** 2)

        # assign
        p_vec[i_wl] = p_value
        s_vec[i_wl] = s_value

        term0 = error_array[:, i_wl][qmin_index : qmax_index + 1]
        term1 = (
            ref_wl_intensities.intensity_vec[qmin_index : qmax_index + 1] * s_value
            - 2.0 * intensity_array[:, i_wl][qmin_index : qmax_index + 1] * p_value
        ) / s_value ** 2
        term2 = ref_wl_intensities.error_vec[qmin_index : qmax_index + 1]
        term3 = intensity_array[:, i_wl][qmin_index : qmax_index + 1] / s_value

        k_error2_vec[i_wl] = np.sum((term0 * term1) ** 2 + (term2 * term3) ** 2)

    # Calculate K
    k_vec = p_vec / s_vec

    return k_vec, np.sqrt(k_error2_vec), p_vec, s_vec


def determine_reference_wavelength_q1d_mesh(
    wavelength_vec,
    q_vec,
    intensity_array,
    error_array,
    qmin_index,
    qmax_index,
    min_wl_index=0,
):
    """Determine the reference wavelength for each Q.

    The reference wavelength of a specific Q or (qx, qy)
    is defined as the shortest wavelength for all the finite I(Q, wavelength) or
    I(qx, qy, wavelength)

    Parameters
    ----------
    wavelength_vec: numpy.ndarray
        ...
    q_vec: numpy.ndarray
        ...
    intensity_array: numpy.ndarray
        ...
    error_array: numpy.ndarray
        ...
    qmin_index: int
        index of qmin in q-vector
    qmax_index: int
        index of qmax in q-vector

    Returns
    -------
    ReferenceWavelengths
        Reference wavelengths for each momentum transfer Q and the corresponding intensity and error

    """
    # Sanity check
    assert wavelength_vec.shape[0] == intensity_array.shape[1], (
        f"Wavelength dimension = {wavelength_vec.shape} ,"
        f"Intensity  dimension = {intensity_array.shape}"
    )

    # Minimum wavelength bin is the reference wavelength
    min_wl_vec = np.zeros_like(q_vec) + wavelength_vec[min_wl_index]

    # Minimum intensity and error
    min_intensity_vec = np.copy(intensity_array[:, min_wl_index])
    min_error_vec = np.copy(error_array[:, min_wl_index])

    # Set the unused defined reference wavelength (outside of qmin and qmax)'s
    # intensity and error to nan
    min_intensity_vec[0:qmin_index] = np.nan
    min_intensity_vec[qmax_index + 1 :] = np.nan
    min_error_vec[0:qmin_index] = np.nan
    min_error_vec[qmax_index + 1 :] = np.nan

    return ReferenceWavelengths(q_vec, min_wl_vec, min_intensity_vec, min_error_vec)


def copy_array_and_mask(array, mask):
    """Make a copy of a Numpy Array but replace items hidden by the mask np.null


    Parameters
    ----------
    array: numpy.ndarray
        ...
    mask: numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Masked copy of the input array

    """

    arrayCopy = array.copy()
    arrayCopy[~mask] = np.nan

    return arrayCopy


def determine_common_mod_q2d_range_mesh(qx, qy, wavelength, intensity_array):
    """Determine mask that represents the common intensities between wavelengths


    Parameters
    ----------
    qx: numpy.ndarray
        ...
    qy: numpy.ndarray
        ...
    wavelength: numpy.ndarray
        ...
    intensity_array: numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Mask for common intensities across wavelengths

    """
    intensity_3d = intensity_array.transpose().reshape((len(np.unique(wavelength)), qx.shape[0], qy.shape[0]))

    # assume all positions exist in each intensity matrix
    mask = np.full((intensity_3d.shape[1], intensity_3d.shape[2]), True)
    # filter out positions by logical and'ing the mask with the non-nan mask of each wl
    for intensity_set in intensity_3d:
        subMask = ~np.isnan(intensity_set)

        mask = np.logical_and(mask, subMask)

    mask = mask.transpose()
    return mask


def calculate_K_2d(i_of_q):
    """Calculates K, K Error, P, and S for a given 2D IQazimuthal

    Parameters
    ----------
    i_of_q: ~drtsans.dataobjects.IQazimuthal
        I(qx, qy, wavelength)

    Returns
    -------
    tuple
        K vector, K error vector, P vector, S vector
    """

    # Calculate P(wl), S(wl)
    p_vec = np.empty(len(np.unique(i_of_q.wavelength)))
    s_vec = np.zeros_like(p_vec)
    k_error2_vec = np.zeros_like(p_vec)

    # will qx ever differ from qy in size/nan ?
    mask = determine_common_mod_q2d_range_mesh(i_of_q.qx, i_of_q.qy, i_of_q.wavelength, i_of_q.intensity)

    ref_wavelength_vec = determine_reference_wavelength_q2d(i_of_q, mask)

    # reshape vectors from 2D to 3D of shape (wavelength, qx_unique.len, qy_unique.len)
    # initial shape of vectors is assumed to be 2D with each additonal wavelength values concated on the end
    # e.g. wavelength1.intensity =  [1,1]     wavelength2.intensity = [.2,.2]
    #                               [1,1]                             [.2,.2]
    #                   i_of_q.intensity =  [1,1,.2,.2]
    #                                       [1,1,.2,.2]
    #
    #                   intensity_3d =      [
    #                                        [[1,1],
    #                                         [1,1]],
    #                                        [[.2,.2],
    #                                         [.2,.2]]
    #                                                 ]
    num_wl = len(np.unique(i_of_q.wavelength))
    intensity_3d = i_of_q.intensity.transpose().reshape((num_wl, i_of_q.qx.shape[0], i_of_q.qy.shape[0]))
    error_3d = i_of_q.error.transpose().reshape((num_wl, i_of_q.qx.shape[0], i_of_q.qy.shape[0]))

    for i_wl, lambda_i in enumerate(np.unique(i_of_q.wavelength)):
        # mask vectors to only be values from the intersection of non nan intensity values
        currentIntensity = copy_array_and_mask(intensity_3d[i_wl].transpose(), mask)
        currentError = copy_array_and_mask(error_3d[i_wl].transpose(), mask)
        refIntensity = copy_array_and_mask(ref_wavelength_vec.intensity_vec, mask)
        refIntensity = refIntensity.transpose()
        refError = copy_array_and_mask(ref_wavelength_vec.error_vec, mask)
        refError = refError.transpose()

        # P(wl) = sum_q I(qx, qy, ref_wl) * I(qx, qy, wl)
        p_value = np.nansum(
            refIntensity
            * currentIntensity
        )

        # S(wl) = sum_q I(qx, qy, wl)**2
        s_value = np.nansum(currentIntensity ** 2)

        # assign
        p_vec[i_wl] = p_value
        s_vec[i_wl] = s_value

        # now calculate error
        term0 = currentError ** 2
        term1 = ((
            refIntensity * s_value
            - 2.0 * currentIntensity * p_value
        ) / (s_value ** 2)) ** 2

        term2 = refError ** 2
        term3 = (currentIntensity / s_value) ** 2

        k_error2_vec[i_wl] = np.nansum((term0 * term1) + (term2 * term3))

    # Calculate K
    k_vec = p_vec / s_vec

    return k_vec, k_error2_vec, p_vec, s_vec


def determine_reference_wavelength_q2d(i_of_q, mask):
    """Determine the reference wavelength for each Q.

    The reference wavelength of a specific Q or (qx, qy)
    is defined as the shortest wavelength for all the finite I(Q, wavelength) or
    I(qx, qy, wavelength)

    Parameters
    ----------
    i_of_q: ~drtsans.dataobjects.IQazimuthal
        I(qx, qy, wavelength)
    ...
    mask: numpy.ndarray

    Returns
    -------
    ReferenceWavelengths2D
        Reference wavelengths for each momentum transfer Q and the corresponding intensity and error

    """
    # Construct 3D versions to easily index by wavelength
    num_wl = len(np.unique(i_of_q.wavelength))
    intensity_3d = i_of_q.intensity.transpose().reshape((num_wl, i_of_q.qx.shape[0], i_of_q.qy.shape[0]))
    error_3d = i_of_q.error.transpose().reshape((num_wl, i_of_q.qx.shape[0], i_of_q.qy.shape[0]))

    # determine index of shortest wavelength
    min_wl = np.min(i_of_q.wavelength)
    min_wl_index = np.where(np.unique(i_of_q.wavelength) == min_wl)[0]

    # pull out properties of min wavelength to store in ReferenceWavelengths2D
    # 2d vector containing intensities for qx and qy
    min_intensity_vec = intensity_3d[min_wl_index].transpose().copy()
    # 2d vector containing intensity errors for qx and qy
    min_error_vec = error_3d[min_wl_index].transpose().copy()
    # 2d vector containing qx values
    min_qx = i_of_q.qx[min_wl_index]
    # 2d vector containing qy values
    min_qy = i_of_q.qy[min_wl_index]

    return ReferenceWavelengths2D(min_qx, min_qy, min_wl, min_intensity_vec, min_error_vec)


def normalize_intensity_q1d(
    wl_vec,
    q_vec,
    intensity_array,
    error_array,
    ref_wl_ints_errs,
    k_vec,
    p_vec,
    s_vec,
    qmin_index,
    qmax_index,
):
    """Normalize Q1D intensities and errors

    Parameters
    ----------
    wl_vec: ~numpy.ndarray
        1D vector of wavelength (in ascending order)
    q_vec: ~numpy.ndarray
        1D vector of Q (in ascending order)
    intensity_array: ~numpy.ndarray
        2D array of intensities, shape[0] = number of Q, shape[1] = number of wavelength
    error_array: ~numpy.ndarray
        2D array of errors, shape[0] = number of Q, shape[1] = number of wavelength
    ref_wl_ints_errs: ReferenceWavelengths
        instance of ReferenceWavelengths containing intensities and errors
    k_vec: ~numpy.ndarray
        calculated K vector
    p_vec: ~numpy.ndarray
        calculated P vector
    s_vec: ~numpy.ndarray
        calculated S vector
    qmin_index: int
        index of common Q min (included)
    qmax_index: int
        index of common Q max (included)

    Returns
    -------
    tuple
        normalized I(Q1D), normalized error(Q1D)

    """

    # Sanity check
    assert wl_vec.shape[0] == intensity_array.shape[1]  # wavelength as lambda
    assert q_vec.shape[0] == error_array.shape[0]       # points as q
    assert intensity_array.shape == error_array.shape

    # Normalized intensities
    normalized_intensity_array = intensity_array * k_vec
    normalized_error2_array = np.zeros_like(error_array)

    # Lowest wavelength bin does not require normalization as K = 1, i_wl = 0
    normalized_error2_array[:, 0] = error_array[:, 0] ** 2

    # Reshape
    ri_vec = ref_wl_ints_errs.intensity_vec.reshape((q_vec.shape[0], 1))
    re_vec = ref_wl_ints_errs.error_vec

    # qmax is included.  need i_qmax to slicing
    i_qmax = qmax_index + 1

    # Loop over wavelength
    num_wl = wl_vec.shape[0]
    for i_wl in range(1, num_wl):

        intensity_vec = intensity_array[:, i_wl].reshape((q_vec.shape[0], 1))

        # Calculate Y: Y_ij = I_i * R_j * s - I_i * 2 * I_j * p
        y_matrix = intensity_vec * (ri_vec.transpose()) * s_vec[
            i_wl
        ] - intensity_vec * (intensity_vec.transpose()) * (2 * p_vec[i_wl])
        y_diag = np.diag(y_matrix)
        # y_matrix[i, :] corresponds to a single q_i/r_i
        # y_matrix[:, j] corresponds to a single q_j/r_j

        # Term 1
        # t2 += [delta I(q', wl)]**2 * Y(q, q'', wl)**2 / S(lw)**4
        t2sum_vec = (
            error_array[qmin_index:i_qmax, i_wl] ** 2
            * y_matrix[:, qmin_index:i_qmax] ** 2
            / s_vec[i_wl] ** 4
        )

        # Term 3
        # t3 += [delta I(q_j, ref_wl[q_j]]^2 * [I(q_j, wl) * I(q, wl)]^2 / S(wl)^2
        t3sum_vec = intensity_array[:, i_wl] ** 2 * np.sum(
            re_vec[qmin_index:i_qmax] ** 2
            * intensity_array[qmin_index:i_qmax, i_wl] ** 2
            / s_vec[i_wl] ** 2
        )

        # outside of qmin and qmax: t1 = [delta I(q, wl)]**2 * [P(wl) / S(wl)]**2
        t1sum_vec = (error_array[:, i_wl] * p_vec[i_wl] / s_vec[i_wl]) ** 2
        # term 2
        t1sum_vec[qmin_index : qmax_index + 1] = (
            error_array[qmin_index : qmax_index + 1, i_wl] ** 2
            * (
                p_vec[i_wl] ** 2 * s_vec[i_wl] ** 2
                + 2 * p_vec[i_wl] * s_vec[i_wl] * y_diag[qmin_index : qmax_index + 1]
            )
            / s_vec[i_wl] ** 4
        )

        # sum up
        normalized_error2_array[:, i_wl] += (
            t1sum_vec + t2sum_vec.sum(axis=1) + t3sum_vec
        )

    return normalized_intensity_array, np.sqrt(normalized_error2_array)


def einsum_multiply(A, B):
    return np.einsum('ij,kl', A, B)


def normalize_intensity_q2d(
    wl_vec,
    qx,
    qy,
    intensity_array,
    error_array,
    ref_wl_ints_errs,
    k_vec,
    p_vec,
    s_vec,
    mask
):
    """Normalize Q1D intensities and errors

    Parameters
    ----------
    wl_vec: ~numpy.ndarray
        1D vector of wavelength (in ascending order)
    qx: ~numpy.ndarray
        2D vector of Q (in ascending order)
    qy: ~numpy.ndarray
        2D vector of Q (in ascending order)
    intensity_array: ~numpy.ndarray
        2D array of intensities, shape[0] = number of Q, shape[1] = number of wavelength
    error_array: ~numpy.ndarray
        2D array of errors, shape[0] = number of Q, shape[1] = number of wavelength
    ref_wl_ints_errs: ReferenceWavelengths
        instance of ReferenceWavelengths containing intensities and errors
    k_vec: ~numpy.ndarray
        calculated K vector
    p_vec: ~numpy.ndarray
        calculated P vector
    s_vec: ~numpy.ndarray
        calculated S vector
    mask: ~numpy.ndarray
        boolean array that represents common intensities across wavelengths

    Returns
    -------
    tuple
        normalized I(Q2D), normalized error(Q2D)

    """

    # Sanity check
    assert wl_vec.shape[0] == intensity_array.shape[0]  # wavelength as lambda
    assert wl_vec.shape[1] == intensity_array.shape[1]
    assert qx.shape[0] == error_array.shape[0]       # points as q
    assert qx.shape[1] == error_array.shape[1]
    assert intensity_array.shape == error_array.shape

    # Reshape vectors to be easily indexed by wavelength
    # Refer to calculate_K_2d() for an input datashape outline
    sizeZ = len(np.unique(wl_vec))
    sizeX = qx.shape[0]
    sizeY = qy.shape[0]
    intensity_3d = intensity_array.transpose().reshape((sizeZ, sizeX, sizeY))
    error_3d = error_array.transpose().reshape((sizeZ, sizeX, sizeY))

    # Init Normalized intensities
    normalized_intensity_array = intensity_3d * k_vec.reshape((3, 1, 1))
    normalized_error2_array = np.zeros_like(error_3d)

    # Reshape
    ri_vec = ref_wl_ints_errs.intensity_vec.transpose().reshape((sizeX, sizeY))
    re_vec = ref_wl_ints_errs.error_vec.transpose().reshape((sizeX, sizeY))

    for i_wl in range(0, sizeZ):
        # Collect current wavelength's properties and mask for intersetion across all wavelengths
        intensity_vec = intensity_3d[i_wl]
        error_vec = error_3d[i_wl]

        i_mn = intensity_vec
        i_kl = intensity_vec.transpose()
        i8_mn = error_vec
        i8_kl = error_vec.transpose()
        iref_mn = ri_vec
        iref_kl = ri_vec.transpose()
        iref8_mn = re_vec

        # Calculate Y: Y_mn = (I_kl * R_mn * S) - (I_kl * P * 2 * I_mn)
        # you would have k,l grid of m,n grids
        y_mn = np.nan_to_num((einsum_multiply(i_kl, iref_mn) * s_vec[i_wl]))
        y_mn -= np.nan_to_num((einsum_multiply(i_kl, i_mn) * p_vec[i_wl] * 2))

        #  what if m,n = 0,0 and k,l = 1,1
        y_kl = np.nan_to_num((i_kl * iref_kl * s_vec[i_wl]))
        y_kl -= np.nan_to_num((i_kl * p_vec[i_wl] * 2 * i_kl))

        # Term 1 for each kl, so it would only be one member of y_mn
        t1sum_vec = np.einsum('klmn->kl',np.nan_to_num(
            (y_mn ** 2) * (i8_mn ** 2)
            / (s_vec[i_wl] ** 4))
        )

        # Term 2
        # outside of kl
        t2sum_vec = (i8_kl ** 2) * (p_vec[i_wl] / s_vec[i_wl]) ** 2

        # inside kl
        kl_in_mn_mask = ~np.isnan(i_mn)
        t2sum_vec[kl_in_mn_mask] = (
            (i8_kl ** 2)
            * (
                ((p_vec[i_wl] ** 2) * (s_vec[i_wl] ** 2))
                + (2 * p_vec[i_wl] * s_vec[i_wl] * y_kl)
            )
            / (s_vec[i_wl] ** 4)
        )[kl_in_mn_mask]

        # Term 3
        # sum over m,n you can pull out kl terms
        t3sum_vec = np.nansum(
            (iref8_mn ** 2)
            * (((i_kl * i_mn) / s_vec[i_wl]) ** 2)
        )
        # sum up
        normalized_error2_array[i_wl] += (t1sum_vec + t2sum_vec + t3sum_vec)

    return normalized_intensity_array, np.sqrt(normalized_error2_array)
