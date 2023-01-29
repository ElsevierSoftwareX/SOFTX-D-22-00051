"""
Module for algorithms to prepare sensitivity for instrument with moving detector
"""
import numpy as np
from drtsans.mask_utils import circular_mask_from_beam_center, apply_mask
import drtsans.mono.gpsans as gp
from mantid.simpleapi import CreateWorkspace, logger


# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/205
def prepare_sensitivity(
    flood_data_matrix, flood_sigma_matrix, threshold_min, threshold_max
):
    """Prepare sensitivity for moving detector

    Data files are processed such that intensities and errors are stored in numpy.ndarray with shape (N, M), where
    - N: number of data files to calculate sensitivities
    - M: number of pixels (aka spectra) in instrument's detector; The 2D data from 2D detector are flattened to 1D

    Prerequisite of the input data:
    - Input data has been normalized by monitor counts already
    - top and bottom of the detector shall be masked (set to value as NaN) due to edge effects
    - in each file, beam center shall be found  and masked out
    - errors are then calculated from the flood intensities

    Workflow to prepare sensitivities
    - normalize the flood field data by monitor
    - find weighted average for each fie and error
    - apply bad pixel threshold to each file
    - correct for beam stop and add all the flood files together to non-normalized sensitivities
    - apply weighted average to sensitivities

    Parameters
    ----------
    flood_data_matrix : ~numpy.ndaray
        multiple set of flood data intensities with shape = N, M
    flood_sigma_matrix : ~numpy.ndaray
        multiple set of flood data intensities' error with shape = N, M
    threshold_min : float
        minimum allowed detector counts to mask out 'bad' pixels
    threshold_max : float
        maximum allowed detector counts to mask out 'bad' pixels

    Returns
    -------
    ~numpy.ndaray, ~numpy.ndaray
        sensitivities, sensitivities error

    """
    # There might some zero-count pixels in the flood data, they shall be masked
    _mask_zero_count_pixel(flood_data_matrix, flood_sigma_matrix)

    # normalize the flood field data by monitor: Normalization is removed from this algorithm to integration
    # inputs: (N, M) array; outputs: (N, M) array
    # flood_data_matrix, flood_sigma_matrix = _normalize_by_monitor(flood_data_matrix, flood_sigma_matrix,
    #                                                               monitor_counts)

    # find weighted average for each fie and error
    # inputs: (N, M) array; outputs: (N, M) array
    returns = _calculate_weighted_average_with_error(
        flood_data_matrix, flood_sigma_matrix
    )
    flood_data_matrix = returns[0]
    flood_sigma_matrix = returns[1]

    # apply bad pixel threshold to each file
    # inputs: (N, M) array; outputs: (N, M) array
    flood_data_matrix, flood_sigma_matrix = _apply_sensitivity_thresholds(
        flood_data_matrix, flood_sigma_matrix, threshold_min, threshold_max
    )

    # correct for beam stop and add all the flood files together to non-normalized sensitivities
    raw_sensitivities, raw_sensitivities_error = _calculate_pixel_wise_sensitivity(
        flood_data_matrix, flood_sigma_matrix
    )

    # apply weighted average to sensitivities
    (
        sensitivities,
        sensitivities_error,
        sens_avg,
        sigma_sens_avg,
    ) = _normalize_sensitivities(raw_sensitivities, raw_sensitivities_error)
    return sensitivities, sensitivities_error


def _mask_zero_count_pixel(flood_data_matrix, flood_sigma_matrix):
    """Mask out pixels/elements in the flood data to NaN.
    Because these zero count is not considered by the equations to calculate weighted average.

    The value will be modified in place to the numpy arrays

    Parameters
    ----------
    flood_data_matrix: numpy.ndarray
            normalized flood data
    flood_sigma_matrix: numpy.ndarray
            normalized flood data's error

    Returns
    -------

    """
    # get the zero count elments
    zero_count_elements = flood_data_matrix < 1e-12
    logger.notice(
        f"Input flood runs: total {len(np.where(zero_count_elements)[0])} are "
        f"masked"
    )

    # set to NaN
    flood_data_matrix[zero_count_elements] = np.nan
    flood_sigma_matrix[zero_count_elements] = np.nan

    return


def _calculate_weighted_average_with_error(normalized_data, normalized_error):
    """Calculated weighted average for normalized flood data and error

    Average = (sum I(m, n) / sigma^(m, n)) / sum 1 / sigma^2

    Parameters
    ----------
    normalized_data : ndarray
        normalized flood data
        multiple set of flood data.  flood_data.shape = (M, N) as M set of N pixels
    normalized_error : ndarray
        normalized flood data's error
        multiple set of flood data.  flood_data.shape = (M, N) as M set of N pixels

    Returns
    -------
    ndarray, ndarray, float, float
        data normalized by average, data's error normalized by average, Average, sigma(Average)

    """
    # Calculate weighted average for each flood file/run (average = sum_{i, j} I(i, j) / sigma^2(i, j))
    # For m-th flood file/run
    # where (i, j) is the index of a pixel on 2D detector and n is the index of same pixel as the 2D array
    # is flattened to 1D.
    # np.nansum() is used to exclude NaN from summation
    # np.nansum():  https://docs.scipy.org/doc/numpy/reference/generated/numpy.nansum.html
    # Summation is done along axis=1, the return, weighted_sum, is a 1D array with shape (M,)

    # calculate:  sum_{n} I(m, n) / sigma^2(m, n)
    weighted_sum = np.nansum(
        normalized_data / normalized_error ** 2, axis=1
    )  # summing in row
    # calculate: sum_n(1 / sigma^2(m, n))
    weights_square = np.nansum(1.0 / normalized_error ** 2, axis=1)
    # average[m] == sum_{n}(I(m, n) / sigma^2(m, n)) / sum_{n}(1 / sigma^2(m, n))
    weighted_average = weighted_sum / weights_square
    # reshape to (M, 1) for division to input 2D array with shape (M, N)
    weighted_average = weighted_average.reshape(
        (normalized_data.shape[0], 1)
    )  # reshape to (N, 1) for division
    # calculate: error for weighted average: sigma_avg[m] = 1 / sqrt(sum_{n}(1 / sigma^2(m, n)))
    weighted_average_error = 1.0 / np.sqrt(weights_square)
    # reshape to (M, 1) for division to input 2D array with shape (M, N)
    weighted_average_error = weighted_average_error.reshape(
        (normalized_data.shape[0], 1)
    )

    # Normalize data by weighted-average
    avg_norm_data = normalized_data / weighted_average

    # Propagate uncertainties: sigma S(n) = I(m, n) / avg * [(error(m, n)/I(m, n))^2 + (sigma Avg/Avg)^2]^1/2
    # in the sqrt operation, first term is a N x M array and second term is a N x 1 array
    avg_norm_error = (
        normalized_data
        / weighted_average
        * np.sqrt(
            (normalized_error / normalized_data) ** 2
            + (weighted_average_error / weighted_average) ** 2
        )
    )

    return avg_norm_data, avg_norm_error, weighted_average, weighted_average_error


def _apply_sensitivity_thresholds(data, data_error, threshold_min, threshold_max):
    """Apply bad pixel threshold to each data set including error

    If any pixel with counts falling out of allowed threshold, i.e., out of range (min, max)
    they will be specified as bad pixels.
    For any bad pixel, the counts will then be set to '-inf'

    Parameters
    ----------
    data : ndarray
        normalized data
    data_error : ndarray
        normalized data error
    threshold_min: float
        minimum value of allowed value
    threshold_max: float
        maximum value of allowed value

    Returns
    -------
    ndarray, ndarray
        data with bad pixels set to INF,
        data error with bad pixels set to INF
    """
    msg = "[drtsans._apply_sensitivity_thresholds]:\n"
    msg += f"\tthreshold_min: {threshold_min}\n"
    msg += f"\tthreshold_max: {threshold_max}\n"
    msg += f"\tnumber of pixels below threshold_min: {len(np.where(data < threshold_min)[0])}\n"
    msg += f"\tnumber of pixels above threshold_max: {len(np.where(data > threshold_max)[0])}\n"
    logger.notice(msg)
    # (data < threshold_min) | (data > threshold_max) returns the list of indexes in array data whose values
    # are either smaller than minimum threshold or larger than maximum threshold.
    data[(data < threshold_min) | (data > threshold_max)] = -np.inf
    data_error[(data < threshold_min) | (data > threshold_max)] = -np.inf

    return data, data_error


def _calculate_pixel_wise_sensitivity(flood_data, flood_error):
    """Calculate pixel-wise average of N files to create the new summed file for doing sensitivity correction

    # data_a, data_a_error, data_b, data_b_error, data_c, data_c_error
    D(m, n) = A_F(m, n) + B_F(m, n) + C_F(m, n) with average weight

    Calculate Pixel-wise Average of 3 files to create the new summed file for
    doing the sensitivity correction

    Parameters
    ----------
    flood_data : ~numpy.ndarray
        processed multiple flood files in an N x M array
        shape[0]: number of flood files
        shape[1]: number of pixels
    flood_error : ~numpy.ndarray
        processed multiple flood files' error in an N x M array

    Returns
    -------
    ~numpy.nparray, ~numpy.nparray
        non-normalized sensitivities, non-normalized sensitivities error
        1D array as all the flood files are summed

    """
    # Keep a record on the array elements with np.inf long axis=0, i.e., same detector pixel among different
    # flood files
    simple_sum = np.sum(flood_data, axis=0)

    # Calculate D'(i, j)    = sum_{k}^{A, B, C}M_k(i, j)/s_k^2(i, j)
    #           1/s^2(i, j) = sum_{k}^{A, B, C}1/s_k^2(i, j)
    # Do weighted summation to the subset and exclude the NaN by np.nansum()
    # np.nansum():  https://docs.scipy.org/doc/numpy/reference/generated/numpy.nansum.html
    # If there is any element in array that is infinity, the summation of all elements on the specified axis
    # i.e., among all flood data files/runs, could be messed up (though every likely the summed value is inf).
    s_ij = np.nansum(
        1.0 / flood_error ** 2, axis=0
    )  # summation along axis 1: among files
    d_ij = np.nansum(flood_data / flood_error ** 2, axis=0) / s_ij
    s_ij = 1.0 / np.sqrt(s_ij)

    # In case there is at least an inf in this subset of data along axis=0, i.e., among various flood runs/files,
    # set sensitivities to -inf in case nansum() messes up
    s_ij[np.isinf(simple_sum)] = -np.inf
    d_ij[np.isinf(simple_sum)] = -np.inf

    # sensitivities = d_ij
    # sensitivities_error = s_ij

    return d_ij, s_ij


def _normalize_sensitivities(d_array, sigam_d_array):
    """Do weighted average to pixel-wise sensitivities and propagate the error
    And then apply the average to sensitivity

    S_avg = sum_{i, j}{D(i, j) / sigma^2(i, j)} / sum_{i, j}{1 / sigma^2(i, j)}

    Parameters
    ----------
    d_array : ndarray
        pixel-wise sensitivities in 1D array with shape (N,) where N is the number of pixels
    sigam_d_array : ndarray
        pixel-wise sensitivities error in 1D array with shape (N,) where N is the number of pixels

    Returns
    -------
    ndarray, ndarray, float, float
        normalized pixel-wise sensitivities, normalized pixel-wise sensitivities error
        scalar sensitivity, error of scalar sensitivity

    """
    # Calculate wighted-average of pixel-wise sensitivities: i.e., do the summation on the all pixels
    # since the 2D detector is treated as a 1D array in this method.
    # Each (i, j) has a unique value p to be mapped to n.

    # Any NaN terms and Infinity terms (for bad pixels) shall be excluded from summation
    # ~(np.isinf(d_array) | np.isnan(d_array) gives out the indexes of elements in d_array that are not NaN or Inf
    # calculate denominator: denominator = sum_{i, j}{D(i, j) / sigma^2(i, j)} = sum_{n}(D(n) / sigma^2(n))
    denominator = np.sum(
        d_array[~(np.isinf(d_array) | np.isnan(d_array))]
        / sigam_d_array[~(np.isinf(d_array) | np.isnan(d_array))] ** 2
    )
    # calculate nominator: nominator = sum_{m, n}{1 / sigma^2(m, n)}
    nominator = np.sum(1 / sigam_d_array[~(np.isinf(d_array) | np.isnan(d_array))] ** 2)
    sens_avg = denominator / nominator

    # Normalize pixel-wise sensitivities
    sensitivities = d_array / sens_avg

    # Calculate scalar sensitivity's error:
    # sigma_S_avg = sqrt(1 / sum_{m, n}(1 / sigma_D(m, n)^2))
    # for all D(m, n) are not NaN
    # Thus, all the NaN terms shall be excluded from summation
    # All the infinity terms shall be ignored because (1/inf) is zero and has no contribution in summation
    # d_array[~(np.isinf(d_array) | np.isnan(d_array))] excludes all items that are either infinity or Nan
    sigma_sens_avg = np.sqrt(
        1 / np.sum(1 / sigam_d_array[~(np.isinf(d_array) | np.isnan(d_array))] ** 2)
    )

    # Propagate the sensitivities
    # sigma_sens(m, n) = D(m, n) / S_avg * [(sigma_D(m, n) / D(m, n))^2 + (sigma_S_avg / S_avg)^2]^{1/2}
    # D(m, n) are the non-normalized sensitivities
    sensitivities_error = (
        d_array
        / sens_avg
        * np.sqrt((sigam_d_array / d_array) ** 2 + (sigma_sens_avg / sens_avg) ** 2)
    )

    # set sensitivities error to -infinity if sensitivities are
    sensitivities_error[np.isinf(sensitivities)] = -np.inf

    return sensitivities, sensitivities_error, sens_avg, sigma_sens_avg


def mask_beam_center(data_ws, beam_center_ws, beam_center_radius):
    """Mask detectors in a workspace

    Mask (1) beam center

    Parameters
    ----------
    data_ws : ~mantid.api.MatrixWorkspace
        Flood data workspace
    beam_center_ws : ~mantid.api.MatrixWorkspace
        Beam center workspace used to generate beam center mask
    beam_center_radius : float
        beam center radius in unit of mm

    Returns
    -------
    ~mantid.api.MatrixWorkspace

    """
    # Use beam center ws to find beam center
    xc, yc = gp.find_beam_center(beam_center_ws)

    # Center detector to the data workspace (change in geometry)
    gp.center_detector(data_ws, xc, yc)

    # Mask the new beam center by 65 mm (Lisa's magic number)
    det = list(circular_mask_from_beam_center(data_ws, beam_center_radius))
    apply_mask(data_ws, mask=det)  # data_ws reference shall not be invalidated here!

    return data_ws


def calculate_sensitivity_correction(flood_run_ws_list, threshold_min, threshold_max):
    """Prepare sensitivities with

    Parameters
    ----------
    flood_run_ws_list : ~list
        List of references to Mantid workspaces for normalized and masked (default and beam center)
    threshold_min : float
        minimum threshold
    threshold_max : float
        maximum threshold

    Returns
    -------
    ~mantid.api.MatrixWorkspace
        Reference to the events workspace

    """
    # Set the flood/beam center pair
    num_ws_pairs = len(flood_run_ws_list)
    # number of histograms
    num_spec = flood_run_ws_list[0].getNumberHistograms()

    # Combine to numpy arrays: N, M
    flood_array = np.ndarray(shape=(num_ws_pairs, num_spec), dtype=float)
    sigma_array = np.ndarray(shape=(num_ws_pairs, num_spec), dtype=float)
    for f_index in range(num_ws_pairs):
        flood_array[f_index][:] = flood_run_ws_list[f_index].extractY().transpose()[0]
        sigma_array[f_index][:] = flood_run_ws_list[f_index].extractE().transpose()[0]

    # Convert all masked pixels' counts to NaN
    masked_items = np.where(sigma_array < 1e-16)
    # Logging
    msg = "[drtsans.calculate_sensitivity_correction]:\n"
    msg += f"\tNumber of zero counts: {np.where(flood_array < 1e-16)[0].shape}\n"
    msg += f"\tNumber of zero sigmas: {np.where(sigma_array < 1e-16)[0].shape}\n"
    logger.notice(msg)
    # set values to masked pixels
    flood_array[masked_items] = np.nan
    sigma_array[masked_items] = np.nan

    # Calculate sensitivities
    sens_array, sens_sigma_array = prepare_sensitivity(
        flood_data_matrix=flood_array,
        flood_sigma_matrix=sigma_array,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
    )

    # Convert all -infinity to nan
    sens_sigma_array[np.where(np.isinf(sens_array))] = np.nan
    sens_array[np.where(np.isinf(sens_array))] = np.nan

    # Export to a MatrixWorkspace
    # Create a workspace for sensitivities
    vec_x = flood_run_ws_list[0].extractX().flatten()
    sens_ws_name = "sensitivities"

    # Create output workspace
    nexus_ws = CreateWorkspace(
        DataX=vec_x,
        DataY=sens_array,
        DataE=sens_sigma_array,
        NSpec=num_spec,
        UnitX="wavelength",
        OutputWorkspace=sens_ws_name,
    )

    return nexus_ws
