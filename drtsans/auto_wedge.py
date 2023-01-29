import numpy as np
import os
from typing import List, Tuple
from drtsans.dataobjects import IQmod
from drtsans.determine_bins import determine_1d_linear_bins
from drtsans.iq import BinningMethod, BinningParams, bin_annular_into_q1d
from drtsans.settings import unique_workspace_dundername

# https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html
from mantid.simpleapi import DeleteWorkspace, logger, Gaussian, FlatBackground

# https://docs.mantidproject.org/nightly/algorithms/Fit-v1.html
from mantid.simpleapi import Fit
import h5py
from matplotlib import pyplot as plt

__all__ = ["getWedgeSelection"]

# factor to convert from a sigma to full width half max
# https://docs.mantidproject.org/nightly/fitting/fitfunctions/Gaussian.html
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

MANTID_FUNCTION_MAP = {Gaussian().name: Gaussian, FlatBackground().name: FlatBackground}


def _create_fit_function(combo_function_str):
    """Create fit function from a Mantid combo function string

    An example:
    name=FlatBackground,A0=5894.8246694221725;name=Gaussian,Height=16626.818784787953,
    PeakCentre=100.43905821664455,Sigma=5.917594016360494;name=Gaussian,Height=18517.052761733517,
    PeakCentre=304.6715713990027,Sigma=36.943377164429435

    Parameters
    ----------
    combo_function_str: str
        function string

    Returns
    -------
    ~list
        list of Mantid IFunction in order of

    """
    # split to multiple functions
    single_function_str_list = combo_function_str.split(";")
    single_functions = list()

    # convert to function
    for single_func in single_function_str_list:
        # split with ','
        params = single_func.split(",")
        # create function
        func_name = params[0].split("=")[1]
        func_i = MANTID_FUNCTION_MAP[func_name]()
        # start to set parameters
        for value_index in range(1, len(params)):
            # split in PARAM_NAME=PARAM_VALUE style
            param_list = params[value_index].split("=")
            param_name_i = param_list[0]
            param_value_i = float(param_list[1])
            func_i.__setattr__(param_name_i, param_value_i)
        # END-FOR
        single_functions.append(func_i)

    return single_functions


def _set_function_param_value(functions, param_name, param_value):
    """set function parameter value from the result of a combo function

    Parameters
    ----------
    functions: ~list
        list of functions
    param_name: str
        function parameter name may have 2 forms: PARAM_NAME or fI.PARAM_NAME, where I is an integer
    param_value: float
        parameter value

    Returns
    -------
    None

    """
    # Check parameter name
    if "." in param_name:
        terms = param_name.split(".")
        function_index = int(terms[0].split("f")[-1])
        param_name = terms[1]
    else:
        function_index = 0

    # set
    functions[function_index].__setattr__(param_name, param_value)


def _calculate_function(fit_functions, vec_x):
    """Calculate a function (set)

    Parameters
    ----------
    fit_functions: ~list
        List of fit function
    vec_x: numpy.ndarray
        1D array for X values

    Returns
    -------
    numpy.ndarray
        1D array for calculated function values

    """
    # Create the combo function from a list of functions
    combo_function = fit_functions[0]
    for f_i in range(1, len(fit_functions)):
        combo_function += fit_functions[f_i]

    # Call the create
    vec_y = combo_function.__call__(vec_x)

    return vec_y


def _plot_fit_results(rings, peak_fit_dict, output_dir):
    """Plot original data and fit result

    Parameters
    ----------
    rings:  ~list
        List of I(Qx, Qy) in rings
    peak_fit_dict: ~dict
        dictionary containing all the peak fitting result
    output_dir: str
        full path of the output directory

    Returns
    -------

    """
    for index, ring in enumerate(rings):
        # add fitting related information to hdf file
        if peak_fit_dict[index]["error"] is not None:
            # bad fit
            continue

        # construct function from estimated
        fit_function_str = peak_fit_dict[index]["fit_function"]
        fit_function_set = _create_fit_function(fit_function_str)
        # calculate estimated peaks
        estimated_y = _calculate_function(fit_function_set, ring.mod_q)

        for param_name in peak_fit_dict[index]:
            if param_name not in ["fit_function", "error", "used"]:
                # parameter value is recorded as tuple as value and error
                _set_function_param_value(
                    fit_function_set, param_name, peak_fit_dict[index][param_name][0]
                )

        # calculate
        model_y = _calculate_function(fit_function_set, ring.mod_q)
        plt.cla()
        plt.plot(ring.mod_q, ring.intensity, label="observed", color="black")
        plt.plot(ring.mod_q, model_y, label="fitted", color="red")
        plt.plot(ring.mod_q, estimated_y, label="estimated", color="blue")
        plt.savefig(os.path.join(output_dir, f"ring_{index:01}.png"))


def _export_to_h5(iq2d, rings, azimuthal_delta, peak_fit_dict, output_dir):
    """write annular binned I(Qx, Qy) and intensity rings to hdf5 for further review

    Parameters
    ----------
    iq2d:  ~drtsans.dataobjects.Azimuthal
    rings: ~list
        List of I(Qx, Qy) in rings
    peak_fit_dict: ~dict
        dictionary containing all the peak fitting result
    output_dir: str
        full path of the output directory

    Returns
    -------

    """
    # annular binning on the full range
    azimuthal_offset = 0.5 * azimuthal_delta
    azimuthal_binning = BinningParams(
        0.0 - azimuthal_offset,
        360.0 - azimuthal_offset,
        bins=int(360.0 / azimuthal_delta),
    )
    q1d = np.sqrt(iq2d.qx ** 2 + iq2d.qy ** 2)
    logger.notice(f"[DEBUG] I(Qx, Qy) Q range: {q1d.min()}, {q1d.max()}")
    full_range_annular = bin_annular_into_q1d(
        iq2d, azimuthal_binning, q1d.min(), q1d.max(), BinningMethod.NOWEIGHT
    )

    # open
    debug_h5 = h5py.File(os.path.join(output_dir, "auto_wedge_fit.h5"), "w")
    # define h5py string type
    h5_string_type = h5py.special_dtype(vlen=bytes)

    # create 3 groups for result
    ring_group = debug_h5.create_group("rings")
    fit_group = debug_h5.create_group("good fit")
    nofit_group = debug_h5.create_group("no fit")

    # write full range
    group = ring_group.create_group("full range")
    group.create_dataset("q", data=full_range_annular.mod_q)
    group.create_dataset("intensity", data=full_range_annular.intensity)

    # write out for each ring
    func_param_dict = dict()

    for index, ring in enumerate(rings):
        # add data to hdf file
        group = ring_group.create_group(f"ring {index}")
        group.create_dataset("q", data=ring.mod_q)
        group.create_dataset("intensity", data=ring.intensity)

        # add fitting related information to hdf file
        if peak_fit_dict[index]["error"] is None:
            # good fit
            group = fit_group.create_group(f"ring {index}")
            for param_name in peak_fit_dict[index]:
                # ignore non function parameter
                if param_name in ["fit_function", "error", "used"]:
                    continue

                # create sub dictionary if it does not exist
                if param_name not in func_param_dict:
                    func_param_dict[param_name] = list()

                # add value to dictionary from fit result dictionary
                param_value, param_error = peak_fit_dict[index][param_name]
                func_param_dict[param_name].append([index, param_value, param_error])
        else:
            # no fit
            group = nofit_group.create_group(f"ring {index}")
            # error message
            string_data_set = group.create_dataset("error", (1,), dtype=h5_string_type)
            string_data_set[0] = peak_fit_dict[index]["error"]

        # fit function
        if "fit_function" in peak_fit_dict[index]:
            function_data_set = group.create_dataset(
                "function", (1,), dtype=h5_string_type
            )
            function_data_set[0] = peak_fit_dict[index]["fit_function"]

    # add peak fitting result
    for param_name in func_param_dict:
        # form data set
        data_set = np.array(func_param_dict[param_name])
        fit_group.create_dataset(param_name, data=data_set)

    # close
    debug_h5.close()


def getWedgeSelection(
    data2d,
    q_min,
    q_delta,
    q_max,
    azimuthal_delta,
    peak_width=0.25,
    background_width=1.5,
    signal_to_noise_min=2.0,
    peak_search_window_size_factor=0.6,
    debug_dir="/tmp/",
) -> List[List[Tuple[float, float]]]:
    """
    Calculate azimuthal binning ranges automatically based on finding peaks in the annular ring. The
    output of this is intended to be used in :py:func:`~drtsans.iq.select_i_of_q_by_wedge`.

    Parameters
    ==========
    data2d: ~drtsans.dataobjects.Azimuthal
    q_min: float
        The left bin boundary for the first Q-bin
    q_delta: float
        The size of the bins in Q
    q_max: float
        The left bin  boundary for the last Q-bin
    azimuthal_delta: float
        The size of the bins in azimuthal angle
    peak_width: float
        Percent of full-width-half-max (FWHM) of the peak to define the signal to be within when
        determining the final range for azimuthal binning.
    background_width: float
        Percent of full-width-half-max (FWHM) of the peak to define the background between peaks
        to be within when determining the final range for azimuthal binning.
    signal_to_noise_min: float
        Minimum signal to noise ratio for the data to be considered "fittable"
    peak_search_window_size_factor: float
        Factor of 360 / (num peaks) to construct the search range for wedge peak
    debug_dir: str
        Full path of the output directory for debugging output files

    Results
    =======
    ~list
      list containing 2 lists each contains 2 2-tuples
      as ``[[(peak1_min, peak1_max), (peak2_min, peak2_max)], [(..., ...), (..., ...)]]``

    """
    # Bin azimuthal
    q, azimuthal_rings = _binInQAndAzimuthal(
        data2d,
        q_min=q_min,
        q_max=q_max,
        q_delta=q_delta,
        azimuthal_delta=azimuthal_delta,
    )

    fit_results_tuple = _fitQAndAzimuthal(
        azimuthal_rings,
        q_bins=q,
        signal_to_noise_min=signal_to_noise_min,
        azimuthal_start=110.0,
        maxchisq=1000.0,
        peak_search_window_size_factor=peak_search_window_size_factor,
    )
    center_vec, fwhm_vec, fit_dict = fit_results_tuple

    # Export fitting result
    _export_to_h5(
        iq2d=data2d,
        rings=azimuthal_rings,
        azimuthal_delta=azimuthal_delta,
        peak_fit_dict=fit_dict,
        output_dir=debug_dir,
    )
    _plot_fit_results(azimuthal_rings, fit_dict, debug_dir)

    # verify that the results didn't predict wedges larger than half of the data
    if np.any(np.array(fwhm_vec) > 360.0 / 2):
        values = ["{:.1f}deg".format(value) for value in fwhm_vec]
        raise RuntimeError(
            "Encountered large fwhm values: {}".format(", ".join(values))
        )

    # convert to min and max ranges
    min_vec, max_vec = [], []

    min_vec.append(center_vec[0] - peak_width * fwhm_vec[0])
    max_vec.append(center_vec[0] + peak_width * fwhm_vec[0])

    min_vec.append(center_vec[0] + background_width * fwhm_vec[0])
    max_vec.append(center_vec[1] - background_width * fwhm_vec[1])

    min_vec.append(center_vec[1] - peak_width * fwhm_vec[1])
    max_vec.append(center_vec[1] + peak_width * fwhm_vec[1])

    min_vec.append(center_vec[1] + background_width * fwhm_vec[1])
    max_vec.append(center_vec[0] - background_width * fwhm_vec[0])

    # clean up the data to be in the form expected by select_i_of_q_by_wedge
    min_vec = np.array(min_vec)
    max_vec = np.array(max_vec)

    min_vec[min_vec < -90.0] += 360.0
    max_vec[max_vec < -90.0] += 360.0

    min_vec[min_vec > 270.0] -= 360.0
    max_vec[max_vec > 270.0] -= 360.0

    # put wedges on opposite sides together
    raw_wedges = list(zip(min_vec, max_vec))
    summing_wedges = []
    for i in range(len(raw_wedges) // 2):  # iterate over half the list
        summing_wedges.append([raw_wedges[i], raw_wedges[i + 2]])

    return summing_wedges


def _binInQAndAzimuthal(data, q_min, q_delta, q_max, azimuthal_delta):
    """This function bins the data in Qmod and azimuthal accoring to the supplied parameters. The maximum
    azimuthal is 540deg to allow for finding a peak at/near azimuthal=0deg.

    Parameters
    ==========
    data: ~drtsans.dataobjects.IQazimuthal
    q_min: float
        The left bin boundary for the first Q-bin
    q_delta: float
        The size of the bins in Q
    q_max: float
        The left bin  boundary for the last Q-bin
    azimuthal_delta: float
        The size of the bins in azimuthal

    Results
    =======
    tuple
        Histogram of ```(intensity, error, azimuthal_bins, q_bins)```
    """
    # Export information for Q
    data_q_vec = np.sqrt(data.qx ** 2 + data.qy ** 2)
    logger.notice(f"Raw I(Q). Q range: {data_q_vec.min()}, {data_q_vec.max()}")
    # the bonus two steps is to get the end-point in the array
    q_bins = np.arange(q_min, q_max + q_delta, q_delta, dtype=float)
    logger.notice(f"1D Q bins: {q_bins}")

    # create azimuthal binning BinningParams takes number of steps
    azimuthal_offset = 0.5 * azimuthal_delta
    azimuthal_binning = BinningParams(
        0.0 - azimuthal_offset,
        360.0 - azimuthal_offset,
        bins=int(360.0 / azimuthal_delta),
    )
    # create the I(azimuthal) for each q-ring
    data_of_q_rings = []
    # debugging output file
    for qmin_ring, qmax_ring in zip(q_bins[:-1], q_bins[1:]):
        # bin into I(azimuthal)
        I_azimuthal = bin_annular_into_q1d(
            data, azimuthal_binning, qmin_ring, qmax_ring, BinningMethod.NOWEIGHT
        )

        # Create a copy of the arrays with the 360->540deg region repeated
        # ignore - delta_mod_q wavelength
        mod_q_new = determine_1d_linear_bins(
            x_min=0.0,
            x_max=540.0 + azimuthal_delta,
            bins=1 + int(540.0 / azimuthal_delta),
        ).centers
        num_orig_bins = I_azimuthal.mod_q.size
        num_repeated_bins = mod_q_new.size - num_orig_bins

        intensity_new = np.zeros(mod_q_new.size)
        intensity_new[:num_orig_bins] = I_azimuthal.intensity
        intensity_new[-1 * num_repeated_bins :] = I_azimuthal.intensity[
            :num_repeated_bins
        ]

        error_new = np.zeros(mod_q_new.size)
        error_new[:num_orig_bins] = I_azimuthal.error
        error_new[-1 * num_repeated_bins :] = I_azimuthal.error[:num_repeated_bins]

        I_azimuthal = IQmod(intensity=intensity_new, error=error_new, mod_q=mod_q_new)

        # append to the list of spectra
        data_of_q_rings.append(I_azimuthal)

    return q_bins, data_of_q_rings


def _estimatePeakParameters(intensity, azimuthal, azimuthal_start, window_half_width):
    """Estimate the peak parameters by determining a window around a bright point in the data then using
    moment calculations to estimate the parameters for a Gaussian that approximates the actual peak.
    This is done to aid the fitting which does better with better starting values.

    Parameters
    ==========
    intensity: numpy.ndarray
        Array of intensities. This must not have nans in it.
    azimuthal: numpy.ndarray
        Array of azimuthal angles centers
    azimuthal_start: float
        Starting guess of peak center
    window_half_width: float
        The window used is this amount on either side of what is determined to be the peak center

    Results
    =======
    tuple
        ``(intensity, center, sigma)`` where intensity is the full height including background
    """
    # Look for the highest point in a section of the data. This is an iterative approach that starts with a window
    # centered at `azimuthal_start`, the repeats until the maximum within the window doesn't move more than 1deg.
    azimuthal_new = azimuthal_start  # where to search around
    # azimuthal_last = azimuthal_start  # last known value
    while True:
        # determine new windows staying at least 90.deg inside the edges
        window_min = np.max((azimuthal_new - window_half_width, azimuthal.min() + 90.0))
        window_max = np.min((azimuthal_new + window_half_width, azimuthal.max() - 90.0))

        # create a search window around azimuthal_new
        left_index = azimuthal.searchsorted(window_min, side="right")
        right_index = azimuthal.searchsorted(window_max, side="right")
        # the highest value in the window
        max_value = intensity[left_index:right_index].max()
        # where that is in the window
        max_index = (
            np.where(intensity[left_index:right_index] == max_value)[0].max()
            + left_index
        )
        # update values
        azimuthal_last = azimuthal_new
        azimuthal_new = azimuthal[max_index]

        # stop searching if the value hasn't changed by less than one degree
        if abs(azimuthal_new - azimuthal_last) < 1.0:
            break
    # output
    print(
        f"[WEDGE FIT] azimuthal: {azimuthal_new}, {azimuthal_last} with "
        f"left and right as {left_index}, {right_index}"
    )

    # now use the first two moments of the data within the window to give an improved center position (first moment)
    # and width (derived from second moment)

    # the position of the center of the peak is the first moment of the data. "mean" can be thought of as the
    # center of mass of the peak in azimuthal angle.
    mean = np.sum(
        intensity[left_index:right_index] * azimuthal[left_index:right_index]
    ) / np.sum(intensity[left_index:right_index])

    # the fit uses sigma rather than fwhm
    # calculate the second moment about the mean as an approximation to a Gaussian's "sigma" parameter
    sigma = np.sum(
        intensity[left_index:right_index]
        * np.square(azimuthal[left_index:right_index] - mean)
    ) / np.sum(intensity[left_index:right_index])
    sigma = np.sqrt(sigma)

    return max_value, mean, sigma


def _fitSpectrum(
    spectrum,
    q_value,
    signal_to_noise_min,
    azimuthal_start,
    peak_search_window_size_factor,
    verbose=True,
):
    """Extract the peak fit parameters for the data. This is done by observing where 2 maxima are in the
    spectrum then fitting for the peak parameters. This makes the assumption that the two peaks are 180deg
    apart.

    Parameters
    ----------
    spectrum:
    q_value: float
        center (of q_min and q_max) for given annular binned Q-ring
    signal_to_noise_min: float
        Minimum signal to noise ratio for the data to be considered "fittable"
    azimuthal_start: float
        First position to look for peaks around
    peak_search_window_size_factor: float
        Factor of 360 / (num peaks) to construct the search range for wedge peak
    verbose: bool
        Flag to output fitting information

    Returns
    -------
    dict
        dict[name] = (value, error) where all of the fit parameters are converted.
        f0 is background, then f1...fn are the fitted peaks

    """
    # define a default window size based on the number of peaks the function supports
    # currently only two peaks that are approximately 180deg apart is supported
    NUM_PEAK = 2
    # window_factor = 0.6  # default is 0.6 about 108 degree with 2 peaks .. for strong anisotropic: 0.1
    peak_search_window_size = peak_search_window_size_factor * (360.0 / NUM_PEAK)
    print(
        f"[WEDGE] Fixed window size = {peak_search_window_size}"
        f"from factor {peak_search_window_size_factor} Number of peaks = {NUM_PEAK}"
    )

    # filter out the nans
    mask = np.logical_not(np.isnan(spectrum.intensity))
    if np.sum(mask) < 10:  # do not allow fitting less points than there are parameters
        raise RuntimeError(
            "Less than 8 points being fit with 7 parameters (found {} points)".format(
                np.sum(mask)
            )
        )

    # first estimate background as minimum value
    # this will be subtracted off from found intensities during estimation
    background = spectrum.intensity[mask].min()

    # check if there is signal to noise greater than 2
    # this calculation assumes that the background is positive
    signal_to_noise = np.sum(spectrum.intensity[mask]) / (
        float(np.sum(mask)) * background
    )
    if signal_to_noise < signal_to_noise_min:
        raise RuntimeError(
            "Estimated signal to noise is smaller than {}: found {:.2f}".format(
                signal_to_noise_min, signal_to_noise
            )
        )

    # start of what will eventually be the fit function by specifying the background
    function = ["name=FlatBackground,A0={}".format(background)]

    # template for describing initial peak guess
    gaussian_str = "name=Gaussian,Height={},PeakCentre={},Sigma={}"

    # guess where one peak might be, start with a window of WINDOW_SIZE each side around 110
    intensity_peak, azimuthal_first, sigma = _estimatePeakParameters(
        spectrum.intensity[mask],
        spectrum.mod_q[mask],
        azimuthal_start=azimuthal_start,
        window_half_width=peak_search_window_size,
    )
    function.append(
        gaussian_str.format(intensity_peak - background, azimuthal_first, sigma)
    )

    for peak_index in range(1, NUM_PEAK):
        # assume the other peak is 360 / NUM_PEAK degrees away
        azimuthal_start = azimuthal_first + (360.0 / NUM_PEAK * peak_index)
        intensity_peak, azimuthal_second, sigma = _estimatePeakParameters(
            spectrum.intensity[mask],
            spectrum.mod_q[mask],
            azimuthal_start=azimuthal_start,
            window_half_width=peak_search_window_size,
        )
        function.append(
            gaussian_str.format(intensity_peak - background, azimuthal_second, sigma)
        )

    # create workspace version of data
    # this includes the nans so `Fit` has to be told to ignore them
    q_azimuthal_workspace = spectrum.to_workspace()

    # fit the positions of the two suspected peaks
    fit_workspace_prefix = unique_workspace_dundername()
    fit_function = ";".join(function)
    try:
        fitresult = Fit(
            Function=";".join(function),
            InputWorkspace=q_azimuthal_workspace,
            Output=fit_workspace_prefix,
            StartX=spectrum.mod_q.min() + 90.0,
            EndX=spectrum.mod_q.min() + 90.0 + 360.0,
            OutputParametersOnly=True,
            IgnoreInvalidData=True,
        )
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to fit Q={} with fit function {}".format(q_value, fit_function)
        ) from e
    finally:
        DeleteWorkspace(q_azimuthal_workspace)

    if fitresult.OutputStatus != "success":
        raise RuntimeError("Failed to fit Q={}".format(q_value))

    # convert the table into a dict[name] = (value, error)
    result = {"fit_function": fit_function}
    for i in range(fitresult.OutputParameters.rowCount()):
        row = fitresult.OutputParameters.row(i)
        print(f"[DEBUG-TEST] row: {row} of type {type(row)}")
        name = row["Name"]
        if name.startswith("Cost function"):
            name = "chisq"
        result[name] = (row["Value"], row["Error"])

    # delete fit results
    for label in ["_Parameters", "_NormalisedCovarianceMatrix"]:
        DeleteWorkspace(Workspace=fit_workspace_prefix + label)

    if verbose:
        print(f"Fit result: {result}")

    return result


def _toPositionAndFWHM(fitresult, peak_label, maxchisq):
    """Returns ((center, center_error), (width, width_error))
    If chisq is too large or any of the errors is nan, all return values are set to nan

    This also generates the weights as height / parameter_error"""
    if fitresult["chisq"][0] > maxchisq:
        return (np.nan, np.nan), (np.nan, np.nan)
    else:
        # height = fitresult[peak_label + '.Height']
        center = fitresult[peak_label + ".PeakCentre"]
        fwhm = tuple(
            [value * SIGMA_TO_FWHM for value in fitresult[peak_label + ".Sigma"]]
        )

    # Anything being nan suggests that a fit failed. Set everything to nan so they do not
    # contribute to the weighted average.
    if np.isnan(center[1]) or np.isnan(fwhm[1]) or center[1] == 0.0 or fwhm[1] == 0.0:
        return (np.nan, np.nan), (np.nan, np.nan)

    # Weights for are height divided by uncertainty. This results in stronger peaks with lower fitting
    # uncertainty contributing more to the parameters in azimuthal angle.
    center = center[0], 1.0 / center[1]
    fwhm = fwhm[0], 1.0 / fwhm[1]

    return center, fwhm


def _weighted_position_and_width(peaks):
    """For a set of peaks, calculate the weighted average position and weighted average fwhm

    Parameters
    ==========
    peaks: list
        [((position, weight), (fwhm, weight)), ...] Each is a peak for a single Q-value
    as a function of azimuthal angle.

    Results
    =======
    tuple
        (position, fwhm)
    """
    if len(peaks) <= 0:
        raise RuntimeError("Encountered zero fitted peaks")
    pos_accum, pos_weight_accum = 0.0, 0.0
    fwhm_accum, fwhm_weight_accum = 0.0, 0.0
    for peak in peaks:
        # friendlier names
        pos, pos_weight = peak[0]  # position and weight
        fwhm, fwhm_weight = peak[1]  # fwhm and weight

        if (
            np.isnan(pos_weight)
            or np.isnan(fwhm_weight)
            or pos_weight <= 0.0
            or fwhm_weight <= 0.0
        ):
            continue  # don't use these points

        pos_accum += pos * pos_weight
        pos_weight_accum += pos_weight

        fwhm_accum += fwhm * fwhm_weight
        fwhm_weight_accum += fwhm_weight

    try:
        return (pos_accum / pos_weight_accum), (fwhm_accum / fwhm_weight_accum)
    except ZeroDivisionError as e:
        raise RuntimeError("Cannot determine fitted positions from zero weights") from e


def _fitQAndAzimuthal(
    azimuthal_rings,
    q_bins,
    signal_to_noise_min,
    azimuthal_start,
    maxchisq,
    peak_search_window_size_factor,
    verbose=True,
):
    """Find the peaks in the azimuthal spectra, then combine them into
    two composite centers and fwhm. This is currently coded to only
    look for two peaks.

    Parameters
    ==========
    intensity: numpy.ndarray
        The intensity as a 2-d array of Q and azimuthal
    error: numpy.ndarray
        The uncertainties as a 2-d array of Q and azimuthal
    azimuthal_bins: numpy.ndarray
        Array of azimuthal angles bin boundaries
    q_bins: numpy.ndarray
        array of Q bin boundaries
    signal_to_noise_min: float
        Minimum signal to noise ratio for the data to be considered "fittable"
    azimuthal_start: float
        First position to look for peaks around
    maxchisq: float
        The maximum chisq value for a fit result to be used in calculating the composite peak
    peak_search_window_size_factor: float
        Factor of 360 / (num peaks) to construct the search range for wedge peak
    verbose: bool
        Flag to turn on fitting information output

    Results
    =======
    list, list, dict
        The first list is the peak centers, the second is the peak fwhm, the third is a dictionary for fit result
    """
    if len(azimuthal_rings) != len(q_bins) - 1:
        raise RuntimeError("must supply q-bin boundaries")

    # change to centers in Q for messages
    q_centers = 0.5 * (q_bins[:-1] + q_bins[1:])

    # select out a single spectrum
    peakResults = [[], []]
    q_centers_used = []

    index = -1  # ring index: set to -1 due to too many 'continue' in the loop
    used_index = list()
    unfit_message = ""
    fitted_peaks_message = ""
    fit_result_dict = dict()

    for spectrum, q_center in zip(azimuthal_rings, q_centers):
        # increase the index number
        index += 1
        # init dict
        fit_result_dict[index] = dict()
        try:
            fitresult = _fitSpectrum(
                spectrum,
                q_center,
                signal_to_noise_min=signal_to_noise_min,
                azimuthal_start=azimuthal_start,
                peak_search_window_size_factor=peak_search_window_size_factor,
                verbose=verbose,
            )
            newlyFittedPeaks = [
                _toPositionAndFWHM(fitresult, label, maxchisq) for label in ["f1", "f2"]
            ]

            # record the fit result
            fit_result_dict[index] = fitresult

            if np.isnan(newlyFittedPeaks[0][0][0]) or np.isnan(
                newlyFittedPeaks[1][0][0]
            ):
                error_reason = (
                    f"spectrum {index}: failed to fit peaks due to NaN in fit result\n"
                )
                unfit_message += error_reason
                fit_result_dict[index]["error"] = error_reason
                continue
            else:
                fitted_peaks_message += (
                    f"spectrum {index-1}: Fitted peaks: {newlyFittedPeaks}\n"
                )
            for i in range(len(peakResults)):
                peakResults[i].append(newlyFittedPeaks[i])
            q_centers_used.append(q_center)
            used_index.append(index)
        except RuntimeError as e:
            error_reason = (
                "spectrum {}: Not using information from Q-slice ({}A):".format(
                    index, q_center
                )
            )
            error_reason += f"Encountered runtime error: {e}\n"  # don't worry about it
            unfit_message += error_reason
            fit_result_dict[index]["error"] = error_reason
            continue
        except ValueError as val_err:
            # in case user specifies a range containing no Q values
            error_reason = f"Spectrum {index}: unable to fit peaks due to {val_err}\n"
            unfit_message += error_reason
            fit_result_dict[index]["error"] = error_reason
            continue
        else:
            # this ring is used
            fit_result_dict[index]["error"] = None

    logger.notice(f"Q-rings used to determine overall wedges: {q_centers_used}")
    logger.information(f"used annular binning index: {used_index}")
    logger.notice(unfit_message)

    peakResults = [_weighted_position_and_width(peak) for peak in peakResults]

    # convert into parallel arrays of centers and fwhm
    center_list = []
    fwhm_list = []
    for center, fwhm in peakResults:
        center_list.append(center)
        fwhm_list.append(fwhm)

    logger.notice(f"Fitted peak centers:\n{fitted_peaks_message}\n")
    logger.notice(
        f"Summed peak centers: {center_list}\nFWHMs              : {fwhm_list}"
    )

    return center_list, fwhm_list, fit_result_dict
