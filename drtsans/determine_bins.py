import numpy as np
from collections import namedtuple


# Define structure (namedtuple) for binning parameters: min, max, number of bins
# bins shall be integer as number of bins
BinningParams = namedtuple("BinningParams", "min max bins")
# Define structure (namedtuple) for bins: bin edges and bin boundaries
# Both bin edge and bin boundaries shall be 1-dimensional 1D array and 'edges' size is 1 larger than centers
Bins = namedtuple("Bins", "edges centers")


def determine_1d_linear_bins(x_min, x_max, bins):
    """Determine linear bin edges and centers

    Parameters
    ----------
    x_min : float
        Q min of bin edge
    x_max : float or None
        Q max of bin edge
    bins : integer
        number of bins

    Returns
    -------
    ~drtsans.iq.Bins
        Bins including bin centers and bin edges

    """
    # Check input x min and x max
    if x_min is None or x_max is None or x_min >= x_max:
        raise RuntimeError(
            "x min {} and x max {} must not be None and x min shall be less than x max"
            "".format(x_min, x_max)
        )
    # force the number of bins to be an integer and error check it
    bins = int(bins)
    if bins <= 0:
        raise ValueError("Encountered illegal number of bins: {}".format(bins))

    # Calculate Q step size
    delta_x = float((x_max - x_min) / bins)
    # Determine bin edges
    bin_edges = np.arange(bins + 1).astype("float") * delta_x + x_min
    # Determine bin centers from edges
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5

    # Construct Bins instance
    linear_bins = Bins(bin_edges, bin_centers)

    return linear_bins


def determine_1d_log_bins(
    x_min, x_max, decade_on_center, n_bins_per_decade=None, n_bins=None
):
    """

    Parameters
    ----------
    x_min: float
        minimum value of X in the bins
    x_max: float
        maximum value of X in the bins
    decade_on_center: bool
        flag that data must be centered on decades
    n_bins_per_decade: int, None
        density of points (number of data points per decade)
    n_bins: int, None
        total number of points in the output

    Returns
    -------

    """
    # Check inputs
    if n_bins_per_decade is None and n_bins is None:
        raise RuntimeError(
            "Density of points (n_bins_per_decade) and total number of bins (n_bins) cannot be "
            "None simultaneously.  One and only one of them must be specified."
        )
    elif n_bins_per_decade is not None and n_bins is not None:
        raise RuntimeError(
            "Density of points (n_bins_per_decade) and total number of bins (n_bins) cannot be "
            "specified simultaneously.  One and only one of them must be specified."
        )
    # only allow either n_bins or n_bins_per_decade

    # Calculate Q min, number of total bins and number of steps
    if n_bins_per_decade is not None:
        # user specifies number of bins per decade

        # determine X min
        if decade_on_center:
            x_ref = _calculate_x_ref(x_min, n_bins_per_decade)
        else:
            x_ref = x_min

        # calculate step size
        n_step = 10 ** (1 / n_bins_per_decade)

        # calculate number of bins
        n_bins = _calculate_n_bins(x_ref, x_max, n_step)
    else:
        # user specifies number of total bins

        # case that is not supported
        if decade_on_center:
            assert n_bins_per_decade is not None, (
                "For option decade_on_center, number of bins per decade " "is required"
            )
        x_ref = x_min

        # calculate bin step size
        # Equation 11.33
        n_step = 10 ** ((np.log10(x_max / x_ref)) / (n_bins - 1))

    # Calculate kay
    kay = (n_step - 1) / (n_step + 1)

    # Calculate bin centers
    # init an array ranging from 0 to (n_bins - 1)
    bin_centers = np.arange(n_bins).astype("float64")
    # Equation 11.34: Q_k = Q_ref * 10^(k * delta L)
    bin_centers = x_ref * n_step ** bin_centers

    # Calculate bin edges (aka boundaries)
    # Equation 11.35
    bin_edges = np.ndarray(shape=(n_bins + 1,), dtype="float64")
    # calculate left edges (i.e., right edges except last one), i.e., Q_{k-1, max} = Q_{k, min}
    bin_edges[:-1] = bin_centers[:] - kay * bin_centers[:]
    # calculate last right edge
    bin_edges[-1] = bin_centers[-1] + kay * bin_centers[-1]

    # Form output as Bins object
    log_bins = Bins(bin_edges, bin_centers)

    return log_bins


def _calculate_x_ref(x_min, n_bins_per_decade):
    """Calculate reference X (minimum X) by Equation in Chapter 11
    x_ref = 10^((1/N_bins_decade)*(round(N_bins_decade*log10(Q_min))))
    Parameters
    ----------
    x_min: float
        minimum x value
    n_bins_per_decade: int
        point density

    Returns
    -------

    """
    ref_x = 10 ** (
        (1.0 / n_bins_per_decade) * (np.round(n_bins_per_decade * np.log10(x_min)))
    )

    return ref_x


def _calculate_n_bins(x_min, x_max, n_step):
    """Calculate number of total bins by implementing
    Equation 11.32

    N_bins = floor(ceil((log10(Q_max/Calc_Q_min) + log10((N_step+1)/2.0))/log10(N_step)))

    Parameters
    ----------
    x_min: float
        minimum value of x to be included in the bin
    x_max: float
        maximum value of x to be included in the bin
    n_step: float
        step size

    Returns
    -------

    """
    n_bins = np.floor(
        np.ceil(
            (np.log10(x_max / x_min) + np.log10((n_step + 1) * 0.5)) / np.log10(n_step)
        )
    )

    # to avoid round off error such that n_bins = |n_bins| + epsilon, where epsilon is an infinitesimally
    # small value
    n_bins = int(n_bins + 1e-5)

    return n_bins
