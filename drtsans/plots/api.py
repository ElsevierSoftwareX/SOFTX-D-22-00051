from enum import Enum
import json
import numpy as np
from typing import List, Any, Dict
import matplotlib
import warnings

warnings.simplefilter("ignore", UserWarning)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa E402
from matplotlib.colors import LogNorm  # noqa E402
import mpld3  # noqa E402
from mpld3 import plugins  # noqa E402

from mantid.api import mtd  # noqa E402
from mantid.simpleapi import logger  # noqa E402
from drtsans.tubecollection import TubeCollection  # noqa E402
from drtsans.dataobjects import DataType, getDataType  # noqa E402
from drtsans.geometry import panel_names  # noqa E402
from drtsans.iq import get_wedges  # noqa E402
from drtsans.iq import validate_wedges_groups  # noqa E402

__all__ = ["plot_IQmod", "plot_IQazimuthal", "plot_detector"]


# mpld3 taken from hack from https://github.com/mpld3/mpld3/issues/434#issuecomment-381964119
if mpld3.__version__ == "0.3":

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    from mpld3 import _display  # noqa E402

    _display.NumpyEncoder = NumpyEncoder


class Backend(Enum):
    """Class for denoting which back-end to save the plot using"""

    MPLD3 = "d3"  # read-only
    MATPLOTLIB = "mpl"  # read and write

    def __str__(self):
        return self.value

    @staticmethod
    def getMode(mode):
        """Public function to convert anything into :py:obj:`Backend`"""
        try:
            if mode in Backend:
                return mode
        except TypeError:
            pass  # intentionally ignore

        mode = str(mode)
        if mode == "mpl" or mode == "matplotlib":
            return Backend.MATPLOTLIB
        elif mode == "mpld3" or mode == "d3":
            return Backend.MPLD3
        else:
            return Backend[mode.upper()]


def _save_file(figure, filename, backend, show=False):
    """Convenience method for the common bits of saving the file based on
    the selected backend.

    Parameters
    ----------
    figure: ~matplotlib.pyplot.figure
        The figure to save to a file
    filename: str
        The name of the file to save to
    backend: Backend
        Which :py:obj:`Backend` to use for saving
    show: bool
        Whether or not to show the figure rather than saving. This is only
        available if the :py:obj:`~Backend.MPLD3` backend is selected.
    """
    if backend == Backend.MATPLOTLIB:
        if filename:
            figure.savefig(filename)
        if show:
            if "inline" in matplotlib.get_backend():  # ipython notebook
                figure.show()
            else:
                raise RuntimeError("Cannot show data with matplotlib backend")
    else:
        if not filename.endswith("json"):
            raise RuntimeError('File "{}" must have ".json" suffix'.format(filename))

        plugins.connect(figure, plugins.MousePosition(fontsize=14, fmt=".0f"))
        with open(filename, "w") as outfile:
            mpld3.save_json(figure, outfile)

        if show:
            mpld3.show(figure)


def _q_label(backend, subscript=""):
    """mpld3 doesn't currently support latex markup. This generates an
    acceptable label for the supplied backend.

    Parameters
    ----------
    backend: Backend
        Which :py:obj:`Backend` to generate the caption for
    subscript: str
        The subscript on the "Q" label. If none is specified then no
        subscript will be displayed
    """
    label = "Q"
    if subscript:
        label += "_" + str(subscript)

    if backend == Backend.MATPLOTLIB:
        return "$" + label + r" (\AA^{-1})$"
    else:  # mpld3
        return label + " (1/{})".format("\u212B")


def plot_IQmod(
    workspaces, filename, loglog=True, backend="d3", errorbar_kwargs={}, **kwargs
):
    """Save a plot representative of the supplied workspaces

    Parameters
    ----------
    workspaces: list, tuple
        A collection of :py:obj:`~drtsans.dataobjects.IQmod` workspaces to
        plot. If only one is desired, it must still be supplied in a
        :py:obj:`list` or :py:obj:`tuple`.
    filename: str
        The name of the file to save to. For the :py:obj:`~Backend.MATPLOTLIB`
        backend, the type of file is determined from the file extension
    loglog: bool
        If true will set both axis to logarithmic, otherwise leave them as linear
    backend: Backend
        Which backend to save the file using
    errorbar_kwargs: dict
        Optional arguments to :py:obj:`matplotlib.axes.Axes.errorbar`
        Can be a comma separated list for each workspace
        e.g. ``{'label':'main,wing,both', 'color':'r,b,g', 'marker':'o,v,.'}``
    kwargs: dict
        Additional key word arguments for :py:obj:`matplotlib.axes.Axes`

    """
    backend = Backend.getMode(backend)
    for workspace in workspaces:
        datatype = getDataType(workspace)
        if datatype != DataType.IQ_MOD:
            raise RuntimeError('Do not know how to plot type="{}"'.format(datatype))

    fig, ax = plt.subplots()
    for n, workspace in enumerate(workspaces):
        eb, _, _ = ax.errorbar(
            workspace.mod_q, workspace.intensity, yerr=workspace.error
        )
        for key in errorbar_kwargs:
            value = [v.strip() for v in errorbar_kwargs[key].split(",")]
            plt.setp(eb, key, value[min(n, len(value) - 1)])
    ax.set_xlabel(_q_label(backend))
    ax.set_ylabel("Intensity")
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    if kwargs:
        plt.setp(ax, **kwargs)

    _save_file(fig, filename, backend)


def _create_ring_roi(iq2d, q_min, q_max, input_roi) -> np.ndarray:
    """Create a mask or ROI by q range and result in a ring (or circle)

    Returns
    -------
    np.ndarray
        Boolean numpy array with same shape as input iq2d.  False for masking, True for ROI

    """
    # create region of interest overlay
    output_roi = input_roi

    # Larger than min Q
    if q_min is not None:
        # roi = np.logical_and(roi, np.square(iq2d.qx) + np.square(iq2d.qy) > np.square(q_min))
        q_min_roi = np.square(iq2d.qx) + np.square(iq2d.qy) > np.square(q_min)
        # act on output
        output_roi = np.logical_and(output_roi, q_min_roi)

    # Less than max Q
    if q_max is not None:
        q_max_roi = np.square(iq2d.qx) + np.square(iq2d.qy) < np.square(q_max)
        # act on output
        output_roi = np.logical_and(output_roi, q_max_roi)

    return output_roi


def _create_wedge_roi(iq2d, wedges, symmetric_wedges: bool,
                      input_roi: np.ndarray, qx_limit: float = 1e10) -> np.ndarray:
    """Create ROI matrix with

    Parameters
    ----------
    iq2d
    wedges
    symmetric_wedges
    input_roi
    qx_limit: float
        Upper limit of possible Qx

    Returns
    -------
    np.ndarray
        boolean array with same shape as intensity.  True for ROI.

    """
    # set output
    output_roi = input_roi

    # Check: wedge must be specified
    if wedges is None:
        return output_roi

    # create bool array selecting nothing
    roi_wedges = np.zeros(iq2d.intensity.shape).astype(bool)
    # expand the supplied variables into an easier form
    # get validated wedge in groups and flatten it to list of wedge angles
    wedge_angles = validate_wedges_groups(wedges, symmetric_wedges)
    wedge_angles = [
        wedge_angle for wedges_group in wedge_angles for wedge_angle in wedges_group
    ]

    # create the individual selections and combine with 'or'
    # Note: qx is in [[qx0, qx0, qx0, ...], [qx1, qx1, qx1, ...], ...]
    #       qy is in [[qy0, qy1, qy2, ...], [qy0, qy1, qy2, ...], ...]
    # this is transposed comparing to how Qx and Qy is plotted for the output
    azimuthal = np.rad2deg(
        np.arctan2(iq2d.qy, iq2d.qx)
    )
    # Try 1azimuthal = np.rad2deg(np.arctan2(workspace.qx, workspace.qy))
    azimuthal[azimuthal <= -90.0] += 360.0
    for lower_boundary_angle, upper_boundary_angle in wedge_angles:
        wedge = np.logical_and((azimuthal > lower_boundary_angle), (azimuthal < upper_boundary_angle))
        roi_wedges = np.logical_or(roi_wedges, wedge)

    # combine with existing roi
    output_roi = np.logical_and(output_roi, roi_wedges)

    return output_roi


def _require_transpose_intensity(iq2d) -> bool:
    """Check whether the intensity/ROI in IQazimuthal shall be transposed to plot
    as Qx in horizontal and Qy in vertical
    """
    # Determine whether intensity matrix shall be inverted or not
    qx2d = iq2d.qx
    qy2d = iq2d.qy
    # set up the flag to transpose ROI if I(Qx, Qy) is to be tranposed
    transpose_flag = False
    if len(qx2d.shape) == 1:
        # No need to transpose if Qx and Qy are given in 1-dim
        transpose_flag = True
    if len(qx2d.shape) == 2 and qx2d.shape[0] > 1 and np.sum(qx2d[0] == qx2d[1]) == qx2d.shape[1]:
        # Input Qx and Qy are 2-dim and
        # I(Qx, Qy) is of same order as meshgrid(Qx, Qy)
        # Qx have identical among rows:
        if qy2d.shape[1] > 1:
            # sanity check
            assert (np.sum(qy2d[:, 0] == qy2d[:, 1]) == qy2d.shape[0]), "Qy shall have identical columns"
    else:
        # I(Qx, Qy) is transposed to meshgrid(Qx, Qy)
        transpose_flag = True

    return transpose_flag


def plot_IQazimuthal(
    workspace,
    filename,
    backend="d3",
    qmin: float = None,
    qmax: float = None,
    wedges: List[Any] = None,
    symmetric_wedges: bool = True,
    mask_alpha=0.6,
    imshow_kwargs: Dict = {},
    **kwargs,
):
    """Save a plot of I(Qx, Qy).
    If qmin is specified, all I(Q) with Q less than qmin will be masked in output plot.
    If qmax is specified, all I(Q) with Q greater than qmax will be masked in output plot.
    If wedges are specified, all I(Q) out side of wedges will be masked in output plot.

    Parameters
    ----------
    workspace: ~drtsans.dataobjects.IQazimuthal
        The workspace (i.e., I(Qx, Qy)) to plot. This assumes the data is binned on a constant grid.
    filename: str
        The name of the file to save to. For the :py:obj:`~Backend.MATPLOTLIB`
        backend, the type of file is determined from the file extension
    qmin: float
        minimum 1D Q for plotting selection area
    qmax: float
        maximum 1D Q for plotting selection area
    wedges: ~list or None
        list of tuples (angle_min, angle_max) for the wedges. Select wedges to plot.
        Both numbers have to be in the [-90,270) range. It will add the wedge offset
        by 180 degrees dependent on ``symmetric_wedges``
    symmetric_wedges: bool
        Add the wedge offset by 180 degrees if True
    mask_alpha: float
        Opacity for for selection area
    backend: Backend
        Which backend to save the file using
    imshow_kwargs: ~dict
        Optional arguments to :py:obj:`matplotlib.axes.Axes.imshow` e.g. ``{"norm": LogNorm()}``
    kwargs: ~dict
        Additional key word arguments for :py:obj:`matplotlib.axes.Axes`
    """
    # Set up backend and verify data type
    backend = Backend.getMode(backend)
    datatype = getDataType(workspace)
    if datatype != DataType.IQ_AZIMUTHAL:
        raise RuntimeError('Do not know how to plot type="{}"'.format(datatype))

    # Set up ROI/mask
    roi = (np.zeros(workspace.intensity.shape) + 1).astype(bool)
    # ROI as a ring (qmin, qmax)
    roi = _create_ring_roi(workspace, qmin, qmax, roi)
    # wedge
    roi = _create_wedge_roi(workspace, wedges, symmetric_wedges, roi)

    # Make sure the orientation of intensity array can be shown correctly with imshow
    # imshow thinks the bottom right corner is (0, n-1) while it is intensity(qx_max, qy_min)
    transpose_flag = _require_transpose_intensity(workspace)
    if transpose_flag:
        # transpose both intensity and ROI
        intensity = workspace.intensity.T
        roi = roi.T
    else:
        intensity = workspace.intensity
    # convert ROI to masks
    roi = np.ma.masked_where(roi, roi.astype(int))

    # put together the plot
    fig, ax = plt.subplots()
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color="grey")
    qxmin = workspace.qx.min()
    qxmax = workspace.qx.max()
    qymin = workspace.qy.min()
    qymax = workspace.qy.max()
    pcm = ax.imshow(
        intensity,
        extent=(qxmin, qxmax, qymin, qymax),
        origin="lower",
        aspect="auto",
        **imshow_kwargs,
    )

    # add calculated region of interest
    ax.imshow(
        roi,
        alpha=mask_alpha,
        extent=(qxmin, qxmax, qymin, qymax),
        cmap="gray",
        vmax=roi.max(),
        interpolation="none",
        origin="lower",
        aspect="auto",
    )
    pcm.cmap.set_bad(alpha=0.5)

    # rest of plotting arguments
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(_q_label(backend, "x"))
    ax.set_ylabel(_q_label(backend, "y"))

    if kwargs:
        plt.setp(ax, **kwargs)

    _save_file(fig, filename, backend)


def plot_detector(
    input_workspace,
    filename=None,
    backend="d3",
    axes_mode="tube-pixel",
    panel_name=None,
    figure_kwargs={"figsize": (8, 6)},
    imshow_kwargs={"norm": LogNorm(vmin=1)},
):
    r"""
    Save a 2D plot representative of the supplied workspace

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
        The workspace to plot
    filename: str
        The name of the file to save to. For the :py:obj:`~Backend.MATPLOTLIB`
        backend, the type of file is determined from the file extension
    backend: Backend
        Which backend to save the file using
    axes_mode: str
        Plot intensities versus different axes. Options are: 'xy' for plotting versus pixel coordinates;
       'tube-pixel' for plotting versus tube and pixel index.
    panel_name: str
        Name of the double panel detector array. If :py:obj:`None`, plots will be generated for all arrays.
    figure_kwargs: str
        Optional arguments to matplotlib.pyplot.figure
    imshow_kwargs: dict
        Optional arguments to matplotlib.axes.Axes.imshow
    """
    workspace = mtd[str(input_workspace)]
    backend = Backend.getMode(backend)
    detector_names = (
        [
            panel_name,
        ]
        if panel_name is not None
        else panel_names(input_workspace)
    )
    fig = plt.figure(**figure_kwargs)
    for i_detector, detector_name in enumerate(detector_names):
        collection = TubeCollection(workspace, detector_name)
        collection = collection.sorted(view="fbfb")
        data = np.sum(
            np.array([tube.readY for tube in collection]), axis=-1
        )  # sum intensities for each pixel
        if isinstance(imshow_kwargs.get("norm", None), LogNorm) is True:
            data[data < 1e-10] = 1e-10  # no negative values when doing a logarithm plot
        mask = np.array([tube.isMasked for tube in collection])
        data = np.ma.masked_where(mask, data)
        # Add subfigure
        axis = fig.add_subplot(len(detector_names), 1, i_detector + 1)
        if axes_mode == "tube-pixel":
            image = axis.imshow(
                np.transpose(data), aspect="auto", origin="lower", **imshow_kwargs
            )
            axis_properties = {
                "set_xlabel": "tube",
                "set_ylabel": "pixel",
                "set_title": f"{detector_name}",
            }
        elif axes_mode == "xy":
            # array x and y denote the boundaries of the pixels when projected on the XY plane
            n_pixels = len(collection[0])  # number of pixels in the first tube
            # Find the "left" sides of the tubes
            x = [tube.x_boundaries[0] * np.ones(n_pixels + 1) for tube in collection]
            # Append the "right" side of the last tube
            x.append(collection[-1].x_boundaries[1] * np.ones(n_pixels + 1))
            x = np.array(x)
            axis.set_xlim(
                max(x.ravel()), min(x.ravel())
            )  # X-axis should plot from larger to smaller values
            y = [tube.pixel_y_boundaries for tube in collection]
            y.append(collection[-1].pixel_y_boundaries)
            y = np.array(y)
            # BOTTLENECK-2 (but 6 times faster than BOTTLENECK-1)
            image = axis.pcolormesh(x, y, data)
            axis_properties = {
                "set_xlabel": "X",
                "set_ylabel": "Y",
                "set_title": f"{detector_name}",
            }
        else:
            raise ValueError(
                'Unrecognized axes_mode. Valid options are "tube-pixel" and "xy"'
            )
        image.cmap.set_bad(alpha=0.5)
        [getattr(axis, prop)(value) for prop, value in axis_properties.items()]
        fig.colorbar(image, ax=axis)
    fig.tight_layout()

    if filename is not None:
        _save_file(fig, filename, backend)
