r""" Links to mantid algorithms
FindCenterOfMassPosition <https://docs.mantidproject.org/nightly/algorithms/FindCenterOfMassPosition-v1.html>
Integration              <https://docs.mantidproject.org/nightly/algorithms/Integration-v1.html>
MoveInstrumentComponent  <https://docs.mantidproject.org/nightly/algorithms/MoveInstrumentComponent-v1.html>
DeleteWorkspace          <https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html>
"""
from mantid.simpleapi import (
    FindCenterOfMassPosition,
    Integration,
    MoveInstrumentComponent,
    DeleteWorkspace,
    mtd,
)
from mantid.kernel import logger

# drtsans imports
from drtsans.settings import unique_workspace_dundername as uwd
from drtsans.mask_utils import apply_mask, mask_spectra_with_special_values
from drtsans.solid_angle import solid_angle_correction
import numpy as np
import lmfit


__all__ = [
    "center_detector",
    "find_beam_center",
    "fbc_options_json",
]  # exports to the drtsans namespace


def _results_to_dict(params):
    fit_results = {}
    for key in params:
        value = params[key]
        fit_results[key] = {
            "value": value.value,
            "stderr": value.stderr,
            "vary": value.vary,
        }
    return fit_results


# defining 2D Gaussian fitting functions
def _Gaussian2D(x1, y1, amp, sigma_x, sigma_y, theta, CenterX, CenterY):
    a = np.cos(theta) ** 2 / (2.0 * sigma_x ** 2) + np.sin(theta) ** 2 / (
        2.0 * sigma_y ** 2
    )
    b = -np.sin(2.0 * theta) / (4.0 * sigma_x ** 2) + np.sin(2.0 * theta) / (
        4.0 * sigma_y ** 2
    )
    c = np.sin(theta) ** 2 / (2.0 * sigma_x ** 2) + np.cos(theta) ** 2 / (
        2.0 * sigma_y ** 2
    )
    amplitude = amp / (np.sqrt(2.0 * np.pi) * np.sqrt(sigma_x * sigma_y))
    val = amplitude * np.exp(
        -(
            a * (x1 - CenterX) ** 2
            + 2.0 * b * (x1 - CenterX) * (y1 - CenterY)
            + c * (y1 - CenterY) ** 2
        )
    )
    return val


def _find_beam_center_gaussian(ws, parameters={}):
    # fitting 2D gaussian to center data
    """Fitting 2D gaussian to workspace for finding the beam center.

    Parameters
    ----------
    ws:  str,  ~mantid.api.MatrixWorkspace
        Input workspace
    parameters: dict
        Fitting parameters passed onto lmfit. Defaults include
        'amp', Amplitude of the Gaussian function. Default: ws.extractY().max()
        'sigma_x', X spead of the Gaussian function. Default: 0.01
        'sigma_y', Y spead of the Gaussian function. Default: 0.01
        'theta', Clockwise rotation angle of Gaussian function. Default: 0.
        'CenterX', Estimate for the beam center in X [m]. Default: 0.
        'CenterY', Estimate for the beam center in Y [m]. Default: 0.

    Returns
    -------
    tuple
        (X, Y) coordinates of the beam center (units in meters).
    """
    ws = mtd[str(ws)]
    ws_size = ws.getNumberHistograms()
    x = np.empty(ws_size)
    y = np.empty(ws_size)
    intes = np.empty(ws_size)
    intes_err = np.empty(ws_size)
    keep = np.empty(ws_size, dtype=np.bool_)

    for i, si in enumerate(ws.spectrumInfo()):
        pos = si.position
        x[i] = pos.X()
        y[i] = pos.Y()
        keep[i] = not si.isMasked and np.isfinite(ws.readY(i)[0])
        intes[i] = ws.readY(i)[0]
        intes_err[i] = ws.readE(i)[0]

    x = x[keep]
    y = y[keep]
    intes = intes[keep]
    intes_err = intes_err[keep]

    model = lmfit.Model(
        _Gaussian2D,
        independent_vars=["x1", "y1"],
        param_names=["amp", "sigma_x", "sigma_y", "theta", "CenterX", "CenterY"],
    )

    params = lmfit.Parameters()
    for key, value in parameters.items():
        params.add(key, **value)

    if "amp" not in params:
        params.add("amp", value=ws.extractY().max())
    if "sigma_x" not in params:
        params.add("sigma_x", value=0.01, min=np.finfo(float).eps)  # width in x
    if "sigma_y" not in params:
        params.add("sigma_y", value=0.01, min=np.finfo(float).eps)  # width in y
    if "theta" not in params:
        params.add("theta", value=0.0, min=-np.pi / 2.0, max=np.pi / 2.0)
    if "CenterX" not in params:
        params.add("CenterX", value=0.0)
    if "CenterY" not in params:
        params.add("CenterY", value=0.0)
    results = model.fit(
        intes, x1=x, y1=y, weights=1.0 / intes_err, params=params, nan_policy="omit"
    )
    fit_params = results.params
    return (
        fit_params["CenterX"].value,
        fit_params["CenterY"].value,
        _results_to_dict(fit_params),
    )


def fbc_options_json(reduction_input):
    fbc_options = {}
    if "method" in reduction_input["beamCenter"].keys():
        method = reduction_input["beamCenter"]["method"]
        fbc_options["method"] = method
        if method == "gaussian":
            if "gaussian_centering_options" in reduction_input["beamCenter"].keys():
                fbc_options["centering_options"] = reduction_input["beamCenter"][
                    "gaussian_centering_options"
                ]
        elif method == "center_of_mass":
            if "com_centering_options" in reduction_input["beamCenter"].keys():
                fbc_options["centering_options"] = reduction_input["beamCenter"][
                    "com_centering_options"
                ]
    return fbc_options


def find_beam_center(
    input_workspace,
    method="center_of_mass",
    mask=None,
    mask_options={},
    centering_options={},
    solid_angle_method="VerticalTube",
):
    r"""
    Calculate absolute coordinates of beam impinging on the detector.
    Usually employed for a direct beam run (no sample and not sample holder).

    **Mantid algorithms used:**
        :ref:`FindCenterOfMassPosition <algm-FindCenterOfMassPosition-v2>`,
        :ref:`Integration <algm-Integration-v1>`,

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventWorkspace
    method: str
        Method to calculate the beam center. Available methods are:
        - 'center_of_mass', invokes :ref:`FindCenterOfMassPosition <algm-FindCenterOfMassPosition-v1>`
        - 'gaussian', 2D Gaussian fit to beam center data
    mask: mask file path, `MaskWorkspace``, :py:obj:`list`.
        Mask to be passed on to ~drtsans.mask_utils.mask_apply.
    mask_options: dict
        Additional arguments to be passed on to ~drtsans.mask_utils.mask_apply.
    centering_options: dict
        Arguments to be passed on to the centering method.
    solid_angle_method: bool, str, specify which solid angle correction is needed

    Returns
    -------
    tuple
        (X, Y, results) coordinates of the beam center (units in meters), dictionary of special parameters
    """
    if method not in ["center_of_mass", "gaussian"]:
        raise NotImplementedError()  # (f'{method} is not implemented')

    # integrate the TOF
    flat_ws = Integration(InputWorkspace=input_workspace, OutputWorkspace=uwd())
    mask_spectra_with_special_values(flat_ws)

    if mask is not None or mask_options != {}:
        apply_mask(flat_ws, mask=mask, **mask_options)

    if solid_angle_method:
        solid_angle_correction(flat_ws, detector_type=solid_angle_method)

    # find center of mass position
    if method == "center_of_mass":
        center = FindCenterOfMassPosition(InputWorkspace=flat_ws, **centering_options)
        x, y = center
        fit_results = {}
    else:  # method == 'gaussian':
        x, y, fit_results = _find_beam_center_gaussian(flat_ws, centering_options)
    logger.information("Found beam position: X={:.3} m, Y={:.3} m.".format(x, y))
    DeleteWorkspace(flat_ws)
    return x, y, fit_results


def center_detector(input_workspace, center_x, center_y, component="detector1"):
    """Translate the beam center currently located at (center_x, center_y) by an amount
    (-center_x, -center_y), so that the beam center is relocated to the origin of coordinates on the XY-plane

    **Mantid algorithms used:**
    :ref:`MoveInstrumentComponent <algm-MoveInstrumentComponent-v1>`,

    Parameters
    ----------
    input_workspace : Workspace2D, str
        The workspace to be centered
    center_x : float
        in meters
    center_y : float
        in meters
    component : string
        name of the detector to be centered
    """
    MoveInstrumentComponent(
        Workspace=input_workspace,
        ComponentName=component,
        X=-center_x,
        Y=-center_y,
        RelativePosition=True,
    )
