from scipy import constants
import numpy as np
import drtsans.beam_finder as bf
from drtsans.beam_finder import fbc_options_json
from typing import Tuple
from mantid import mtd
from mantid.kernel import logger
from drtsans.samplelogs import SampleLogs

__all__ = ["center_detector", "find_beam_center", "fbc_options_json"]


def _calculate_neutron_drop(path_length, wavelength):
    """Calculate the gravitational drop of the neutrons from the wing detector
    (which is closer) to the main detector.

    Parameters
    ----------
    path_length : float
        path_length in meters
    wavelength : float
        wavelength in Angstrom

    Returns
    -------
    float
        Return the Y drop in meters
    """
    wavelength *= 1e-10
    neutron_mass = constants.neutron_mass
    gravity = constants.g
    h_planck = constants.Planck
    y_drop = (gravity * neutron_mass ** 2 / (2.0 * h_planck ** 2)) * path_length ** 2
    return wavelength ** 2 * y_drop


def _beam_center_gravitational_drop(
    ws,
    beam_center_y,
    sample_det_cent_main_detector,
    sample_det_cent_wing_detector,
    vertical_offset=0.0135,
):
    """This method is used for correcting for gravitational drop by
    finding the difference in drop between the main and wing
    detectors. The center in the wing detector will be higher because
    the neutrons fall due to gravity until they hit the main detector.

    Parameters
    ----------
    ws : ~mantid.api.MatrixWorkspace, str
        The workspace to get the sample to the main detector distance
    beam_center_y : float
        The Y center coordinate in the main detector in meters
    sample_det_cent_main_detector: float
        :ref:`sample to detector center distance <devdocs-standardnames>` of the main detector
        in meters
    sample_det_cent_wing_detector : float
        :ref:`sample to detector center distance <devdocs-standardnames>` of the wing detector
        in meters
    vertical_offset: float
        :vertical offset between main detector and wing detector in m

    Returns
    -------
    float
        The new y beam center of the wing detector

    """
    sl = SampleLogs(ws)

    wavelength = np.mean(sl.wavelength.value)

    # this comes back as a positive number
    drop_main = _calculate_neutron_drop(sample_det_cent_main_detector, wavelength)
    drop_wing = _calculate_neutron_drop(sample_det_cent_wing_detector, wavelength)

    new_beam_center_y = beam_center_y + drop_main - drop_wing + (vertical_offset)
    logger.information(
        "Beam Center Y before gravity (drop = {:.3}): {:.3}"
        " after = {:.3} and vertical offset between main and wing detector= {:.3}.".format(
            drop_main - drop_wing, beam_center_y, new_beam_center_y, vertical_offset
        )
    )

    return new_beam_center_y


def find_beam_center(
    input_workspace,
    method="center_of_mass",
    mask=None,
    mask_options={},
    centering_options={},
    sample_det_cent_main_detector=None,
    sample_det_cent_wing_detector=None,
    solid_angle_method="VerticalTube",
) -> Tuple[float, float, float]:
    """Finds the beam center in a 2D SANS data set.
    This is based on (and uses) :func:`drtsans.find_beam_center`

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventWorkspace
    method: str
        Method to calculate the beam center. Available methods are:
        - 'center_of_mass', invokes :ref:`FindCenterOfMassPosition <algm-FindCenterOfMassPosition-v1>`.
    mask: mask file path, `MaskWorkspace``, :py:obj:`list`.
        Mask to be passed on to ~drtsans.mask_utils.mask_apply.
    mask_options: dict
        Additional arguments to be passed on to ~drtsans.mask_utils.mask_apply.
    centering_options: dict
        Arguments to be passed on to the centering method.
    sample_det_cent_wing_detector : float
        :ref:`sample to detector center distance <devdocs-standardnames>`,
        in meters, of the wing detector.
    solid_angle_method: bool, str specify which solid angle correction is needed

    Returns
    -------
    tuple
        Three float numbers:
        ``(center_x, center_y, center_y_wing)`` corrected for gravity.
        ``center_y_wing`` is used to correct BIOSANS wing detector Y position.
    """
    ws = mtd[str(input_workspace)]

    # find the center on the main detector
    center_x, center_y, fit_results = bf.find_beam_center(
        ws,
        method,
        mask,
        mask_options=mask_options,
        centering_options=centering_options,
        solid_angle_method=solid_angle_method,
    )

    # get the distance to center of the main and wing detectors
    if sample_det_cent_main_detector is None or sample_det_cent_main_detector == 0.0:
        sample_det_cent_main_detector = (
            ws.getInstrument().getComponentByName("detector1").getPos().norm()
        )

    if sample_det_cent_wing_detector is None or sample_det_cent_wing_detector == 0.0:
        sample_det_cent_wing_detector = (
            ws.getInstrument().getComponentByName("wing_detector").getPos().norm()
        )
        if sample_det_cent_wing_detector == 0.0:
            try:  # old IDF
                sample_det_cent_wing_detector = (
                    ws.getInstrument().getComponentByName("wing_tube_0").getPos().norm()
                )
            except AttributeError:  # new IDF
                sample_det_cent_wing_detector = (
                    ws.getInstrument().getComponentByName("bank49").getPos().norm()
                )

    center_y_wing = _beam_center_gravitational_drop(
        ws, center_y, sample_det_cent_main_detector, sample_det_cent_wing_detector
    )

    logger.information(
        "Beam Center: x={:.3} y={:.3} y_gravity={:.3}.".format(
            center_x, center_y, center_y_wing
        )
    )
    return center_x, center_y, center_y_wing, fit_results


def center_detector(input_workspace, center_x, center_y, center_y_wing):
    """Center the detector and adjusts the height for both the main and wing detectors
    This uses :func:`drtsans.center_detector`

    **Mantid algorithms used:**
    :ref:`MoveInstrumentComponent <algm-MoveInstrumentComponent-v1>`

    Parameters
    ----------
    input_workspace : ~mantid.api.MatrixWorkspace, str
        The workspace to be centered
    center_x : float
        The x-coordinate of the beam center in meters
    center_y : float
        The y-coordinate of the beam center on the main detector in meters
    center_y_wing : float
        The y-coordinate of the beam center on the wing detector in meters

    Returns
    -------
    ~mantid.api.MatrixWorkspace
        reference to the corrected ``input_workspace``
    """
    # move the main detector
    bf.center_detector(input_workspace, center_x, center_y)

    # move the wing detector for the gravity drop
    # the x position of the wing is calibrated differently
    bf.center_detector(input_workspace, 0, center_y_wing, component="wing_detector")
