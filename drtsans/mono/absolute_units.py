import sys

import drtsans.geometry

# https://docs.mantidproject.org/nightly/algorithms/CreateSingleValuedWorkspace-v1.html
# https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html
# https://docs.mantidproject.org/nightly/algorithms/Divide-v1.html
# https://docs.mantidproject.org/nightly/algorithms/GroupDetectors-v1.html
from mantid.simpleapi import (
    mtd,
    CreateSingleValuedWorkspace,
    DeleteWorkspace,
    Divide,
    GroupDetectors,
)

from drtsans.settings import unique_workspace_dundername
from drtsans.mask_utils import circular_mask_from_beam_center, masked_detectors

__all__ = [
    "empty_beam_scaling",
]


def empty_beam_intensity(
    empty_beam_workspace,
    beam_radius=None,
    unit="mm",
    roi=None,
    attenuator_coefficient=1.0,
    attenuator_error=0.0,
    output_workspace=None,
):
    r"""Calculate the intensity impinging on the detector, taking into account attenuation.

    It is assumed the center of the detector has been moved to coincide with the center of the beam.

    Parameters
    ----------
    empty_beam_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        Attenuated intensities collected at the detector with and empty beam.
    beam_radius: float
        Radius of the beam at the detector. If None, it will be estimated with the sample and source apertures.
    unit: str
        Units for the beam radius, either meters ('m') or mili-miters ('mm').
    roi: file path, MaskWorkspace, list
        Region of interest where to collect intensities. If :py:obj:`list`, it is a list of detector ID's.
        This option overrides beam radius.
    attenuator_coefficient: float
        Fraction of the neutrons allowed to pass through the attenuator. Assumed wavelength independent.
    attenuator_error: float
        Estimated error for the attenuator coefficient.
    output_workspace: str
        Name of the workspace containing the calculated intensity. If :py:obj:`None`, a random hidden name
        will be automatically provided.

    Returns
    -------
    ~mantid.api.MatrixWorkspace
        intensity spectrum with one bin only
    """
    if output_workspace is None:
        output_workspace = unique_workspace_dundername()

    # Obtain the beam radius from the logs or calculate from the source and sample apertures
    if beam_radius is None:
        beam_radius = drtsans.geometry.beam_radius(empty_beam_workspace, unit=unit)

    det_ids = circular_mask_from_beam_center(
        empty_beam_workspace, beam_radius, unit=unit
    )

    # Warn when having too many pixels masked within the beam radius
    if len(masked_detectors(empty_beam_workspace, det_ids)) > len(det_ids) / 2:
        msg = (
            "More than half of the detectors within a radius of {:.2f} {} ".format(
                beam_radius, unit
            )
            + "from the beam center are masked in the empty beam workspace"
        )
        sys.stdout.write("Warning: " + msg)

    # Integrate the intensity within the beam radius, then divide by the attenuation factor
    GroupDetectors(
        InputWorkspace=empty_beam_workspace,
        DetectorList=det_ids,
        OutputWorkspace=output_workspace,
    )
    chi = CreateSingleValuedWorkspace(
        DataValue=attenuator_coefficient, ErrorValue=attenuator_error
    )
    Divide(
        LHSWorkspace=output_workspace,
        RHSWorkspace=chi,
        OutputWorkspace=output_workspace,
    )

    return mtd[output_workspace]


def empty_beam_scaling(
    input_workspace,
    empty_beam_workspace,
    beam_radius=None,
    unit="mm",
    attenuator_coefficient=1.0,
    attenuator_error=0.0,
    output_workspace=None,
):
    r"""
    Normalize input workspace by the intensity impinging on the detector for an empty beam run,
    taking into account attenuation.

    **Mantid Algorithms used:**
    :ref:`Divide <algm-Divide-v1>`,
    :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`,

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        Workspace to be normalized
    empty_beam_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        Attenuated intensities collected at the detector with and empty beam.
    beam_radius: float
        Radius of the beam at the detector. If None, it will be estimated with the sample and source apertures.
    unit: str
        Units for the beam radius, either meters ('m') or mili-miters ('mm').
    attenuator_coefficient: float
        Fraction of the neutrons allowed to pass through the attenuator. Assumed wavelength independent.
    attenuator_error: float
        Estimated error for the attenuator coefficient.
    output_workspace: str
        Name of the normalized workspace. If ``None``, then the name of ``input_workspace`` will be used,
        thus overwriting ``input_workspace``.

    Returns
    -------
    ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)

    # Calculate the intensity impinging on the detector, taking into account attenuation.
    beam_intensity = unique_workspace_dundername()  # temporary workspace
    empty_beam_intensity(
        empty_beam_workspace,
        beam_radius=beam_radius,
        unit=unit,
        attenuator_coefficient=attenuator_coefficient,
        attenuator_error=attenuator_error,
        output_workspace=beam_intensity,
    )

    # Divide the sample intensity by the empty beam intensity
    Divide(
        LHSWorkspace=input_workspace,
        RHSWorkspace=beam_intensity,
        OutputWorkspace=output_workspace,
    )

    DeleteWorkspace(
        Workspace=beam_intensity
    )  # the temporary workspace is not needed anymore
    return str(output_workspace)
