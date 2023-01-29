import numpy as np

r""" Links to Mantid algorithms
ExtractMask          <https://docs.mantidproject.org/nightly/algorithms/ExtractMask-v1.html>
FindDetectorsInShape <https://docs.mantidproject.org/nightly/algorithms/FindDetectorsInShape-v1.html>
LoadMask             <https://docs.mantidproject.org/nightly/algorithms/LoadMask-v1.html>
LoadNexusProcessed   <https://docs.mantidproject.org/nightly/algorithms/LoadNexusProcessed-v2.html>
MaskAngle            <https://docs.mantidproject.org/nightly/algorithms/MaskAngle-v1.html>
MaskBTP              <https://docs.mantidproject.org/nightly/algorithms/MaskBTP-v1.html>
MaskDetectors        <https://docs.mantidproject.org/nightly/algorithms/MaskDetectors-v1.html>
MaskSpectra          <https://docs.mantidproject.org/nightly/algorithms/MaskSpectra-v1.html>
"""
from mantid.simpleapi import (
    ExtractMask,
    FindDetectorsInShape,
    LoadMask,
    logger,
    MaskBTP,
    MaskDetectors,
    MaskSpectra,
    LoadNexusProcessed,
    MaskAngle,
)
from mantid.api import mtd, MatrixWorkspace, IEventWorkspace
import os

# drtsans imports
from drtsans.settings import (
    unique_workspace_dundername,
    unique_workspace_dundername as uwd,
)

__all__ = ["apply_mask", "circular_mask_from_beam_center"]


def mask_as_numpy_array(input_workspace, invert=False):
    r"""
    Array of mask or roi (region-of-interest) boolean values for the pixel detectors.

    When ``invert=False``, a ``True`` value indicates a masked pixel detector. When ``invert=True``, a ``True``
    value indicates an unmasked pixel detector. Option ``invert=True`` in indicated when working with
    region-of-interest areas of the detector.

    Parameters
    ----------
    input_workspace: str, , ~mantid.api.MatrixWorkspace, ~mantid.api.IEventWorkspace
    invert: bool
        Invert mask values.

    Returns
    -------
    :ref:`~numpy.ndarray`
        Array of boolean values, with ``True`` representing masked spectra.
    """
    input_workspace = mtd[str(input_workspace)]  # handle to the workspace
    mask = [
        input_workspace.getDetector(i).isMasked()
        for i in range(input_workspace.getNumberHistograms())
    ]
    mask = np.asarray(mask)
    return mask if invert is False else np.invert(mask)


def masked_indexes(input_workspace, invert=False):
    r"""
    Return indexes for either masked or unmasked pixel detectors.

    When ``invert=False``, indexes for all masked pixel detectors are returned. When ``invert=True``, indexes for
    all un masked pixel detectors are returned. Option ``invert=True`` in indicated when working with
    region-of-interest areas of the detector.

    Parameters
    ----------
    input_workspace: str, , ~mantid.api.MatrixWorkspace, ~mantid.api.IEventWorkspace
    invert: bool
        return indexes for unmasked pixel detectors.

    Returns
    -------
    :ref:`~numpy.ndarray`
        Array of integers
    """
    mask_array = mask_as_numpy_array(input_workspace, invert=invert)
    return np.where(mask_array)[0]


def apply_mask(input_workspace, mask=None, panel=None, **btp):
    r"""
    Apply a mask to a workspace.

    The function accepts a path to a mask file, a MaskWorkspace, or options
    to algorithm :ref:`MaskBTP <algm-MaskBTP-v1>`.

    Parameters
    ----------
    input_workspace: str, ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Workspace to be masked
    mask: mask file path, ~mantid.api.MaskWorkspace, :py:obj:`list`
        Additional mask to be applied. If :py:obj:`list`, it is a list of
        detector ID's. If `None`, it is expected that `maskbtp` is not empty.
    panel: str
        Either 'front' or 'back' to mask a whole panel
    btp: dict
        Options to Mantid algorithm :ref:`MaskBTP <algm-MaskBTP-v1>` or :ref:`MaskAngle <algm-MaskAngle-v1>`.
        Will be used if  ``mask=None``
    """
    input_workspace = str(input_workspace)
    # instrument = mtd[input_workspace].getInstrument().getName()
    if mask is not None:
        if isinstance(mask, str):
            if os.path.splitext(mask)[1] == ".xml":
                # mask_workspace = LoadMask(Instrument=instrument, InputFile=mask,
                #                           RefWorkspace=input_workspace,
                #                           OutputWorkspace=unique_workspace_dundername())
                mask_workspace = load_mask_xml(mask, input_workspace)
            else:
                mask_workspace = load_mask(mask)
            MaskDetectors(Workspace=input_workspace, MaskedWorkspace=mask_workspace)
            mask_workspace.delete()  # delete temporary workspace
        elif isinstance(mask, MatrixWorkspace):
            MaskDetectors(Workspace=input_workspace, MaskedWorkspace=mask)
        elif isinstance(mask, list):
            MaskDetectors(Workspace=input_workspace, DetectorList=mask)
    if panel:
        MaskBTP(Workspace=input_workspace, Components=panel + "-panel")
    if bool(btp):
        min_angle = btp.pop("MinAngle", None)
        max_angle = btp.pop("MaxAngle", None)
        angle = btp.pop("Angle", "TwoTheta")
        if min_angle is not None or max_angle is not None:
            MaskAngle(
                Workspace=input_workspace,
                MinAngle=min_angle,
                MaxAngle=max_angle,
                Angle=angle,
            )
        if bool(btp):  # see if any parameters are left
            print(
                "Try to mask BTP to workspace {} with {}".format(input_workspace, btp)
            )
            MaskBTP(Workspace=input_workspace, **btp)


def load_mask(mask_file="", output_workspace=None):
    r"""
    Load mask file in a workspace

    Parameters
    ----------

    mask_file: mask file path
        path to mask file

    output_workspace:
        Name of the output ~mantid.api.MatrixWorkspace. If ``None``, a random name will be provided for the workspace.

    Returns
    --------
    str, ~mantid.api.MatrixWorkspace Workspace
    """
    if not output_workspace:
        output_workspace = unique_workspace_dundername()
    mask_workspace = LoadNexusProcessed(
        Filename=mask_file, OutputWorkspace=output_workspace
    )
    if isinstance(mask_workspace, IEventWorkspace):
        logger.warning(
            "Storing the mask on an EventWorkspace is inefficient. \
                        Consider saving as a histogram with one bin."
        )
    return mask_workspace


def load_mask_xml(mask_file, ref_workspace, output_workspace=None):
    """Load mask file in a workspace

    Parameters
    ----------
    mask_file
    ref_workspace
    output_workspace

    Returns
    -------

    """

    # Create output MaskWorkspace name if not specified
    if not output_workspace:
        output_workspace = unique_workspace_dundername()

    # Get instrument name
    ref_workspace = str(ref_workspace)
    instrument = mtd[ref_workspace].getInstrument().getName()

    # Load
    mask_workspace = LoadMask(
        Instrument=instrument,
        InputFile=mask_file,
        RefWorkspace=ref_workspace,
        OutputWorkspace=output_workspace,
    )

    return mask_workspace


def mask_spectra_with_special_values(input_workspace, output_workspace=None):
    r"""
    Mask spectra in a workspace containing non-finite values.

    Non-finite values are evaluated with `numpy.isfinite`

    Parameters
    ----------
    input_workspace: str, MatrixWorkspace
    special_values: list
        List of string representations for special `float` values. The special value can be obtained by applying
        `float` to the string, e.g. float('nan').
    output_workspace : str
        Name of the normalized workspace. If None, the name of the input
        workspace is chosen (the input workspace is overwritten).

    Returns
    -------
    list
        Workspace indexes masked. Returns zero if no spectra are masked.
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)
    workspace = mtd[str(input_workspace)]
    intensities = workspace.extractY()
    non_finite_indexes = np.argwhere(
        np.isfinite(np.sum(intensities, axis=-1))
        == False  # NOTE: this is a awkward way to do it
    )  # noqa: E712
    non_finite_indexes = non_finite_indexes.flatten().tolist()
    if len(non_finite_indexes) > 0:
        MaskSpectra(
            InputWorkspace=input_workspace,
            InputWorkspaceIndexType="WorkspaceIndex",
            InputWorkspaceIndexSet=non_finite_indexes,
            OutputWorkspace=output_workspace,
        )
    return len(non_finite_indexes)


def circular_mask_from_beam_center(input_workspace, radius, unit="mm"):
    """
    Find the detectors ID's within a certain radius from the beam center

    Parameters
    ----------
    input_workspace: MatrixWorkspace
        Workspace containing the detector already beam-centered
    radius: float
        Radius of the circle encompassing the detectors of interest.
    unit: str
        Either 'mm' or 'm', unit of the `radius` option.

    Returns
    -------
    numpy.ndarray
        List of detector ID's
    """
    r = radius * 1e-3 if unit == "mm" else radius

    cylinder = r"""
    <infinite-cylinder id="shape">
        <centre x="0.0" y="0.0" z="0.0" />
        <axis x="0.0" y="0.0" z="1" />
        <radius val="{}" />
    </infinite-cylinder>
    <algebra val="shape" />
    """.format(
        r
    )
    det_ids = FindDetectorsInShape(Workspace=input_workspace, ShapeXML=cylinder)
    return det_ids


def masked_detectors(input_workspace, query_ids=None):
    r"""
    List of detector ID's that are masked

    Parameters
    ----------
    input_workspace: str, MatrixWorkspace
        Input workspace to find the detectors
    query_ids: list
        Restrict the search to this list of detector ID's. If `None`, query
        all detectors.

    Returns
    -------
    list
    """
    mask_ws, det_ids = ExtractMask(input_workspace, OutputWorkspace=uwd())
    if query_ids is not None:
        det_ids = sorted(list(set(det_ids) & set(query_ids)))
    mask_ws.delete()
    return det_ids
