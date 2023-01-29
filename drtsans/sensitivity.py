import os
import numpy as np
from drtsans.settings import unique_workspace_name as uwn
from drtsans.path import exists as path_exists

r"""
Links to mantid algorithms
https://docs.mantidproject.org/nightly/algorithms/CloneWorkspace-v1.html
https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html
https://docs.mantidproject.org/nightly/algorithms/Divide-v1.html
https://docs.mantidproject.org/nightly/algorithms/LoadNexusProcessed-v2.html
https://docs.mantidproject.org/nightly/algorithms/MaskDetectors-v1.html
https://docs.mantidproject.org/nightly/algorithms/MaskDetectorsIf-v1.html
https://docs.mantidproject.org/nightly/algorithms/ReplaceSpecialValues-v1.html
https://docs.mantidproject.org/nightly/algorithms/SaveNexusProcessed-v1.html
https://docs.mantidproject.org/nightly/algorithms/Integration-v1.html
https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html
"""
from mantid.simpleapi import (
    mtd,
    CloneWorkspace,
    CalculateEfficiency,
    DeleteWorkspace,
    Divide,
    LoadNexusProcessed,
    MaskDetectors,
    MaskDetectorsIf,
    ReplaceSpecialValues,
    SaveNexusProcessed,
    Integration,
    CreateWorkspace,
)

__all__ = ["load_sensitivity_workspace", "apply_sensitivity_correction"]


def load_sensitivity_workspace(sensitivity_filename, output_workspace):
    """Loads sensitivity workspace. Useful to avoid multiple loads from disk.

    **Mantid algorithms used:**
    :ref:`LoadNexusProcessed <algm-LoadNexusProcessed-v1>`,

    Parameters
    ----------
    sensitivity_filename: str
        file containing previously calculated sensitivity correction
    output_workspace: str, ~mantid.api.MatrixWorkspace
        workspace containing previously calculated sensitivity correction. This
        overrides the sensitivity_filename if both are provided.
    """
    if not path_exists(sensitivity_filename):
        msg = 'Cannot find file "{}"'.format(sensitivity_filename)
        raise RuntimeError(msg)
    LoadNexusProcessed(
        Filename=sensitivity_filename,
        OutputWorkspace=output_workspace,
        LoadHistory=False,
    )

    # nans in workspace to masked pixels
    mask_pixels_with_nan(output_workspace)

    return mtd[output_workspace]


# flake8: noqa: C901
def apply_sensitivity_correction(
    input_workspace,
    sensitivity_filename=None,
    sensitivity_workspace=None,
    min_threshold=None,
    max_threshold=None,
    output_workspace=None,
):
    """Apply a previously calculated sensitivity correction

    **Mantid algorithms used:**
    :ref:`CloneWorkspace <algm-CloneWorkspace-v1>`,
    :ref:`DeleteWorkspace <algm-DeleteWorkspace-v1>`,
    :ref:`Divide <algm-Divide-v1>`,
    :ref:`LoadNexusProcessed <algm-LoadNexusProcessed-v1>`,
    :ref:`MaskDetectors <algm-MaskDetectors-v1>`
    :ref:`MaskDetectorsIf <algm-MaskDetectorsIf-v1>`

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
        workspace to apply the correction to
    sensitivity_filename: str
        file containing previously calculated sensitivity correction
    sensitivity_workspace: str, ~mantid.api.MatrixWorkspace
        workspace containing previously calculated sensitivity correction. This
        overrides the sensitivity_filename if both are provided.
    min_threshold: float or None
        if not None, the data will be masked if the sensitivity
        is below this threshold
    max_threshold: float or None
        if not None, the data will be masked if the sensitivity
        is above this threshold
    output_workspace:  ~mantid.api.MatrixWorkspace
        corrected workspace. This is the input workspace by default
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)

    cleanupSensitivity = False
    if (
        sensitivity_workspace is None or str(sensitivity_workspace) not in mtd
    ):  # load the file
        if sensitivity_workspace is None:
            sensitivity_workspace = os.path.split(sensitivity_filename)[-1]
            sensitivity_workspace = sensitivity_workspace.split(".")[0]
        cleanupSensitivity = True
        load_sensitivity_workspace(sensitivity_filename, sensitivity_workspace)
    if (not sensitivity_workspace) or (str(sensitivity_workspace) not in mtd):
        raise RuntimeError("No sensitivity workspace provided")

    if str(input_workspace) != str(output_workspace):
        CloneWorkspace(InputWorkspace=input_workspace, OutputWorkspace=output_workspace)
    MaskDetectors(Workspace=output_workspace, MaskedWorkspace=sensitivity_workspace)

    temp_sensitivity = uwn(prefix="__sensitivity_")
    CloneWorkspace(
        InputWorkspace=sensitivity_workspace, OutputWorkspace=temp_sensitivity
    )
    if min_threshold is not None:
        MaskDetectorsIf(
            InputWorkspace=temp_sensitivity,
            Operator="LessEqual",
            Value=min_threshold,
            OutputWorkspace=temp_sensitivity,
        )
    if max_threshold is not None:
        MaskDetectorsIf(
            InputWorkspace=temp_sensitivity,
            Operator="GreaterEqual",
            Value=max_threshold,
            OutputWorkspace=temp_sensitivity,
        )
    Divide(
        LHSWorkspace=output_workspace,
        RHSWorkspace=temp_sensitivity,
        OutputWorkspace=output_workspace,
    )
    DeleteWorkspace(temp_sensitivity)

    if cleanupSensitivity:
        DeleteWorkspace(sensitivity_workspace)

    # set empty units
    mtd[output_workspace].setYUnit("")

    return mtd[output_workspace]


def mask_pixels_with_nan(sensitivity_workspace):
    """Mask pixels of data set to NaN.

    Parameters
    ----------
    sensitivity_workspace :  ~mantid.api.MatrixWorkspace
        sensitivity workspace

    Returns
    -------
    ~mantid.api.MatrixWorkspace
        Workspace with the mask bit set
    """
    # value to convert nans to as an intermediate step
    BAD_PIXEL = 1.0e10

    # convert nan's to non-physical value for sensitivity
    ReplaceSpecialValues(
        InputWorkspace=sensitivity_workspace,
        OutputWorkspace=sensitivity_workspace,
        NaNValue=BAD_PIXEL,
    )

    # mask the "bad" pixels
    temp_sensitivity = MaskDetectorsIf(
        InputWorkspace=sensitivity_workspace,
        Operator="GreaterEqual",
        Value=BAD_PIXEL,
        Mode="SelectIf",
        OutputWorkspace=sensitivity_workspace,
    )

    ReplaceSpecialValues(
        InputWorkspace=sensitivity_workspace,
        OutputWorkspace=sensitivity_workspace,
        BigNumberThreshold=BAD_PIXEL - 1.0,
        BigNumberValue=1.0,
    )
