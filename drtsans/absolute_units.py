r""" Links to Mantid algorithms
DeleteWorkspace <https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html>
Divide          <https://docs.mantidproject.org/nightly/algorithms/Divide-v1.html>
Multiply        <https://docs.mantidproject.org/nightly/algorithms/Multiply-v1.html>
"""
from mantid.simpleapi import DeleteWorkspace, Divide, Multiply
from mantid.dataobjects import WorkspaceSingleValue

# drtsans imports
from drtsans.settings import unique_workspace_dundername as uwd  # pylint: disable=W0404

__all__ = [
    "standard_sample_scaling",
]


def standard_sample_scaling(input_workspace, f, f_std, output_workspace=None):
    r"""
    Normalize input workspace using a calibrated standard sample

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
        Workspace to be normalized
    f: ~mantid.api.WorkspaceSingleValue
        Level of flat scattering
    f_std: ~mantid.api.WorkspaceSingleValue
        Known value of the scattering level of the material
    output_workspace: ~mantid.api.MatrixWorkspace
        Name of the normalized workspace. If ``None``, then the name of ``input_workspace`` will be used,
        thus overwriting ``input_workspace``.
    Returns
    -------
        ~mantid.api.MatrixWorkspace
    """
    if not isinstance(f, WorkspaceSingleValue):
        raise TypeError("f is not of type WorkspaceSingleValue")
    if not isinstance(f_std, WorkspaceSingleValue):
        raise TypeError("f_std is not of type WorkspaceSingleValue")

    if output_workspace is None:
        output_workspace = str(input_workspace)

    scaling_factor = Divide(LHSWorkspace=f_std, RHSWorkspace=f, OutputWorkspace=uwd())
    output_workspace = Multiply(
        LHSWorkspace=input_workspace, RHSWorkspace=scaling_factor
    )
    DeleteWorkspace(Workspace=scaling_factor)
    return output_workspace
