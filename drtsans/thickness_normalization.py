from mantid.api import mtd

__all__ = ["normalize_by_thickness"]


def normalize_by_thickness(input_workspace, thickness):
    r"""Normalize input workspace by thickness

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
    thickness: float
        Thickness of the sample in centimeters.

    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """
    if thickness <= 0.0:
        msg = "Sample thickness should be positive. Got {}".format(thickness)
        raise ValueError(msg)
    out = mtd[str(input_workspace)]
    out /= thickness
    return out
